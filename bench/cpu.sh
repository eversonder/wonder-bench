#!/usr/bin/env bash
# "Squeeze the CPU": software-only experiments from the week-1 plan. No reboot, no firmware or boot changes.
# Each step is skipped if its result exists, and starts from a cooled SoC (no cooler yet: treat 4-thread rows as heat-limited).
set -euo pipefail
cd "$(dirname "$0")/.."
. .venv/bin/activate
export PYTHONUNBUFFERED=1
out=results/$(hostname)-$(date +%Y%m%d)
mkdir -p "$out" models/qwen/sweep data/text
bin=llama.cpp/build-kai/bin
temp() { cat /sys/class/thermal/thermal_zone0/temp; }
cool() { while [ "$(temp)" -gt 60000 ]; do sleep 10; done; }
step() { [ -s "$out/$1.json" ] && echo "skip $1" || { cool; echo "== $1 ($(date +%T), $(( $(temp) / 1000 )) °C)"; shift; "$@"; }; }

# tools the main suite didn't need
[ -x $bin/llama-server ] || cmake --build llama.cpp/build-kai -j4 --target llama-server llama-quantize llama-perplexity >/dev/null

# memory bandwidth: the ceiling on token generation
stream() {
  [ -x data/stream ] || { curl -fsSL -o data/stream.c https://www.cs.virginia.edu/stream/FTP/Code/stream.c && gcc -O2 -fopenmp -DSTREAM_ARRAY_SIZE=80000000 data/stream.c -o data/stream; }
  for t in 1 2 4; do echo "threads $t"; OMP_NUM_THREADS=$t data/stream | grep -E "^(Copy|Scale|Add|Triad):"; done > "$out/stream.txt"
  python - "$out" <<'EOF'
import json, re, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows, t = [], None
for line in (out / "stream.txt").read_text().splitlines():
    if line.startswith("threads"): t = int(line.split()[1]); continue
    k, mbs = re.match(r"(\w+):\s+([\d.]+)", line).groups(); rows.append({"threads": t, "kernel": k, "gb_s": round(float(mbs) / 1000, 2)})
(out / "stream.json").write_text(json.dumps({"rows": rows}, indent=2)); print(rows)
EOF
}

# one bf16 source, quantised every way we want to compare (no imatrix, so the comparison is like for like)
quants() {
  src=models/qwen/sweep/Qwen3-0.6B-BF16.gguf
  [ -s $src ] || curl -fsSL -o $src https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-BF16.gguf
  for q in Q4_0 Q4_K_M IQ4_NL Q8_0; do
    [ -s models/qwen/sweep/q-$q.gguf ] || $bin/llama-quantize $src models/qwen/sweep/q-$q.gguf $q >/dev/null 2>&1
  done
  # Qwen3-0.6B ties embeddings: the default keeps that big tensor at Q6_K and reads it every token
  [ -s models/qwen/sweep/q-Q4_0-embd.gguf ] || $bin/llama-quantize --token-embedding-type q4_0 --output-tensor-type q4_0 $src models/qwen/sweep/q-Q4_0-embd.gguf Q4_0 >/dev/null 2>&1
  ls -la models/qwen/sweep/
  m=$(ls models/qwen/sweep/q-*.gguf | paste -sd, -)
  : > "$out/cpu-quants.raw.jsonl"
  for t in 1 2 4; do
    cool
    $bin/llama-bench -m "$m" -p 256 -n 64 -t $t -fa 0,1 -r 3 -o jsonl >> "$out/cpu-quants.raw.jsonl"
  done
  cool
  $bin/llama-bench -m models/qwen/sweep/q-Q4_0.gguf -p 256 -n 0 -t 4 -ub 64,128,256,512 -r 3 -o jsonl >> "$out/cpu-quants.raw.jsonl"
  cool
  $bin/llama-bench -m models/qwen/sweep/q-Q4_0.gguf -p 0 -n 64 -t 2 -mmp 0,1 -r 3 -o jsonl >> "$out/cpu-quants.raw.jsonl"
  python - "$out" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for line in (out / "cpu-quants.raw.jsonl").read_text().splitlines():
    r = json.loads(line)
    rows.append({"model": pathlib.Path(r["model_filename"]).stem, "size_mb": round(r["model_size"] / 2**20), "threads": r["n_threads"],
                 "test": f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}", "flash_attn": r.get("flash_attn"),
                 "n_ubatch": r.get("n_ubatch"), "mmap": r.get("use_mmap"), "tok_per_s": round(r["avg_ts"], 1)})
(out / "cpu-quants.json").write_text(json.dumps({"rows": rows}, indent=2)); print(f"{len(rows)} rows")
EOF
}

# quality cost of each quant: KL divergence from bf16 on public-domain text (Pride and Prejudice, Project Gutenberg)
kld() {
  txt=data/text/pg1342.txt
  [ -s $txt ] || curl -fsSL https://www.gutenberg.org/cache/epub/1342/pg1342.txt | head -c 60000 > $txt
  $bin/llama-perplexity -m models/qwen/sweep/Qwen3-0.6B-BF16.gguf -f $txt -c 512 --chunks 12 -t 4 --kl-divergence-base data/text/base.kld >/dev/null 2>&1
  : > "$out/cpu-kld.txt"
  for f in models/qwen/sweep/q-*.gguf; do
    cool
    echo "== $(basename $f .gguf)" >> "$out/cpu-kld.txt"
    $bin/llama-perplexity -m $f -c 512 --chunks 12 -t 4 --kl-divergence-base data/text/base.kld --kl-divergence 2>&1 | grep -E "Mean +KLD|Same top p|Mean PPL" >> "$out/cpu-kld.txt" || true
  done
  python - "$out" <<'EOF'
import json, re, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows, cur = [], None
for line in (out / "cpu-kld.txt").read_text().splitlines():
    if line.startswith("== "): cur = {"model": line[3:]}; rows.append(cur); continue
    m = re.search(r"Mean +KLD:\s+([\d.]+)", line) or None
    if m: cur["mean_kld"] = float(m.group(1))
    m = re.search(r"Same top p:\s+([\d.]+)", line)
    if m: cur["same_top_token_pct"] = float(m.group(1))
(out / "cpu-kld.json").write_text(json.dumps({"rows": rows, "text": "Pride and Prejudice (public domain), first 60 KB"}, indent=2)); print(rows)
EOF
}

train_variants() { # at 2 threads, where the board holds 2.4 GHz without a cooler
  cd bench
  base="--threads 2 --kinds tts,asr --modes full --no-matmul --steps 5"
  python train.py $base --tag base
  python train.py $base --compile --tag compile
  for d in 0 64 256; do TORCH_MKLDNN_MATMUL_MIN_DIM=$d python train.py $base --tag mindim-$d; done
  [ -e /usr/lib/aarch64-linux-gnu/libjemalloc.so.2 ] || sudo -n apt-get install -y -qq libjemalloc2 >/dev/null
  LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2 OMP_PROC_BIND=close OMP_PLACES=cores python train.py $base --tag jemalloc-bind
  for b in 1 4 8; do python train.py --threads 2 --kinds tts --modes full --no-matmul --steps 5 --batch $b --tag batch-$b; done
  cd ..
  python - "$out" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for f in sorted(out.glob("train-*.json")):
    for r in json.loads(f.read_text())["rows"]:
        rows.append({k: r.get(k) for k in ("tag", "kind", "mode", "threads", "batch", "step_s_median", "gflops_effective", "temp_max_c", "arm_mhz_min")})
(out / "cpu-train.json").write_text(json.dumps({"rows": rows}, indent=2)); print(f"{len(rows)} rows")
EOF
}

step stream         stream
step cpu-quants     quants
step cpu-kld        kld
step cpu-server     bash -c 'cd bench && python cpu_server.py'
step cpu-contention bash -c 'cd bench && python cpu_contention.py'
step cpu-train      train_variants
echo "cpu experiments done"
