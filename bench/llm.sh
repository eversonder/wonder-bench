#!/usr/bin/env bash
# Qwen3 0.6B / 1.7B at Q4_0 and Q8_0, threads 1-4, with and without KleidiAI. pp512 = prompt, tg128 = generation.
# Each (build, thread count) starts from a cooled SoC, so thread scaling isn't confounded with heat.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${WB_RESULTS:-results}/$(hostname)-$(date +%Y%m%d)
mkdir -p "$out"
models=$(ls models/qwen/*.gguf | paste -sd, -)
temp() { cat /sys/class/thermal/thermal_zone0/temp; }
cool() { while [ "$(temp)" -gt 60000 ]; do sleep 10; done; }
: > "$out/llm.raw.jsonl"
for v in kai ref; do
  for t in 1 2 3 4; do
    cool
    start=$(temp)
    echo "llama-bench build-$v -t $t (start $((start / 1000)) °C)"
    llama.cpp/build-$v/bin/llama-bench -m "$models" -p 512 -n 128 -t "$t" -r 3 -o jsonl |
      sed "s/^{/{\"wb_build\":\"$v\",\"wb_start_c\":$((start / 1000)),\"wb_end_c\":TEMP_END,/" > "$out/llm.part"
    sed "s/TEMP_END/$(( $(temp) / 1000 ))/" "$out/llm.part" >> "$out/llm.raw.jsonl"
  done
done
rm -f "$out/llm.part"
python - "$out" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for line in (out / "llm.raw.jsonl").read_text().splitlines():
    r = json.loads(line)
    rows.append({"build": r["wb_build"], "model": pathlib.Path(r["model_filename"]).stem, "threads": r["n_threads"],
                 "test": f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}",
                 "tok_per_s": round(r["avg_ts"], 1), "stddev": round(r["stddev_ts"], 1),
                 "start_c": r["wb_start_c"], "end_c": r["wb_end_c"]})
(out / "llm.json").write_text(json.dumps({"rows": rows}, indent=2)); print(f"wrote {out}/llm.json")
EOF
