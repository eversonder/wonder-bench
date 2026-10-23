#!/usr/bin/env bash
# Qwen3 0.6B / 1.7B at Q4_0 and Q8_0, threads 1-4, with and without KleidiAI. pp512 = prompt, tg128 = generation.
set -euo pipefail
cd "$(dirname "$0")/.."
out=${WB_RESULTS:-results}/$(hostname)-$(date +%Y%m%d)
mkdir -p "$out"
models=$(ls models/qwen/*.gguf | paste -sd, -)
for v in kai ref; do
  echo "llama-bench build-$v"
  llama.cpp/build-$v/bin/llama-bench -m "$models" -p 512 -n 128 -t 1,2,3,4 -r 3 -o json > "$out/llm-$v.raw.json"
done
python - "$out" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1]); rows = []
for v in ("kai", "ref"):
    for r in json.loads((out / f"llm-{v}.raw.json").read_text()):
        rows.append({"build": v, "model": pathlib.Path(r["model_filename"]).stem, "threads": r["n_threads"],
                     "test": f"pp{r['n_prompt']}" if r["n_gen"] == 0 else f"tg{r['n_gen']}",
                     "tok_per_s": round(r["avg_ts"], 1), "stddev": round(r["stddev_ts"], 1)})
(out / "llm.json").write_text(json.dumps({"rows": rows}, indent=2)); print(f"wrote {out}/llm.json")
EOF
