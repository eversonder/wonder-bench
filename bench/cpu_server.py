"""Prompt caching and prompt/generation thread splits, measured through llama-server the way a voice loop would call it.

Each config starts a server, sends a ~300-token system prompt + question, then the same system prompt + a new question.
Turn 2 shows what caching saves: only the new tokens should be prefilled.
"""
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from common import save, sysinfo

ROOT = Path(__file__).resolve().parents[1]
SERVER = ROOT / "llama.cpp/build-kai/bin/llama-server"
MODEL = ROOT / "models/qwen/Qwen3-0.6B-Q4_0.gguf"
PORT = 8089
SYSTEM = ("You are Wonder, a calm voice assistant that runs entirely on a small computer in someone's home. "
          "You speak in short, warm sentences. You never pretend to be human. You keep answers under three sentences "
          "unless asked for more. You can set timers, read the time, tell short stories, and answer everyday questions. ") * 5
TURNS = ["How long should I boil an egg?", "And what about a soft one?"]


def post(path, body):
    req = urllib.request.Request(f"http://127.0.0.1:{PORT}{path}", json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.loads(r.read())


def run(t, tb, cache):
    proc = subprocess.Popen([str(SERVER), "-m", str(MODEL), "-t", str(t), "-tb", str(tb), "-c", "2048", "--port", str(PORT)],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        for _ in range(120):  # wait for the model to load
            try:
                if json.loads(urllib.request.urlopen(f"http://127.0.0.1:{PORT}/health", timeout=2).read()).get("status") == "ok":
                    break
            except Exception:
                time.sleep(1)
        rows = []
        for i, q in enumerate(TURNS):
            r = post("/completion", {"prompt": f"{SYSTEM}\nUser: {q}\nWonder:", "n_predict": 64, "temperature": 0,
                                     "cache_prompt": cache, "seed": 1})
            tm = r["timings"]
            rows.append({"threads_gen": t, "threads_batch": tb, "cache_prompt": cache, "turn": i + 1,
                         "prompt_tokens_processed": tm["prompt_n"], "prompt_ms": round(tm["prompt_ms"]),
                         "prompt_tok_s": round(tm["prompt_per_second"], 1), "gen_tok_s": round(tm["predicted_per_second"], 1)})
            print(rows[-1])
        return rows
    finally:
        proc.terminate()
        proc.wait()


def main():
    rows = []
    for t, tb in ((4, 4), (2, 4), (1, 4)):
        rows += run(t, tb, cache=True)
    rows += run(2, 4, cache=False)
    save("cpu-server", {"sysinfo": sysinfo(), "model": MODEL.name, "system_prompt_chars": len(SYSTEM), "rows": rows})


if __name__ == "__main__":
    main()
