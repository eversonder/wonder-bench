"""Listening while thinking: Moonshine on core 0 while Qwen3 decodes on cores 1-2, against each alone.

The voice loop runs these at the same time, so the number that matters is how much each slows the other.
"""
import json
import subprocess
import time
from pathlib import Path

import soundfile as sf
from moonshine_voice import ModelArch, Transcriber, get_model_for_language

from common import save, sysinfo

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data/librispeech-test-clean-20"
BENCH = ROOT / "llama.cpp/build-kai/bin/llama-bench"
MODEL = ROOT / "models/qwen/Qwen3-0.6B-Q4_0.gguf"


def asr_rtf(arch):
    path, a = get_model_for_language("en", arch)
    t = Transcriber(model_path=path, model_arch=a)
    utts = [sf.read(DATA / f"{l.split(chr(9))[0]}.flac", dtype="float32")[0] for l in (DATA / "transcripts.tsv").read_text().splitlines()]
    t.transcribe_without_streaming(utts[0].tolist(), 16000)
    wall = audio = 0.0
    for w in utts:
        buf = w.tolist()
        t0 = time.perf_counter()
        t.transcribe_without_streaming(buf, 16000)
        wall += time.perf_counter() - t0
        audio += len(w) / 16000
    t.close()
    return round(wall / audio, 4)


def llm_tg(extra=None):
    cmd = ["taskset", "-c", "1,2", str(BENCH), "-m", str(MODEL), "-t", "2", "-p", "0", "-n", "256", "-r", "5", "-o", "json"]  # outlasts the ASR pass
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)


def main():
    import os
    os.sched_setaffinity(0, {0})  # listening gets core 0, as in the plan's layout
    rows = []
    for arch in (ModelArch.TINY_STREAMING, ModelArch.BASE):
        alone = asr_rtf(arch)
        p = llm_tg()
        time.sleep(2)
        busy = asr_rtf(arch)
        out, _ = p.communicate()
        tg_busy = json.loads(out)[0]["avg_ts"]
        p = llm_tg()
        out, _ = p.communicate()
        tg_alone = json.loads(out)[0]["avg_ts"]
        rows.append({"asr": arch.name, "asr_rtf_alone": alone, "asr_rtf_with_llm": busy,
                     "llm_tg_alone": round(tg_alone, 1), "llm_tg_with_asr": round(tg_busy, 1)})
        print(rows[-1])
    save("cpu-contention", {"sysinfo": sysinfo(), "layout": "ASR pinned to core 0, Qwen3-0.6B Q4_0 generation on cores 1-2", "rows": rows})


if __name__ == "__main__":
    main()
