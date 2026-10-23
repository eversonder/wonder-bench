"""Text-to-speech on the Pi: time-to-first-audio (TTFA) and real-time factor (RTF), idle and with the LM decoding.

The plan's go/no-go: a TTS that holds RTF <= 0.3 while llama.cpp decodes on two other cores.
Core layout from PLAN.md §6: core 0 I/O+ASR, cores 1-2 LM, core 3 TTS.
Each engine is optional: if its package or model is missing it is reported as skipped, not fatal.
"""
import argparse
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

from common import Telemetry, save, sysinfo

MODELS = Path(os.environ.get("WB_MODELS", Path(__file__).resolve().parents[1] / "models"))
LLAMA = Path(os.environ.get("WB_LLAMA", Path(__file__).resolve().parents[1] / "llama.cpp/build-kai/bin"))

TEXTS = {
    "short": "Forty minutes on the clock. I'll call you when the bread's ready.",
    "medium": ("The kettle is on. Your sister called this morning and said she'd ring back after lunch. "
               "It's going to rain later, so the washing might want to come in. "
               "Would you like me to read the next chapter of your book?"),
    "numbers": "Your appointment is on the 14th of November at 3:45 pm, room 212.",
}


def kokoro(threads):
    import onnxruntime as ort
    from kokoro_onnx import Kokoro
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    sess = ort.InferenceSession(str(MODELS / "kokoro/kokoro-v1.0.int8.onnx"), so, providers=["CPUExecutionProvider"])
    k = Kokoro.from_session(sess, str(MODELS / "kokoro/voices-v1.0.bin"))

    def run(text):  # sentence by sentence, as a real-time voice loop would
        t0, ttfa, n, sr = time.perf_counter(), None, 0, 24000
        for s in [s.strip() + "." for s in text.split(".") if s.strip()]:
            a, sr = k.create(s, voice="af_sarah", speed=1.0, lang="en-us")
            ttfa, n = ttfa or time.perf_counter() - t0, n + len(a)
        return ttfa, time.perf_counter() - t0, n / sr
    return run


def piper(threads):
    from piper import PiperVoice, SynthesisConfig
    v = PiperVoice.load(str(MODELS / "piper/en_US-libritts_r-medium.onnx"))
    cfg = SynthesisConfig(speaker_id=0)

    def run(text):
        t0, ttfa, n, sr = time.perf_counter(), None, 0, 22050
        for ch in v.synthesize(text, syn_config=cfg):
            ttfa, n, sr = ttfa or time.perf_counter() - t0, n + len(ch.audio_float_array), ch.sample_rate
        return ttfa, time.perf_counter() - t0, n / sr
    return run


def pocket(threads):
    import torch
    from pocket_tts import TTSModel
    torch.set_num_threads(threads)
    m = TTSModel.load_model(language="english", quantize=False)
    voice = m.get_state_for_audio_prompt("alba")

    def run(text):
        t0, ttfa, n = time.perf_counter(), None, 0
        for ch in m.generate_audio_stream(voice, text):
            ttfa, n = ttfa or time.perf_counter() - t0, n + ch.numel()
        return ttfa, time.perf_counter() - t0, n / m.sample_rate
    return run


def phonon(threads):
    import ptts
    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))  # ponytail: thread knob undocumented; we pin with taskset regardless
    tts = ptts.TTS("hf://gradium/phonon-31m-en", lang="en", quant="q8")

    def run(text):
        t0, ttfa, n = time.perf_counter(), None, 0
        with tts.stream(text, voice="Easton") as audio:
            for pcm in audio:
                ttfa, n = ttfa or time.perf_counter() - t0, n + len(pcm)
            sr = audio.sample_rate
        return ttfa, time.perf_counter() - t0, n / sr
    return run


ENGINES = {"kokoro-int8": kokoro, "piper-libritts_r": piper, "pocket-tts": pocket, "phonon-31m-q8": phonon}


def lm_load():
    """Keep Qwen3-0.6B decoding on cores 1-2 for the whole measurement, like a live conversation."""
    model = MODELS / "qwen/Qwen3-0.6B-Q8_0.gguf"
    cmd = ["taskset", "-c", "1,2", str(LLAMA / "llama-bench"), "-m", str(model), "-t", "2", "-p", "0", "-n", "256", "-r", "1000"]
    return subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def measure(name, threads, reps):
    try:
        run = ENGINES[name](threads)
        run("Warm up.")
    except Exception as e:  # missing package/model: report, don't die
        return [{"engine": name, "skipped": f"{type(e).__name__}: {e}"[:200]}]
    rows = []
    for label, text in TEXTS.items():
        ttfas, rtfs = [], []
        for _ in range(reps):
            ttfa, wall, secs = run(text)
            ttfas.append(ttfa)
            rtfs.append(wall / secs)
        rows.append({"engine": name, "text": label, "audio_s": round(secs, 2),
                     "ttfa_ms_median": round(sorted(ttfas)[reps // 2] * 1000),
                     "rtf_median": round(sorted(rtfs)[reps // 2], 3), "threads": threads})
    rows[-1]["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", help="run one engine (used internally so each gets a fresh process and clean RSS)")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--load", action="store_true", help="decode Qwen3-0.6B on cores 1-2 while measuring")
    args = ap.parse_args()

    if args.engine:  # child: print rows as JSON on the last line
        print(json.dumps(measure(args.engine, args.threads, args.reps)))
        return

    rows = []
    for load in (False, True):
        bg = lm_load() if load else None
        time.sleep(3 if load else 0)
        try:
            with Telemetry() as tel:
                for name in ENGINES:
                    # TTS gets core 3 to itself, as in the plan's layout
                    out = subprocess.run(["taskset", "-c", "3", sys.executable, __file__, "--engine", name,
                                          "--threads", str(args.threads), "--reps", str(args.reps)],
                                         capture_output=True, text=True)
                    lines = [l for l in out.stdout.splitlines() if l.startswith("[")]
                    got = json.loads(lines[-1]) if lines else [{"engine": name, "skipped": out.stderr[-300:]}]
                    for r in got:
                        r["lm_load"] = load
                        print(r)
                    rows += got
        finally:
            if bg:
                bg.kill()
        rows.append({"engine": "telemetry", "lm_load": load, **tel.summary()})
    save("tts", {"sysinfo": sysinfo(), "texts": TEXTS, "rows": rows,
                 "gate": "RTF <= 0.3 under LM load on one core (PLAN.md week 1 go/no-go)"})


if __name__ == "__main__":
    main()
