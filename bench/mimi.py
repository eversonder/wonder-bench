"""Mimi decoder on the Pi, frame by frame (80 ms of 24 kHz audio per frame), the way a streaming TTS would call it.

The plan bootstraps Wonder TTS v0 on frozen Mimi, so its decoder has to fit inside the 80 ms frame budget
with room to spare for the TTS backbone. Uses rustymimi (Rust/candle, no torch) because moshi pins old torch.
"""
import os
import time
from pathlib import Path

import numpy as np
import rustymimi

from common import Telemetry, save, sysinfo

MODEL = Path(os.environ.get("WB_MODELS", Path(__file__).resolve().parents[1] / "models")) / "mimi/tokenizer-e351c8d8-checkpoint125.safetensors"


def main(frames=125):
    tok = rustymimi.Tokenizer(str(MODEL))
    codes = np.random.default_rng(0).integers(0, 2048, (1, 8, frames), dtype=np.uint32)
    tok.reset()
    for i in range(5):  # warm-up
        tok.decode_step(np.ascontiguousarray(codes[:, :, i:i + 1]))
    tok.reset()
    lat = []
    with Telemetry() as tel:
        for i in range(frames):
            t0 = time.perf_counter()
            tok.decode_step(np.ascontiguousarray(codes[:, :, i:i + 1]))
            lat.append((time.perf_counter() - t0) * 1000)
    s = sorted(lat)
    save("mimi", {"sysinfo": sysinfo(), "threads_env": os.environ.get("RAYON_NUM_THREADS", "default"),
                  "frames": frames, "frame_budget_ms": 80,
                  "ms_p50": round(s[frames // 2], 1), "ms_p95": round(s[int(frames * 0.95)], 1),
                  "rtf": round(sum(lat) / (frames * 80), 3), **tel.summary()})
    print(f"mimi decode: p50 {s[frames // 2]:.1f} ms, p95 {s[int(frames * 0.95)]:.1f} ms per 80 ms frame")


if __name__ == "__main__":
    main()
