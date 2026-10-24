"""Decoder bake-off: which audio decoder fits Wonder's frame budget on one Pi 5 core, and what it costs in quality.

Mimi's decoder takes 132 ms per 80 ms frame (mimi.py), so a streaming voice can't be built on it as-is.
Every candidate here turns a compact representation (mel frames or codec tokens) back into a waveform.
  speed:   offline RTF (whole utterance) and streaming latency (80 ms of new frames + 240 ms of left context
           per call, or the decoder's own streaming step), on core 3, idle and while Qwen3 decodes on cores 1-2
  quality: resynthesise 20 LibriTTS-R test-clean clips (24 kHz) from their own mels or codes (copy synthesis, an upper
           bound: in TTS a backbone predicts these frames), score UTMOS22 (predicted MOS) and Moonshine-base WER at 16 kHz
Pretrained weights are only measured here, never shipped: Wonder retrains its decoder on Tier A data.
Resynthesised audio lands in out/decoder/<name>/ (untracked) for listening.
"""
import argparse
import json
import math
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from common import Telemetry, save, sysinfo

ROOT = Path(__file__).resolve().parents[1]
MODELS = Path(os.environ.get("WB_MODELS", ROOT / "models"))
DATA = ROOT / "data/libritts-r-test-clean-20"
OUT = ROOT / "out/decoder"


class Dec:
    """sr, fps (frames per second of the decoder input), feats(wav at sr) -> frames on the last axis,
    decode(frames) -> mono float32 audio; step(one frame) for decoders with their own streaming state."""
    step = None
    reset = None

    def chunk(self, f, a, b):
        return f[..., a:b]

    def frames(self, f):
        return f.shape[-1]


def _torch(threads):
    import torch
    torch.set_num_threads(threads)
    return torch


def vocos_mel(threads, onnx=False):
    torch = _torch(threads)
    from vocos import Vocos
    v = Vocos.from_pretrained("charactr/vocos-mel-24khz").eval()
    d = Dec()
    d.sr, d.fps, d.params_m = 24000, 24000 / 256, sum(p.numel() for p in [*v.backbone.parameters(), *v.head.parameters()]) / 1e6
    d.feats = lambda w: v.feature_extractor(torch.from_numpy(w)[None]).numpy()
    if not onnx:
        d.decode = lambda f: v.decode(torch.from_numpy(f))[0].numpy()
        return d
    import onnxruntime as ort
    # sherpa-onnx export of the same weights; ONNX has no iSTFT, so the graph stops at magnitude and phase
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    s = ort.InferenceSession(str(MODELS / "vocoders/vocos_24khz.onnx"), so, providers=["CPUExecutionProvider"])
    win = torch.hann_window(1024)

    def decode(f):
        mag, x, y = s.run(None, {"mels": f})
        spec = torch.from_numpy(mag) * torch.complex(torch.from_numpy(x), torch.from_numpy(y))
        return torch.istft(spec, 1024, 256, 1024, win, center=True)[0].numpy()
    d.decode = decode
    return d


def vocos_encodec(threads):
    torch = _torch(threads)
    from vocos import Vocos
    v = Vocos.from_pretrained("charactr/vocos-encodec-24khz").eval()
    bw = torch.tensor([2])  # 6 kbps, 8 codebooks
    d = Dec()
    d.sr, d.fps, d.params_m = 24000, 75, sum(p.numel() for p in [*v.backbone.parameters(), *v.head.parameters()]) / 1e6
    d.feats = lambda w: v.feature_extractor(torch.from_numpy(w)[None], bandwidth_id=bw).numpy()
    d.decode = lambda f: v.decode(torch.from_numpy(f), bandwidth_id=bw)[0].numpy()
    return d


def hifigan(version):
    def load(threads):
        torch = _torch(threads)
        import librosa
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
        s = ort.InferenceSession(str(MODELS / f"vocoders/hifigan_{version}.onnx"), so, providers=["CPUExecutionProvider"])
        fb = torch.from_numpy(librosa.filters.mel(sr=22050, n_fft=1024, n_mels=80, fmin=0, fmax=8000))

        def feats(w):  # jik876/hifi-gan meldataset.py: reflect pad, |STFT|, slaney mel, log clamp 1e-5
            y = torch.nn.functional.pad(torch.from_numpy(w)[None, None], (384, 384), mode="reflect")[0]
            spec = torch.stft(y, 1024, 256, 1024, torch.hann_window(1024), center=False, return_complex=True).abs()
            return torch.log((fb @ spec).clamp(min=1e-5)).numpy()
        d = Dec()
        d.sr, d.fps = 22050, 22050 / 256
        d.params_m = os.path.getsize(MODELS / f"vocoders/hifigan_{version}.onnx") / 4e6  # fp32 weights, 4 bytes each
        d.feats = feats
        d.decode = lambda f: s.run(None, {"mel": f})[0][0]
        return d
    return load


def wavtokenizer(threads):
    torch = _torch(threads)
    sys.path.insert(0, str(ROOT / "WavTokenizer"))
    from decoder.pretrained import WavTokenizer
    wt = WavTokenizer.from_pretrained0802(
        str(ROOT / "WavTokenizer/configs/wavtokenizer_smalldata_frame40_3s_nq1_code4096_dim512_kmeans200_attn.yaml"),
        str(MODELS / "wavtokenizer/WavTokenizer_small_600_24k_4096.ckpt")).eval()
    bw = torch.tensor([0])
    d = Dec()
    d.sr, d.fps, d.params_m = 24000, 40, sum(p.numel() for p in [*wt.backbone.parameters(), *wt.head.parameters()]) / 1e6
    d.feats = lambda w: wt.encode_infer(torch.from_numpy(w)[None], bandwidth_id=bw)[1].numpy()  # (1, 1, T) codes
    d.decode = lambda c: wt.decode(wt.codes_to_features(torch.from_numpy(c)), bandwidth_id=bw)[0].numpy()
    return d


def snac(threads):
    torch = _torch(threads)
    from snac import SNAC
    m = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()

    class D(Dec):  # three code levels at 12/23/47 Hz; one frame here is one coarse frame and its finer children
        def chunk(self, f, a, b):
            return [c[..., a << i:b << i] for i, c in enumerate(f)]

        def frames(self, f):
            return f[0].shape[-1]
    d = D()
    d.sr, d.fps, d.params_m = 24000, 24000 / 2048, sum(p.numel() for p in [*m.decoder.parameters(), *m.quantizer.parameters()]) / 1e6
    d.feats = lambda w: [c.numpy() for c in m.encode(torch.from_numpy(w)[None, None])]
    d.decode = lambda f: m.decode([torch.from_numpy(c) for c in f])[0, 0].numpy()
    return d


def mimi(threads):
    import rustymimi
    os.environ.setdefault("RAYON_NUM_THREADS", str(threads))
    tok = rustymimi.Tokenizer(str(MODELS / "mimi/tokenizer-e351c8d8-checkpoint125.safetensors"))
    d = Dec()
    d.sr, d.fps, d.params_m = 24000, 12.5, None
    d.feats = lambda w: tok.encode(np.ascontiguousarray(w[None, None]))  # (1, 8, T) codes
    d.decode = lambda c: tok.decode(np.ascontiguousarray(c))[0, 0]
    d.step = lambda c: tok.decode_step(np.ascontiguousarray(c))
    d.reset = tok.reset
    return d


DECODERS = {
    "vocos-mel": vocos_mel,
    "vocos-mel-onnx": lambda t: vocos_mel(t, onnx=True),
    "vocos-encodec-6k": vocos_encodec,
    "wavtokenizer-40hz": wavtokenizer,
    "hifigan-v1": hifigan("v1"),
    "hifigan-v2": hifigan("v2"),
    "hifigan-v3": hifigan("v3"),
    "snac-24k": snac,
    "mimi": mimi,
}
NOTES = {  # weights licence / training data, from each model card; the decoder is retrained before anything ships
    "vocos-mel": "MIT; LibriTTS", "vocos-mel-onnx": "MIT; LibriTTS (sherpa-onnx export)",
    "vocos-encodec-6k": "MIT; DNS Challenge", "wavtokenizer-40hz": "MIT; LibriTTS",
    "hifigan-v1": "MIT; LJSpeech, one speaker", "hifigan-v2": "MIT; LJSpeech, one speaker",
    "hifigan-v3": "MIT; LJSpeech, one speaker", "snac-24k": "MIT; data not stated", "mimi": "CC-BY-4.0",
}


def utts():
    out = []
    for u, t in (l.split("\t") for l in (DATA / "transcripts.tsv").read_text().splitlines()):
        w = sf.read(DATA / f"{u}.wav", dtype="float32")[0]
        out.append((u, 0.7 * w / max(np.abs(w).max(), 1e-6), t))  # -3 dBFS peak: inside every decoder's training levels
    return out


def at(wav, sr):
    import librosa
    return librosa.resample(wav, orig_sr=24000, target_sr=sr) if sr != 24000 else wav


def speed(name, threads, n_chunks=100):
    import torch
    d = DECODERS[name](threads)
    clips = [at(w, d.sr) for _, w, _ in utts()[:3]]
    with torch.inference_mode():
        feats = [d.feats(w) for w in clips]  # encoding is not timed: in TTS the backbone makes these frames
        Path("/proc/self/clear_refs").write_text("5")  # restart the peak-RSS count: encoder activations aren't decoder cost
        d.decode(d.chunk(feats[0], 0, 8))  # warm-up
        wall = audio = 0.0
        for w, f in zip(clips, feats):
            t0 = time.perf_counter()
            y = d.decode(f)
            wall += time.perf_counter() - t0
            audio += len(y) / d.sr
        f = feats[0]
        new = math.ceil(0.08 * d.fps)
        ctx = 0 if d.step else math.ceil(0.24 * d.fps)
        lat = []
        if d.step:
            d.reset()
        for i in range(n_chunks):
            a = ctx + (i * new) % (d.frames(f) - ctx - new)
            t0 = time.perf_counter()
            if d.step:
                d.step(d.chunk(f, a, a + new))
            else:
                d.decode(d.chunk(f, a - ctx, a + new))
            # cost of a call yielding exactly 80 ms; ponytail: assumes cost is linear in frames decoded
            lat.append((time.perf_counter() - t0) * 1000 * (ctx + 0.08 * d.fps) / (ctx + new))
    lat.sort()
    return {"decoder": name, "params_m": round(d.params_m, 1) if d.params_m else None, "sr": d.sr, "frame_hz": round(d.fps, 2),
            "offline_rtf": round(wall / audio, 3),
            "stream_ms_per_80ms_p50": round(lat[n_chunks // 2], 1), "stream_ms_per_80ms_p95": round(lat[int(n_chunks * 0.95)], 1),
            "stream_chunk": f"{new} new + {ctx} context frames" if not d.step else "decoder's own streaming step",
            "threads": threads, "peak_rss_mb": round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024),
            "weights": NOTES[name]}


def resynth(name):
    import torch
    d = DECODERS[name](4)
    (OUT / name).mkdir(parents=True, exist_ok=True)
    with torch.inference_mode():
        for u, w, _ in utts():
            sf.write(OUT / name / f"{u}.wav", np.clip(d.decode(d.feats(at(w, d.sr))), -1, 1), d.sr)
    return {"decoder": name}


def score(names):
    """UTMOS22-strong and Moonshine-base WER on each folder of resynthesised clips, plus the original recordings."""
    import librosa
    import torch
    from moonshine_voice import ModelArch, Transcriber, get_model_for_language
    from asr import norm, wer
    utmos = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    path, arch = get_model_for_language("en", ModelArch.BASE)
    asr = Transcriber(model_path=path, model_arch=arch)
    data = utts()
    rows = []
    for name in ["original", *names]:
        mos, hyps = [], []
        for u, w, _ in data:
            sr = 24000
            if name != "original":
                w, sr = sf.read(OUT / name / f"{u}.wav", dtype="float32")
            w = librosa.resample(w, orig_sr=sr, target_sr=16000)  # both scorers are 16 kHz models
            with torch.inference_mode():
                mos.append(utmos(torch.from_numpy(w)[None], 16000).item())
            hyps.append(" ".join(l.text for l in asr.transcribe_without_streaming(w.tolist(), 16000).lines))
        rows.append({"decoder": name, "utmos": round(float(np.mean(mos)), 2),
                     "wer_pct": round(100 * wer([norm(t) for *_, t in data], [norm(h) for h in hyps]), 1)})
        print(rows[-1], flush=True)
    asr.close()
    return rows


def child(args):
    """Run one job in this process and print its result as JSON on the last line."""
    try:
        out = speed(args.engine, args.threads) if args.job == "speed" else resynth(args.engine)
    except Exception as e:  # missing package or model: report, don't die
        out = {"decoder": args.engine, "skipped": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(out))


def run(name, job, cores, threads=1):
    out = subprocess.run(["taskset", "-c", cores, sys.executable, "-W", "ignore", __file__, "--engine", name, "--job", job,
                          "--threads", str(threads)], capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"decoder": name, "skipped": out.stderr[-300:]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", help=argparse.SUPPRESS)
    ap.add_argument("--job", default="speed", help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--only", help="comma-separated decoders (saves as decoder-<names>.json)")
    args = ap.parse_args()
    if args.engine:
        return child(args)
    from tts import lm_load
    names = args.only.split(",") if args.only else list(DECODERS)
    rows = []
    os.sched_setaffinity(0, {0})  # telemetry's vcgencmd forks stay off the timed core 3
    for load in (False, True):
        bg = lm_load() if load else None
        time.sleep(3 if load else 0)
        try:
            with Telemetry() as tel:
                for name in names:  # core 3 to itself, as in the plan's layout
                    rows.append({**run(name, "speed", "3", args.threads), "lm_load": load and bg.poll() is None})
                    print(rows[-1], flush=True)
        finally:
            if bg:
                bg.kill()
        rows.append({"decoder": "telemetry", "lm_load": load, **tel.summary()})
    os.sched_setaffinity(0, {0, 1, 2, 3})
    try:
        made = {n: run(n, "resynth", "0-3") for n in names}
        quality = score([n for n, r in made.items() if "skipped" not in r]) + [r for r in made.values() if "skipped" in r]
    except Exception as e:  # scoring needs the network and two more models; never lose the speed rows to it
        quality = [{"decoder": "score", "skipped": f"{type(e).__name__}: {e}"[:300]}]
    save(f"decoder-{'-'.join(names)}" if args.only else "decoder", {
        "sysinfo": sysinfo(), "rows": rows, "quality": quality, "budget": "80 ms of audio per 80 ms frame on one core, "
        "shared with the TTS backbone; the whole voice must hold RTF <= 0.3 under LM load",
        "data": "LibriTTS-R test-clean (24 kHz, CC BY 4.0), peak -3 dBFS, resampled to each decoder's rate; copy "
                "synthesis: mel rows decode exact mels, codec rows their own quantised codes (vocos-encodec 6 kbps, "
                "wavtokenizer 1x4096 at 40 Hz, snac 3 levels, mimi 8 codebooks); scored at 16 kHz (content above 8 kHz "
                "is not judged) with UTMOS22-strong via SpeechMOS v1.2.0 and Moonshine base WER; hifigan weights are "
                "single-speaker LJSpeech"})


if __name__ == "__main__":
    main()
