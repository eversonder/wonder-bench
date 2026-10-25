"""Exact streaming Vocos: per-layer caches so chunked decoding equals offline Vocos.decode, and what that costs on one core.

Vocos is not causal: a kernel-7 embed conv and 8 ConvNeXt blocks with kernel-7 depthwise convs each look 3 frames ahead,
and the iSTFT's overlap-add needs the next frames too. decoder.py's streaming number re-decoded 240 ms of left context per
call with no right context, so its chunk edges were approximations. Here each same-padded conv keeps its last k-1 inputs,
the residual is taken from the same buffer (so it is delayed like the conv), and the iSTFT overlap-adds across calls with
the offline window-envelope normalisation: streamed audio equals offline decoding to float rounding. The price is a fixed
algorithmic lookahead (27 backbone frames plus the iSTFT overlap).
  --check   assert streamed == offline on the 20 LibriTTS-R clips, both variants, several chunk sizes, flush at the end
  default   ms of compute per 80 ms of audio on core 3 for 80/160/320 ms chunks, idle and while Qwen3 decodes on cores 1-2,
            against decoder.py's 240 ms-context method in the same session; the old method's error vs offline (SNR,
            UTMOS22); the streaming step exported to ONNX (caches as explicit inputs/outputs) when the onnx module exists
"""
import argparse
import json
import math
import os
import subprocess
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

from common import Telemetry, _temp_c, save, sysinfo
from decoder import ROOT, utts

VARIANTS = {"vocos-mel": ("charactr/vocos-mel-24khz", None), "vocos-encodec-6k": ("charactr/vocos-encodec-24khz", 2)}  # 2 = 6 kbps
CHUNKS_MS = (80, 160, 320)
CTX_MS = 240  # decoder.py's left context
OUT = ROOT / "out/vocos-stream"


def load(name):
    from vocos import Vocos
    repo, bw = VARIANTS[name]
    v = Vocos.from_pretrained(repo).eval()
    kw = {} if bw is None else {"bandwidth_id": torch.tensor([bw])}
    return v, kw, lambda w: v.feature_extractor(torch.from_numpy(w)[None], **kw)


class Stream:
    """feed(frames (1, C, n)) -> the audio those frames complete; flush() -> the rest. Concatenated == v.decode(all frames)."""

    def __init__(self, v, bandwidth_id=None):
        self.v, self.bw = v, bandwidth_id
        self.convs = [v.backbone.embed, *(b.dwconv for b in v.backbone.convnext)]
        ist = v.head.istft
        self.n, self.hop, self.win = ist.n_fft, ist.hop_length, ist.window
        self.center = ist.padding == "center"  # mel: torch.istft(center=True); encodec: vocos's 'same' ISTFT
        self.pad = self.n // 2 if self.center else (self.n - self.hop) // 2
        self.rf = sum(c.kernel_size[0] // 2 for c in self.convs)  # backbone right context, frames
        # audio that offline decoding of the frames so far would already give but exact streaming must hold back
        self.lookahead = self.rf * self.hop + self.pad - self.hop * self.center
        self.reset()

    def reset(self):
        self.cache = [torch.zeros(1, c.in_channels, c.kernel_size[0] // 2) for c in self.convs]  # = the left zero padding
        self.tail = torch.zeros(2, self.n - self.hop)  # overlap-add carry: signal, window envelope
        self.trim = self.pad

    def conv(self, i, x, final):
        """Conv i over cache + x: every output whose k-wide window is complete, and the matching centre inputs (residual)."""
        c = self.convs[i]
        p = c.kernel_size[0] // 2
        buf = torch.cat([self.cache[i], x, *([x.new_zeros(1, x.shape[1], p)] if final else [])], -1)
        m = max(0, buf.shape[-1] - 2 * p)
        self.cache[i] = buf[..., m:]
        y = F.conv1d(buf, c.weight, c.bias, groups=c.groups) if m else buf.new_zeros(1, c.out_channels, 0)
        return y, buf[..., p:p + m]

    def spec(self, x, final):
        """Backbone and head of VocosBackbone.forward / ISTFTHead.forward, up to (mag, cos, sin) of the frames x completes."""
        bb, h = self.v.backbone, self.v.head
        x = self.conv(0, x, final)[0].transpose(1, 2)
        x = (bb.norm(x, cond_embedding_id=self.bw) if bb.adanorm else bb.norm(x)).transpose(1, 2)
        for i, b in enumerate(bb.convnext, 1):
            y, res = self.conv(i, x, final)
            y = y.transpose(1, 2)
            y = b.pwconv2(b.act(b.pwconv1(b.norm(y, self.bw) if b.adanorm else b.norm(y))))
            x = res + (y if b.gamma is None else b.gamma * y).transpose(1, 2)
        mag, p = h.out(bb.final_layer_norm(x.transpose(1, 2))).transpose(1, 2).chunk(2, dim=1)
        return torch.exp(mag).clip(max=1e2), torch.cos(p), torch.sin(p)

    def ola(self, mag, cos, sin, final):
        """Streaming iSTFT: overlap-add signal and window envelope together, emit the samples no later frame touches."""
        S, n, hop = mag * (cos + 1j * sin), self.n, self.hop
        m = S.shape[-1]
        if m:
            fr = torch.fft.irfft(S, n, dim=1, norm="backward")[0] * self.win[:, None]
            y = F.fold(torch.stack([fr, self.win.square()[:, None].expand(-1, m)]), (1, (m - 1) * hop + n), (1, n),
                       stride=(1, hop))[:, 0, 0]
            y[:, :n - hop] += self.tail
        else:
            y = self.tail
        k = y.shape[1] - (self.pad if final else n - hop)
        out, self.tail = y[:, :k], y[:, k:]
        d = min(self.trim, k)
        out, self.trim = out[:, d:], self.trim - d
        return out[0] / out[1]

    def feed(self, x, final=False):
        return self.ola(*self.spec(x, final), final)

    def flush(self):
        return self.feed(torch.zeros(1, self.convs[0].in_channels, 0), final=True)


class Step(torch.nn.Module):
    """One steady-state call (all caches full) with the caches as explicit inputs and outputs, for ONNX export."""

    def __init__(self, st):
        super().__init__()
        self.st, self.v = st, st.v

    def forward(self, x, *cache):
        self.st.cache = list(cache)
        return *self.st.spec(x, False), *self.st.cache


def onnx_feed(st, new, threads):
    """Export Step for chunks of `new` frames and return feed(x) that runs it in onnxruntime (iSTFT stays in torch, as in
    decoder.py's ONNX row); torch fills the caches during the first calls, while their shapes still change."""
    import onnxruntime as ort
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{'mel' if st.center else 'encodec'}-{new}.onnx"
    full = [torch.zeros(1, c.in_channels, 2 * (c.kernel_size[0] // 2)) for c in st.convs]
    names = [f"c{i}" for i in range(len(full))]
    with torch.no_grad():
        torch.onnx.export(Step(st), (torch.zeros(1, st.convs[0].in_channels, new), *full), str(path), dynamo=False,
                          input_names=["x", *names], output_names=["mag", "cos", "sin", *[f"o{n}" for n in names]])
    st.reset()
    so = ort.SessionOptions()
    so.intra_op_num_threads, so.inter_op_num_threads = threads, 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])

    def feed(x):
        if any(c.shape[-1] < f.shape[-1] for c, f in zip(st.cache, full)) or x.shape[-1] != new:
            return st.feed(x)
        o = s.run(None, {"x": x.numpy(), **{k: c.numpy() for k, c in zip(names, st.cache)}})
        st.cache = [torch.from_numpy(c) for c in o[3:]]
        return st.ola(*map(torch.from_numpy, o[:3]), False)
    return feed


def chunked(v, kw, f, new, ctx):
    """decoder.py's streaming call, spliced into audio: re-decode ctx frames of left context, keep the new frames' samples."""
    hop, T = v.head.istft.hop_length, f.shape[-1]
    d = int(v.head.istft.padding == "center")  # torch.istft(center) ends a chunk at its last frame's centre: lag one hop
    total, out = (T - d) * hop, []
    for a in range(0, T, new):
        s = max(0, a - ctx)
        y = v.decode(f[..., s:a + new], **kw)[0]
        out.append(y[max(0, a - d) * hop - s * hop:min(total, (a + new - d) * hop) - s * hop])
    return torch.cat(out)


def snr(ref, y):
    return float(10 * torch.log10(ref.double().square().sum() / (ref - y).double().square().sum().clamp(min=1e-30)))


def err(ref, y):
    """max abs error, SNR (dB) and |STFT| SNR (dB, phase-blind) of y against ref"""
    assert ref.shape == y.shape, (ref.shape, y.shape)
    mag = lambda a: torch.stft(a, 1024, 256, window=torch.hann_window(1024), return_complex=True).abs()
    return float((ref - y).abs().max()), snr(ref, y), snr(mag(ref), mag(y))


def frames(ms, fps):
    return math.ceil(ms / 1000 * fps)


def exactness(name, n_clips=20, check=False):
    """Streamed (plus ONNX if available) audio against offline v.decode, per clip. check: several chunk sizes, ONNX at every
    timed chunk (1 frame on two clips only: ~45 ms a call); else the 80 ms chunk plus decoder.py's old method (SNR, UTMOS22)."""
    v, kw, feats = load(name)
    st = Stream(v, kw.get("bandwidth_id"))
    fps = 24000 / st.hop
    new = frames(80, fps)
    sizes = [1, new - 1, *(frames(ms, fps) for ms in CHUNKS_MS), 10**6] if check else [new]
    res = {"variant": name, "frame_hz": round(fps, 2), "backbone_rf_frames": st.rf, "lookahead_samples": st.lookahead,
           "lookahead_frames": round(st.lookahead / st.hop, 2), "lookahead_ms": round(st.lookahead / 24, 1)}
    feeds = {s: (st.feed, s) for s in sizes}
    try:
        import onnx  # noqa: F401  torch's exporter needs it
        for ms in CHUNKS_MS if check else CHUNKS_MS[:1]:
            n = frames(ms, fps)
            feeds["onnx" if n == new else f"onnx-{ms}ms"] = (onnx_feed(st, n, torch.get_num_threads()), n)
    except Exception as e:
        res["onnx"] = f"skipped: {type(e).__name__}: {e}"[:200]
    stream, old, wavs = {k: [] for k in feeds}, {ms: [] for ms in CHUNKS_MS}, {"offline": [], "context-80ms": []}
    with torch.inference_mode():
        for i, (_, w, _) in enumerate(utts()[:n_clips]):
            f = feats(w)
            ref = v.decode(f, **kw)[0]
            for k, (step, s) in feeds.items():
                if s == 1 and i >= 2:
                    continue
                st.reset()
                parts = [step(f[..., a:a + s]) for a in range(0, f.shape[-1], s)]
                if k == new:  # nothing flushed yet: exactly `lookahead` samples are held back
                    assert ref.shape[0] - sum(map(len, parts)) == st.lookahead, (ref.shape[0], sum(map(len, parts)))
                stream[k].append(err(ref, torch.cat([*parts, st.flush()])))
            for ms in () if check else CHUNKS_MS:
                y = chunked(v, kw, f, frames(ms, fps), frames(CTX_MS, fps))
                old[ms].append(err(ref, y))
                if ms == 80:
                    wavs["offline"].append(ref.numpy())
                    wavs["context-80ms"].append(y.numpy())
    res["stream"] = {str(k): {"max_abs_err": max(e for e, *_ in r), "snr_db_min": round(min(db for _, db, _ in r), 1)} for k, r in stream.items()}
    if check:
        return res
    res[f"context_{CTX_MS}ms"] = {f"{ms}ms": {"max_abs_err": round(max(e for e, *_ in r), 4),
                                              "snr_db_median": round(float(np.median([db for _, db, _ in r])), 1),
                                              "snr_db_min": round(min(db for _, db, _ in r), 1),
                                              "mag_snr_db_median": round(float(np.median([db for *_, db in r])), 1)} for ms, r in old.items()}
    res["utmos"] = utmos(wavs)
    return res


def utmos(wavs):
    """UTMOS22-strong at 16 kHz, as decoder.score"""
    import librosa
    m = torch.hub.load("tarepan/SpeechMOS:v1.2.0", "utmos22_strong", trust_repo=True).eval()
    with torch.inference_mode():
        return {k: round(float(np.mean([m(torch.from_numpy(librosa.resample(w, orig_sr=24000, target_sr=16000))[None], 16000).item()
                                        for w in ws])), 2) for k, ws in wavs.items()}


def check(n_clips):
    for name in VARIANTS:
        r = exactness(name, n_clips, check=True)
        print(json.dumps(r))
        for s, e in r["stream"].items():
            assert e["max_abs_err"] < 1e-3 and e["snr_db_min"] > 60, (name, s, e)
    print(f"ok: streamed == offline on {n_clips} clips, both variants")


def lat_stream(step, st, fs, new, n=100):
    """per-call seconds once every cache is full, chunks of `new` frames through successive clips"""
    lat = []
    while True:
        for f in fs:
            st.reset()
            for a in range(0, f.shape[-1] - new + 1, new):
                t0 = time.perf_counter()
                step(f[..., a:a + new])
                if a > st.rf + new:
                    lat.append(time.perf_counter() - t0)
                    if len(lat) == n:
                        return sorted(lat)


def lat_context(v, kw, f, new, ctx, n=100):
    """decoder.py's loop: decode ctx + new frames per call"""
    lat = []
    for i in range(n):
        a = ctx + (i * new) % (f.shape[-1] - ctx - new)
        t0 = time.perf_counter()
        v.decode(f[..., a - ctx:a + new], **kw)
        lat.append(time.perf_counter() - t0)
    return sorted(lat)


def speed(name, threads):
    torch.set_num_threads(threads)
    v, kw, feats = load(name)
    st = Stream(v, kw.get("bandwidth_id"))
    fps = 24000 / st.hop
    try:
        import onnx  # noqa: F401
        ofeeds = {ms: onnx_feed(st, frames(ms, fps), threads) for ms in CHUNKS_MS}
    except Exception as e:
        ofeeds = {"skipped": f"{type(e).__name__}: {e}"[:200]}
    rows = []
    with torch.inference_mode():
        fs = [feats(w) for _, w, _ in utts()[:3]]  # encoding is not timed: in TTS the backbone makes these frames
        v.decode(fs[0][..., :8], **kw)  # warm-up
        wall = audio = 0.0
        for f in fs:
            t0 = time.perf_counter()
            audio += len(v.decode(f, **kw)[0]) / 24000
            wall += time.perf_counter() - t0
        rows.append({"variant": name, "method": "offline", "offline_rtf": round(wall / audio, 3)})

        def row(method, ms, new, lat):
            k = 80 / (1000 * new / fps)  # ms per call -> ms per 80 ms of audio produced
            return {"variant": name, "method": method, "chunk_ms": ms, "chunk_frames": new, "lookahead_ms": round(st.lookahead / 24, 1),
                    "ms_per_call_p50": round(lat[len(lat) // 2] * 1000, 2), "ms_per_80ms_p50": round(lat[len(lat) // 2] * 1000 * k, 1),
                    "ms_per_80ms_p95": round(lat[int(len(lat) * 0.95)] * 1000 * k, 1), "threads": threads}
        for ms in CHUNKS_MS:
            new = frames(ms, fps)
            rows.append(row("stream", ms, new, lat_stream(st.feed, st, fs, new)))
            if ms in ofeeds:
                rows.append(row("stream-onnx", ms, new, lat_stream(ofeeds[ms], st, fs, new)))
            rows.append({**row(f"context-{CTX_MS}ms", ms, new, lat_context(v, kw, fs[0], new, frames(CTX_MS, fps))), "lookahead_ms": 0})
    if "skipped" in ofeeds:
        rows.append({"variant": name, "method": "stream-onnx", "skipped": ofeeds["skipped"]})
    return {"rows": rows}


def child(args):
    torch.set_num_threads(args.threads)
    try:
        out = speed(args.engine, args.threads) if args.job == "speed" else exactness(args.engine)
    except Exception as e:
        out = {"variant": args.engine, "skipped": f"{type(e).__name__}: {e}"[:300]}
    print(json.dumps(out))


def run(job, name, cores, threads=1):
    out = subprocess.run(["taskset", "-c", cores, sys.executable, "-W", "ignore", __file__, "--engine", name, "--job", job,
                          "--threads", str(threads)], capture_output=True, text=True)
    lines = [l for l in out.stdout.splitlines() if l.startswith("{")]
    return json.loads(lines[-1]) if lines else {"variant": name, "skipped": out.stderr[-300:]}


def cool(limit=60):  # run.sh's cool(): start each pass from a similar temperature
    while (_temp_c() or 0) > limit:
        time.sleep(10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="assert streamed == offline (both variants, several chunk sizes)")
    ap.add_argument("--clips", type=int, default=20)
    ap.add_argument("--engine", help=argparse.SUPPRESS)
    ap.add_argument("--job", default="speed", help=argparse.SUPPRESS)
    ap.add_argument("--threads", type=int, default=1)
    args = ap.parse_args()
    if args.check:
        torch.set_num_threads(args.threads)
        return check(args.clips)
    if args.engine:
        return child(args)
    from tts import lm_load
    rows = []
    os.sched_setaffinity(0, {0})  # telemetry's vcgencmd forks stay off the timed core 3
    for load in (False, True):
        cool()
        bg = lm_load() if load else None
        time.sleep(3 if load else 0)
        try:
            with Telemetry() as tel:
                for name in VARIANTS:
                    out = run("speed", name, "3", args.threads)
                    for r in out.get("rows", [out]):
                        rows.append({**r, "lm_load": load and bg.poll() is None})
                        print(rows[-1], flush=True)
        finally:
            if bg:
                bg.kill()
        rows.append({"variant": "telemetry", "lm_load": load, **tel.summary()})
    os.sched_setaffinity(0, {0, 1, 2, 3})
    exact = [run("quality", n, "0-3") for n in VARIANTS]  # 1 thread: these small calls run slower on 4
    save("vocos-stream", {
        "sysinfo": sysinfo(), "rows": rows, "exactness": exact,
        "method": "stream = this file's cached streaming decoder (exact); stream-onnx = its steady-state step in onnxruntime, "
                  f"iSTFT overlap-add in torch; context-{CTX_MS}ms = decoder.py's call (re-decode {CTX_MS} ms of left context, "
                  "no right context); ms_per_80ms = ms per call x 80 / chunk audio ms; lookahead_ms = audio held back behind "
                  "the frames received (add the chunk itself and compute for latency)",
        "data": "LibriTTS-R test-clean (24 kHz, CC BY 4.0), peak -3 dBFS; timing on 3 clips, exactness on 20; errors vs "
                "offline Vocos.decode of the same frames (mel: exact mels; encodec: 6 kbps codes); UTMOS22-strong via "
                "SpeechMOS v1.2.0 at 16 kHz"})


if __name__ == "__main__":
    main()
