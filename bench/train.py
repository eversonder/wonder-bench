"""How fast can a Pi 5 actually train? Forward+backward+optimizer steps on skeletons shaped like Wonder Nano.

Replaces the plan's unverified ~30 GFLOPS planning number with a measured one.
  asr: ~30M streaming encoder over 50 Hz frames (CTC head)
  tts: ~40M causal transformer over 12.5 Hz continuous latents
Modes: full (all weights train, ~6*N*D FLOPs) and lora (frozen base + rank-r adapters, ~4*N*D).
"""
import argparse
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from common import Telemetry, save, sysinfo


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r=8):
        super().__init__()
        self.base, self.a, self.b = base, nn.Linear(base.in_features, r, bias=False), nn.Linear(r, base.out_features, bias=False)
        nn.init.zeros_(self.b.weight)

    def forward(self, x):
        return self.base(x) + self.b(self.a(x))


class Block(nn.Module):
    def __init__(self, d, heads, ff, causal):
        super().__init__()
        self.h, self.causal = heads, causal
        self.n1, self.n2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.up, self.down = nn.Linear(d, ff), nn.Linear(ff, d)

    def forward(self, x):
        B, T, D = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, T, 3, self.h, D // self.h).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=self.causal).transpose(1, 2).reshape(B, T, D)
        x = x + self.o(a)
        return x + self.down(F.gelu(self.up(self.n2(x))))


class Skeleton(nn.Module):
    def __init__(self, d_in, d, layers, heads, ff, d_out, causal):
        super().__init__()
        self.inp, self.out = nn.Linear(d_in, d), nn.Linear(d, d_out)
        self.blocks = nn.ModuleList(Block(d, heads, ff, causal) for _ in range(layers))

    def forward(self, x):
        x = self.inp(x)
        for b in self.blocks:
            x = b(x)
        return self.out(x)


# ponytail: shapes, not real models; FLOPs/step is what we're measuring, the recipe comes later
SHAPES = {
    "asr": dict(d_in=80, d=384, layers=16, heads=6, ff=1536, d_out=256, causal=False, hz=50),
    "tts": dict(d_in=32, d=512, layers=12, heads=8, ff=2048, d_out=32, causal=True, hz=12.5),
}


def build(kind, mode, scale):
    s = dict(SHAPES[kind])
    hz = s.pop("hz")
    if scale != 1:
        s["d"] = max(64, int(s["d"] * scale) // s["heads"] * s["heads"])
        s["ff"], s["layers"] = s["d"] * 4, max(2, int(s["layers"] * scale))
    m = Skeleton(**s)
    n_all = sum(p.numel() for p in m.parameters())
    if mode == "lora":
        for p in m.parameters():
            p.requires_grad_(False)
        for b in m.blocks:
            b.qkv, b.o, b.up, b.down = (LoRALinear(l) for l in (b.qkv, b.o, b.up, b.down))
    return m, n_all, hz, s


def step_fn(kind, m, opt, x):
    y = m(x)
    loss = F.cross_entropy(y.reshape(-1, y.shape[-1]), torch.randint(0, y.shape[-1], y.shape[:2]).reshape(-1)) if kind == "asr" else F.mse_loss(y, x)
    opt.zero_grad(set_to_none=True)
    loss.backward()
    opt.step()


def bench(kind, mode, threads, seconds_audio, batch, steps, scale):
    torch.set_num_threads(threads)
    torch.manual_seed(0)
    m, n, hz, shape = build(kind, mode, scale)
    trainable = [p for p in m.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=1e-4)
    frames = int(seconds_audio * hz)
    x = torch.randn(batch, frames, shape["d_in"])
    step_fn(kind, m, opt, x)  # warm-up
    times = []
    with Telemetry() as tel:
        for _ in range(steps):
            t0 = time.perf_counter()
            step_fn(kind, m, opt, x)
            times.append(time.perf_counter() - t0)
    t = sorted(times)[len(times) // 2]
    tokens = batch * frames
    flops = (6 if mode == "full" else 4) * n * tokens  # ponytail: 6ND/4ND ignores attention's T² term; fine at these lengths
    return {
        "kind": kind, "mode": mode, "threads": threads, "params_m": round(n / 1e6, 1),
        "trainable_m": round(sum(p.numel() for p in trainable) / 1e6, 2),
        "batch": batch, "frames": frames, "step_s_median": round(t, 3),
        "frames_per_s": round(tokens / t), "gflops_effective": round(flops / t / 1e9, 1),
        **tel.summary(),
    }


def matmul_peak(threads, n=1024, reps=20):
    torch.set_num_threads(threads)
    a, b = torch.randn(n, n), torch.randn(n, n)
    a @ b
    t0 = time.perf_counter()
    for _ in range(reps):
        a @ b
    return round(2 * n**3 * reps / (time.perf_counter() - t0) / 1e9, 1)


def extrapolate(gflops):
    """Wall-clock for the plan's jobs at the measured rate (same 6ND/4ND rules as PLAN.md §5)."""
    g = gflops * 1e9
    jobs = {
        "tts_lora_30min_30ep": 4 * 40e6 * (1800 * 12.5) * 30,
        "asr_full_10h_10ep": 6 * 30e6 * (36000 * 50) * 10,
        "tts_pretrain_1kh_100ep": 6 * 40e6 * (3.6e6 * 12.5) * 100,
    }
    return {k: f"{v / g / 3600:.1f} h" if v / g < 3.6e6 else f"{v / g / 3.15e7:.1f} years" for k, v in jobs.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true", help="tiny shapes, for checking the harness off-Pi")
    ap.add_argument("--threads", default="1,2,4")
    ap.add_argument("--steps", type=int, default=5)
    args = ap.parse_args()
    scale = 0.25 if args.smoke else 1
    # seconds of audio per sequence, batch: ~750-800 frames per step for both shapes
    audio = {"asr": (2, 2), "tts": (8, 2)} if args.smoke else {"asr": (8, 2), "tts": (30, 2)}
    threads = [int(t) for t in args.threads.split(",")]

    rows = []
    for th in threads:
        rows.append({"kind": "matmul_fp32", "threads": th, "gflops_effective": matmul_peak(th)})
        print(rows[-1])
        for kind in ("asr", "tts"):
            for mode in ("full", "lora"):
                rows.append(bench(kind, mode, th, *audio[kind], args.steps, scale))
                print({k: rows[-1][k] for k in ("kind", "mode", "threads", "params_m", "step_s_median", "gflops_effective")})
    best = max(r["gflops_effective"] for r in rows if r["kind"] in ("asr", "tts") and r["mode"] == "full")
    save("train", {"sysinfo": sysinfo(), "torch": torch.__version__, "smoke": args.smoke, "rows": rows,
                   "best_training_gflops": best, "extrapolated": extrapolate(best)})


if __name__ == "__main__":
    main()
