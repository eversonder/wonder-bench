# wonder-bench

Week 1 of [EVERSONDER Wonder](https://eversonder.com): measure, on a real Raspberry Pi 5, every number the plan currently only estimates. Nothing about Wonder's speed goes on the website until it comes out of this repo.

| Script | Measures | Replaces the plan's guess |
|---|---|---|
| `bench/train.py` | Training throughput: forward + backward + AdamW on Nano-shaped skeletons (28.5M listen, 37.9M speak), full and LoRA, 1/2/4 threads; extrapolates the plan's Pi training jobs | "~30 GFLOPS effective" |
| `bench/llm.sh` | Qwen3 0.6B / 1.7B at Q4_0 and Q8_0, prompt and generation tokens/s, threads 1–4, llama.cpp with and without KleidiAI | "~18–28 tok/s" |
| `bench/asr.py` | Moonshine tiny, base, tiny-streaming, small-streaming: WER on 20 LibriSpeech utterances, RTF, streaming update latency | "240 ms final transcript" |
| `bench/mimi.py` | Mimi decoder per 80 ms frame (p50/p95), the codec Wonder TTS v0 builds on | "unmeasured" |
| `bench/tts.py` | Kokoro int8, Piper (LibriTTS-R voice), Pocket TTS, Phonon-31m: time-to-first-audio and RTF on one core, idle and while Qwen3 decodes on two others | the go/no-go gate: **RTF ≤ 0.3 under LM load** |

`bench/cpu.sh` runs the "squeeze the CPU" experiments: memory bandwidth (STREAM), a quantisation sweep with flash attention, micro-batch and mmap options, KL divergence of each quant from bf16, prompt caching and prompt/generation thread splits through llama-server, listening-while-thinking contention, and PyTorch training variants (torch.compile, small-GEMM routing, jemalloc + thread binding, batch size). Software only: no reboot, firmware or boot changes.

Every result records power (PMIC rails), SoC temperature, ARM clock and throttle flags, so a throttled run shows up as throttled rather than as "the Pi is slow".

## Run it

```bash
git clone https://github.com/eversonder/wonder-bench && cd wonder-bench
./setup.sh      # apt deps, Python 3.12 venv, two llama.cpp builds, ~5 GB of models, 20 test utterances
./run.sh        # all benchmarks in sequence, waits for the SoC to cool between them
```

Results land in `results/<host>-<date>/` as JSON plus a `REPORT.md`. Off-Pi smoke test for the training bench: `python bench/train.py --smoke`.

## First results

[`results/gold-20261024/REPORT.md`](results/gold-20261024/REPORT.md): Raspberry Pi 5 8 GB, Ubuntu 26.04, no cooler. Headlines:

- **Training:** 35 GFLOPS effective (fp32, 4 cores, throttled to 1.5 GHz); 2 cores reach ~90% of that.
- **Thinking:** Qwen3-0.6B Q4_0 writes 26 tok/s on one core; KleidiAI reads prompts 1.5–1.9× faster than the reference build at Q8_0.
- **Listening:** Moonshine base 2.7% WER at 0.53 RTF; tiny-streaming 4.3% at 0.23.
- **Speaking (one core, while Qwen3 decodes on two):** Piper 0.25–0.30 RTF, Phonon-31m ~1.05, Pocket TTS ~10. Only Piper clears the 0.3 gate. Kokoro int8 runs at 1.7–2.7 RTF alone and 3.2–4.9 under load, with 3.4–9.6 s to first audio.
- **Mimi decoder:** 132 ms per 80 ms frame on one core: too slow to build a streaming voice on as-is.

### Squeezing the CPU (`bench/cpu.sh`)

- **Memory bandwidth caps writing.** STREAM Triad 8.0 / 7.1 / 5.7 GB/s at 1 / 2 / 4 threads: one core already saturates memory, which is why token generation gets slower with more threads.
- **Flash attention is free speed:** +2–18% writing and +5–31% reading, for every quant at every thread count.
- **Speed against quality** (Qwen3-0.6B, writing at 1 thread with flash attention; KL divergence from bf16 on public-domain text, lower is better):

  | Quant | MiB | Writes tok/s | Reads tok/s (4 threads) | KLD | Same top token |
  |---|---|---|---|---|---|
  | Q4_0 + Q4_0 embeddings | 320 | **30.6** | 178 | 0.249 | 72.4% |
  | Q4_0 | 358 | 26.8 | 168 | 0.220 | 75.2% |
  | Q4_K_M | 373 | 23.5 | 132 | 0.148 | 79.0% |
  | IQ4_NL | 359 | 22.7 | 156 | 0.165 | 76.9% |
  | Q8_0 | 604 | 20.3 | **215** | **0.027** | **90.4%** |

  Q8_0 is close to lossless and still writes several times faster than anyone speaks; the Q4_0-embeddings variant is fastest but drifts most.
- **Prompt caching** cuts the second turn of a conversation from 2,083 ms of prompt reading to 127 ms (only the 9 new tokens are read). Best server split: generation on 1 thread, prompt reading on 4.
- **Listening while thinking:** with Qwen3 decoding on two cores, Moonshine tiny-streaming slows from 0.31 to 0.46 RTF and Qwen3 from 21.4 to 15.8 tok/s; with Moonshine base, Qwen3 drops to 13.9 tok/s.
- **Load mode:** mmap or a full read makes no difference to writing speed (24.8 vs 24.5 tok/s at 2 threads).
- **Training variants** (2 threads): `torch.compile` +3–6%; small-GEMM routing, jemalloc and thread binding don't help; batch 1 loses ~18% efficiency, batch 4–8 matches batch 2.

## Rules

- **Same conditions, stated.** Board, RAM, OS, kernel, governor, cooling and temperature go with every number. Default board is a Pi 5 8 GB; note the cooler (or its absence).
- **Licence:** the harness is Apache-2.0 (see `LICENSE`); models and data it downloads keep their own licences, listed below.
- **Under load or it didn't happen.** Voice-loop numbers are taken with the LM decoding, because that's how they run in a conversation.
- **Baselines are for comparison, never initialisation.** Kokoro and Piper phonemise with espeak-ng (GPL-3); Piper's runtime is GPL-3. They are benchmarked, not shipped.
- **Licences of what's measured:** Qwen3 Apache-2.0 · Kokoro weights Apache-2.0 · Piper `en_US-libritts_r-medium` trained on LibriTTS-R (CC BY 4.0) · Pocket TTS weights CC BY 4.0 · Phonon-31m Apache-2.0 · Mimi CC BY 4.0 · Moonshine English MIT · LibriSpeech CC BY 4.0.
