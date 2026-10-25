# wonder-bench

Week 1 of [EVERSONDER Wonder](https://eversonder.com): measure, on a real Raspberry Pi 5, every number the plan currently only estimates. Nothing about Wonder's speed goes on the website until it comes out of this repo.

| Script | Measures | Replaces the plan's guess |
|---|---|---|
| `bench/train.py` | Training throughput: forward + backward + AdamW on Nano-shaped skeletons (28.5M listen, 37.9M speak), full and LoRA, 1/2/4 threads; extrapolates the plan's Pi training jobs | "~30 GFLOPS effective" |
| `bench/llm.sh` | Qwen3 0.6B / 1.7B at Q4_0 and Q8_0, prompt and generation tokens/s, threads 1–4, llama.cpp with and without KleidiAI | "~18–28 tok/s" |
| `bench/asr.py` | Moonshine tiny, base, tiny-streaming, small-streaming: WER on 20 LibriSpeech utterances, RTF, streaming update latency | "240 ms final transcript" |
| `bench/mimi.py` | Mimi decoder per 80 ms frame (p50/p95), the codec Wonder TTS v0 builds on | "unmeasured" |
| `bench/tts.py` | Kokoro int8, Piper (LibriTTS-R voice), Pocket TTS, Phonon-31m: time-to-first-audio and RTF on one core, idle and while Qwen3 decodes on two others | the go/no-go gate: **RTF ≤ 0.3 under LM load** |
| `bench/decoder.py` | Decoder bake-off: Vocos (mel, EnCodec), WavTokenizer, HiFi-GAN v1–3, SNAC and Mimi on one core, offline RTF and streaming ms per 80 ms of audio, idle and under LM load; quality by copy synthesis of 20 LibriTTS-R clips (UTMOS22, Moonshine WER) | which decoder replaces Mimi |
| `bench/vocos_stream.py` | Exact streaming Vocos (mel, EnCodec): per-layer caches so chunked output equals offline decoding; ms per 80 ms at 80/160/320 ms chunks in PyTorch and ONNX Runtime, idle and under LM load; the lookahead that exactness costs; the error and UTMOS22 of the bake-off's 240 ms-context call, hard-spliced into audio | "240 ms of context" streaming in the bake-off |

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

### Decoder bake-off (`bench/decoder.py`)

Mimi's decoder misses the 80 ms frame budget, so every open candidate raced on core 3. Streaming = one call per 80 ms of new audio with 240 ms of left context (Mimi uses its own streaming step). Quality = copy synthesis of 20 LibriTTS-R test-clean clips (24 kHz) from each decoder's own mels or codes, scored at 16 kHz.

| Decoder | Params | ms per 80 ms, idle | … under LM load* | Offline RTF, idle | UTMOS22 | WER |
|---|---|---|---|---|---|---|
| Original recording | | | | | 4.33 | 3.1% |
| Vocos EnCodec, 6 kbps | 8.0M | **30** | 87 | **0.054** | **4.21** | 4.4% |
| Vocos mel, ONNX + torch iSTFT | 13.5M | 32 | **72** | 0.101 | 4.04 | 3.8% |
| Vocos mel, PyTorch | 13.5M | 54 | 129 | 0.115 | 4.04 | **3.5%** |
| HiFi-GAN v2 (LJSpeech) | 0.9M | 48 | 111 | 0.166 | 3.30 | 3.5% |
| HiFi-GAN v3 (LJSpeech) | 1.5M | 51 | 124 | 0.172 | 3.06 | 4.4% |
| Mimi (rustymimi) | | 117 | 251 | 0.449 | 3.91 | 5.2% |
| WavTokenizer, 40 Hz | 63.8M | 162 | 402 | 0.201 | 3.84 | 14.6% |
| HiFi-GAN v1 (LJSpeech) | 13.9M | 594 | 1,488 | 1.869 | 3.53 | 4.4% |
| SNAC 24 kHz | 13.2M | 706 | 1,323 | 1.985 | 4.15 | 5.0% |

\* Under LM load the board reached 85 °C and dropped to 1.5 GHz; the idle pass held 2.4 GHz at up to 71 °C.

- **Vocos wins on both axes:** about 4× faster than Mimi with higher predicted quality. The EnCodec-token variant has the best quality per millisecond; the mel variant has the lowest WER.
- **Context is the cost, not the network.** Offline, Vocos EnCodec decodes at 0.054 RTF (4 ms per 80 ms); re-decoding 240 ms of context per call makes it 30 ms. A decoder with its own streaming state would recover most of that, which is a design target for Wonder's decoder.
- **ONNX Runtime beats PyTorch for small chunks** (32 vs 54 ms for the same Vocos weights).
- **Caveats:** 20 clips, so 1 point of WER is a handful of words; HiFi-GAN's checkpoints are single-speaker LJSpeech, so its quality on 10 LibriTTS-R speakers understates the architecture; UTMOS and WER are judged at 16 kHz, so the 8–12 kHz band isn't scored. Listen to `out/decoder/` before deciding.

### Streaming Vocos (`bench/vocos_stream.py`)

Vocos isn't causal: a kernel-7 embed conv and 8 ConvNeXt blocks with kernel-7 depthwise convs each look 3 frames ahead, and the iSTFT overlap-add needs the next frames too. The bake-off's streaming call had left context only, so its chunk edges were approximations. This decoder keeps each conv's last 6 inputs, takes residuals from the same buffer, and overlap-adds the iSTFT across calls with the offline window-envelope normalisation, so its output equals `Vocos.decode` on the whole utterance. Same session as the old method; one core, ms of compute per 80 ms of audio (p50) for 80 / 160 / 320 ms chunks.

| Vocos variant, method | Lookahead | ms per 80 ms, idle | … under LM load* | Error vs offline (SNR) | UTMOS22 |
|---|---|---|---|---|---|
| EnCodec 6 kbps, exact stream, ONNX step | 380 ms | **6.2** / 4.8 / 4.1 | **19.0** / 12.0 / 9.7 | 90 dB | not scored (offline 4.21) |
| EnCodec 6 kbps, exact stream, PyTorch | 380 ms | 20.7 / 12.3 / 8.0 | 70.0 / 38.4 / 22.6 | 94 dB | not scored (offline 4.21) |
| EnCodec 6 kbps, 240 ms left context, hard-spliced (bake-off's call, re-timed) | 0 | 31.0 / 17.3 / 10.5 | 92.4 / 49.1 / 28.9 | 10.9 / 12.9 / 17.2 dB | 3.93 |
| mel, exact stream, ONNX step | 299 ms | 10.0 / 9.1 / 7.9 | 25.2 / 24.1 / 17.7 | 88 dB | not scored (offline 4.04) |
| mel, exact stream, PyTorch | 299 ms | 29.7 / 21.0 / 14.1 | 92.2 / 53.8 / 35.2 | 94 dB | not scored (offline 4.04) |
| mel, 240 ms left context, hard-spliced (bake-off's call, re-timed) | 0 | 52.0 / 30.6 / 19.0 | 125.5 / 74.9 / 44.0 | −0.8 / 1.1 / 4.2 dB | 2.71 |

\* The load pass peaked at 82.6 °C and 2.31 GHz; idle peaked at 71.6 °C and held 2.4 GHz. SNR is the median over 20 clips for the context rows and the worst clip for the exact rows. UTMOS is at 80 ms chunks and only scored for offline and the context method; the exact rows are within 88–94 dB of offline, so they weren't re-scored. Offline RTF in the same session: EnCodec 0.055, mel 0.115.

- **The bake-off's streaming call gives approximate audio, and for mel it's bad.** `decoder.py` only timed its calls and never joined their output, so this file splices them: each call's new samples, hard-cut and concatenated with no crossfade. Spliced, the 240 ms-context chunks score −0.8 dB waveform SNR against offline Vocos mel, 12 dB on the magnitude spectrum, and UTMOS 2.71 against 4.04. EnCodec degrades less (3.93 against 4.21). The error comes from the missing right context: on one clip, 100 frames of left context give the same SNR as 23.
- **Exact streaming works.** It matches offline to 86–94 dB SNR on the worst clip (max error 1e-4 to 5e-4 of full scale; 88–94 dB at the 80 ms chunk), checked on all 20 clips at 80 ms minus one frame, 80, 160 and 320 ms and whole-utterance chunks, and the ONNX step at 80, 160 and 320 ms, each with a flush at the end (`--check`); 1-frame chunks are checked on 2 clips.
- **Each call costs weight traffic, not frames.** PyTorch's GEMM is slow at a few rows: an 8-row 512×1536 linear takes 1.4 ms, about 2 GB/s of weights. So at 80 ms chunks the exact PyTorch stream is only 1.5–1.8× faster than re-decoding context idle, and 1.3–1.4× under LM load. The same step in ONNX Runtime, with caches as explicit inputs and outputs, is 3.0–3.7× faster again at 80 ms chunks and 1.8–3.2× at 160 and 320 ms. EnCodec at 6.2 ms per 80 ms idle and 19 ms under load is the first exact streaming decoder here with headroom for a backbone on the same core.
- **Exactness costs lookahead:** 27 frames from the 9 kernel-7 convs plus the iSTFT overlap, 380 ms for EnCodec and 299 ms for mel. That delay sits on top of the chunk and the compute, so it takes about a third of a 1 s voice-to-voice budget. Only a decoder trained with causal or short-lookahead convs removes it, and that's a design target for Wonder's.
- **Caveats:** the ONNX graph is fixed per chunk size and runs only in steady state. PyTorch handles each utterance's first ~30 frames while the caches fill (untimed), and the iSTFT overlap-add stays in torch, as in the bake-off's ONNX row. Timing covers steady-state calls on 3 clips. "Per 80 ms" is ms per call × 80 / the chunk's audio length (mel's 8-frame chunk is 85 ms long; `decoder.py` scales by ms per call × (context + 7.5) / (context + 8) frames, so on its scale mel's 80 ms context row reads 54.5 idle and 131.8 under load against the bake-off's 53.5 and 128.8; EnCodec's chunk is exactly 80 ms, so both scales agree). The ONNX export needs `onnx` 1.23.2 in the venv.

## Rules

- **Same conditions, stated.** Board, RAM, OS, kernel, governor, cooling and temperature go with every number. Default board is a Pi 5 8 GB; note the cooler (or its absence).
- **Licence:** the harness is Apache-2.0 (see `LICENSE`); models and data it downloads keep their own licences, listed below.
- **Under load or it didn't happen.** Voice-loop numbers are taken with the LM decoding, because that's how they run in a conversation.
- **Baselines are for comparison, never initialisation.** Kokoro and Piper phonemise with espeak-ng (GPL-3); Piper's runtime is GPL-3. They are benchmarked, not shipped.
- **Licences of what's measured:** Qwen3 Apache-2.0 · Kokoro weights Apache-2.0 · Piper `en_US-libritts_r-medium` trained on LibriTTS-R (CC BY 4.0) · Pocket TTS weights CC BY 4.0 · Phonon-31m Apache-2.0 · Mimi CC BY 4.0 · Moonshine English MIT · LibriSpeech CC BY 4.0.
