# wonder-bench

Week 1 of [EVERSONDER Wonder](https://eversonder.com): measure, on a real Raspberry Pi 5, every number the plan currently only estimates. Nothing about Wonder's speed goes on the website until it comes out of this repo.

| Script | Measures | Replaces the plan's guess |
|---|---|---|
| `bench/train.py` | Training throughput: forward + backward + AdamW on Nano-shaped skeletons (28.5M listen, 37.9M speak), full and LoRA, 1/2/4 threads; extrapolates the plan's Pi training jobs | "~30 GFLOPS effective" |
| `bench/llm.sh` | Qwen3 0.6B / 1.7B at Q4_0 and Q8_0, prompt and generation tokens/s, threads 1–4, llama.cpp with and without KleidiAI | "~18–28 tok/s" |
| `bench/asr.py` | Moonshine tiny, base, tiny-streaming, small-streaming: WER on 20 LibriSpeech utterances, RTF, streaming update latency | "240 ms final transcript" |
| `bench/mimi.py` | Mimi decoder per 80 ms frame (p50/p95), the codec Wonder TTS v0 builds on | "unmeasured" |
| `bench/tts.py` | Kokoro int8, Piper (LibriTTS-R voice), Pocket TTS, Phonon-31m: time-to-first-audio and RTF on one core, idle and while Qwen3 decodes on two others | the go/no-go gate: **RTF ≤ 0.3 under LM load** |

Every result records power (PMIC rails), SoC temperature, ARM clock and throttle flags, so a throttled run shows up as throttled rather than as "the Pi is slow".

## Run it

```bash
git clone https://github.com/eversonder/wonder-bench && cd wonder-bench
./setup.sh      # apt deps, Python 3.12 venv, two llama.cpp builds, ~5 GB of models, 20 test utterances
./run.sh        # all benchmarks in sequence, waits for the SoC to cool between them
```

Results land in `results/<host>-<date>/` as JSON plus a `REPORT.md`. Off-Pi smoke test for the training bench: `python bench/train.py --smoke`.

## Rules

- **Same conditions, stated.** Board, RAM, OS, kernel, governor, cooling and temperature go with every number. Default board is a Pi 5 8 GB; note the cooler (or its absence).
- **Under load or it didn't happen.** Voice-loop numbers are taken with the LM decoding, because that's how they run in a conversation.
- **Baselines are for comparison, never initialisation.** Kokoro and Piper phonemise with espeak-ng (GPL-3); Piper's runtime is GPL-3. They are benchmarked, not shipped.
- **Licences of what's measured:** Qwen3 Apache-2.0 · Kokoro weights Apache-2.0 · Piper `en_US-libritts_r-medium` trained on LibriTTS-R (CC BY 4.0) · Pocket TTS weights CC BY 4.0 · Phonon-31m Apache-2.0 · Mimi CC BY 4.0 · Moonshine English MIT · LibriSpeech CC BY 4.0.
