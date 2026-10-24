# wonder-bench: gold-20261024

## asr

Raspberry Pi 5 Model B Rev 1.0 · 7.7 GB · kernel 7.0.0-1020-raspi · governor performance · 2026-10-24T01:42:14+0100

| model | wer_pct | rtf | audio_s | update_ms_p50 | update_ms_p95 |
|---|---|---|---|---|---|
| TINY | 5.4 | 0.364 | 133.2 |  |  |
| BASE | 2.7 | 0.533 | 133.2 |  |  |
| TINY_STREAMING | 4.3 | 0.234 | 133.2 | 7.6 | 344.3 |
| SMALL_STREAMING | 3.8 | 0.581 | 133.2 | 7.9 | 832.6 |

- **utterances**: 20
- **watts_mean**: 5.7
- **watts_peak**: 9.393
- **temp_max_c**: 79.8
- **arm_mhz_min**: 2400
- **throttled_after**: throttled=0xe0000
- **samples**: 417
- **data**: LibriSpeech test-clean, Panayotov et al., CC BY 4.0

## llm

| build | model | threads | test | tok_per_s | stddev |
|---|---|---|---|---|---|
| kai | Qwen3-0.6B-Q4_0 | 1 | pp512 | 54.6 | 0.0 |
| kai | Qwen3-0.6B-Q4_0 | 1 | tg128 | 26.1 | 0.0 |
| kai | Qwen3-0.6B-Q4_0 | 2 | pp512 | 100.3 | 0.4 |
| kai | Qwen3-0.6B-Q4_0 | 2 | tg128 | 24.7 | 0.4 |
| kai | Qwen3-0.6B-Q4_0 | 3 | pp512 | 132.7 | 2.4 |
| kai | Qwen3-0.6B-Q4_0 | 3 | tg128 | 20.8 | 0.0 |
| kai | Qwen3-0.6B-Q4_0 | 4 | pp512 | 133.4 | 0.9 |
| kai | Qwen3-0.6B-Q4_0 | 4 | tg128 | 18.3 | 0.1 |
| kai | Qwen3-0.6B-Q8_0 | 1 | pp512 | 85.9 | 0.0 |
| kai | Qwen3-0.6B-Q8_0 | 1 | tg128 | 19.7 | 0.1 |
| kai | Qwen3-0.6B-Q8_0 | 2 | pp512 | 153.9 | 0.2 |
| kai | Qwen3-0.6B-Q8_0 | 2 | tg128 | 17.0 | 0.0 |
| kai | Qwen3-0.6B-Q8_0 | 3 | pp512 | 173.9 | 2.9 |
| kai | Qwen3-0.6B-Q8_0 | 3 | tg128 | 14.0 | 0.0 |
| kai | Qwen3-0.6B-Q8_0 | 4 | pp512 | 194.9 | 1.4 |
| kai | Qwen3-0.6B-Q8_0 | 4 | tg128 | 12.2 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 1 | pp512 | 21.5 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 1 | tg128 | 9.9 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 2 | pp512 | 40.6 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 2 | tg128 | 9.2 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 3 | pp512 | 48.1 | 0.6 |
| kai | Qwen3-1.7B-Q4_0 | 3 | tg128 | 7.5 | 0.0 |
| kai | Qwen3-1.7B-Q4_0 | 4 | pp512 | 56.4 | 0.2 |
| kai | Qwen3-1.7B-Q4_0 | 4 | tg128 | 6.4 | 0.0 |
| kai | Qwen3-1.7B-Q8_0 | 1 | pp512 | 39.7 | 0.0 |
| kai | Qwen3-1.7B-Q8_0 | 1 | tg128 | 7.3 | 0.0 |
| kai | Qwen3-1.7B-Q8_0 | 2 | pp512 | 64.6 | 0.3 |
| kai | Qwen3-1.7B-Q8_0 | 2 | tg128 | 6.0 | 0.0 |
| kai | Qwen3-1.7B-Q8_0 | 3 | pp512 | 76.5 | 0.1 |
| kai | Qwen3-1.7B-Q8_0 | 3 | tg128 | 4.9 | 0.0 |
| kai | Qwen3-1.7B-Q8_0 | 4 | pp512 | 88.6 | 1.5 |
| kai | Qwen3-1.7B-Q8_0 | 4 | tg128 | 4.2 | 0.0 |
| ref | Qwen3-0.6B-Q4_0 | 1 | pp512 | 58.9 | 0.1 |
| ref | Qwen3-0.6B-Q4_0 | 1 | tg128 | 21.2 | 0.0 |
| ref | Qwen3-0.6B-Q4_0 | 2 | pp512 | 99.6 | 0.4 |
| ref | Qwen3-0.6B-Q4_0 | 2 | tg128 | 24.3 | 0.2 |
| ref | Qwen3-0.6B-Q4_0 | 3 | pp512 | 121.5 | 0.8 |
| ref | Qwen3-0.6B-Q4_0 | 3 | tg128 | 20.7 | 0.1 |
| ref | Qwen3-0.6B-Q4_0 | 4 | pp512 | 141.2 | 0.5 |
| ref | Qwen3-0.6B-Q4_0 | 4 | tg128 | 17.8 | 0.1 |
| ref | Qwen3-0.6B-Q8_0 | 1 | pp512 | 55.3 | 0.1 |
| ref | Qwen3-0.6B-Q8_0 | 1 | tg128 | 18.4 | 0.0 |
| ref | Qwen3-0.6B-Q8_0 | 2 | pp512 | 93.3 | 0.3 |
| ref | Qwen3-0.6B-Q8_0 | 2 | tg128 | 15.7 | 0.1 |
| ref | Qwen3-0.6B-Q8_0 | 3 | pp512 | 112.9 | 1.2 |
| ref | Qwen3-0.6B-Q8_0 | 3 | tg128 | 12.6 | 0.0 |
| ref | Qwen3-0.6B-Q8_0 | 4 | pp512 | 132.8 | 0.8 |
| ref | Qwen3-0.6B-Q8_0 | 4 | tg128 | 10.9 | 0.0 |
| ref | Qwen3-1.7B-Q4_0 | 1 | pp512 | 23.8 | 0.0 |
| ref | Qwen3-1.7B-Q4_0 | 1 | tg128 | 8.6 | 0.0 |
| ref | Qwen3-1.7B-Q4_0 | 2 | pp512 | 40.6 | 0.4 |
| ref | Qwen3-1.7B-Q4_0 | 2 | tg128 | 9.2 | 0.1 |
| ref | Qwen3-1.7B-Q4_0 | 3 | pp512 | 48.9 | 0.8 |
| ref | Qwen3-1.7B-Q4_0 | 3 | tg128 | 7.6 | 0.1 |
| ref | Qwen3-1.7B-Q4_0 | 4 | pp512 | 55.4 | 0.3 |
| ref | Qwen3-1.7B-Q4_0 | 4 | tg128 | 6.5 | 0.0 |
| ref | Qwen3-1.7B-Q8_0 | 1 | pp512 | 21.1 | 0.0 |
| ref | Qwen3-1.7B-Q8_0 | 1 | tg128 | 6.7 | 0.0 |
| ref | Qwen3-1.7B-Q8_0 | 2 | pp512 | 35.5 | 0.5 |
| ref | Qwen3-1.7B-Q8_0 | 2 | tg128 | 5.6 | 0.0 |
| ref | Qwen3-1.7B-Q8_0 | 3 | pp512 | 40.6 | 0.5 |
| ref | Qwen3-1.7B-Q8_0 | 3 | tg128 | 4.5 | 0.0 |
| ref | Qwen3-1.7B-Q8_0 | 4 | pp512 | 26.1 | 0.1 |
| ref | Qwen3-1.7B-Q8_0 | 4 | tg128 | 3.9 | 0.0 |


## mimi

Raspberry Pi 5 Model B Rev 1.0 · 7.7 GB · kernel 7.0.0-1020-raspi · governor performance · 2026-10-24T01:44:03+0100

- **threads_env**: default
- **frames**: 125
- **frame_budget_ms**: 80
- **ms_p50**: 132.3
- **ms_p95**: 150.0
- **rtf**: 1.689
- **watts_mean**: 4.78
- **watts_peak**: 8.436
- **temp_max_c**: 71.6
- **arm_mhz_min**: 2400
- **throttled_after**: throttled=0xe0000
- **samples**: 28

## train

Raspberry Pi 5 Model B Rev 1.0 · 7.7 GB · kernel 7.0.0-1020-raspi · governor performance · 2026-10-24T00:28:08+0100

| kind | threads | gflops_effective | mode | params_m | trainable_m | batch | frames | step_s_median | frames_per_s | watts_mean | watts_peak | temp_max_c | arm_mhz_min | throttled_after | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| matmul_fp32 | 1 | 32.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| asr | 1 | 21.5 | full | 28.5 | 28.52 | 2 | 400 | 6.355 | 126 | 4.76 | 10.256 | 65.5 | 2400 | throttled=0x0 | 56 |
| asr | 1 | 18.6 | lora | 28.5 | 0.79 | 2 | 400 | 4.908 | 163 | 4.65 | 8.381 | 67.2 | 2400 | throttled=0x0 | 43 |
| tts | 1 | 22.3 | full | 37.9 | 37.86 | 2 | 375 | 7.654 | 98 | 4.55 | 6.708 | 67.2 | 2400 | throttled=0x0 | 67 |
| tts | 1 | 20.2 | lora | 37.9 | 0.79 | 2 | 375 | 5.617 | 134 | 4.47 | 6.469 | 67.2 | 2400 | throttled=0x0 | 50 |
| matmul_fp32 | 2 | 60.1 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| asr | 2 | 31.7 | full | 28.5 | 28.52 | 2 | 400 | 4.318 | 185 | 5.83 | 7.478 | 74.9 | 2400 | throttled=0x0 | 37 |
| asr | 2 | 24.9 | lora | 28.5 | 0.79 | 2 | 400 | 3.673 | 218 | 6.3 | 10.668 | 75.5 | 2400 | throttled=0x0 | 31 |
| tts | 2 | 31.4 | full | 37.9 | 37.86 | 2 | 375 | 5.417 | 138 | 6.19 | 10.165 | 77.1 | 2400 | throttled=0x0 | 45 |
| tts | 2 | 27.5 | lora | 37.9 | 0.79 | 2 | 375 | 4.124 | 182 | 5.96 | 7.526 | 77.1 | 2400 | throttled=0x0 | 35 |
| matmul_fp32 | 4 | 81.9 |  |  |  |  |  |  |  |  |  |  |  |  |  |
| asr | 4 | 33.6 | full | 28.5 | 28.52 | 2 | 400 | 4.074 | 196 | 6.96 | 11.271 | 84.2 | 2201 | throttled=0x80000 | 34 |
| asr | 4 | 25.0 | lora | 28.5 | 0.79 | 2 | 400 | 3.654 | 219 | 6.22 | 8.88 | 82.0 | 2311 | throttled=0x80000 | 30 |
| tts | 4 | 35.4 | full | 37.9 | 37.86 | 2 | 375 | 4.82 | 156 | 7.12 | 11.543 | 84.8 | 1500 | throttled=0xe0000 | 41 |
| tts | 4 | 29.1 | lora | 37.9 | 0.79 | 2 | 375 | 3.904 | 192 | 6.49 | 10.574 | 83.7 | 1500 | throttled=0xe0000 | 32 |

- **torch**: 2.14.1+cpu
- **smoke**: False
- **best_training_gflops**: 35.4
- **extrapolated**: {"tts_lora_30min_30ep": "0.8 h", "asr_full_10h_10ep": "25.4 h", "tts_pretrain_1kh_100ep": "1.0 years"}

## tts

Raspberry Pi 5 Model B Rev 1.0 · 7.7 GB · kernel 7.0.0-1020-raspi · governor performance · 2026-10-24T01:59:35+0100

| engine | skipped | lm_load | text | audio_s | ttfa_ms_median | rtf_median | threads | peak_rss_mb | watts_mean | watts_peak | temp_max_c | arm_mhz_min | throttled_after | samples |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| kokoro-int8 | ValueError: zero-size array to reduction operation maximum which has no identity | False |  |  |  |  |  |  |  |  |  |  |  |  |
| piper-libritts_r |  | False | short | 2.36 | 171 | 0.152 | 1 |  |  |  |  |  |  |  |
| piper-libritts_r |  | False | medium | 7.78 | 128 | 0.148 | 1 |  |  |  |  |  |  |  |
| piper-libritts_r |  | False | numbers | 4.35 | 655 | 0.15 | 1 | 273 |  |  |  |  |  |  |
| pocket-tts |  | False | short | 3.04 | 718 | 3.948 | 1 |  |  |  |  |  |  |  |
| pocket-tts |  | False | medium | 8.96 | 827 | 3.948 | 1 |  |  |  |  |  |  |  |
| pocket-tts |  | False | numbers | 4.56 | 739 | 3.869 | 1 | 1161 |  |  |  |  |  |  |
| phonon-31m-q8 |  | False | short | 3.44 | 116 | 0.483 | 1 |  |  |  |  |  |  |  |
| phonon-31m-q8 |  | False | medium | 10.16 | 186 | 0.474 | 1 |  |  |  |  |  |  |  |
| phonon-31m-q8 |  | False | numbers | 5.52 | 137 | 0.482 | 1 | 293 |  |  |  |  |  |  |
| telemetry |  | False |  |  |  |  |  |  | 4.13 | 8.351 | 79.3 | 2400 | throttled=0xe0000 | 477 |
| kokoro-int8 | ValueError: zero-size array to reduction operation maximum which has no identity | True |  |  |  |  |  |  |  |  |  |  |  |  |
| piper-libritts_r |  | True | short | 2.29 | 300 | 0.3 | 1 |  |  |  |  |  |  |  |
| piper-libritts_r |  | True | medium | 7.56 | 234 | 0.245 | 1 |  |  |  |  |  |  |  |
| piper-libritts_r |  | True | numbers | 4.26 | 1066 | 0.249 | 1 | 272 |  |  |  |  |  |  |
| pocket-tts |  | True | short | 3.04 | 1643 | 9.974 | 1 |  |  |  |  |  |  |  |
| pocket-tts |  | True | medium | 9.04 | 2118 | 10.131 | 1 |  |  |  |  |  |  |  |
| pocket-tts |  | True | numbers | 4.72 | 1770 | 10.037 | 1 | 1039 |  |  |  |  |  |  |
| phonon-31m-q8 |  | True | short | 3.44 | 209 | 1.044 | 1 |  |  |  |  |  |  |  |
| phonon-31m-q8 |  | True | medium | 10.16 | 289 | 1.082 | 1 |  |  |  |  |  |  |  |
| phonon-31m-q8 |  | True | numbers | 5.52 | 232 | 1.059 | 1 | 271 |  |  |  |  |  |  |
| telemetry |  | True |  |  |  |  |  |  | 6.02 | 9.058 | 84.2 | 1500 | throttled=0xe0008 | 950 |

- **texts**: {"short": "Forty minutes on the clock. I'll call you when the bread's ready.", "medium": "The kettle is on. Your sister called this morning and said she'd ring back after lunch. It's going to rain later, so the washing might want to come in. Would you like me to read the next chapter of your book?", "numbers": "Your appointment is on the 14th of November at 3:45 pm, room 212."}
- **gate**: RTF <= 0.3 under LM load on one core (PLAN.md week 1 go/no-go)
