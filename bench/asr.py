"""Speech recognition on the Pi: Moonshine baselines on 20 LibriSpeech test-clean utterances (CC BY 4.0).

Reports WER (normalised), RTF for whole-utterance decoding, and per-update latency for the streaming models.
"""
import re
import time
from pathlib import Path

import soundfile as sf
from moonshine_voice import ModelArch, Transcriber, get_model_for_language

from common import Telemetry, save, sysinfo

DATA = Path(__file__).resolve().parents[1] / "data/librispeech-test-clean-20"


def norm(s):
    return " ".join(re.sub(r"[^a-z' ]", " ", s.lower()).split())


def wer(refs, hyps):
    E = N = 0
    for r, h in zip(refs, hyps):
        r, h = r.split(), h.split()
        d = list(range(len(h) + 1))
        for i, rw in enumerate(r, 1):
            prev, d[0] = d[0], i
            for j, hw in enumerate(h, 1):
                prev, d[j] = d[j], min(d[j] + 1, d[j - 1] + 1, prev + (rw != hw))
        E += d[-1]
        N += len(r)
    return E / N


def load():
    rows = [l.split("\t") for l in (DATA / "transcripts.tsv").read_text().splitlines()]
    return [(sf.read(DATA / f"{u}.flac", dtype="float32")[0], t) for u, t in rows]


def bench(arch, utts):
    path, a = get_model_for_language("en", arch)
    t = Transcriber(model_path=path, model_arch=a)
    t.transcribe_without_streaming(utts[0][0].tolist(), 16000)  # warm-up
    hyps, wall, audio = [], 0.0, 0.0
    for wav, _ in utts:
        buf = wav.tolist()  # conversion kept outside the timed region
        t0 = time.perf_counter()
        tr = t.transcribe_without_streaming(buf, 16000)
        wall += time.perf_counter() - t0
        audio += len(wav) / 16000
        hyps.append(" ".join(l.text for l in tr.lines))
    row = {"model": arch.name, "wer_pct": round(100 * wer([norm(r) for _, r in utts], [norm(h) for h in hyps]), 1),
           "rtf": round(wall / audio, 3), "audio_s": round(audio, 1)}
    if "STREAMING" in arch.name:  # 100 ms chunks, time each update like a live mic would
        wav = utts[0][0]
        s = t.create_stream()
        s.start()
        ups = []
        for i in range(0, len(wav), 1600):
            s.add_audio(wav[i:i + 1600].tolist(), 16000)
            t0 = time.perf_counter()
            s.update_transcription()
            ups.append((time.perf_counter() - t0) * 1000)
        s.stop()
        ups.sort()
        row |= {"update_ms_p50": round(ups[len(ups) // 2], 1), "update_ms_p95": round(ups[int(len(ups) * 0.95)], 1)}
    t.close()
    return row


def main():
    utts = load()
    rows = []
    with Telemetry() as tel:
        for arch in (ModelArch.TINY, ModelArch.BASE, ModelArch.TINY_STREAMING, ModelArch.SMALL_STREAMING):
            try:
                rows.append(bench(arch, utts))
            except Exception as e:
                rows.append({"model": arch.name, "skipped": f"{type(e).__name__}: {e}"[:200]})
            print(rows[-1])
    save("asr", {"sysinfo": sysinfo(), "utterances": len(utts), "rows": rows, **tel.summary(),
                 "data": "LibriSpeech test-clean, Panayotov et al., CC BY 4.0"})


if __name__ == "__main__":
    main()
