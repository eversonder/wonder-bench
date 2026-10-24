"""Keep 20 test-clean utterances (2 per speaker, 10 speakers) from LibriSpeech (16 kHz, for ASR) and LibriTTS-R
(24 kHz, for the decoder bake-off), without downloading either corpus. Each set is skipped once it exists."""
import json
import os
import tarfile
import urllib.request

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
N, PER_SPEAKER = 20, 2


def write(out, texts, licence):
    with open(os.path.join(out, "transcripts.tsv"), "w") as f:
        for u in sorted(texts):
            f.write(f"{u}\t{texts[u]}\n")
    with open(os.path.join(out, "LICENSE"), "w") as f:
        f.write(licence + "\n")
    print(f"{len(texts)} utterances in {out}")


def librispeech(out):
    """Stream the 346 MB tarball and stop once 20 clips and their transcripts are in."""
    flacs, trans, per = {}, {}, {}
    with urllib.request.urlopen("https://www.openslr.org/resources/12/test-clean.tar.gz") as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if name.endswith(".trans.txt"):
                for line in tf.extractfile(m).read().decode().splitlines():
                    u, t = line.split(" ", 1)
                    trans[u] = t
            elif name.endswith(".flac") and len(flacs) < N:
                spk = name.split("-")[0]
                if per.get(spk, 0) < PER_SPEAKER:
                    per[spk] = per.get(spk, 0) + 1
                    with open(os.path.join(out, name), "wb") as f:
                        f.write(tf.extractfile(m).read())
                    flacs[name[:-5]] = True
            if len(flacs) >= N and all(u in trans for u in flacs):
                break
    write(out, {u: trans[u] for u in flacs}, "LibriSpeech ASR corpus, Panayotov et al. 2015, https://www.openslr.org/12, CC BY 4.0")


def libritts_r(out):
    """Page through the Hugging Face datasets-server and download only the clips we keep."""
    rows_url = "https://datasets-server.huggingface.co/rows?dataset=mythicinfinity/libritts_r&config=clean&split=test.clean"
    keep, per = {}, {}
    for offset in range(0, 5000, 100):
        with urllib.request.urlopen(f"{rows_url}&offset={offset}&length=100") as r:
            rows = [x["row"] for x in json.load(r)["rows"]]
        for row in rows:
            spk, text = str(row["speaker_id"]), row["text_normalized"]
            if len(keep) < N and per.get(spk, 0) < PER_SPEAKER and 80 <= len(text) <= 250:  # roughly 5-15 s
                per[spk] = per.get(spk, 0) + 1
                urllib.request.urlretrieve(row["audio"][0]["src"], os.path.join(out, f"{row['id']}.wav"))
                keep[row["id"]] = text
        if len(keep) >= N or not rows:
            break
    write(out, keep, "LibriTTS-R, Koizumi et al. 2023, https://www.openslr.org/141, CC BY 4.0")


for name, fetch in (("librispeech-test-clean-20", librispeech), ("libritts-r-test-clean-20", libritts_r)):
    out = os.path.join(DATA, name)
    if not os.path.exists(os.path.join(out, "transcripts.tsv")):
        os.makedirs(out, exist_ok=True)
        fetch(out)
