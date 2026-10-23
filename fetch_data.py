"""Stream LibriSpeech test-clean and keep 20 utterances (2 per speaker, 10 speakers) without downloading all 346 MB."""
import os
import tarfile
import urllib.request

URL = "https://www.openslr.org/resources/12/test-clean.tar.gz"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data/librispeech-test-clean-20")
N, PER_SPEAKER = 20, 2

os.makedirs(OUT, exist_ok=True)
flacs, trans, per = {}, {}, {}
with urllib.request.urlopen(URL) as r, tarfile.open(fileobj=r, mode="r|gz") as tf:
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
                path = os.path.join(OUT, name)
                with open(path, "wb") as f:
                    f.write(tf.extractfile(m).read())
                flacs[name[:-5]] = path
        if len(flacs) >= N and all(u in trans for u in flacs):
            break

with open(os.path.join(OUT, "transcripts.tsv"), "w") as f:
    for u in sorted(flacs):
        f.write(f"{u}\t{trans[u]}\n")
with open(os.path.join(OUT, "LICENSE"), "w") as f:
    f.write("LibriSpeech ASR corpus, Panayotov et al. 2015, https://www.openslr.org/12, CC BY 4.0\n")
print(f"{len(flacs)} utterances from {len(per)} speakers in {OUT}")
