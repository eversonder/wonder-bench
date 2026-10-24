#!/usr/bin/env bash
# One-time setup on a Raspberry Pi 5 (tested on Ubuntu 26.04; Raspberry Pi OS Bookworm should work the same).
# Everything lands inside this folder except the apt packages.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=$PWD

sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq build-essential cmake git ffmpeg libopenblas-dev libsndfile1 libcurl4-openssl-dev pkg-config espeak-ng

command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH=$HOME/.local/bin:$PATH

# Python 3.12: system Python on newer distros (3.14) is ahead of several wheels
[ -d .venv ] || uv venv -q --python 3.12 .venv
. .venv/bin/activate
uv pip install -q torch==2.14.1 torchaudio==2.11.0 --index-url https://download.pytorch.org/whl/cpu  # PyPI's torchaudio is a CUDA build
# --index-strategy lets pocket-tts see the CPU torch above instead of pulling CUDA wheels
uv pip install -q --index-strategy unsafe-best-match --extra-index-url https://download.pytorch.org/whl/cpu \
  numpy soundfile onnxruntime huggingface_hub \
  kokoro-onnx==0.6.1 piper-tts==1.8.0 pocket-tts==4.0.1 ptts==0.4.1 rustymimi==0.4.1 moonshine-voice==0.1.5 \
  vocos==0.1.0 snac==1.2.1 librosa==1.0.0 pyyaml

# llama.cpp twice: with Arm's KleidiAI kernels and without, so we A/B rather than assume
[ -d llama.cpp ] || git clone -q --depth 1 https://github.com/ggml-org/llama.cpp
for v in kai ref; do
  flags="-DCMAKE_BUILD_TYPE=Release -DGGML_NATIVE=ON -DLLAMA_CURL=OFF"
  [ $v = kai ] && flags="$flags -DGGML_CPU_KLEIDIAI=ON"
  [ -x llama.cpp/build-$v/bin/llama-bench ] || {
    start=$(date +%s)
    cmake -S llama.cpp -B llama.cpp/build-$v $flags >/dev/null
    cmake --build llama.cpp/build-$v -j4 --target llama-bench llama-cli >/dev/null
    echo "llama.cpp build-$v: $(( $(date +%s) - start )) s"
  }
done

mkdir -p models/{qwen,kokoro,piper,mimi}
get() { [ -s "$2" ] || curl -fsSL -o "$2" "$1"; }
get https://huggingface.co/Qwen/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q8_0.gguf models/qwen/Qwen3-0.6B-Q8_0.gguf
get https://huggingface.co/Qwen/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q8_0.gguf models/qwen/Qwen3-1.7B-Q8_0.gguf
# ponytail: official repos only publish Q8_0; Unsloth's Q4_0 builds stand in until we quantize from bf16 ourselves
get https://huggingface.co/unsloth/Qwen3-0.6B-GGUF/resolve/main/Qwen3-0.6B-Q4_0.gguf models/qwen/Qwen3-0.6B-Q4_0.gguf
get https://huggingface.co/unsloth/Qwen3-1.7B-GGUF/resolve/main/Qwen3-1.7B-Q4_0.gguf models/qwen/Qwen3-1.7B-Q4_0.gguf || echo "no Q4_0 for 1.7B upstream; skipping"
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/kokoro-v1.0.int8.onnx models/kokoro/kokoro-v1.0.int8.onnx
get https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.1/voices-v1.0.bin models/kokoro/voices-v1.0.bin
for ext in onnx onnx.json; do
  get https://huggingface.co/rhasspy/piper-voices/resolve/main/en/en_US/libritts_r/medium/en_US-libritts_r-medium.$ext models/piper/en_US-libritts_r-medium.$ext
done
get https://huggingface.co/kyutai/moshiko-pytorch-bf16/resolve/main/tokenizer-e351c8d8-checkpoint125.safetensors models/mimi/tokenizer-e351c8d8-checkpoint125.safetensors

# decoder bake-off: sherpa-onnx exports of HiFi-GAN (LJSpeech) and Vocos mel, plus WavTokenizer's code and checkpoint
mkdir -p models/vocoders models/wavtokenizer
for f in hifigan_v1 hifigan_v2 hifigan_v3 vocos_24khz; do
  get https://github.com/k2-fsa/sherpa-onnx/releases/download/vocoder-models/$f.onnx models/vocoders/$f.onnx
done
get https://huggingface.co/novateur/WavTokenizer/resolve/main/WavTokenizer_small_600_24k_4096.ckpt models/wavtokenizer/WavTokenizer_small_600_24k_4096.ckpt
[ -d WavTokenizer ] || { git clone -q https://github.com/jishengpeng/WavTokenizer && git -C WavTokenizer checkout -q 5cf440d; }

python fetch_data.py  # skips sets it already has
echo "setup done: $(du -sh models | cut -f1) of models"
