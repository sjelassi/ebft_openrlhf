#!/bin/bash

conda create -n openrlhf python==3.12.11
conda activate openrlhf

pip install vllm==0.8.4
pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu124
pip install psutil
python -m pip install "flash-attn==2.7.4post1" --no-build-isolation -v

pip install sacrebleu
pip install sentence-transformers
pip install unbabel-comet==2.2.7
pip install humanize
pip install loguru
pip install rouge-score
pip install unbabel-comet
pip install datatrove
pip install math-verify omegaconf==2.4.0dev3
pip install setuptools==81.0.0 --upgrade
pip install -e .