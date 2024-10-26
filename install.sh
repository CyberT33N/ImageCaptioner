#!/bin/bash

# Virtuelle Umgebung löschen, falls sie existiert
rm -rf venv

# Virtuelle Umgebung erstellen
python3 -m venv venv
source venv/bin/activate

# CUDA-Version erkennen
CUDA_VERSION=$(nvcc --version | grep -oP 'release \K[0-9]+\.[0-9]+' | head -1)
echo "CUDA-Version: $CUDA_VERSION"

# Installationsbefehl für PyTorch basierend auf der CUDA-Version
if [[ $CUDA_VERSION == 12.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu124
elif [[ $CUDA_VERSION == 11.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu117
elif [[ $CUDA_VERSION == 10.* ]]; then
    pip install torch --extra-index-url https://download.pytorch.org/whl/cu102
else
    echo "CUDA-Version nicht unterstützt oder nicht gefunden."
    exit 1
fi

# Installiere weitere benötigte Pakete
pip install transformers Pillow
