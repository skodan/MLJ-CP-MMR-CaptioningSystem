# MLJ-CP-MMR-CaptioningSystem
This is a Capstone project on Multi Model Retrieval and Captioning System

## Setup

Install the Python dependencies. On Windows, using `conda` is recommended for PyTorch and Faiss:

```
# Create and activate a conda environment (optional)
conda create -n cpmmr python=3.10 -y
conda activate cpmmr

# Install PyTorch + torchvision (CPU)
conda install pytorch torchvision torchaudio cpuonly -c pytorch -y

# For GPU (example with CUDA 11.8):
conda install pytorch torchvision torchaudio cudatoolkit=11.8 -c pytorch -c nvidia -y

# Install Faiss (CPU)
conda install -c pytorch faiss-cpu -y

# Install the other Python dependencies
pip install -r requirements.txt
```

If you prefer `pip` only, install `torch` and `torchvision` using the PyTorch selector at https://pytorch.org/get-started/locally/, then run:

```
pip install -r requirements.txt
```

Note: Windows `pip` wheels for `faiss` are limited; use `conda` for Faiss on Windows if possible.

## Quick Test
After installing the dependencies, verify imports in Python:

```
python - <<'PY'
import torch
import torch.nn as nn
import torchvision.models as models
import faiss

print('PyTorch version:', torch.__version__)
print('Torchvision version:', models.__name__)
print('FAISS imported:', 'faiss' in globals())
PY
```


