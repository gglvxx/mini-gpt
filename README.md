# Mini-GPT

A character-level GPT language model built from scratch in PyTorch.
It is trained on Shakespeare's works and generates new text in the same style.

![Tests](https://github.com/gglvxx/mini-gpt/actions/workflows/tests.yml/badge.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c)
![License](https://img.shields.io/badge/License-MIT-green)

## Features

- **Decoder-only Transformer** implemented from scratch: causal multi-head self-attention, feed-forward layers, residual connections, LayerNorm
- **Character-level tokenizer** with encode/decode roundtrip
- **Hardware-aware config**: automatically uses a larger model when a CUDA GPU is available
- **Training loop** with train/val evaluation, gradient clipping and best-checkpoint saving
- **Loss tracking** with an automatically generated training curve
- **Streaming text generation** with temperature and top-k sampling
- **Unit tests** for the tokenizer, dataset and model (including a causal-mask test)

## Architecture

```
Input IDs
   │
Token Embedding + Position Embedding
   │
┌──────────────────────────────┐
│  Transformer Block  × N      │
│   ├─ LayerNorm               │
│   ├─ Causal Self-Attention   │  + residual
│   ├─ LayerNorm               │
│   └─ Feed-Forward (GELU)     │  + residual
└──────────────────────────────┘
   │
Final LayerNorm → Linear → Logits (next-character probabilities)
```

## Project Structure

```
mini-gpt/
├── src/
│   ├── config.py        # Hyperparameters and paths
│   ├── tokenizer.py     # Character-level tokenizer
│   ├── dataset.py       # Train/val split and batching
│   └── model.py         # Transformer architecture
├── tests/               # Unit tests (pytest)
├── assets/              # Images used in this README
├── data/                # Dataset (downloaded, not tracked)
├── checkpoints/         # Trained models and loss history (not tracked)
├── prepare_data.py      # Downloads the dataset
├── train.py             # Trains the model
├── generate.py          # Generates text
├── plot_loss.py         # Plots the training curve
├── pytest.ini
└── requirements.txt
```

## Installation

```bash
git clone https://github.com/gglvxx/mini-gpt.git
cd mini-gpt
python3 -m venv .venv
source .venv/bin/activate

# CPU only (smaller download)
pip install torch --index-url https://download.pytorch.org/whl/cpu
# Or, with an NVIDIA GPU:
# pip install torch

pip install -r requirements.txt
```

## Usage

```bash
# 1. Download the Tiny Shakespeare dataset (~1 MB)
python prepare_data.py

# 2. Train the model
python train.py                    # full training
python train.py --max-iters 300    # quick test run

# 3. Plot the loss curve
python plot_loss.py

# 4. Generate text
python generate.py --prompt "ROMEO:" --tokens 300
python generate.py --temperature 0.5 --top-k 20 --seed 42

# 5. Run tests
pytest -v
```

### Generation options

| Argument | Default | Description |
|---|---|---|
| `--prompt` | `\n` | Starting text |
| `--tokens` | `500` | Number of characters to generate |
| `--temperature` | `0.8` | Lower = safer, higher = more