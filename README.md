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
- **Streaming text generation** with temperature and top-k sampling, from the CLI or a web interface
- **Web demo** built with Gradio
- **Unit tests** for the tokenizer, dataset and model (including a causal-mask test), run automatically with GitHub Actions

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
├── .github/workflows/   # CI: runs the tests on every push
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
├── generate.py          # Generates text from the command line
├── app.py               # Web interface (Gradio)
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

# 4. Generate text from the command line
python generate.py --prompt "ROMEO:" --tokens 300
python generate.py --temperature 0.5 --top-k 20 --seed 42

# 5. Launch the web interface
python app.py

# 6. Run tests
pytest -v
```

### Generation options

| Argument | Default | Description |
|---|---|---|
| `--prompt` | `\n` | Starting text |
| `--tokens` | `500` | Number of characters to generate |
| `--temperature` | `0.8` | Lower = safer, higher = more creative |
| `--top-k` | `40` | Sample only from the K most likely characters |
| `--seed` | none | Fixed seed for reproducible output |

## Web Demo

An interactive web interface built with Gradio, with live streaming generation.

```bash
python app.py
# then open http://127.0.0.1:7860
```

![Web demo](assets/demo.png)

## Results

| Setting | Value |
|---|---|
| Hardware | CPU (Intel, Lenovo IdeaPad Slim 3) |
| Parameters | 0.82M |
| Layers / Heads / Embedding | 4 / 4 / 128 |
| Context length | 128 characters |
| Training steps | 3,000 |
| Training time | ~X min |
| **Validation loss** | **1.597** (random baseline: 4.17) |

![Loss curve](assets/loss_curve.png)

### Sample output

```
PASTE YOUR GENERATED TEXT HERE
```

The model learns the **form** of the text (dialogue format, character names, spelling and punctuation), not its meaning. This is expected at this scale.

## Experiments

| Change | Best val loss | Kept? |
|---|---|---|
| Baseline (constant LR 1e-3) | **1.597** | ✅ |
| Cosine LR schedule with warmup (peak 1e-3) | 1.609 | ❌ |

With only 3,000 steps the model is still underfitting, so decaying the learning rate slowed learning down instead of helping.

## Configuration

All hyperparameters live in [`src/config.py`](src/config.py). A small preset is used on CPU, and a larger one (10.8M parameters, 6 layers, 384 embedding) is applied automatically when CUDA is available.

## Acknowledgements

- [Attention Is All You Need](https://arxiv.org/abs/1706.03762) (Vaswani et al., 2017)
- Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) and the *Let's build GPT* lecture
- Dataset: [Tiny Shakespeare](https://github.com/karpathy/char-rnn)

## License

[MIT](LICENSE)