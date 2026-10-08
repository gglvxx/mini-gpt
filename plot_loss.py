"""Desenează curba de loss (train vs val) din istoricul salvat de train.py."""
import json
import math

import matplotlib

matplotlib.use("Agg")  # salvează direct imaginea, fără să deschidă o fereastră
import matplotlib.pyplot as plt

from src.config import ROOT_DIR, config

OUTPUT_PATH = ROOT_DIR / "assets" / "loss_curve.png"


def main():
    if not config.history_path.exists():
        raise SystemExit(f"Nu există {config.history_path}. Rulează întâi: python train.py")

    history = json.loads(config.history_path.read_text())
    evals = history["evals"]
    steps = [e["step"] for e in evals]
    train = [e["train"] for e in evals]
    val = [e["val"] for e in evals]

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, train, label="Train loss", marker="o", markersize=3)
    ax.plot(steps, val, label="Validation loss", marker="o", markersize=3)

    # Linia de referință: loss-ul unui model care ghicește complet aleatoriu
    baseline = math.log(history["vocab_size"])
    ax.axhline(baseline, color="gray", linestyle="--", linewidth=1,
               label=f"Random guess ({baseline:.2f})")

    # Marchează cel mai bun punct de validare (cel salvat în checkpoint)
    best_i = min(range(len(val)), key=val.__getitem__)
    ax.annotate(f"Best val: {val[best_i]:.3f}",
                xy=(steps[best_i], val[best_i]),
                xytext=(0, 25), textcoords="offset points", ha="center",
                arrowprops=dict(arrowstyle="->"))

    ax.set_title("Mini-GPT training on Tiny Shakespeare")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.grid(alpha=0.3)
    ax.legend()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Grafic salvat în: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()