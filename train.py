"""Antrenează MiniGPT și salvează cel mai bun model în checkpoints/model.pt."""
import argparse
import json  # NOU
import time

import torch
from tqdm import tqdm

from src.config import config
from src.dataset import TextDataset
from src.model import MiniGPT


@torch.no_grad()
def estimate_loss(model, dataset) -> dict[str, float]:
    """Media loss-ului pe mai multe batch-uri. Un singur batch ar da o valoare prea 'zgomotoasă'."""
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(config.eval_iters)
        for i in range(config.eval_iters):
            x, y = dataset.get_batch(split)
            _, loss = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def save_checkpoint(model, dataset, step: int, val_loss: float) -> None:
    """Salvează tot ce trebuie ca generate.py să reconstruiască modelul fără dataset."""
    config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "chars": dataset.tokenizer.chars,
            "model_args": {
                k: getattr(config, k)
                for k in ("block_size", "n_embd", "n_head", "n_layer", "dropout")
            },
            "step": step,
            "val_loss": val_loss,
        },
        config.checkpoint_path,
    )


def record_history(history: dict, step: int, losses: dict) -> None:  # NOU
    """Adaugă o evaluare în istoric și rescrie fișierul JSON imediat.
    Scriere la fiecare evaluare -> datele rămân salvate chiar dacă oprești cu Ctrl+C."""
    history["evals"].append({"step": step, "train": losses["train"], "val": losses["val"]})
    config.history_path.parent.mkdir(parents=True, exist_ok=True)
    config.history_path.write_text(json.dumps(history, indent=2))


def main():
    parser = argparse.ArgumentParser(description="Antrenare MiniGPT")
    parser.add_argument("--max-iters", type=int, default=config.max_iters,
                        help="Număr de pași de antrenare (default din config.py)")
    args = parser.parse_args()
    config.max_iters = args.max_iters

    torch.manual_seed(config.seed)
    dataset = TextDataset(config)
    model = MiniGPT(config, dataset.tokenizer.vocab_size).to(config.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)

    print(f"Device: {config.device} | Parametri: {model.num_params() / 1e6:.2f}M | "
          f"Pași: {config.max_iters}")

    best_val = float("inf")
    history = {"vocab_size": dataset.tokenizer.vocab_size, "evals": []}  # NOU
    start_time = time.time()

    try:
        pbar = tqdm(range(config.max_iters), desc="Antrenare")
        for step in pbar:
            if step % config.eval_interval == 0:
                losses = estimate_loss(model, dataset)
                record_history(history, step, losses)  # NOU
                msg = f"Pas {step:5d} | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}"
                if losses["val"] < best_val:
                    best_val = losses["val"]
                    save_checkpoint(model, dataset, step, best_val)
                    msg += "  -> checkpoint salvat"
                tqdm.write(msg)

            x, y = dataset.get_batch("train")
            _, loss = model(x, y)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % 10 == 0:
                pbar.set_postfix(loss=f"{loss.item():.3f}")

        losses = estimate_loss(model, dataset)
        record_history(history, config.max_iters, losses)  # NOU
        print(f"Final    | train loss {losses['train']:.4f} | val loss {losses['val']:.4f}")
        if losses["val"] < best_val:
            best_val = losses["val"]
            save_checkpoint(model, dataset, config.max_iters, best_val)

    except KeyboardInterrupt:
        print("\nOprit manual. Cel mai bun checkpoint a fost deja salvat.")

    elapsed = (time.time() - start_time) / 60
    print(f"\nDurată: {elapsed:.1f} min | Cel mai bun val loss: {best_val:.4f}")
    print(f"Model salvat în: {config.checkpoint_path}")
    print(f"Istoric salvat în: {config.history_path}")  # NOU

    model.eval()
    start = torch.zeros((1, 1), dtype=torch.long, device=config.device)
    print("\n--- Mostră generată ---")
    print(dataset.tokenizer.decode(model.generate(start, 300, top_k=10)[0]))


if __name__ == "__main__":
    main()