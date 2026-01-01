"""
Utility functions for Khmer space injection RNN model
"""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional, Iterable

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.net import KhmerRNN


# ======================================================
# Reproducibility
# ======================================================
def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Deterministic behavior (may reduce performance slightly)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ======================================================
# Text Utils
# ======================================================
def char_to_idx(char: str, char_to_index: dict) -> int:
    if char not in char_to_index:
        raise ValueError(f"Character '{char}' not found in char_to_index dictionary")
    return char_to_index[char]


def idx_to_char(idx: int, index_to_char: dict) -> str:
    if idx not in index_to_char:
        raise ValueError(f"Index '{idx}' not found in index_to_char dictionary")
    return index_to_char[idx]


def build_vocab(texts: Iterable[str]) -> tuple[dict, dict]:
    chars = set()
    for text in texts:
        chars.update(text)

    chars = sorted(chars)
    char_to_index = {ch: i for i, ch in enumerate(chars)}
    index_to_char = {i: ch for ch, i in char_to_index.items()}
    return char_to_index, index_to_char


# ======================================================
# Logging
# ======================================================
def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def log(msg: str) -> None:
    print(f"[{now_str()}] {msg}")


# ======================================================
# Small helpers (I/O)
# ======================================================
def chunk_text(text: str, max_length: int) -> list[str]:
    return [text[i : i + max_length] for i in range(0, len(text), max_length)]


def read_text_file(path: Optional[str]) -> str:
    if not path:
        return ""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Text file not found: {path}")
    if not path.lower().endswith(".txt"):
        raise ValueError("Inference input must be a .txt file (e.g., --text_path input.txt)")
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def ensure_parent_dir(path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)


# ======================================================
# Vocab + checkpoint
# ======================================================
def save_checkpoint(model: nn.Module, path: str) -> None:
    ensure_parent_dir(path)
    torch.save(model.state_dict(), path)


def save_vocab(vocab: dict, path: str) -> None:
    ensure_parent_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ======================================================
# Device / metrics
# ======================================================
def select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def compute_token_accuracy(
    logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100
) -> float:
    pred = logits.argmax(dim=-1)
    mask = targets != ignore_index
    if mask.sum().item() == 0:
        return 0.0
    correct = (pred[mask] == targets[mask]).sum().item()
    total = mask.sum().item()
    return correct / total


# ======================================================
# Khmer decoding (space insertion)
# ======================================================
_KHMER_COMBINING = {
    "្",  # coeng (subscript marker)
    "់", "ៈ", "៎", "៏", "័", "៌", "៍", "៑", "៓", "៕", "។", "៘",
    "ា", "ិ", "ី", "ឹ", "ឺ", "ុ", "ូ", "ួ", "ើ", "ឿ", "ៀ", "េ", "ែ", "ៃ", "ោ", "ៅ",
    "ំ", "ះ",
}


def decode_spaces(text: str, pred_labels: list[int]) -> str:
    """
    label=1 means insert a space AFTER this character.
    Avoid inserting spaces before Khmer combining marks to not split grapheme clusters.
    """
    out: list[str] = []
    n = min(len(text), len(pred_labels))

    for i in range(n):
        ch = text[i]
        out.append(ch)

        if pred_labels[i] != 1:
            continue

        # If next codepoint is combining, don't insert a space
        if i + 1 < n and text[i + 1] in _KHMER_COMBINING:
            continue

        out.append(" ")

    return "".join(out).strip()


# ======================================================
# Optimizer / loss
# ======================================================
def make_optimizer(args: Any, model: nn.Module) -> torch.optim.Optimizer:
    opt_name = args.optimizer.lower()

    if opt_name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if opt_name == "adam":
        return torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if opt_name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    raise ValueError(f"Unknown optimizer: {args.optimizer}")


def make_criterion(args: Any) -> nn.Module:
    if args.loss.lower() == "ce":
        return nn.CrossEntropyLoss(ignore_index=-100)
    raise ValueError(f"Unknown loss: {args.loss}")


# ======================================================
# W&B
# ======================================================
def maybe_init_wandb(args: Any) -> Optional[Any]:
    if not getattr(args, "use_wandb", False):
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        log("[WARN] wandb not available. Continuing without wandb logging.")
        return None

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return wandb


# ======================================================
# Argparse helper (keep your behavior)
# ======================================================
def str2bool(v):
    # accepts True/False from wandb like "--flag=True"
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    s = str(v).strip().lower()
    if s in ("1", "true", "t", "yes", "y", "on"):
        return True
    if s in ("0", "false", "f", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean expected, got: {v}")


# ======================================================
# Train / inference
# ======================================================
def train(args: Any) -> None:
    from src.dataloader import load_data
    device = select_device(args.device)
    wandb = maybe_init_wandb(args)

    log("Loading training data...")
    train_loader, vocab = load_data(
        data_dir=args.train_path,
        batch_size=args.batch_size,
        max_length=args.max_length,
        char_to_index=None,
        shuffle=True,
        num_workers=args.num_workers,
        skip_invalid=True,
        return_vocab=True,
    )

    # config logs
    log("=== Training config ===")
    log(f"device={device} | epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}")
    log(f"optimizer={args.optimizer} | weight_decay={args.weight_decay} | loss={args.loss} | grad_clip={args.grad_clip}")
    log(f"max_length={args.max_length} | num_workers={args.num_workers}")
    log(f"model: rnn_type={args.rnn_type} | emb={args.embedding_dim} | hid={args.hidden_dim} | layers={args.num_layers}")
    log(f"dropout={args.dropout} | bidirectional={args.bidirectional} | residual={args.residual}")
    log(f"vocab_size={len(vocab)}")

    try:
        log(f"train_batches_per_epoch={len(train_loader)}")
    except Exception:
        log("train_batches_per_epoch=unknown (DataLoader has no __len__)")

    if args.vocab_path:
        save_vocab(vocab, args.vocab_path)
        log(f"[Saved vocab] {args.vocab_path}")

    model = KhmerRNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        rnn_type=args.rnn_type,
        residual=args.residual,
        use_crf=False,
    ).to(device)

    # parameter count log
    try:
        n_params = sum(p.numel() for p in model.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"model_params_total={n_params:,} | trainable={n_train:,}")
    except Exception:
        pass

    criterion = make_criterion(args)
    optimizer = make_optimizer(args, model)

    best_loss = float("inf")
    global_step = 0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n_steps = 0
        epoch_start = time.time()

        epoch_iter = train_loader
        if args.tqdm:
            epoch_iter = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)

        for step, (x, y) in enumerate(epoch_iter, start=1):
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad(set_to_none=True)

            logits = model(x)  # (B,T,2)
            loss = criterion(logits.view(-1, 2), y.view(-1))
            loss.backward()

            if args.grad_clip and args.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)

            optimizer.step()

            acc = compute_token_accuracy(logits, y, ignore_index=-100)

            running_loss += loss.item()
            running_acc += acc
            n_steps += 1
            global_step += 1

            if args.tqdm:
                epoch_iter.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

            if args.log_every and args.log_every > 0 and (step % args.log_every == 0):
                try:
                    total_steps = len(train_loader)
                    log(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} token_acc={acc:.4f}")
                except Exception:
                    log(f"epoch={epoch} step={step} loss={loss.item():.4f} token_acc={acc:.4f}")

            if wandb is not None:
                wandb.log(
                    {
                        "train/loss": loss.item(),
                        "train/token_acc": acc,
                        "train/step": global_step,
                        "train/epoch": epoch,
                    },
                    step=global_step,
                )

        epoch_loss = running_loss / max(1, n_steps)
        epoch_acc = running_acc / max(1, n_steps)
        elapsed = time.time() - epoch_start
        log(f"[Epoch {epoch}/{args.epochs}] loss={epoch_loss:.4f} token_acc={epoch_acc:.4f} time={elapsed:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if args.ckpt_path:
                save_checkpoint(model, args.ckpt_path)
                log(f"[Saved] {args.ckpt_path} (best_loss={best_loss:.4f})")

        if wandb is not None:
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_token_acc": epoch_acc,
                    "train/best_loss": best_loss,
                },
                step=global_step,
            )

    log(f"Training done. Total time={(time.time() - total_start) / 60:.1f} min")

    if wandb is not None:
        wandb.finish()


def inference(args: Any) -> None:
    device = select_device(args.device)
    wandb = maybe_init_wandb(args)

    if not args.ckpt_path or not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    if not args.vocab_path or not os.path.isfile(args.vocab_path):
        raise FileNotFoundError(
            f"Vocab not found: {args.vocab_path}. Train first (it saves vocab), or provide --vocab_path."
        )

    text = read_text_file(args.text_path).strip()
    if not text:
        raise ValueError("Provide a non-empty .txt file with --text_path")

    log("Loading vocab + checkpoint...")
    vocab = load_vocab(args.vocab_path)

    model = KhmerRNN(
        vocab_size=len(vocab),
        embedding_dim=args.embedding_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout,
        bidirectional=args.bidirectional,
        rnn_type=args.rnn_type,
        residual=args.residual,
        use_crf=False,
    ).to(device)

    state = torch.load(args.ckpt_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    unk = vocab.get("<UNK>", 1)

    chunks = chunk_text(text, args.max_length)
    log(f"Inference on {len(chunks)} chunks (max_length={args.max_length})")

    outputs: list[str] = []
    for chunk in chunks:
        ids = [vocab.get(ch, unk) for ch in chunk]
        x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(x)
            pred = logits.argmax(dim=-1).squeeze(0).tolist()

        outputs.append(decode_spaces(chunk, pred))

    segmented = "".join(outputs)

    out_path = args.output_path or "inference_output.txt"
    ensure_parent_dir(out_path)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(segmented)

    log(f"[Saved inference output] {out_path}")
    log(f"input_chars={len(text)} | output_chars={len(segmented)}")

    if wandb is not None:
        wandb.log({"infer/input_len": len(text), "infer/output_len": len(segmented)})
        wandb.finish()
