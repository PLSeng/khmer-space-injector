"""
Main entry point for Khmer space injection RNN training and inference
"""

import argparse
from argparse import BooleanOptionalAction
import json
import os
import time
from datetime import datetime
from typing import Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.net import KhmerRNN
from src.dataloader import load_data
from src.utils import str2bool


# ======================================================
# Logging helpers
# ======================================================
def _now() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def _log(msg: str) -> None:
    print(f"[{_now()}] {msg}")


# ======================================================
# W&B helpers
# ======================================================
def _maybe_init_wandb(args) -> Optional[object]:
    if not args.use_wandb:
        return None
    try:
        import wandb  # type: ignore
    except Exception:
        _log("[WARN] wandb not available. Continuing without wandb logging.")
        return None

    wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config=vars(args),
    )
    return wandb


def _select_device(device_arg: str) -> torch.device:
    if device_arg == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _save_checkpoint(model: nn.Module, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(model.state_dict(), path)


def _save_vocab(vocab: dict, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)


def _load_vocab(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _compute_token_accuracy(logits: torch.Tensor, targets: torch.Tensor, ignore_index: int = -100) -> float:
    with torch.no_grad():
        pred = logits.argmax(dim=-1)
        mask = (targets != ignore_index)
        if mask.sum().item() == 0:
            return 0.0
        correct = (pred[mask] == targets[mask]).sum().item()
        total = mask.sum().item()
        return correct / total


def _decode_spaces(text: str, pred_labels: list[int]) -> str:
    """
    label=1 means insert a space AFTER this character
    """
    out = []
    for ch, yhat in zip(text, pred_labels):
        out.append(ch)
        if yhat == 1:
            out.append(" ")
    return "".join(out).strip()


def _make_optimizer(args, model: nn.Module):
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


def _make_criterion(args):
    if args.loss.lower() == "ce":
        return nn.CrossEntropyLoss(ignore_index=-100)
    raise ValueError(f"Unknown loss: {args.loss}")


# ======================================================
# Train / Infer
# ======================================================
def train(args):
    device = _select_device(args.device)
    wandb = _maybe_init_wandb(args)

    _log("Loading training data...")
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
    _log("=== Training config ===")
    _log(f"device={device} | epochs={args.epochs} | batch_size={args.batch_size} | lr={args.lr}")
    _log(f"optimizer={args.optimizer} | weight_decay={args.weight_decay} | loss={args.loss} | grad_clip={args.grad_clip}")
    _log(f"max_length={args.max_length} | num_workers={args.num_workers}")
    _log(f"model: rnn_type={args.rnn_type} | emb={args.embedding_dim} | hid={args.hidden_dim} | layers={args.num_layers}")
    _log(f"dropout={args.dropout} | bidirectional={args.bidirectional} | residual={args.residual}")
    _log(f"vocab_size={len(vocab)}")

    try:
        _log(f"train_batches_per_epoch={len(train_loader)}")
    except Exception:
        _log("train_batches_per_epoch=unknown (DataLoader has no __len__)")

    if args.vocab_path:
        _save_vocab(vocab, args.vocab_path)
        _log(f"[Saved vocab] {args.vocab_path}")

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
        _log(f"model_params_total={n_params:,} | trainable={n_train:,}")
    except Exception:
        pass

    criterion = _make_criterion(args)
    optimizer = _make_optimizer(args, model)

    best_loss = float("inf")
    global_step = 0
    total_start = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        n = 0
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

            acc = _compute_token_accuracy(logits, y, ignore_index=-100)

            running_loss += loss.item()
            running_acc += acc
            n += 1
            global_step += 1

            # tqdm live metrics
            if args.tqdm:
                epoch_iter.set_postfix(loss=f"{loss.item():.4f}", acc=f"{acc:.4f}")

            # periodic print logs (avoid spamming)
            if args.log_every and args.log_every > 0 and (step % args.log_every == 0):
                try:
                    total_steps = len(train_loader)
                    _log(f"epoch={epoch} step={step}/{total_steps} loss={loss.item():.4f} token_acc={acc:.4f}")
                except Exception:
                    _log(f"epoch={epoch} step={step} loss={loss.item():.4f} token_acc={acc:.4f}")

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

        epoch_loss = running_loss / max(1, n)
        epoch_acc = running_acc / max(1, n)
        elapsed = time.time() - epoch_start
        _log(f"[Epoch {epoch}/{args.epochs}] loss={epoch_loss:.4f} token_acc={epoch_acc:.4f} time={elapsed:.1f}s")

        if epoch_loss < best_loss:
            best_loss = epoch_loss
            if args.ckpt_path:
                _save_checkpoint(model, args.ckpt_path)
                _log(f"[Saved] {args.ckpt_path} (best_loss={best_loss:.4f})")

        if wandb is not None:
            wandb.log(
                {
                    "train/epoch_loss": epoch_loss,
                    "train/epoch_token_acc": epoch_acc,
                    "train/best_loss": best_loss,
                },
                step=global_step,
            )

    _log(f"Training done. Total time={(time.time() - total_start) / 60:.1f} min")

    if wandb is not None:
        wandb.finish()


def inference(args):
    device = _select_device(args.device)
    wandb = _maybe_init_wandb(args)

    if not args.ckpt_path or not os.path.isfile(args.ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {args.ckpt_path}")

    if not args.vocab_path or not os.path.isfile(args.vocab_path):
        raise FileNotFoundError(
            f"Vocab not found: {args.vocab_path}. Train first (it saves vocab), or provide --vocab_path."
        )

    text = (args.text or "").strip()
    if not text:
        raise ValueError("Provide input text with --text")

    _log("Loading vocab + checkpoint...")
    vocab = _load_vocab(args.vocab_path)

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
    ids = [vocab.get(ch, unk) for ch in text]

    if len(ids) > args.max_length:
        _log(f"[WARN] input_len={len(ids)} > max_length={args.max_length}. Truncating.")
        ids = ids[: args.max_length]
        text = text[: args.max_length]

    x = torch.tensor(ids, dtype=torch.long).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        pred = logits.argmax(dim=-1).squeeze(0).tolist()

    segmented = _decode_spaces(text, pred)
    print(segmented)

    if wandb is not None:
        wandb.log({"infer/input_len": len(text), "infer/output_len": len(segmented)})
        wandb.finish()


# ======================================================
# CLI
# ======================================================
def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Khmer Space Injector - Train / Inference")

    p.add_argument("--mode", type=str, default="train", choices=["train", "infer"])

    # data
    p.add_argument(
        "--train_path",
        type=str,
        default="data",  # folder or file/glob depending on your dataloader
        help="Path to training data (folder, file, or glob).",
    )
    p.add_argument("--text", type=str, default=None, help="Input Khmer text (no spaces) for inference.")

    # checkpoint + vocab
    p.add_argument("--ckpt_path", type=str, default="checkpoints/khmer_rnn.pt")
    p.add_argument("--vocab_path", type=str, default="checkpoints/vocab.json")

    # training
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight_decay", type=float, default=1e-2)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--max_length", type=int, default=128)

    # logging
    p.add_argument("--log_every", type=int, default=50, help="Print a log line every N steps (0 disables).")
    p.add_argument("--tqdm", action="store_true", default=True, help="Enable tqdm progress bars.")

    # optimizer/loss tuning
    p.add_argument("--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"])
    p.add_argument("--loss", type=str, default="ce", choices=["ce"])
    p.add_argument("--momentum", type=float, default=0.9)  # for SGD
    p.add_argument("--nesterov", action="store_true", default=False)  # for SGD

    # model hparams
    p.add_argument("--embedding_dim", type=int, default=128)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--num_layers", type=int, default=2)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--bidirectional", type=str2bool, default=True)
    p.add_argument("--residual", type=str2bool, default=True)
    p.add_argument("--rnn_type", type=str, default="lstm", choices=["rnn", "gru", "lstm"])

    # system
    p.add_argument("--device", type=str, default="cuda", choices=["cpu", "cuda"])

    # wandb
    p.add_argument("--use_wandb", action="store_true", default=False)
    p.add_argument("--wandb_project", type=str, default="khmer-space-injector")
    p.add_argument("--wandb_run_name", type=str, default=None)

    return p


def main():
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
