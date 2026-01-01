"""
Main entry point for Khmer space injection RNN training and inference
"""

import argparse

from src.utils import str2bool, train, inference


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Khmer Space Injector - Train / Inference")

    p.add_argument("--mode", type=str, default="train", choices=["train", "infer"])

    # data
    p.add_argument("--train_path", type=str, default="data", help="Path to training data (folder, file, or glob).")
    p.add_argument("--text_path", type=str, default=None, help="Path to input .txt file for inference.")
    p.add_argument("--output_path", type=str, default="inference_output.txt", help="Path to save inference output (.txt).")

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
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--nesterov", action="store_true", default=False)

    # model hparams (DO NOT change str2bool usage)
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


def main() -> None:
    args = build_arg_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        inference(args)


if __name__ == "__main__":
    main()
