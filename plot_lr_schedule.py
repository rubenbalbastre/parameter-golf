from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib.pyplot as plt


def lr_mul(
    step: int,
    iterations: int,
    warmdown_iters: int,
    lr_warmup_iters: int,
    min_lr_mul: float,
) -> float:
    if warmdown_iters <= 0:
        if lr_warmup_iters > 0 and step < lr_warmup_iters:
            return step / max(lr_warmup_iters, 1)
        return 1.0

    if lr_warmup_iters > 0 and step < lr_warmup_iters:
        return step / max(lr_warmup_iters, 1)

    warmdown_start = max(iterations - warmdown_iters, 0)
    if step < warmdown_start:
        return 1.0

    progress = min((step - warmdown_start) / max(warmdown_iters, 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr_mul + (1.0 - min_lr_mul) * cosine


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot the train_gpt.py LR multiplier schedule.")
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmdown-iters", type=int, default=2000)
    parser.add_argument("--lr-warmup-iters", type=int, default=250)
    parser.add_argument("--min-lr-mul", type=float, default=0.1)
    parser.add_argument("--output", type=Path, default=Path("lr_schedule.png"))
    args = parser.parse_args()

    steps = list(range(args.iterations + 1))
    lrs = [
        lr_mul(
            step,
            iterations=args.iterations,
            warmdown_iters=args.warmdown_iters,
            lr_warmup_iters=args.lr_warmup_iters,
            min_lr_mul=args.min_lr_mul,
        )
        for step in steps
    ]

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(steps, lrs, linewidth=2)
    ax.set_title("Learning Rate Multiplier Schedule")
    ax.set_xlabel("Step")
    ax.set_ylabel("LR Multiplier")
    ax.set_xlim(0, args.iterations)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)

    warmdown_start = max(args.iterations - args.warmdown_iters, 0)
    if args.lr_warmup_iters > 0:
        ax.axvline(args.lr_warmup_iters, color="tab:green", linestyle="--", alpha=0.7, label="warmup end")
    ax.axvline(warmdown_start, color="tab:red", linestyle="--", alpha=0.7, label="warmdown start")
    ax.legend()
    fig.tight_layout()
    fig.savefig(args.output, dpi=160)
    print(f"saved {args.output}")


if __name__ == "__main__":
    main()
