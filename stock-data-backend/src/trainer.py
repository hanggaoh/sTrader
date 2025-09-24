from __future__ import annotations

import argparse
import logging

from ml.utils import run_training, Config

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument('--symbols', type=str, nargs='+', default=None, help='Train on a specific list of stock symbols.')
    p.add_argument('--sample-fraction', type=float, default=None, help='Randomly train on a fraction of all available symbols.')
    p.add_argument('--artifacts-dir', type=str, default='artifacts')
    p.add_argument('--logs-dir', type=str, default='logs')
    p.add_argument('--seed', type=int, default=1337)
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--use-returns', action='store_true', default=True)
    p.add_argument('--no-returns', dest='use_returns', action='store_false')
    p.add_argument('--min-per-symbol-rows', type=int, default=200)
    p.add_argument('--const-sentiment', type=float, default=None)
    p.add_argument('--hidden', type=int, default=256)
    p.add_argument('--layers', type=int, default=3)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--bidir', action='store_true', default=False)
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--val-size', type=float, default=0.1)
    p.add_argument('--test-size', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=0)

    args, _ = p.parse_known_args()
    return Config(
        symbols=args.symbols,
        train_sample_fraction=args.sample_fraction,
        artifacts_dir=args.artifacts_dir,
        logs_dir=args.logs_dir,
        seed=args.seed,
        window=args.window,
        horizon=args.horizon,
        use_returns=args.use_returns,
        min_per_symbol_rows=args.min_per_symbol_rows,
        const_sentiment=args.const_sentiment,
        hidden_size=args.hidden,
        num_layers=args.layers,
        dropout=args.dropout,
        bidirectional=args.bidir,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        epochs=args.epochs,
        val_size=args.val_size,
        test_size=args.test_size,
        early_stopping_patience=args.patience,
        clip_grad=args.clip_grad,
        num_workers=args.workers,
    )

def main():
    """Sets up configuration and starts the training process."""
    # --- Basic Logging Setup ---
    # This ensures all informational messages are visible in the console.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    # ---------------------------

    # For easy debugging, use a more powerful default config
    cfg = Config(
        train_sample_fraction=0.2, # Use 20% of stocks
        epochs=20,                 # Train for more epochs
        hidden_size=256,           # Deeper model
        num_layers=3               # Deeper model
    )
    # To use command-line arguments, comment the line above and uncomment the one below
    # cfg = parse_args()

    run_training(cfg)

if __name__ == "__main__":
    main()
