from __future__ import annotations

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import logging

from config import config
from data.storage import Storage
from ml.config import Config
from ml.utils import save_artifacts, prepare_data, train_loop


def run_phase1(storage: Storage, logger: logging.Logger):
    logger.info("--- Starting Phase 1: Long-term training ---")
    phase1_cfg = Config(
        window=60,
        epochs=15,
        train_sample_fraction=0.1,
        horizon=5,
        hidden_size=128,       # Reduced model complexity to prevent overfitting
        num_layers=2,
        dropout=0.4,           # Increased dropout for even stronger regularization
        bidirectional=True,    # Use a bidirectional LSTM to capture more context
        lr=2e-4,               # A slightly smaller learning rate for more stable training
        weight_decay=1e-4,
        batch_size=256,
        early_stopping_patience=10, # Give the model more time to find a good solution
        label_smoothing=0.1,
        num_workers=2,
        seed=1337,
    )

    df, seq_feats, static_feats = prepare_data(phase1_cfg, storage, logger)
    if df is None or not seq_feats:
        logger.error("Data preparation for Phase 1 failed.")
        return

    model, scaler, metrics = train_loop(phase1_cfg, df, seq_feats, static_feats, logger, debug_overfit=False)

    if model:
        outdir = save_artifacts(model, scaler, phase1_cfg, metrics, logger)
        logger.info(f"--- Phase 1 complete. Artifacts: {outdir} ---")
    else:
        logger.error("Phase 1 training failed or was in debug mode.")


def main():
    """Sets up configuration and starts the training process for Phase 1."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase1_trainer")

    storage = None
    try:
        storage = Storage(config)
        run_phase1(storage, logger)
    finally:
        if storage:
            storage.close()


if __name__ == "__main__":
    main()
