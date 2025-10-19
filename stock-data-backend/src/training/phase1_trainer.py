from __future__ import annotations

import logging

from config import config
from data.storage import Storage
from ml.config import Config
from ml.utils import save_artifacts, prepare_data, train_loop


def run_phase1(storage: Storage, logger: logging.Logger):
    logger.info("--- Starting Phase 1: Long-term training ---")
    phase1_cfg = Config(
        window=60,  # Reduced window size
        epochs=15,
        train_sample_fraction=0.8, # Increased data sample
        horizon=5,
        hidden_size=128, # Simplified model
        num_layers=1, # Simplified model
        lr=1e-4,
        weight_decay=1e-2,
        batch_size=512,
        early_stopping_patience=7,
        label_smoothing=0.1,
        num_workers=4,
    )

    df, feats = prepare_data(phase1_cfg, storage, logger)
    if df is None or not feats:
        logger.error("Data preparation for Phase 1 failed.")
        return

    # The 'sentiment' feature is now included to ensure model compatibility with Phase 2.

    # --- Overfit Sanity Check ---
    # Set debug_overfit=False to run the full training.
    model, scaler, metrics = train_loop(phase1_cfg, df, feats, logger, debug_overfit=False)

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
