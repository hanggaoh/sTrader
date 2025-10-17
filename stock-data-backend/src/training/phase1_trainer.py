from __future__ import annotations

import logging

from config import config
from data.storage import Storage
from ml.config import Config
from ml.utils import save_artifacts, prepare_data, train_loop


def run_phase1(storage: Storage, logger: logging.Logger):
    logger.info("--- Starting Phase 1: Long-term training ---")
    phase1_cfg = Config(
        window=120,  # Big window size
        epochs=15,
        train_sample_fraction=0.1,
        horizon=5,
        hidden_size=512,
        num_layers=2,
        lr=5e-4,
        batch_size=512,
        early_stopping_patience=7,
        label_smoothing=0.1,
        num_workers=4,
    )

    df, feats = prepare_data(phase1_cfg, storage, logger)
    if df is None or not feats:
        logger.error("Data preparation for Phase 1 failed.")
        return

    # Exclude sentiment for Phase 1
    feats = [f for f in feats if f != 'sentiment']

    model, scaler, metrics = train_loop(phase1_cfg, df, feats, logger)
    if model:
        outdir = save_artifacts(model, scaler, phase1_cfg, metrics, logger)
        logger.info(f"--- Phase 1 complete. Artifacts: {outdir} ---")
    else:
        logger.error("Phase 1 training failed.")


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
