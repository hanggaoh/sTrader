from __future__ import annotations

import argparse
import json
import logging
import os

import torch

from config import config
from data.storage import Storage
from ml.config import Config
from ml.utils import save_artifacts, prepare_data, train_loop, LSTMClassifier


def run_phase2(storage: Storage, logger: logging.Logger, model: torch.nn.Module):
    logger.info("--- Starting Phase 2: Short-term fine-tuning ---")
    phase2_cfg = Config(
        window=30,  # Smaller window size
        epochs=10,
        train_sample_fraction=0.2,
        horizon=5,
        hidden_size=512,
        num_layers=2,
        lr=1e-4,  # Lower learning rate for fine-tuning
        batch_size=512,
        early_stopping_patience=5,
        label_smoothing=0.1,
        num_workers=4,
    )

    df, feats = prepare_data(phase2_cfg, storage, logger)
    if df is None or not feats:
        logger.error("Data preparation for Phase 2 failed.")
        return

    # BUG FIX: Ensure feature set is consistent with phase 1, excluding sentiment.
    feats = [f for f in feats if f != 'sentiment']

    model, scaler, metrics = train_loop(phase2_cfg, df, feats, logger, model=model)
    if model:
        outdir = save_artifacts(model, scaler, phase2_cfg, metrics, logger)
        logger.info(f"--- Phase 2 complete. Artifacts: {outdir} ---")
    else:
        logger.error("Phase 2 training failed.")


def main():
    """Sets up configuration and starts the training process for Phase 2."""
    parser = argparse.ArgumentParser(description="Run Phase 2 training (fine-tuning).")
    parser.add_argument("--phase1-artifacts", required=True, help="Path to the artifacts directory from phase 1.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)-15s - %(levelname)-8s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger = logging.getLogger("phase2_trainer")

    storage = None
    try:
        # --- Load Model from Phase 1 ---
        phase1_artifacts_path = args.phase1_artifacts
        phase1_config_path = os.path.join(phase1_artifacts_path, "config.json")
        phase1_model_path = os.path.join(phase1_artifacts_path, "model.pt")

        with open(phase1_config_path, "r") as f:
            phase1_cfg = json.load(f)

        # To instantiate the model, we need the input size, which depends on the number of features.
        # We determine this by running a "dry run" of prepare_data for phase 2.
        storage = Storage(config)
        temp_phase2_cfg = Config(train_sample_fraction=0.01) # Small sample to get features fast
        _, feats = prepare_data(temp_phase2_cfg, storage, logger)
        
        if not feats:
            logger.error("Could not determine feature set for model loading. Aborting.")
            return
            
        feats = [f for f in feats if f != 'sentiment']
        input_size = len(feats)

        model = LSTMClassifier(
            input_size=input_size,
            hidden_size=phase1_cfg['hidden_size'],
            num_layers=phase1_cfg['num_layers'],
            dropout=phase1_cfg['dropout'],
            bidirectional=phase1_cfg['bidirectional']
        )
        model.load_state_dict(torch.load(phase1_model_path))
        logger.info(f"Successfully loaded model from {phase1_model_path}")

        # --- Run Phase 2 ---
        run_phase2(storage, logger, model)

    except FileNotFoundError:
        logger.error(f"Could not find phase 1 artifacts at the specified path: {args.phase1_artifacts}")
    finally:
        if storage:
            storage.close()


if __name__ == "__main__":
    main()
