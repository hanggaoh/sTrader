from __future__ import annotations
import os
import pickle
import json
from typing import Dict

import torch
import pandas as pd
import numpy as np

# Add src directory to path to allow for imports from the src module.
import sys
try:
    # Assumes the script is in <project_root>/src/ml/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
except NameError:
    # Assumes the interactive session is run from the project root.
    sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "src")))


from ml.utils import LSTMRegressor, Config

def load_artifacts(run_id: str, artifacts_dir: str = "artifacts") -> Dict:
    """
    Loads the artifacts for a given training run.

    Args:
        run_id: The ID of the training run (the timestamped folder name).
        artifacts_dir: The root directory where artifacts are stored.

    Returns:
        A dictionary containing the loaded model, scalers, and config.
    """
    outdir = os.path.join(artifacts_dir, run_id)
    if not os.path.exists(outdir):
        raise FileNotFoundError(f"Artifacts directory not found for run_id: {run_id}")

    # Load config
    with open(os.path.join(outdir, "config.json"), "r") as f:
        cfg_dict = json.load(f)
        cfg = Config(**cfg_dict)

    # Load model
    device = torch.device(cfg.device)
    model = LSTMRegressor(
        input_size=len(cfg.features) if cfg.features else 6, # Default to 6 features if not specified
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    )
    model.load_state_dict(torch.load(os.path.join(outdir, "model.pt"), map_location=device))
    model.to(device)
    model.eval()

    # Load scalers
    with open(os.path.join(outdir, "scalers.pkl"), "rb") as f:
        scalers = pickle.load(f)

    return {"model": model, "scalers": scalers, "config": cfg}


@torch.no_grad()
def predict_next(artifacts: Dict, recent_df: pd.DataFrame) -> float:
    """
    Predicts the next value using a loaded model and recent data.

    Args:
        artifacts: The dictionary of loaded artifacts from load_artifacts.
        recent_df: A DataFrame with the most recent `window` rows of data for a SINGLE symbol.
                   Must contain the feature columns used during training.

    Returns:
        The predicted value (e.g., next return or price), inverse-transformed.
    """
    model = artifacts["model"]
    scalers = artifacts["scalers"]
    cfg = artifacts["config"]
    device = torch.device(cfg.device)

    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]
    
    # Ensure the features are present, default if necessary
    feats = cfg.features or ["open", "high", "low", "close", "volume", "sentiment"]
    
    # Create a copy to avoid the SettingWithCopyWarning
    data_for_pred = recent_df.copy()
    if "sentiment" not in data_for_pred.columns:
        data_for_pred["sentiment"] = 0.0

    if len(data_for_pred) < cfg.window:
        raise ValueError(f"Input data must have at least `window` rows. Got {len(data_for_pred)}, expected {cfg.window}.")

    # Prepare the input tensor
    X = data_for_pred[feats].values.astype(np.float32)
    X_scaled = x_scaler.transform(X)
    x_t = torch.from_numpy(X_scaled[-cfg.window:][None, ...]).to(device)

    # Make prediction
    pred_scaled = model(x_t).item()

    # Inverse transform to get the real value
    prediction = float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])

    return prediction
