from __future__ import annotations
import argparse
import json
import os
import random
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sqlalchemy import create_engine, text

# Project-specific imports.
# Note: This script should be run as a module from the project's root directory
# to ensure these imports work correctly, e.g., `python -m src.ml.trainer`.
from .. import logger_config
from ..config import config

# ----------------------
# Config
# ----------------------
@dataclass
class Config:
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    seed: int = 1337

    # data
    window: int = 60
    horizon: int = 1  # predict t+1 close (or return)
    use_returns: bool = True  # predict next return if True; else scaled close
    min_per_symbol_rows: int = 200
    const_sentiment: Optional[float] = None  # if set, will ignore sentiment column and use this value
    features: Optional[List[str]] = None  # if None: auto ['open','high','low','close','volume','sentiment?']

    # model
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False

    # train
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    val_size: float = 0.1
    test_size: float = 0.1
    early_stopping_patience: int = 7
    clip_grad: float = 1.0
    num_workers: int = 0
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

# ----------------------
# Repro
# ----------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

# ----------------------
# Data utilities
# ----------------------
REQUIRED_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]


def fetch_data(db_url: str, sql: str, logger) -> pd.DataFrame:
    logger.info("Fetching data via SQLAlchemy…")
    engine = create_engine(db_url)
    with engine.connect() as conn:
        df = pd.read_sql(text(sql), conn)
    # normalize column names
    df.columns = [c.lower() for c in df.columns]
    # basic checks
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}' in query result")
    # ensure timestamp is datetime
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])  # drop rows with invalid timestamps
    # sort
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df


def attach_sentiment(df: pd.DataFrame, cfg: Config, logger) -> pd.DataFrame:
    if cfg.const_sentiment is not None:
        logger.warning(
            f"Using constant sentiment={cfg.const_sentiment} for ALL rows (placeholder mode)."
        )
        df["sentiment"] = float(cfg.const_sentiment)
    else:
        if "sentiment" not in df.columns:
            logger.warning(
                "No 'sentiment' column found and no --const-sentiment provided. Defaulting to 0.0."
            )
            df["sentiment"] = 0.0
        # else trust provided sentiment column
    return df


def build_features(df: pd.DataFrame, cfg: Config, logger) -> Tuple[pd.DataFrame, List[str]]:
    # compute returns if requested
    if cfg.use_returns:
        logger.info("Computing log returns for close price…")
        df["close_ret"] = df.groupby("symbol")["close"].apply(lambda s: np.log(s).diff())
        # forward-looking target: next period return
        df["target"] = df.groupby("symbol")["close_ret"].shift(-cfg.horizon)
    else:
        logger.info("Using raw close (scaled) and predicting future close.")
        df["target"] = df.groupby("symbol")["close"].shift(-cfg.horizon)

    # choose features
    base_feats = ["open", "high", "low", "close", "volume", "sentiment"]
    feats = cfg.features or base_feats

    # drop early NaNs due to diff/shift
    df = df.dropna(subset=feats + ["target"]).copy()
    return df, feats


class SequenceDataset(Dataset):
    def __init__(
        self,
        df: pd.DataFrame,
        feats: List[str],
        window: int,
        scalers: Optional[dict] = None,
    ):
        # We'll build sequences per symbol to preserve time order
        self.samples = []  # list of (np.array[window, F], float)
        self.feats = feats
        self.window = window

        # Fit/Use scalers per-feature global (across symbols)
        if scalers is None:
            self.x_scaler = StandardScaler()
            self.y_scaler = StandardScaler()
            fit_x = True
        else:
            self.x_scaler = scalers["x_scaler"]
            self.y_scaler = scalers["y_scaler"]
            fit_x = False

        # Build sequences symbol-wise
        for sym, g in df.groupby("symbol"):
            if len(g) < window + 1:
                continue
            X = g[self.feats].values.astype(np.float32)
            y = g["target"].values.astype(np.float32)
            # standardize X globally (fit only once on full training set outside if needed)
            if fit_x:
                self.x_scaler.partial_fit(X)
            # y scaler fit later when data collated; to keep simple we fit here as partial
            self.y_scaler.partial_fit(y.reshape(-1, 1))
            # create sliding windows
            for i in range(window, len(g)):
                self.samples.append((X[i - window : i], y[i]))

        # After building, transform in-place
        if fit_x:
            # Second pass to actually transform
            transformed = []
            for seq, tgt in self.samples:
                seq_t = self.x_scaler.transform(seq)
                transformed.append((seq_t.astype(np.float32), tgt))
            self.samples = transformed

        # Always transform y
        self.samples = [
            (seq, float(self.y_scaler.transform([[tgt]])[0, 0])) for seq, tgt in self.samples
        ]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)

    def scalers_dict(self):
        return {"x_scaler": self.x_scaler, "y_scaler": self.y_scaler}


# ----------------------
# Model
# ----------------------
class LSTMRegressor(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        self.head = nn.Sequential(
            nn.LayerNorm(d * hidden_size),
            nn.Linear(d * hidden_size, d * hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d * hidden_size // 2, 1),
        )

    def forward(self, x):
        # x: [B, T, F]
        out, _ = self.lstm(x)
        last = out[:, -1, :]  # [B, H]
        return self.head(last).squeeze(-1)


# ----------------------
# Train / Eval
# ----------------------
@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    mse, n = 0.0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb)
        mse += torch.nn.functional.mse_loss(pred, yb, reduction='sum').item()
        n += len(xb)
    return mse / max(n, 1)


def train_loop(cfg: Config, df: pd.DataFrame, feats: List[str], logger):
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # Split by time globally to avoid leakage (simple approach)
    # We'll do a random split on sequences later; here just reserve a test tail by timestamp
    # but since we already built sequences in the Dataset, we can do a standard split

    # Build full dataset (fit scalers inside)
    full_ds = SequenceDataset(df, feats, cfg.window, scalers=None)
    scalers = full_ds.scalers_dict()

    # Train/Val/Test split on indices
    idx = np.arange(len(full_ds))
    train_idx, tmp_idx = train_test_split(idx, test_size=cfg.val_size + cfg.test_size, random_state=cfg.seed, shuffle=True)
    rel_val = cfg.val_size / (cfg.val_size + cfg.test_size)
    val_idx, test_idx = train_test_split(tmp_idx, test_size=1 - rel_val, random_state=cfg.seed, shuffle=True)

    def subset(ds, indices):
        class _Subset(Dataset):
            def __init__(self, base, ids):
                self.base = base
                self.ids = ids
            def __len__(self):
                return len(self.ids)
            def __getitem__(self, i):
                return self.base[self.ids[i]]
        return _Subset(ds, indices)

    train_ds = subset(full_ds, train_idx)
    val_ds = subset(full_ds, val_idx)
    test_ds = subset(full_ds, test_idx)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = LSTMRegressor(
        input_size=len(feats),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    ).to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val = float('inf')
    best_state = None
    patience = cfg.early_stopping_patience

    logger.info(f"Training started: N_train={len(train_ds)}, N_val={len(val_ds)}, N_test={len(test_ds)}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        running = 0.0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = nn.functional.mse_loss(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            running += loss.item() * len(xb)
        train_mse = running / max(1, len(train_ds))
        val_mse = evaluate(model, val_loader, device)
        logger.info(f"Epoch {epoch:03d} | train_mse={train_mse:.6f} | val_mse={val_mse:.6f}")

        if val_mse < best_val - 1e-6:
            best_val = val_mse
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            patience = cfg.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping triggered.")
                break

    # Load best
    if best_state is not None:
        model.load_state_dict(best_state)

    test_mse = evaluate(model, test_loader, device)
    logger.info(f"Test MSE (scaled target): {test_mse:.6f}")
    return model, scalers, {"test_mse": test_mse}


# ----------------------
# Persistence
# ----------------------
import pickle

def save_artifacts(model, scalers, cfg: Config, metrics: dict, logger) -> str:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(cfg.artifacts_dir, run_id)
    os.makedirs(outdir, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "scalers.pkl"), "wb") as f:
        pickle.dump(scalers, f)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    logger.info(f"Artifacts saved to {outdir}")
    return outdir


# ----------------------
# Inference helper (predict next‑horizon return/close depending on cfg)
# ----------------------
@torch.no_grad()
def predict_next(model_path: str, scalers_path: str, recent_df: pd.DataFrame, feats: List[str], cfg: Config) -> float:
    """
    recent_df: latest chronological rows for ONE symbol, containing feature columns.
    Returns the **scaled** prediction. To invert scaling, use scalers['y_scaler'].inverse_transform.
    """
    device = torch.device(cfg.device)
    model = LSTMRegressor(
        input_size=len(feats),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        dropout=cfg.dropout,
        bidirectional=cfg.bidirectional,
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    with open(scalers_path, "rb") as f:
        scalers = pickle.load(f)
    x_scaler = scalers["x_scaler"]
    y_scaler = scalers["y_scaler"]

    X = recent_df[feats].values.astype(np.float32)
    X = x_scaler.transform(X)
    x_t = torch.from_numpy(X[-cfg.window:][None, ...]).to(device)
    pred_scaled = model(x_t).item()
    return float(y_scaler.inverse_transform([[pred_scaled]])[0, 0])


# ----------------------
# CLI
# ----------------------

def parse_args() -> Config:
    p = argparse.ArgumentParser()
    p.add_argument('--artifacts-dir', type=str, default='artifacts')
    p.add_argument('--logs-dir', type=str, default='logs')
    p.add_argument('--seed', type=int, default=1337)

    # data
    p.add_argument('--window', type=int, default=60)
    p.add_argument('--horizon', type=int, default=1)
    p.add_argument('--use-returns', action='store_true', default=True)
    p.add_argument('--no-returns', dest='use_returns', action='store_false')
    p.add_argument('--min-per-symbol-rows', type=int, default=200)
    p.add_argument('--const-sentiment', type=float, default=None)

    # model
    p.add_argument('--hidden', type=int, default=128)
    p.add_argument('--layers', type=int, default=2)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--bidir', action='store_true', default=False)

    # train
    p.add_argument('--batch-size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight-decay', type=float, default=1e-5)
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--val-size', type=float, default=0.1)
    p.add_argument('--test-size', type=float, default=0.1)
    p.add_argument('--patience', type=int, default=7)
    p.add_argument('--clip-grad', type=float, default=1.0)
    p.add_argument('--workers', type=int, default=0)

    args = p.parse_args()
    cfg = Config(
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
    return cfg


def main():
    cfg = parse_args()
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    logger = logging.getLogger("lstm_trainer")

    set_seed(cfg.seed)

    # Use project's database configuration
    db_url = f"postgresql+psycopg://{config.db_user}:{config.db_password}@{config.db_host}:{config.db_port}/{config.db_name}"
    sql_query = "SELECT ts as timestamp, symbol, open, high, low, close, volume FROM kline_day WHERE ts BETWEEN '2020-01-01' AND '2024-12-31'"

    df = fetch_data(db_url, sql_query, logger)

    # Filter out symbols with too few rows
    logger.info("Filtering thin symbols…")
    counts = df.groupby('symbol').size()
    keep_syms = counts[counts >= cfg.min_per_symbol_rows].index
    df = df[df['symbol'].isin(keep_syms)].copy()

    df = attach_sentiment(df, cfg, logger)
    df, feats = build_features(df, cfg, logger)

    logger.info(f"Final features: {feats} | window={cfg.window} | horizon={cfg.horizon}")
    model, scalers, metrics = train_loop(cfg, df, feats, logger)

    outdir = save_artifacts(model, scalers, cfg, metrics, logger)
    logger.info(f"Run complete. Artifacts: {outdir}")


if __name__ == "__main__":
    main()
