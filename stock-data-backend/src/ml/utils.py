from __future__ import annotations

import json
import logging
import os
import pickle
import random
import unittest
from dataclasses import asdict, dataclass
from datetime import datetime
from typing import List, Optional, Tuple, Dict

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, Dataset

from config import config
from data.storage import Storage

# ----------------------
# CONFIGURATION
# ----------------------
@dataclass
class Config:
    symbols: Optional[List[str]] = None
    train_sample_fraction: Optional[float] = None
    artifacts_dir: str = "artifacts"
    logs_dir: str = "logs"
    seed: int = 1337
    window: int = 60
    horizon: int = 1
    # --- Task Type ---
    # `use_returns` is now repurposed for classification vs regression
    # True = Classification (predict UP/DOWN), False = Regression (predict price)
    use_returns: bool = True 
    min_per_symbol_rows: int = 200
    const_sentiment: Optional[float] = None
    features: Optional[List[str]] = None
    hidden_size: int = 128
    num_layers: int = 2
    dropout: float = 0.2
    bidirectional: bool = False
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-5
    epochs: int = 30
    val_size: float = 0.1
    test_size: float = 0.1
    early_stopping_patience: int = 7
    clip_grad: float = 1.0
    num_workers: int = 0
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )

# ----------------------
# MODEL DEFINITION
# ----------------------
class LSTMClassifier(nn.Module):
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
            nn.Linear(d * hidden_size // 2, 2),  # Output 2 logits for UP/DOWN classes
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        return self.head(last)

# ----------------------
# DATA UTILITIES
# ----------------------
REQUIRED_COLS = ["timestamp", "symbol", "open", "high", "low", "close", "volume"]

def fetch_data(storage: Storage, sql: str, logger, params: Optional[list] = None) -> pd.DataFrame:
    logger.info("Fetching data via Storage class…")
    with storage.pool.connection() as conn:
        df = pd.read_sql(sql, conn, params=params)
    df.columns = [c.lower() for c in df.columns]
    df = df.rename(columns={"time": "timestamp", "stock_symbol": "symbol"})
    for c in REQUIRED_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing required column '{c}'. Columns found: {list(df.columns)}")
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.dropna(subset=["timestamp"])
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)
    return df

def attach_sentiment(df: pd.DataFrame, cfg: Config, logger) -> pd.DataFrame:
    if cfg.const_sentiment is not None:
        df["sentiment"] = float(cfg.const_sentiment)
    elif "sentiment" not in df.columns:
        df["sentiment"] = 0.0
    return df

def build_features(df: pd.DataFrame, cfg: Config, logger) -> Tuple[pd.DataFrame, List[str]]:
    logger.info("Building features...")
    df = df.copy()

    # --- Technical Indicators ---
    df['sma_20'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(20).mean())
    df['sma_50'] = df.groupby('symbol')['close'].transform(lambda s: s.rolling(50).mean())
    df['ema_20'] = df.groupby('symbol')['close'].transform(lambda s: s.ewm(span=20, adjust=False).mean())
    delta = df.groupby('symbol')['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['rsi'] = 100 - (100 / (1 + rs))

    # --- Target Variable (Classification) ---
    # Target is 1 if the price goes up in the next `horizon` periods, 0 otherwise.
    df["target"] = (df.groupby('symbol')['close'].shift(-cfg.horizon) > df['close']).astype(int)
    
    # --- Diagnostic Logging: Target Distribution ---
    # Check if the target variable is imbalanced. If it's all 0s or 1s, the model can't learn.
    target_dist = df['target'].value_counts(normalize=True)
    logger.info(f"Target variable distribution:\n{target_dist}")
    if len(target_dist) < 2:
        logger.warning("The target variable has only one class! The model will not be able to learn.")

    base_feats = ['open', 'high', 'low', 'close', 'volume', 'sentiment', 'sma_20', 'sma_50', 'ema_20', 'rsi']
    feats = cfg.features or base_feats
    df = df.dropna(subset=feats + ["target"]).copy()
    logger.info(f"Finished building features. Final dataset shape: {df.shape}")
    return df, feats

class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, feats: List[str], window: int):
        self.samples = []
        # The dataframe is now expected to be pre-scaled before being passed here.
        for sym, g in df.groupby("symbol"):
            if len(g) < window + 1:
                continue
            X = g[feats].values.astype(np.float32)
            y = g["target"].values.astype(np.int64) # Target for classification should be int
            for i in range(window, len(g)):
                self.samples.append((X[i - window : i], y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x, y = self.samples[idx]
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.long) # Use torch.long for CrossEntropyLoss

# ----------------------
# TRAINING & ORCHESTRATION
# ----------------------
@torch.no_grad()
def evaluate_classification(model, loader, device):
    model.eval()
    all_preds, all_labels, total_loss = [], [], 0.0
    criterion = nn.CrossEntropyLoss() # To calculate validation loss

    for xb, yb in loader:
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        preds = torch.argmax(logits, dim=1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_loss = total_loss / len(all_labels)
    return avg_loss, accuracy, f1

def train_loop(cfg: Config, df: pd.DataFrame, feats: List[str], logger):
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # --- Data Splitting and Scaling (Time-based split) ---
    # This approach is more realistic for financial time series.
    # We train on all stocks up to a certain point in time and validate/test on a later period.
    df = df.sort_values('timestamp')
    last_date = df['timestamp'].max()
    test_split_date = last_date - pd.DateOffset(days=int(len(df['timestamp'].unique()) * cfg.test_size))
    val_split_date = test_split_date - pd.DateOffset(days=int(len(df['timestamp'].unique()) * cfg.val_size))

    train_df = df[df['timestamp'] < val_split_date].copy()
    val_df = df[(df['timestamp'] >= val_split_date) & (df['timestamp'] < test_split_date)].copy()
    test_df = df[df['timestamp'] >= test_split_date].copy()

    logger.info(f"Splitting data by time:")
    logger.info(f"  - Train: Before {val_split_date.date()}")
    logger.info(f"  - Val:   {val_split_date.date()} to {test_split_date.date()}")
    logger.info(f"  - Test:  After {test_split_date.date()}")

    # Fit scaler ONLY on training data to prevent leakage
    scaler = StandardScaler()
    train_df.loc[:, feats] = scaler.fit_transform(train_df[feats])
    # Use the same scaler to transform validation and test data
    val_df.loc[:, feats] = scaler.transform(val_df[feats])
    test_df.loc[:, feats] = scaler.transform(test_df[feats])

    train_ds = SequenceDataset(train_df, feats, cfg.window)
    val_ds = SequenceDataset(val_df, feats, cfg.window)
    test_ds = SequenceDataset(test_df, feats, cfg.window)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    # --- Diagnostic Logging: Check a single batch ---
    try:
        x_sample, y_sample = next(iter(train_loader))
        logger.info(f"Sample batch shapes: X={x_sample.shape}, y={y_sample.shape}")
        logger.info(f"Sample batch types: X={x_sample.dtype}, y={y_sample.dtype}")
    except StopIteration:
        logger.error("Training loader is empty! Cannot proceed with training.")
        return None, None, {}

    model = LSTMClassifier(len(feats), cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.bidirectional).to(device)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    best_val_acc, best_state, patience = 0.0, None, cfg.early_stopping_patience

    logger.info(f"Training started: N_train={len(train_ds)}, N_val={len(val_ds)}, N_test={len(test_ds)}")
    logger.info(f"Model details: {model}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for i, (xb, yb) in enumerate(train_loader):
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1 = evaluate_classification(model, val_loader, device)
        logger.info(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f} | val_f1={val_f1:.4f}")

        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state, patience = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, cfg.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping triggered.")
                break
    if best_state:
        model.load_state_dict(best_state)
    
    test_loss, test_acc, test_f1 = evaluate_classification(model, test_loader, device)
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f} | Final Test F1-Score: {test_f1:.4f}")
    return model, scaler, {"test_loss": test_loss, "test_accuracy": test_acc, "test_f1": test_f1}

def prepare_data(cfg: Config, storage: Storage, logger) -> tuple[pd.DataFrame | None, list | None]:
    sql_query = "SELECT time, stock_symbol, open, high, low, close, volume FROM stock_data WHERE time BETWEEN '2020-01-01' AND '2024-12-31'"
    params = []
    if cfg.symbols:
        placeholders = ",".join(["%s"] * len(cfg.symbols))
        sql_query += f" AND stock_symbol IN ({placeholders})"
        params.extend(cfg.symbols)
    df = fetch_data(storage, sql_query, logger, params=params)
    if df.empty:
        return None, None
    if cfg.train_sample_fraction and not cfg.symbols:
        unique_symbols = df['symbol'].unique()
        sample_size = int(len(unique_symbols) * cfg.train_sample_fraction)
        if sample_size > 0:
            sampled_symbols = random.sample(list(unique_symbols), sample_size)
            df = df[df['symbol'].isin(sampled_symbols)]
            logger.info(f"Randomly sampled {len(df['symbol'].unique())} symbols for training.")
    counts = df.groupby('symbol').size()
    keep_syms = counts[counts >= cfg.min_per_symbol_rows].index
    df = df[df['symbol'].isin(keep_syms)].copy()
    if df.empty:
        return None, None
    df = attach_sentiment(df, cfg, logger)
    df, feats = build_features(df, cfg, logger)
    return df, feats

def run_training(cfg: Config):
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    logger = logging.getLogger("lstm_trainer")
    storage = None
    try:
        storage = Storage(config)
        df, feats = prepare_data(cfg, storage, logger)
        if df is not None and feats:
            model, scaler, metrics = train_loop(cfg, df, feats, logger)
            outdir = save_artifacts(model, scaler, cfg, metrics, logger)
            logger.info(f"Run complete. Artifacts: {outdir}")
        else:
            logger.info("Data preparation did not yield any data. Aborting training.")
    finally:
        if storage:
            storage.close()

def save_artifacts(model, scaler, cfg: Config, metrics: dict, logger) -> str:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(cfg.artifacts_dir, run_id)
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Artifacts saved to {outdir}")
    return outdir


# ----------------------
# UNIT TESTS
# ----------------------
class TestDataUtils(unittest.TestCase):

    def test_build_features_classification(self):
        """Tests the feature engineering for classification (UP/DOWN target)."""
        data = {
            'symbol': ['A'] * 100,
            'close':  np.linspace(100, 150, 100),
        }
        df = pd.DataFrame(data)
        cfg = Config(horizon=1)
        logger = logging.getLogger("test_logger")

        features_df, _ = build_features(df.copy(), cfg, logger)

        # The last row will have a NaN target because of the shift
        # The first 49 rows will have NaN from sma_50
        self.assertEqual(len(features_df), 50)
        
        # In a steady uptrend, all targets should be 1 (UP)
        self.assertTrue((features_df['target'] == 1).all())

    def test_sequence_dataset_classification(self):
        """Tests the SequenceDataset for a classification task."""
        data = {
            'symbol': ['A'] * 20,
            'open':   np.arange(20, dtype=float),
            'close':  np.arange(20, dtype=float),
            'target':   np.random.randint(0, 2, 20)
        }
        # Add other required columns with dummy data
        for col in ['high', 'low', 'volume', 'sentiment', 'sma_20', 'sma_50', 'ema_20', 'rsi']:
            data[col] = np.zeros(20)
            
        df = pd.DataFrame(data)
        feats = ['open', 'high', 'low', 'close', 'volume', 'sentiment']
        window_size = 5

        dataset = SequenceDataset(df, feats, window_size)
        self.assertEqual(len(dataset), 15)
        x, y = dataset[0]
        self.assertEqual(x.shape, (window_size, len(feats)))
        self.assertEqual(y.dtype, torch.long) # Target type should be long


if __name__ == '__main__':
    unittest.main()
