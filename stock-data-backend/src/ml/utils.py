from __future__ import annotations

import json
from dataclasses import asdict
import logging
import os
import pickle
import random
import unittest
from datetime import datetime
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader, Dataset

from config import config
from data.storage import Storage
from ml.config import Config

from ml.features import add_target_variable

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

class HybridLSTMClassifier(nn.Module):
    """
    A hybrid model that uses an LSTM for sequential features and a separate 
    Feed-Forward Network (FFN) for static/contextual features.
    """
    def __init__(self, seq_input_size: int, static_input_size: int, hidden_size: int, num_layers: int, dropout: float, bidirectional: bool):
        super().__init__()
        # --- LSTM Branch for Sequential Data ---
        self.lstm = nn.LSTM(
            input_size=seq_input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        d = 2 if bidirectional else 1
        lstm_output_size = d * hidden_size

        # --- FFN Branch for Static Data ---
        self.static_head = nn.Sequential(
            nn.LayerNorm(static_input_size),
            nn.Linear(static_input_size, static_input_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        static_output_size = static_input_size * 2

        # --- Combined Head for Final Prediction ---
        self.combined_head = nn.Sequential(
            nn.LayerNorm(lstm_output_size + static_output_size),
            nn.Linear(lstm_output_size + static_output_size, (lstm_output_size + static_output_size) // 2),
            nn.ReLU(),
            nn.Linear((lstm_output_size + static_output_size) // 2, 2),
        )

    def forward(self, x_seq, x_static):
        lstm_out, _ = self.lstm(x_seq)
        lstm_last = lstm_out[:, -1, :]  # Get the last output from the sequence
        static_out = self.static_head(x_static)
        combined = torch.cat((lstm_last, static_out), dim=1)
        return self.combined_head(combined)

# ----------------------
# DATA UTILITIES
# ----------------------
class SequenceDataset(Dataset):
    def __init__(self, df: pd.DataFrame, seq_feats: List[str], static_feats: List[str], window: int):
        self.samples = []
        # The dataframe is now expected to be pre-scaled before being passed here.
        for sym, g in df.groupby("symbol"):
            if len(g) < window + 1:
                continue
            X_seq = g[seq_feats].values.astype(np.float32)
            X_static = g[static_feats].values.astype(np.float32)
            y = g["target"].values.astype(np.int64) # Target for classification should be int
            for i in range(window, len(g)):
                self.samples.append((X_seq[i - window : i], X_static[i], y[i]))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        x_seq, x_static, y = self.samples[idx]
        return torch.from_numpy(x_seq), torch.from_numpy(x_static), torch.tensor(y, dtype=torch.long)

# ----------------------
# TRAINING & ORCHESTRATION
# ----------------------
@torch.no_grad()
def evaluate_classification(model, loader, device, logger, log_report: bool = False):
    model.eval()
    all_preds, all_labels, all_confidences, total_loss = [], [], [], 0.0
    criterion = nn.CrossEntropyLoss()

    for batch in loader:
        if len(batch) == 3: # Hybrid model
            xb_seq, xb_static, yb = batch
            xb_seq, xb_static, yb = xb_seq.to(device), xb_static.to(device), yb.to(device)
            logits = model(xb_seq, xb_static)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb_seq.size(0)
        else: # Standard LSTM model
            xb, yb = batch
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            loss = criterion(logits, yb)
            total_loss += loss.item() * xb.size(0)

        # Convert logits to probabilities (confidence scores)
        probs = torch.softmax(logits, dim=1)
        confidences, preds = torch.max(probs, dim=1)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(yb.cpu().numpy())
        all_confidences.extend(confidences.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='weighted')
    avg_confidence = np.mean(all_confidences)
    
    if log_report:
        logger.info(f"Classification Report:\n{classification_report(all_labels, all_preds, target_names=['DOWN', 'UP'])}")
        logger.info(f"Confusion Matrix:\n{confusion_matrix(all_labels, all_preds)}")
        logger.info(f"Average prediction confidence: {avg_confidence:.4f}")

    avg_loss = total_loss / len(all_labels)
    return avg_loss, accuracy, f1, avg_confidence

def train_loop(cfg: Config, df: pd.DataFrame, seq_feats: List[str], static_feats: List[str], logger, model: nn.Module = None, debug_overfit: bool = False):
    device = torch.device(cfg.device)
    logger.info(f"Using device: {device}")

    # --- Data Splitting and Scaling (Time-based split) ---
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
    all_feats = seq_feats + static_feats
    train_df.loc[:, all_feats] = scaler.fit_transform(train_df[all_feats])
    # Use the same scaler to transform validation and test data
    val_df.loc[:, all_feats] = scaler.transform(val_df[all_feats])
    test_df.loc[:, all_feats] = scaler.transform(test_df[all_feats])

    train_ds = SequenceDataset(train_df, seq_feats, static_feats, cfg.window)
    val_ds = SequenceDataset(val_df, seq_feats, static_feats, cfg.window)
    test_ds = SequenceDataset(test_df, seq_feats, static_feats, cfg.window)

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    try:
        x_seq_sample, x_static_sample, y_sample = next(iter(train_loader))
        logger.info(f"Sample batch shapes: X_seq={x_seq_sample.shape}, X_static={x_static_sample.shape}, y={y_sample.shape}")
        logger.info(f"Sample batch types: X_seq={x_seq_sample.dtype}, X_static={x_static_sample.dtype}, y={y_sample.dtype}")
    except StopIteration:
        logger.error("Training loader is empty! Cannot proceed with training.")
        return None, None, {}

    class_counts = train_df['target'].value_counts().sort_index().values
    if len(class_counts) == 2:
        weights = 1.0 / torch.tensor(class_counts, dtype=torch.float)
        weights = weights / weights.sum()
        class_weights = weights.to(device)
        logger.info(f"Applying class weights to handle imbalance: {class_weights.cpu().numpy().tolist()}")
    else:
        class_weights = None

    if model is None:
        model = HybridLSTMClassifier(len(seq_feats), len(static_feats), cfg.hidden_size, cfg.num_layers, cfg.dropout, cfg.bidirectional).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=cfg.label_smoothing)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = ReduceLROnPlateau(opt, mode='max', factor=0.2, patience=2)

    if debug_overfit:
        logger.warning("--- RUNNING IN OVERFIT DEBUG MODE ---")
        try:
            xb_seq, xb_static, yb = next(iter(train_loader))
        except StopIteration:
            logger.error("Training loader is empty! Cannot run overfit test.")
            return None, None, {}
        
        xb_seq, xb_static, yb = xb_seq.to(device), xb_static.to(device), yb.to(device)
        logger.info(f"Attempting to overfit on a single batch of size: X_seq={xb_seq.shape}, X_static={xb_static.shape}, y={yb.shape}")

        model.train()
        for i in range(100):
            opt.zero_grad()
            logits = model(xb_seq, xb_static)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            if (i + 1) % 10 == 0:
                preds = torch.argmax(logits, dim=1)
                acc = accuracy_score(yb.cpu().numpy(), preds.cpu().numpy())
                logger.info(f"Overfit Iter {i+1:03d} | Loss={loss.item():.4f} | Accuracy={acc:.4f}")
        
        logger.warning("--- OVERFIT DEBUG MODE COMPLETE ---")
        return None, None, {}

    best_val_acc, best_state, patience = 0.0, None, cfg.early_stopping_patience

    logger.info(f"Training started: N_train={len(train_ds)}, N_val={len(val_ds)}, N_test={len(test_ds)}")
    logger.info(f"Model details: {model}")
    for epoch in range(1, cfg.epochs + 1):
        model.train()
        total_train_loss = 0.0
        for i, (xb_seq, xb_static, yb) in enumerate(train_loader):
            xb_seq, xb_static, yb = xb_seq.to(device), xb_static.to(device), yb.to(device)
            opt.zero_grad()
            logits = model(xb_seq, xb_static)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), cfg.clip_grad)
            opt.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        val_loss, val_acc, val_f1, val_conf = evaluate_classification(model, val_loader, device, logger)
        logger.info(f"Epoch {epoch:03d} | train_loss={avg_train_loss:.4f} | val_loss={val_loss:.4f} | val_accuracy={val_acc:.4f} | val_f1={val_f1:.4f} | val_confidence={val_conf:.4f}")

        scheduler.step(val_acc)

        if val_acc > best_val_acc + 1e-4:
            best_val_acc, best_state, patience = val_acc, {k: v.cpu() for k, v in model.state_dict().items()}, cfg.early_stopping_patience
        else:
            patience -= 1
            if patience <= 0:
                logger.info("Early stopping triggered.")
                break
    if best_state:
        model.load_state_dict(best_state)
    
    test_loss, test_acc, test_f1, test_conf = evaluate_classification(model, test_loader, device, logger, log_report=True)
    logger.info(f"Final Test Loss: {test_loss:.4f}")
    logger.info(f"Final Test Accuracy: {test_acc:.4f} | Final Test F1-Score: {test_f1:.4f} | Final Test Confidence: {test_conf:.4f}")
    
    # Convert all metric values to standard Python floats to ensure JSON serializability.
    metrics = {
        "test_loss": float(test_loss), 
        "test_accuracy": float(test_acc), 
        "test_f1": float(test_f1), 
        "test_confidence": float(test_conf)
    }
    return model, scaler, metrics

def set_seed(seed: int):
    """Sets the seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def prepare_data(cfg: Config, storage: Storage, logger) -> tuple[pd.DataFrame | None, list | None, list | None]:
    set_seed(cfg.seed)
    logger.info(f"Using random seed: {cfg.seed}")
    symbols_to_use = cfg.symbols
    if not symbols_to_use:
        all_symbols = storage.get_all_distinct_symbols()
        if cfg.train_sample_fraction:
            sample_size = int(len(all_symbols) * cfg.train_sample_fraction)
            symbols_to_use = random.sample(all_symbols, sample_size)
            logger.info(f"Randomly sampled {len(symbols_to_use)} symbols for training.")
        else:
            symbols_to_use = all_symbols

    if not symbols_to_use:
        logger.warning("No symbols found to train on.")
        return None, None, None

    df = storage.query_features_for_symbols(symbols_to_use)
    if df.empty:
        logger.warning("Feature query returned no data.")
        return None, None, None

    df = df.rename(columns={"time": "timestamp", "stock_symbol": "symbol"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.sort_values(["symbol", "timestamp"]).reset_index(drop=True)

    # Add target variable here to avoid storing it and prevent lookahead bias in feature store.
    df = add_target_variable(df, cfg.horizon)

    # Define which features are sequential and which are static
    # Define all possible static features, then filter by what's available in the dataframe.
    # This prevents KeyErrors if features haven't been calculated and stored yet.
    possible_static_feats = ['sentiment', 'sentiment_3d_avg', 'sentiment_7d_avg', 'sentiment_momentum', 'obv', 'bb_percent_b', 'bb_width', 'skew_21d', 'kurt_21d']
    all_current_feats = [col for col in df.columns if col not in ['timestamp', 'symbol', 'target']]
    
    static_feats = [f for f in possible_static_feats if f in all_current_feats]
    seq_feats = [f for f in all_current_feats if f not in static_feats]

    logger.info(f"Identified {len(seq_feats)} sequential features and {len(static_feats)} static features.")
    feats = seq_feats + static_feats
    
    df = df.dropna(subset=feats + ['target'])
    if df.empty:
        logger.warning("No rows left after dropping NaNs from features and target.")
        return None, None

    df['target'] = df['target'].astype(int)

    # Log class distribution
    class_dist = df['target'].value_counts(normalize=True)
    logger.info(f"Target class distribution:\n{class_dist}")

    counts = df.groupby('symbol').size()
    keep_syms = counts[counts >= cfg.min_per_symbol_rows].index
    df = df[df['symbol'].isin(keep_syms)].copy()
    
    if df.empty:
        logger.warning(f"No symbols with at least {cfg.min_per_symbol_rows} rows of feature data.")
        return None, None, None

    logger.info(f"Prepared data for {len(df['symbol'].unique())} symbols. Total rows: {len(df)}")
    
    return df, seq_feats, static_feats

def run_training(cfg: Config):
    os.makedirs(cfg.logs_dir, exist_ok=True)
    os.makedirs(cfg.artifacts_dir, exist_ok=True)
    logger = logging.getLogger("lstm_trainer")
    storage = None
    try:
        storage = Storage(config)
        df, seq_feats, static_feats = prepare_data(cfg, storage, logger)
        if df is not None and seq_feats:
            model, scaler, metrics = train_loop(cfg, df, seq_feats, static_feats, logger)
            outdir = save_artifacts(model, scaler, cfg, metrics, logger)
            logger.info(f"Run complete. Artifacts: {outdir}")
        else:
            logger.info("Data preparation did not yield any data. Aborting training.")
    finally:
        if storage:
            storage.close()

class NumpyEncoder(json.JSONEncoder):
    """ Custom JSON encoder for NumPy types. """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

def save_artifacts(model, scaler, cfg: Config, metrics: dict, logger) -> str:
    run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    outdir = os.path.join(cfg.artifacts_dir, run_id)
    os.makedirs(outdir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(outdir, "model.pt"))
    with open(os.path.join(outdir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
    # Use the custom NumpyEncoder to handle NumPy types in config and metrics
    with open(os.path.join(outdir, "config.json"), "w") as f:
        json.dump(asdict(cfg), f, indent=2, cls=NumpyEncoder)
    with open(os.path.join(outdir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2, cls=NumpyEncoder)
    logger.info(f"Artifacts saved to {outdir}")
    return outdir


# ----------------------
# UNIT TESTS
# ----------------------
class TestDataUtils(unittest.TestCase):

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
