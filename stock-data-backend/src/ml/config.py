from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch


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
    label_smoothing: float = 0.0
    clip_grad: float = 1.0
    num_workers: int = 0
    device: str = (
        "cuda"
        if torch.cuda.is_available()
        else ("mps" if torch.backends.mps.is_available() else "cpu")
    )
