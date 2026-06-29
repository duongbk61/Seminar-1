"""Small shared helpers: reproducibility and device selection."""
from __future__ import annotations

import os
import random

import numpy as np


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def pick_device(prefer: str = "auto") -> str:
    """Return 'cuda' when available (e.g. the user's GTX 1650), else 'cpu'."""
    try:
        import torch

        if prefer == "cpu":
            return "cpu"
        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"
