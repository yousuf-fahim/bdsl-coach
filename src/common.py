"""Shared paths, seed and helpers used by every script."""
import json
import random
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw" / "Recognition_Processed_Compressed"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODELS = ROOT / "models"
SEED = 42  # CLAUDE.md: seed 42 everywhere

# Windows font with Bengali glyphs, used by matplotlib for labels.
BENGALI_FONT = "Nirmala UI"


def set_seed(seed: int = SEED):
    """Fix every random number generator we use."""
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def load_labels():
    """List of class dicts, indexed by class index 0..48."""
    data = json.loads((PROCESSED / "labels.json").read_text(encoding="utf-8"))
    return data["classes"]


def save_json(obj, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
