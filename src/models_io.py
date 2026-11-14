"""Load the trained models (shared by evaluate.py, robustness.py, error analysis, export)."""
import torch

from common import MODELS


def load_mlp():
    from train_landmark_mlp import LandmarkMLP
    ck = torch.load(MODELS / "landmark_mlp.pt", map_location="cpu")
    m = LandmarkMLP(ck["d_in"], ck["n_classes"], ck["h1"], ck["h2"], ck["p"])
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def load_cnn(name):
    from train_cnn import build
    m, _ = build(name) if name == "custom" else _mobilenet_no_download()
    ck = torch.load(MODELS / f"cnn_{name}.pt", map_location="cpu")
    m.load_state_dict(ck["state_dict"])
    return m.eval()


def _mobilenet_no_download():
    """Same architecture as training, without fetching ImageNet weights again."""
    import torch.nn as nn
    from torchvision.models import mobilenet_v3_small
    m = mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, 49)
    return m, None


MODEL_FILES = {"custom": MODELS / "cnn_custom.pt", "mobilenet": MODELS / "cnn_mobilenet.pt",
               "landmark_mlp": MODELS / "landmark_mlp.pt"}
