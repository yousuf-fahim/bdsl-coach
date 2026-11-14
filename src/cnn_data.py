"""Image pipeline for approach A (CNNs).

Cache: every image is letterboxed (aspect ratio kept, black padding) to
CACHE x CACHE pixels once and stored in data/processed/cache/img{CACHE}.npy,
because decoding thousands of JPEGs of up to 5570 px every epoch would make CPU
training far too slow.

Training batches get augmentation (applied to the whole batch with torch ops):
  random crop 128 from 144   -> the hand is not always perfectly centred
  rotation +-10 degrees      -> slightly tilted camera or hand
  colour jitter              -> different lighting, skin tones, white balance
  NO horizontal flip         -> flipping changes which hand is which and can turn
                                one sign into another (or into a left-handed version)
Validation/test: centre crop 128 only.
"""
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from common import PROCESSED, RAW

CACHE, CROP = 144, 128
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)  # ImageNet statistics
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)


def letterbox(path, size=CACHE):
    """Resize the longest side to `size`, pad the rest with black: no distortion."""
    im = Image.open(RAW / path).convert("RGB")
    im.thumbnail((size, size), Image.BILINEAR)
    canvas = Image.new("RGB", (size, size))
    canvas.paste(im, ((size - im.width) // 2, (size - im.height) // 2))
    return np.asarray(canvas)


def load_cache():
    """Returns (images uint8 [N,144,144,3], dataframe of splits.csv rows not excluded)."""
    df = pd.read_csv(PROCESSED / "splits.csv")
    df = df[df.split != "excluded"].reset_index(drop=True)
    f = PROCESSED / "cache" / f"img{CACHE}.npy"
    if f.exists():
        imgs = np.load(f)
        if len(imgs) == len(df):
            return imgs, df
    f.parent.mkdir(parents=True, exist_ok=True)
    with ProcessPoolExecutor() as ex:
        imgs = np.stack(list(ex.map(letterbox, df.path, chunksize=64)))
    np.save(f, imgs)
    return imgs, df


def to_tensor(batch_uint8):
    """uint8 NHWC numpy -> float NCHW tensor in [0, 1]."""
    return torch.from_numpy(batch_uint8).permute(0, 3, 1, 2).float().div_(255)


def normalize(x):
    return (x - MEAN) / STD


def center_crop(x):
    o = (CACHE - CROP) // 2
    return x[:, :, o:o + CROP, o:o + CROP]


def augment(x, gen):
    """Batch augmentation on float NCHW tensors in [0,1]. `gen` is a torch.Generator."""
    n = x.shape[0]
    # random crop (per sample)
    ox = torch.randint(0, CACHE - CROP + 1, (n,), generator=gen)
    oy = torch.randint(0, CACHE - CROP + 1, (n,), generator=gen)
    x = torch.stack([x[i, :, oy[i]:oy[i] + CROP, ox[i]:ox[i] + CROP] for i in range(n)])
    # rotation +-10 degrees via an affine grid
    ang = (torch.rand(n, generator=gen) * 20 - 10) * torch.pi / 180
    cos, sin = torch.cos(ang), torch.sin(ang)
    theta = torch.zeros(n, 2, 3)
    theta[:, 0, 0], theta[:, 0, 1], theta[:, 1, 0], theta[:, 1, 1] = cos, -sin, sin, cos
    grid = F.affine_grid(theta, x.shape, align_corners=False)
    x = F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)
    # colour jitter: brightness, contrast, saturation (each +-30%), small hue-like channel gain
    b = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.6
    c = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.6
    s = 1 + (torch.rand(n, 1, 1, 1, generator=gen) - 0.5) * 0.6
    g = 1 + (torch.rand(n, 3, 1, 1, generator=gen) - 0.5) * 0.1
    x = x * b
    m = x.mean(dim=(1, 2, 3), keepdim=True)
    x = (x - m) * c + m
    gray = x.mean(dim=1, keepdim=True)
    x = (x - gray) * s + gray
    x = (x * g).clamp_(0, 1)
    return x
