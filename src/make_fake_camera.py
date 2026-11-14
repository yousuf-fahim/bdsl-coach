"""Phase 8: build a fake webcam video (.y4m) from random VALIDATION images.

Chromium/Edge can use a .y4m file as the camera (--use-file-for-fake-video-capture),
which lets the automated smoke test (web/tests/smoke.mjs) drive the real demo page
end-to-end and capture screenshots for docs/demo.gif.
Images are picked at random (seed 42) among validation images with a detected hand;
they are NOT filtered by whether the model gets them right.

Output: logs/fake_camera.y4m (gitignored; ~80 MB) and logs/fake_camera.json (which images)
Run: .venv\\Scripts\\python src\\make_fake_camera.py
"""
import json

import numpy as np
from PIL import Image

from common import PROCESSED, RAW, ROOT, SEED

W, H, FPS, SECONDS_PER_IMAGE, N_IMAGES = 640, 480, 15, 3, 6


def rgb_to_yuv420(rgb):
    a = rgb.astype(np.float32)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.168736 * r - 0.331264 * g + 0.5 * b + 128
    v = 0.5 * r - 0.418688 * g - 0.081312 * b + 128
    u = u.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    v = v.reshape(H // 2, 2, W // 2, 2).mean((1, 3))
    return [np.clip(c, 0, 255).astype(np.uint8).tobytes() for c in (y, u, v)]


def main():
    d = np.load(PROCESSED / "landmarks.npz", allow_pickle=True)
    rng = np.random.default_rng(SEED)
    ok = np.flatnonzero((d["split"] == "val") & (d["n_hands"] > 0))
    pick = rng.choice(ok, N_IMAGES, replace=False)
    out = ROOT / "logs" / "fake_camera.y4m"
    out.parent.mkdir(exist_ok=True)
    with open(out, "wb") as f:
        f.write(f"YUV4MPEG2 W{W} H{H} F{FPS}:1 Ip A1:1 C420jpeg\n".encode())
        for i in pick:
            im = Image.open(RAW / d["path"][i]).convert("RGB")
            im.thumbnail((W - 160, H - 80))
            canvas = Image.new("RGB", (W, H), (70, 70, 70))
            canvas.paste(im, ((W - im.width) // 2, (H - im.height) // 2))
            planes = rgb_to_yuv420(np.asarray(canvas))
            for _ in range(FPS * SECONDS_PER_IMAGE):
                f.write(b"FRAME\n")
                for p in planes:
                    f.write(p)
    (ROOT / "logs" / "fake_camera.json").write_text(json.dumps(
        {"images": [str(d["path"][i]) for i in pick], "labels": [int(d["y"][i]) for i in pick],
         "fps": FPS, "seconds_per_image": SECONDS_PER_IMAGE}, indent=1))
    print("wrote", out)


if __name__ == "__main__":
    main()
