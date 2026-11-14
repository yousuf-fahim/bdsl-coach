"""Phase 8: assemble docs/demo.gif from the smoke-test screenshots (logs/smoke/*.png).

The screenshots come from web/tests/smoke.mjs: the BUILT demo running in headless
Edge with a fake camera made of random validation images (src/make_fake_camera.py).
It shows the real UI and real predictions, but not a live person signing.
Run: .venv\\Scripts\\python src\\make_demo_gif.py
"""
from PIL import Image

from common import ROOT


def main():
    shots = sorted((ROOT / "logs" / "smoke").glob("[rp]*.png"))
    frames = []
    for p in shots:
        im = Image.open(p).convert("RGB").crop((0, 0, 1100, 600))
        im.thumbnail((720, 720))
        frames.append(im.convert("P", palette=Image.ADAPTIVE, colors=128))
    out = ROOT / "docs" / "demo.gif"
    frames[0].save(out, save_all=True, append_images=frames[1:], duration=500, loop=0, optimize=True)
    print(f"wrote {out} ({len(frames)} frames, {out.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
