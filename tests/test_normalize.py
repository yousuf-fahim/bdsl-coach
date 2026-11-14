"""Unit tests for landmarks.normalize_hands (the function the browser re-implements).

Run: .venv\\Scripts\\python -m pytest tests
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from landmarks import FEAT_DIM, MIDDLE_MCP, normalize_hands  # noqa: E402

RNG = np.random.default_rng(42)


def hand(offset=(0.3, 0.4), size=0.1):
    """A random-ish hand in normalised coords around offset."""
    p = RNG.random((21, 3)).astype(np.float32) * size
    p[:, 0] += offset[0]
    p[:, 1] += offset[1]
    return p


def test_empty_is_zero_vector():
    f = normalize_hands([], 640, 480)
    assert f.shape == (FEAT_DIM,) and not f.any()


def test_wrist_at_origin_and_unit_scale():
    f = normalize_hands([(hand(), "Right")], 640, 480)
    q = f[:63].reshape(21, 3)
    assert np.allclose(q[0], 0)
    assert np.isclose(np.linalg.norm(q[MIDDLE_MCP, :2]), 1.0, atol=1e-5)
    assert f[126] == 1 and f[127] == 0


def test_translation_and_scale_invariance():
    h = hand((0.2, 0.2))
    moved = h.copy()
    moved[:, 0] += 0.3
    moved[:, 1] += 0.1
    bigger = (h - h[0]) * 2 + h[0]
    a = normalize_hands([(h, "Right")], 640, 480)
    assert np.allclose(a, normalize_hands([(moved, "Right")], 640, 480), atol=1e-5)
    assert np.allclose(a, normalize_hands([(bigger, "Right")], 640, 480), atol=1e-5)


def test_single_left_hand_is_mirrored():
    h = hand()
    r = normalize_hands([(h, "Right")], 640, 480)
    l = normalize_hands([(h, "Left")], 640, 480)
    assert np.allclose(l[0:63:3], -r[0:63:3])     # x flipped
    assert np.allclose(l[1:63:3], r[1:63:3])      # y unchanged


def test_two_hands_ordered_left_to_right_and_not_mirrored():
    left_in_image, right_in_image = hand((0.1, 0.4)), hand((0.7, 0.4))
    a = normalize_hands([(right_in_image, "Left"), (left_in_image, "Right")], 640, 480)
    b = normalize_hands([(left_in_image, "Right"), (right_in_image, "Left")], 640, 480)
    assert np.allclose(a, b)                       # input order doesn't matter
    assert a[126] == 1 and a[127] == 1
    assert a[128] > 0                              # second hand is to the right


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
