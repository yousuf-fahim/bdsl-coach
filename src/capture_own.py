"""Phase 6: capture your own webcam photos for an out-of-distribution test.

Usage
  .venv\\Scripts\\python src\\capture_own.py                 # default 10 signs, 10 photos each
  .venv\\Scripts\\python src\\capture_own.py --classes 0 6 24 --per-class 5

Keys (click the preview window first)
  SPACE  take a photo of the current sign
  n      skip to the next sign
  b      go back one sign
  q      quit

Photos go to data/own/<class index>/<timestamp>.jpg (never committed to git).
The preview is mirrored like a selfie so it feels natural, but the SAVED photo
is the raw, un-mirrored camera frame: that is what the models and the web demo see.
The target sign is shown by index and romanisation (OpenCV can't draw Bangla text);
look up the handshape in results/sample_grid.png.
Then run: .venv\\Scripts\\python src\\eval_own.py
"""
import argparse
import time

import cv2

from common import ROOT, load_labels

# Default: 10 signs that are visually distinct and mostly one-handed (easy to self-capture):
# অ, ক, ব, ম, ০, ১, ২, ৩, ৪, ৫  (folder indices)
DEFAULT = [0, 6, 24, 26, 36, 37, 38, 39, 40, 41]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, nargs="+", default=DEFAULT)
    ap.add_argument("--per-class", type=int, default=10)
    ap.add_argument("--camera", type=int, default=0)
    args = ap.parse_args()
    labels = load_labels()
    out_root = ROOT / "data" / "own"
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise SystemExit("Could not open the webcam. Try --camera 1, or close other apps using it.")
    k = 0
    try:
        while 0 <= k < len(args.classes):
            c = args.classes[k]
            folder = out_root / str(c)
            folder.mkdir(parents=True, exist_ok=True)
            have = len(list(folder.glob("*.jpg")))
            ok, frame = cap.read()
            if not ok:
                raise SystemExit("Webcam stopped sending frames.")
            view = cv2.flip(frame, 1)  # mirrored preview only
            text = f"sign {k + 1}/{len(args.classes)}: class {c} ({labels[c]['roman']})  {have}/{args.per_class}"
            cv2.putText(view, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(view, "SPACE=capture  n=next  b=back  q=quit", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            cv2.imshow("BdSL capture", view)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                cv2.imwrite(str(folder / f"{time.strftime('%Y%m%d_%H%M%S')}_{int(time.time() * 1000) % 1000:03d}.jpg"), frame)
                if have + 1 >= args.per_class:
                    k += 1
            elif key == ord("n"):
                k += 1
            elif key == ord("b"):
                k = max(0, k - 1)
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"Saved photos under {out_root}. Now run: .venv\\Scripts\\python src\\eval_own.py")


if __name__ == "__main__":
    main()
