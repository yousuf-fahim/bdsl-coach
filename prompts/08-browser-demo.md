# Phase 8: Browser demo and deploy (120 min) PLAN

Run in plan mode (Shift+Tab) after /clear. Run `vercel login` yourself in a terminal when Claude Code asks. Paste everything below the line.

Checkpoint: why was the Python/JS parity test needed? What would have happened without it?

---

Phase 8: ship the demo. Plan first. Constraints: no backend, everything runs in the browser, the camera feed never leaves the user's device.

1. Export the best landmark model to ONNX (models/landmark_mlp.onnx) and verify it matches the PyTorch outputs on 20 saved samples. If onnxruntime-web causes trouble, fall back to exporting weights as JSON and implementing the forward pass by hand in JavaScript (it is only a few matrix multiplications).
2. Build the app in web/ using the simplest stack that deploys on Vercel (plain Vite or static HTML/JS is fine). Use @mediapipe/tasks-vision HandLandmarker for landmarks.
3. The biggest risk is a preprocessing mismatch between Python and JavaScript. Port the exact normalization from src/landmarks.py to JS and write a test that feeds the same 20 raw landmark sets through both and compares outputs. Do not proceed until they match.
4. UI: live webcam with landmark overlay; top-3 predictions with confidence bars; prediction smoothing over about 5 frames; a Practice mode that shows a target Bangla letter and turns green when I hold the correct sign above 70% confidence for half a second. Use the Noto Sans Bengali font for letters. Make it work on a phone browser too.
5. Mind the webcam mirror and left/right hand handling.
6. Test it locally and tell me how to run it. I will run `vercel login` myself, then deploy with the Vercel CLI and give me the URL.
7. Record a short demo GIF into docs/, append the Phase 8 entry to docs/learning-log.md and commit.
