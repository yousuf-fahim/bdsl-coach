# Project: BdSL Learning Coach

Goal: compare two neural-network approaches for Bangla Sign Language alphabet recognition and ship the better one as an in-browser webcam demo.

- Approach A: CNN on raw images (small custom CNN plus pretrained MobileNetV3-small).
- Approach B: MediaPipe hand landmarks fed into a small MLP trained from scratch.

Dataset: BDSL 49 (Mendeley Data), fallback BAUST Lipi. Raw data lives in data/raw and is never committed.

The full plan, time estimate and phase prompts are in PLAN.md and prompts/.

## Rules

- Random seed 42 everywhere.
- Never random-split frames that could come from the same signer or video. Group by signer if recoverable, otherwise by near-duplicate cluster. The test set is used only once, for the final evaluation.
- Every number in the report must come from a script in src/ that writes to results/ (CSV, JSON or PNG). No hand-typed numbers.
- Keep code simple and commented. Clarity over cleverness.
- After each phase, append an entry to docs/learning-log.md: what we did, why, the key concept in plain language (5-8 lines), and 3 questions a faculty member might ask with short answers.
- Ask before installing anything heavy or changing the plan. Explain errors in plain language.
- Commit at the end of each phase with a clear message.

## About the user

Fahim, final-year CSE student at BUBT. Strong at web dev and Vercel, has done ML labs, new to computer vision. Explain concepts briefly but do not skip them. This project is for his Neural Network lab and will be presented to faculty.
