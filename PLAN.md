# BdSL Learning Coach: Plan

Prepared 10 Nov 2026 for Fahim.

## What you're building

A browser app that recognizes Bangla Sign Language (BdSL) alphabet signs from your webcam, plus a small research study behind it that compares two neural-network approaches on the same data:

- **Approach A:** a CNN trained on raw images (a small custom CNN and a pretrained MobileNet).
- **Approach B:** MediaPipe hand landmarks (21 points per hand) fed into a small MLP that you train from scratch.

**Final deliverables:** a GitHub repo, a live Vercel demo, a results table with charts, a 6-8 page report, and a viva cheat sheet.

**Honest expectation:** approach B should be far smaller and faster, and more robust to backgrounds and lighting, but its accuracy depends on how often MediaPipe detects the hand. Either outcome is a valid result as long as it is measured properly and explained.

## Time estimate

| Phase | What happens | Your active time | Notes |
|---|---|---|---|
| 0 | Environment setup | 20-30 min | |
| 1 | Data audit, leakage check, splits | 60-75 min | Dataset download is manual |
| 2 | Landmark extraction | 45 min | Extraction runs in the background |
| 3 | Landmark MLP (approach B) | 60 min | |
| 4 | CNN baselines (approach A) | 75-90 min | Training wait; faster with a GPU |
| 5 | Final comparison | 45 min | Test set used once |
| 6 | Robustness tests | 60 min | Includes your own webcam photos |
| 7 | Error analysis | 40 min | |
| 8 | Browser demo and deploy | 120 min | Most likely place for bugs |
| 9 | Report and viva prep | 90-120 min | |

**Total: about 9.5-11.5 hours of your time. Budget 12-13 hours to leave room for debugging.**

**Suggested split:** Saturday for phases 0-5 (about 6 hours), Sunday for phases 6-9 (about 6 hours). If you run short, trim phases 6 and 7. Phases 1-5, 8 and 9 are the core.

## Before you start

- Claude Code installed and signed in.
- Python 3.10-3.12. MediaPipe does not always support the newest Python, so check this first.
- Node 20 or newer, git, a GitHub account, a Vercel account, and a webcam.
- **Dataset (manual step):** download BDSL 49 yourself and put it in `data/raw/`. Claude Code may not be able to fetch it. The paper lists the repository as Mendeley Data: https://data.mendeley.com/datasets/k5yk4j8z8s/5
  Fallback: BAUST Lipi, described at https://arxiv.org/html/2408.10518v1
- A GPU is optional. The CNN baselines use 128x128 inputs, so they should be manageable on CPU, and Claude Code will measure and tell you. Google Colab is the fallback.

## How to work with Claude Code

- Open a terminal in this folder and run `claude`.
- Paste one prompt file from `prompts/` at a time, in order. Files marked PLAN are best run in plan mode (press Shift+Tab to switch). Read the plan, then approve.
- Run `/clear` between phases. `CLAUDE.md` and `docs/learning-log.md` carry the context forward.
- Each phase ends with a git commit and a learning-log entry. Read the entry before you move on. This is how you learn and how you prepare for the viva.
- If Claude Code starts doing something you did not ask for, say: Stop. Summarize what you changed so far and what you plan next. You can always roll back with git.

## If something goes wrong

| Problem | What to do |
|---|---|
| MediaPipe will not install | Almost always the Python version. Ask Claude Code to create the venv with Python 3.11 |
| Hand detection rate is low | Try a lower confidence threshold, and report the rate honestly. It is a finding, not a failure |
| Landmark model scores lower than the CNN | Fine. Report it, then use the size, speed and robustness results to explain the trade-off |
| Demo disagrees with Python results | Preprocessing mismatch. Re-run the parity test from Phase 8 |
| Predictions look mirrored in the demo | Webcam mirroring or handedness flip. Ask Claude Code to check both |
| Training is too slow | Smaller images, fewer epochs, or move the CNN phase to Colab |
| Claude Code wandered off-plan | Ask it to summarize its changes, then use `git diff` or `git checkout` to roll back |

## Dataset notes

- **BDSL 49:** per the paper abstract (https://arxiv.org/abs/2208.06827), 49 labels, 29,490 images, 14 signers; the specifications table lists Mendeley Data as the host. Only the abstract was checked, so confirm the folder layout and license on the download page.
- **BAUST Lipi (fallback):** per its paper (https://arxiv.org/html/2408.10518v1), 36 classes, 18,000 images (600 per class), 15 signers, 224x224. The paper says the data is public on GitHub and Kaggle but states no license. Verify the link before relying on it.
- Cite whichever dataset you use in the report.

## Final checklist

- Repo pushed to GitHub, README renders, demo link works on your phone.
- Every number in the report traces to a file in `results/`.
- You can explain every item in `docs/viva-cheatsheet.md` in your own words. Read the learning log once top to bottom the night before.
