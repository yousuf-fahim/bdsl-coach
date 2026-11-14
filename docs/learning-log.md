# Learning log

One entry per phase, written by Claude Code: what we did, why, the key concept in plain language, and three likely faculty questions with short answers. Read it top to bottom before your viva.

## Phase 0: Setup (2026-11-14)

**What we did:** created a Python 3.13 virtual environment (`.venv`), installed the 14 libraries in `requirements.txt` (exact versions are pinned in `requirements-lock.txt`), initialised git, and recorded the machine's specs in `results/environment.json` via `src/check_env.py`. The machine has a 6-core AMD Ryzen 5 7500F and no CUDA GPU, so all training runs on the CPU. Kaggle is the fallback if a model is too slow to train.

**Why:** a virtual environment keeps this project's library versions separate from the rest of the computer, so someone else (or a faculty member) can rebuild the exact same setup from `requirements-lock.txt`.

**Key concept: reproducibility.** An ML result is only trustworthy if someone else can get the same number again. That needs three things fixed: the code (git), the library versions (lock file) and the randomness (seed 42 everywhere). If a library version changes, results can shift slightly, which is why we pin versions and record the environment the numbers came from.

**Likely faculty questions**
1. *Why a virtual environment?* It isolates the project's dependencies, so upgrading a library for another project can't silently change our results.
2. *Why fix the random seed?* Weight initialisation, data shuffling and augmentation are random. A fixed seed makes runs repeatable, so differences between models come from the models and not from luck.
3. *Does training without a GPU matter?* Only for speed. The maths is identical. A GPU just runs the matrix operations in parallel. Our models are small enough for a CPU.
