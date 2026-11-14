# Decisions made while you were away

Each entry: what came up, the options, what I chose, and how to change it.

## Q0.1 Python version (Phase 0)
- **What came up:** PLAN.md asks for Python 3.10-3.12, but only Python 3.13.3 is installed.
- **Options:** (a) install Python 3.12 alongside it, or (b) use 3.13 if MediaPipe supports it.
- **Chosen:** (b). MediaPipe 1.2.3 publishes a `py3-none-win_amd64` wheel, which works on any Python 3, including 3.13. No extra install needed.
- **To change:** install Python 3.12, delete `.venv`, run `py -3.12 -m venv .venv`, then `.venv\Scripts\pip install -r requirements.txt`.

## Q0.2 Which part of the download to use (Phase 0)
- **What came up:** the Mendeley zip holds a Recognition set (1.8 GB, one image per sign, classification) and a Detection set (8.4 GB, scenes with bounding boxes).
- **Chosen:** Recognition only. It is the classification task this project is about.
- **To change:** extract `Detection Dataset with Annotation.zip` from the outer zip yourself. Nothing in the pipeline uses it.

## Q0.3 Git workflow (Phase 0)
- **What came up:** your global rules say: never commit to `main`, use one branch per task, open a PR. The project rules say: commit at the end of each phase. There's no GitHub remote yet, so I can't push or open a PR.
- **Chosen:** the scaffold you already wrote (CLAUDE.md, PLAN.md, prompts/) is the first commit on `main`, as a baseline. All phase work goes on branch `feat/bdsl-pipeline`, with one commit per phase. I didn't create a GitHub repo, push or open a PR.
- **To change:** create the GitHub repo, then run `git remote add origin <url>`, `git push -u origin main feat/bdsl-pipeline`, and `gh pr create` from the branch. If you'd rather have a branch per phase, I can split the history.

## Q0.4 OneDrive sync of the dataset (Phase 0)
- **What came up:** the project folder is inside OneDrive, so `data/raw/` (1.8 GB) gets uploaded.
- **Chosen:** left as is. 1.8 GB is manageable. git ignores `data/` either way.
- **To change:** in File Explorer, right-click `data` and choose "Free up space", or pause OneDrive.
