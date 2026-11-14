# STATUS

> ## ✅ FINISHED AND DEPLOYED. Last updated: 2026-11-15 01:47
> **Live demo: https://yousuf-fahim.github.io/bdsl-coach/** (headless Edge check: models load, 0 failed requests)
> Repo: https://github.com/yousuf-fahim/bdsl-coach. PRs #1–#4 were merged at your request.
> **3 items are still waiting on you (1, 2 and 5 below).** Items 3 and 4 are done.

## Final summary

### What's complete
| Phase | Result |
|---|---|
| 0 Setup | Python 3.13 venv, pinned `requirements-lock.txt`, CPU only (Ryzen 5 7500F) |
| 1 Data | 14,714 images, 49 classes. Leakage-aware 70/15/15 split (68.7/14.8/15.0%, 228 near-duplicates excluded). Labels **PROVISIONAL** |
| 2 Landmarks | MediaPipe found a hand in 89.2% of images (padding + 0.3 threshold) |
| 3 Landmark MLP | 79.0% validation accuracy (detected images), 74k parameters |
| 4 CNNs | validation: MobileNetV3-small 97.5%, custom CNN 73.8% (retrained for 45 epochs) |
| 5 Test comparison | run once, with bootstrap CIs |
| 6 Robustness | 7 perturbations × 5 severities. Own-webcam test **PENDING** your photos |
| 7 Error analysis | confusion matrices, confused pairs + landmark distances, Grad-CAM, per-block accuracy |
| 8 Web demo | static, self-hosted, `web/dist` (49 MB). All parity and smoke tests pass. Builds from a fresh clone and loads from a sub-path. `docs/DEPLOY.md` covers Cloudflare and GitHub Pages |
| 9 Write-up | `docs/report.md` + `docs/report.pdf` (8 pages), `docs/viva-cheatsheet.md`, README, learning log for every phase |

### Headline results (test set, 2,200 images; `results/comparison.csv`, `results/comparison_ci.csv`)
- **MobileNetV3-small: 97.2%** (95% CI 96.6–97.9)
- **Custom CNN:** 73.2%
- **Landmark MLP:** 75.7% on images where a hand was detected (90.7% of them), **68.6% end-to-end**
- **MLP vs custom CNN on detected images:** a statistical tie (+2.4 points, CI −0.1 to +5.0)
- **Size and speed:** the MLP classifier is 297 KB and takes 0.05 ms, but needs the 7.6 MB MediaPipe model and about 15.4 ms end-to-end
- **Robustness** (`results/robustness_summary.csv`): MobileNet stays the most accurate at every severity. Approach B fails mainly because MediaPipe stops detecting hands (noise, over-exposure). **The hypothesis "landmarks are more robust" was not supported**, and the report says so.
- **Leakage demo:** colour-only 1-NN scores 41.9% on a random split vs 17.4% on ours (`results/leakage_knn_demo.json`).

## Waiting on you, with exact steps

**1. Verify the label mapping (Q1.1).**
- Open `results/sample_grid.png` and check that the Bangla letter above each row matches the handshape.
- If something is wrong:
  - edit `TABLE` in `src/make_labels.py`
  - run `.venv\Scripts\python src\make_labels.py`
  - run `.venv\Scripts\python src\export_onnx.py` (updates `web/public/models/config.json`)
  - run `cd web; npm run build`
- No retraining is needed: only the display names change.
- When it's verified, change `"status"` in `make_labels.py` from PROVISIONAL and re-run the steps above.
- Also glance at the suspicious pair in Q1.3.

**2. Own-webcam photos (Phase 6, PENDING).**
```powershell
cd C:\Users\FAHIM\OneDrive\Documents\PROJECTS\bdsl-coach
.venv\Scripts\python src\capture_own.py      # SPACE = capture, n = next sign, q = quit (10 signs x 10 photos)
.venv\Scripts\python src\eval_own.py         # writes results/own_eval.json + own_predictions.csv
```
Then ask me to "fill in the own-photo results". That means report Section 6.3, the README line, the viva sheet and a re-rendered PDF (`cd web; npm run report:pdf`).

**3. Deploy: ✅ DONE.** The site is on GitHub Pages and redeploys automatically on every push to `main` that touches `web/`. The first two builds failed: Windows npm left two packages out of the lock file, fixed in PR #3. For reference, the original instructions are kept below.
- **Quickest:** `cd web; npm run build`, then on dash.cloudflare.com go to **Workers & Pages → Create → Pages → Upload assets** and upload `web\dist`.
- **Or GitHub Pages:** create the repo, push, set **Settings → Pages → Source = GitHub Actions**, then merge. The workflow is already in `.github/workflows/deploy-pages.yml`. **No base-path setting is needed**, because the build uses relative `./` paths and was tested under `/bdsl-coach/`.
- Then put the URL in `README.md` (the line marked TODO) and in the report header.

**4. GitHub repo, PR and merge: ✅ DONE.** The public repo is on `yousuf-fahim`, and the PRs were merged at your request. For reference, the commands used were:
```powershell
gh repo create bdsl-coach --public --source . --remote origin
git push -u origin main feat/bdsl-pipeline
gh pr create --base main --head feat/bdsl-pipeline --fill
```
Review the PR and merge it yourself.

**5. Decide the demo model (Q6.1).** The demo ships the landmark MLP, as planned. The evidence favours MobileNet (more accurate and more robust), but on webcam frames MobileNet needs a hand cropper. I suggest deciding after step 2. If you want the hybrid (MediaPipe crop → MobileNet ONNX), ask me.

## Things I'd double-check
- **Dataset license** before making the repo public. The paper says the data is free for research and education, but I didn't verify the Mendeley page's license. `results/sample_grid.png`, `near_duplicate_examples.png`, `landmark_failures.png`, `confused_pairs_examples.png`, `gradcam.png` and `docs/demo.gif` contain dataset images (faces are blurred in the dataset). The raw images themselves are never committed.
- **MobileNet's 97%** may be partly inflated: the same 14 signers appear in every split (Q4.3). Step 2 is the real test.
- **The noise robustness test** may be harsh: noise is added to small crops (Q6.2).
- **The custom CNN was simplified** to one conv per block to fit the CPU time budget (Q4.1).
- **Files outside the project folder:**
  - torchvision cached the MobileNet ImageNet weights in `C:\Users\FAHIM\.cache\torch` (10 MB)
  - pip and npm used their normal user caches
  - nothing else outside the folder was changed. The Downloads zip is untouched.
- **Tests:** the parity and smoke tests use your installed Microsoft Edge (no browser download). On a machine without Edge, set `PW_CHANNEL=chromium` after `npx playwright install chromium`.
- **Not re-run end-to-end from scratch:** the full Python pipeline from a fresh clone. It needs the dataset and about 1.5 h of CPU time. Each script was run here, and the MLP training reproduced its numbers exactly. The web part was verified from a fresh clone.

## Where things are
- `docs/QUESTIONS.md`: every decision I made while you were away (Q0.1–Q6.2), with how to change it
- `docs/learning-log.md`: Phases 0–9, your viva notes
- `docs/plans/`: phase plans
- `logs/` (not committed): run logs
