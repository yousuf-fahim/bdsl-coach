// Headless-browser parity test (Phase 8).
// Starts the Vite dev server, opens tests/parity.html in Chromium and checks:
//   A) same raw landmarks: JS normalisation == Python (1e-5) and ONNX Runtime Web
//      logits == PyTorch logits (1e-4), identical predictions          -> must pass
//   B) end-to-end: MediaPipe JS vs MediaPipe Python on the same padded image
//      -> reported (different runtimes can differ slightly), with agreement rates
// Writes ../results/web_parity.json.
import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { createServer } from "vite";

const webRoot = fileURLToPath(new URL("..", import.meta.url));
const server = await createServer({ root: webRoot, configFile: fileURLToPath(new URL("../vite.config.js", import.meta.url)), server: { port: 5174 } });
await server.listen();
// Uses the installed Microsoft Edge (Chromium) so no browser download is needed.
// Set PW_CHANNEL=chromium after `npx playwright install chromium` to use Playwright's own build.
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL || "msedge" });
let failed = false;
try {
  const page = await browser.newPage();
  page.on("console", (m) => m.type() === "error" && console.error("[browser]", m.text()));
  await page.goto("http://localhost:5174/tests/parity.html");
  await page.waitForFunction(() => window.__parity, null, { timeout: 180000 });
  const res = await page.evaluate(() => window.__parity);
  if (!res.ok) throw new Error(res.error + "\n" + res.stack);
  const r = res.rows;
  const withHands = r.filter((x) => x.py_pred !== null);
  const A = {
    samples: r.length,
    feature_max_diff: Math.max(...r.map((x) => x.feature_max_diff)),
    logit_max_diff: Math.max(...withHands.map((x) => x.logit_max_diff)),
    same_predictions: withHands.every((x) => x.py_pred === x.js_pred),
  };
  const e = r.map((x) => x.e2e).filter((x) => x && !x.error);
  const B = {
    samples: e.length,
    same_hand_count: e.filter((x) => x.js_hands === x.py_hands).length,
    both_detected: e.filter((x) => x.js_hands > 0 && x.py_hands > 0).length,
    same_prediction_when_both_detected: r.filter((x) => x.e2e?.js_pred != null && x.py_pred != null &&
      x.e2e.js_pred === x.py_pred).length,
    median_feature_diff_same_hand_count: median(e.filter((x) => x.js_hands === x.py_hands && x.py_hands > 0)
      .map((x) => x.max_feature_diff)),
    errors: r.map((x) => x.e2e?.error).filter(Boolean),
  };
  const passA = A.feature_max_diff < 1e-5 && A.logit_max_diff < 1e-4 && A.same_predictions;
  const out = { A_same_landmarks: A, A_pass: passA, B_mediapipe_js_vs_python: B, rows: r };
  writeFileSync(new URL("../../results/web_parity.json", import.meta.url), JSON.stringify(out, null, 2));
  console.log(JSON.stringify({ A, A_pass: passA, B }, null, 2));
  failed = !passA;
} catch (err) {
  console.error(err);
  failed = true;
} finally {
  await browser.close();
  await server.close();
}
process.exit(failed ? 1 : 0);

function median(a) {
  if (!a.length) return null;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}
