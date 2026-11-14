// Smoke test of the BUILT static site (web/dist) in headless Edge with a fake camera.
//   - serves dist/ with `vite preview` under a sub-path-like URL check (relative base)
//   - fake camera = logs/fake_camera.y4m if present (src/make_fake_camera.py), else Chromium's test pattern
//   - checks: models load, frames are processed, NO request leaves localhost (self-hosted),
//     no console errors
//   - saves screenshots to ../logs/smoke/ for docs/demo.gif
// Writes ../results/web_smoke.json. Run after `npm run build`.
import { existsSync, mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { chromium } from "playwright";
import { preview } from "vite";

const here = (p) => fileURLToPath(new URL(p, import.meta.url));
const fake = here("../../logs/fake_camera.y4m");
const shots = here("../../logs/smoke/");
mkdirSync(shots, { recursive: true });

const server = await preview({ root: here(".."), configFile: here("../vite.config.js"), preview: { port: 4174 } });
const args = ["--use-fake-device-for-media-stream", "--use-fake-ui-for-media-stream"];
if (existsSync(fake)) args.push(`--use-file-for-fake-video-capture=${fake}`);
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL || "msedge", args });
const result = { fake_video: existsSync(fake), requests: [], external_requests: [], console_errors: [], failed_requests: [] };
let ok = false;
try {
  const context = await browser.newContext({ viewport: { width: 1100, height: 760 }, permissions: ["camera"] });
  const page = await context.newPage();
  page.on("request", (r) => {
    const u = new URL(r.url());
    result.requests.push(u.pathname);
    if (!["localhost", "127.0.0.1"].includes(u.hostname) && !["data:", "blob:"].includes(u.protocol)) {
      result.external_requests.push(r.url());
    }
  });
  page.on("requestfailed", (r) => result.failed_requests.push(r.url()));
  page.on("console", (m) => {
    // MediaPipe logs INFO lines through console.error; only keep real errors.
    if (m.type() === "error" && !/^INFO:|^W\d{4}|^I\d{4}/.test(m.text())) result.console_errors.push(m.text());
  });
  const t0 = Date.now();
  await page.goto("http://localhost:4174/");
  await page.waitForFunction(() => window.__bdslReady || window.__bdslError, null, { timeout: 120000 });
  result.load_error = await page.evaluate(() => window.__bdslError || null);
  result.models_loaded_ms = Date.now() - t0;
  await page.screenshot({ path: `${shots}00_loaded.png` });
  await page.click("#start");
  await page.waitForFunction(() => (window.__bdslFrames || 0) >= 20, null, { timeout: 60000 });
  // Recognise mode: a screenshot every 0.5 s for ~9 s
  for (let i = 0; i < 18; i++) {
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${shots}r${String(i).padStart(2, "0")}.png` });
  }
  const f0 = await page.evaluate(() => window.__bdslFrames);
  await page.waitForTimeout(3000);
  result.fps = ((await page.evaluate(() => window.__bdslFrames)) - f0) / 3;
  await page.click("#tab-practice");
  for (let i = 0; i < 6; i++) {
    await page.waitForTimeout(500);
    await page.screenshot({ path: `${shots}p${String(i).padStart(2, "0")}.png` });
  }
  result.top3_visible = await page.evaluate(() => document.querySelectorAll("#top3 li").length);
  ok = !result.load_error && result.external_requests.length === 0 && result.console_errors.length === 0 &&
    result.failed_requests.length === 0;
} catch (err) {
  result.exception = String(err);
} finally {
  result.pass = ok;
  result.requests = [...new Set(result.requests)].sort();
  writeFileSync(here("../../results/web_smoke.json"), JSON.stringify(result, null, 2));
  console.log(JSON.stringify({ ...result, requests: result.requests.length }, null, 2));
  await browser.close();
  server.httpServer.close();
}
process.exit(ok ? 0 : 1);
