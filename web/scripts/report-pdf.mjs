// Convert a Markdown file in docs/ to PDF with headless Edge (no pandoc/LaTeX needed).
// Usage (from web/): node scripts/report-pdf.mjs ../docs/report.md  -> ../docs/report.pdf
// Relative image links (e.g. ../results/x.png) resolve from the Markdown file's folder.
import { readFileSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, join, resolve } from "node:path";
import { pathToFileURL } from "node:url";
import { marked } from "marked";
import { chromium } from "playwright";

const src = resolve(process.argv[2] || "../docs/report.md");
const out = src.replace(/\.md$/, ".pdf");
const html = `<!doctype html><html><head><meta charset="utf-8"><style>
  @page { size: A4; margin: 18mm 16mm; }
  body { font: 10.5pt/1.45 "Segoe UI", "Nirmala UI", "Noto Sans Bengali", sans-serif; color: #111; }
  h1 { font-size: 18pt; margin: 0 0 6pt; } h2 { font-size: 13.5pt; margin: 16pt 0 6pt; border-bottom: 1px solid #ccc; }
  h3 { font-size: 11.5pt; margin: 12pt 0 4pt; } p, li { text-align: justify; }
  table { border-collapse: collapse; margin: 8pt 0; font-size: 9.5pt; } th, td { border: 1px solid #bbb; padding: 3pt 6pt; }
  th { background: #f0f0f0; } img { max-width: 100%; max-height: 95mm; display: block; margin: 6pt auto; }
  code { font-size: 9pt; background: #f4f4f4; padding: 0 2pt; } pre { background: #f4f4f4; padding: 6pt; font-size: 8.5pt; white-space: pre-wrap; }
  blockquote { border-left: 3px solid #999; margin: 6pt 0; padding: 2pt 10pt; color: #333; }
  em.caption, p > em:only-child { display: block; text-align: center; font-size: 9pt; color: #444; }
</style></head><body>${marked.parse(readFileSync(src, "utf8"))}</body></html>`;
const tmp = join(dirname(src), `.${basename(src)}.tmp.html`);
writeFileSync(tmp, html);
const browser = await chromium.launch({ channel: process.env.PW_CHANNEL || "msedge" });
try {
  const page = await browser.newPage();
  await page.goto(pathToFileURL(tmp).href, { waitUntil: "load" });
  await page.pdf({ path: out, format: "A4", printBackground: true, displayHeaderFooter: true,
    headerTemplate: "<span></span>",
    footerTemplate: '<div style="font-size:8pt;width:100%;text-align:center;color:#666"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
    margin: { top: "16mm", bottom: "16mm", left: "15mm", right: "15mm" } });
  console.log("wrote", out);
} finally {
  await browser.close();
  unlinkSync(tmp);
}
