// Node unit test: web/src/normalize.js must reproduce Python's normalize_hands()
// on the same raw landmarks (fixtures written by src/export_onnx.py).
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { test } from "node:test";
import { normalizeHands } from "../src/normalize.js";

const { samples } = JSON.parse(readFileSync(new URL("./fixtures/parity_samples.json", import.meta.url), "utf8"));

test("20 fixtures: JS features == Python features", () => {
  assert.equal(samples.length, 20);
  let worst = 0;
  for (const s of samples) {
    const f = normalizeHands(s.hands, s.width, s.height);
    assert.equal(f.length, s.py_features.length);
    for (let i = 0; i < f.length; i++) worst = Math.max(worst, Math.abs(f[i] - s.py_features[i]));
  }
  console.log(`max |JS - Python| = ${worst.toExponential(2)}`);
  assert.ok(worst < 1e-5, `features differ by ${worst}`);
});

test("covers 0, 1 and 2 hands", () => {
  const counts = new Set(samples.map((s) => s.hands.length));
  assert.deepEqual([...counts].sort(), [0, 1, 2]);
});
