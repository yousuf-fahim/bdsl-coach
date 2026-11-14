// Line-by-line port of normalize_hands() in src/landmarks.py.
// Any change here must be mirrored there (and vice versa). tests/parity.mjs checks it.

export const N_LM = 21;
export const FEAT_DIM = 2 * N_LM * 3 + 2 + 3; // 131
const WRIST = 0;
const MIDDLE_MCP = 9;

/**
 * @param {Array<{landmarks: Array<{x:number,y:number,z:number}>|number[][], side: string}>} hands
 *        up to two hands; landmarks in MediaPipe normalised coords; side "Left" | "Right"
 * @param {number} width  width of the image MediaPipe saw
 * @param {number} height height of the image MediaPipe saw
 * @returns {Float32Array} 131 features
 */
export function normalizeHands(hands, width, height) {
  const feat = new Float32Array(FEAT_DIM);
  if (!hands || hands.length === 0) return feat;

  // a) to pixel-like units (float64 like numpy)
  let pts = hands.slice(0, 2).map(({ landmarks, side }) => ({
    side,
    p: landmarks.map((l) => {
      const [x, y, z] = Array.isArray(l) ? l : [l.x, l.y, l.z];
      return [x * width, y * height, z * width];
    }),
  }));
  // two hands: order left-to-right by wrist x in the image
  // (Python: list.sort is stable; Array.prototype.sort is stable too)
  if (pts.length === 2) pts.sort((a, b) => a.p[WRIST][0] - b.p[WRIST][0]);
  const onlyOne = pts.length === 1;

  const wrists = [];
  const scales = [];
  pts.forEach(({ p, side }, slot) => {
    const wrist = p[WRIST].slice();
    let scale = Math.hypot(p[MIDDLE_MCP][0] - wrist[0], p[MIDDLE_MCP][1] - wrist[1]);
    if (!scale) scale = 1.0; // Python: `or 1.0`
    const mirror = onlyOne && side === "Left";
    for (let i = 0; i < N_LM; i++) {
      for (let k = 0; k < 3; k++) {
        let v = (p[i][k] - wrist[k]) / scale; // b) translate, c) scale
        if (k === 0 && mirror) v = -v; // d) mirror a lone left hand
        feat[slot * 63 + i * 3 + k] = v;
      }
    }
    feat[126 + slot] = 1.0;
    wrists.push(wrist);
    scales.push(scale);
  });
  if (pts.length === 2) {
    for (let k = 0; k < 3; k++) feat[128 + k] = (wrists[1][k] - wrists[0][k]) / scales[0];
  }
  return feat;
}
