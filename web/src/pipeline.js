// Inference pipeline: MediaPipe HandLandmarker -> normalizeHands -> ONNX MLP -> probabilities.
// Every file is loaded from this site (relative URLs); nothing comes from a CDN,
// and camera frames never leave the device.
import { FilesetResolver, HandLandmarker } from "@mediapipe/tasks-vision";
import * as ort from "onnxruntime-web/wasm";
import { normalizeHands } from "./normalize.js";

export async function createPipeline({ mode = "VIDEO", base = "./" } = {}) {
  // All assets are resolved relative to the page, so the site works under any sub-path.
  const BASE = new URL(base, document.baseURI);
  const url = (p) => new URL(p, BASE).href;
  const config = await (await fetch(url("models/config.json"))).json();

  const fileset = await FilesetResolver.forVisionTasks(url("mediapipe-wasm"));
  const landmarker = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: url("models/hand_landmarker.task"), delegate: "CPU" },
    runningMode: mode,
    numHands: 2,
    minHandDetectionConfidence: config.min_detection_confidence,
    minHandPresenceConfidence: config.min_presence_confidence,
    minTrackingConfidence: 0.5,
  });

  // Single-threaded wasm: static hosts can't send the COOP/COEP headers threads need.
  ort.env.wasm.numThreads = 1;
  ort.env.wasm.wasmPaths = url("ort/");
  const session = await ort.InferenceSession.create(url("models/landmark_mlp.onnx"), {
    executionProviders: ["wasm"],
  });

  /** MediaPipe result -> [{landmarks, side, score}] */
  function toHands(result) {
    return (result.landmarks || []).map((lm, i) => ({
      landmarks: lm,
      side: result.handedness?.[i]?.[0]?.categoryName ?? "Right",
      score: result.handedness?.[i]?.[0]?.score ?? 0,
    }));
  }

  async function logits(features) {
    const out = await session.run({ features: new ort.Tensor("float32", features, [1, features.length]) });
    return Array.from(out.logits.data);
  }

  function softmax(z) {
    const m = Math.max(...z);
    const e = z.map((v) => Math.exp(v - m));
    const s = e.reduce((a, b) => a + b, 0);
    return e.map((v) => v / s);
  }

  /** Full pipeline on one frame. Returns {hands, features, probs|null}. */
  async function run(source, width, height, timestampMs) {
    const result = mode === "VIDEO" ? landmarker.detectForVideo(source, timestampMs) : landmarker.detect(source);
    const hands = toHands(result);
    const features = normalizeHands(hands, width, height);
    const probs = hands.length ? softmax(await logits(features)) : null;
    return { hands, features, probs };
  }

  return { config, landmarker, session, run, logits, softmax, toHands };
}
