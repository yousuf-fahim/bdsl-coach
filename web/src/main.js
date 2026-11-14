import "@fontsource/noto-sans-bengali/400.css";
import "@fontsource/noto-sans-bengali/700.css";
import "./style.css";
import { createPipeline } from "./pipeline.js";

const $ = (id) => document.getElementById(id);
const video = $("video");
const canvas = $("overlay");
const ctx = canvas.getContext("2d");

const SMOOTH_FRAMES = 5; // average probabilities over the last 5 frames with a hand
const PRACTICE_CONF = 0.7; // target must be top-1 with >= 70% ...
const PRACTICE_HOLD_MS = 500; // ... for half a second

// MediaPipe hand skeleton
const CONNECTIONS = [[0, 1], [1, 2], [2, 3], [3, 4], [0, 5], [5, 6], [6, 7], [7, 8], [5, 9], [9, 10], [10, 11],
  [11, 12], [9, 13], [13, 14], [14, 15], [15, 16], [13, 17], [17, 18], [18, 19], [19, 20], [0, 17]];

let pipe = null;
let labels = [];
let mode = "recognise";
let history = [];
let missed = 0;
let target = 0;
let holdStart = null;
let succeeded = false;
let busy = false;

function setStatus(text) {
  $("status").textContent = text;
}

async function init() {
  try {
    pipe = await createPipeline({ mode: "VIDEO" });
    labels = pipe.config.labels;
    if (String(pipe.config.label_status).startsWith("PROVISIONAL")) {
      $("label-note").textContent = "Letter labels are provisional and still being verified.";
    }
    const sel = $("target-select");
    labels.forEach((l) => sel.add(new Option(`${l.display}  (${l.roman})`, l.index)));
    pickTarget(0);
    $("start").disabled = false;
    $("start").textContent = "Start camera";
    setStatus("Models loaded. Start the camera to begin.");
    window.__bdslReady = true; // used by the automated smoke test
  } catch (err) {
    console.error(err);
    setStatus(`Could not load the models: ${err.message}`);
    window.__bdslError = String(err);
  }
}

async function startCamera() {
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 640 }, height: { ideal: 480 } },
      audio: false,
    });
    video.srcObject = stream;
    await video.play();
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    $("camera-cta").classList.add("hidden");
    setStatus("Show a sign to the camera.");
    requestAnimationFrame(loop);
  } catch (err) {
    setStatus(`Camera unavailable: ${err.message}. Allow camera access and use https or localhost.`);
  }
}

let lastVideoTime = -1;
async function loop() {
  requestAnimationFrame(loop);
  if (busy || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  busy = true;
  try {
    // Raw (un-mirrored) frame goes to MediaPipe, exactly like the dataset photos.
    const { hands, probs } = await pipe.run(video, video.videoWidth, video.videoHeight, performance.now());
    draw(hands);
    update(probs);
    window.__bdslFrames = (window.__bdslFrames || 0) + 1;
  } finally {
    busy = false;
  }
}

function draw(hands) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const w = canvas.width;
  const h = canvas.height;
  hands.forEach((hand, k) => {
    ctx.strokeStyle = k === 0 ? "#38e07b" : "#3ccbf0";
    ctx.lineWidth = 3;
    for (const [a, b] of CONNECTIONS) {
      ctx.beginPath();
      ctx.moveTo(hand.landmarks[a].x * w, hand.landmarks[a].y * h);
      ctx.lineTo(hand.landmarks[b].x * w, hand.landmarks[b].y * h);
      ctx.stroke();
    }
    ctx.fillStyle = "#ff4d4d";
    for (const p of hand.landmarks) {
      ctx.beginPath();
      ctx.arc(p.x * w, p.y * h, 3.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
}

function smoothed(probs) {
  if (probs) {
    history.push(probs);
    if (history.length > SMOOTH_FRAMES) history.shift();
    missed = 0;
  } else if (++missed > SMOOTH_FRAMES) {
    history = [];
  }
  if (!history.length) return null;
  const avg = new Array(history[0].length).fill(0);
  for (const p of history) p.forEach((v, i) => (avg[i] += v / history.length));
  return avg;
}

function update(probs) {
  const avg = smoothed(probs);
  const top = avg ? avg.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3) : [];
  $("top3").innerHTML = top
    .map(([p, i]) => `<li><span class="letter">${labels[i].display}</span>` +
      `<span class="bar"><div style="width:${(p * 100).toFixed(1)}%"></div></span>` +
      `<span class="pct">${(p * 100).toFixed(0)}%</span></li>`)
    .join("");
  if (!avg) setStatus(probs === null ? "No hand detected. Move your hand into view." : "");
  else setStatus("");
  if (mode === "practice") practice(top);
}

function practice(top) {
  if (succeeded) return;
  const ok = top.length && top[0][1] === target && top[0][0] >= PRACTICE_CONF;
  const now = performance.now();
  if (ok) {
    holdStart ??= now;
    const frac = Math.min(1, (now - holdStart) / PRACTICE_HOLD_MS);
    $("hold-bar").style.width = `${frac * 100}%`;
    if (frac >= 1) {
      succeeded = true;
      $("target").classList.add("success");
      $("practice-msg").textContent = "Correct! Next sign in a moment…";
      setTimeout(() => pickTarget(), 1500);
    }
  } else {
    holdStart = null;
    $("hold-bar").style.width = "0%";
  }
}

function pickTarget(i) {
  target = i ?? (target + 1 + Math.floor(Math.random() * (labels.length - 1))) % labels.length;
  succeeded = false;
  holdStart = null;
  $("target").textContent = labels[target].display;
  $("target-roman").textContent = labels[target].roman;
  $("target").classList.remove("success");
  $("hold-bar").style.width = "0%";
  $("practice-msg").textContent = "Hold the sign above 70% confidence for half a second.";
  $("target-select").value = String(target);
}

function setMode(m) {
  mode = m;
  $("tab-recognise").setAttribute("aria-selected", String(m === "recognise"));
  $("tab-practice").setAttribute("aria-selected", String(m === "practice"));
  $("panel-recognise").classList.toggle("hidden", m !== "recognise");
  $("panel-practice").classList.toggle("hidden", m !== "practice");
}

$("start").addEventListener("click", startCamera);
$("tab-recognise").addEventListener("click", () => setMode("recognise"));
$("tab-practice").addEventListener("click", () => setMode("practice"));
$("next").addEventListener("click", () => pickTarget());
$("target-select").addEventListener("change", (e) => pickTarget(Number(e.target.value)));
init();
