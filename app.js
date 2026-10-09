// Facial Nerve Palsy – tablet experience
// The front camera takes a photo of the visitor, then one side of the face in
// that photo is changed to show a symptom of Bell's palsy. A new photo is taken
// for every symptom. At the end: normal face next to the full Bell's palsy face.

import { FaceLandmarker, FilesetResolver } from "./lib/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// EASY SETTINGS – change these numbers to change the timing (in seconds)
// ---------------------------------------------------------------------------
const CONFIG = {
  countdownSeconds: 3,     // the "3 2 1" countdown at the start
  titleSeconds: 4,         // the "Facial Nerve Palsy" title
  photoCountdownSeconds: 3,// "look at the camera" countdown before each photo
  symptomSeconds: 30,      // how long each symptom is shown
  finalSeconds: 5,         // the final normal vs. Bell's palsy snapshot
  thanksSeconds: 15,       // thank-you screen, then back to the start
};

// Add ?quick to the web address to preview everything fast (5 s per symptom).
if (new URLSearchParams(location.search).has("quick")) {
  CONFIG.symptomSeconds = 5;
  CONFIG.thanksSeconds = 5;
}

const SYMPTOMS = [
  { key: "brow",  title: "Drooping eyebrow",
    text: "The forehead muscle is paralysed, so the eyebrow sags and the forehead cannot wrinkle on one side." },
  { key: "eye",   title: "Eye cannot close fully",
    text: "The eyelid muscle is weak: the lower lid droops, the eye stays open and waters (tearing)." },
  { key: "cheek", title: "Flattened cheek and smile line",
    text: "The cheek sags and the fold between the nose and mouth (nasolabial fold) flattens." },
  { key: "mouth", title: "Drooping mouth corner",
    text: "One corner of the mouth droops and the mouth is pulled toward the healthy side." },
  { key: "drool", title: "Drooling",
    text: "The weak lips cannot seal, so saliva escapes from the corner of the mouth." },
];
const ALL = { brow: 1, eye: 1, cheek: 1, mouth: 1, drool: 1 };

// Face-mesh landmark numbers for the person's LEFT and RIGHT side.
const SIDE = {
  left: {
    brow: [300, 334], upperLid: 386, lowerLid: 374,
    lowerLidOuter: 373, lowerLidInner: 380, eyeOuter: 263,
    mouthCorner: 291, lowerLip: 321, upperLip: 270,
    cheek: 280, fold: 425, noseWing: 358,
  },
  right: {
    brow: [70, 105], upperLid: 159, lowerLid: 145,
    lowerLidOuter: 144, lowerLidInner: 153, eyeOuter: 33,
    mouthCorner: 61, lowerLip: 91, upperLip: 40,
    cheek: 50, fold: 205, noseWing: 129,
  },
};

// ---------------------------------------------------------------------------
// Page elements
// ---------------------------------------------------------------------------
const canvas = document.getElementById("screen");
const ctx = canvas.getContext("2d");
const video = document.getElementById("video");
const menu = document.getElementById("menu");
const statusEl = document.getElementById("status");
const btnStart = document.getElementById("start");

let faceLandmarker = null;
let stream = null;
let running = false;
let affected = SIDE.left, healthy = SIDE.right;
let strength = 1;
let wakeLock = null;

let liveLandmarks = null;    // smoothed landmarks of the live camera [{x,y}]
let lastFaceTime = -1e9;
let lastVideoTime = -1;
let lm = null;               // landmarks currently used for warping
let photo = null;            // { canvas, landmarks, crop, before }

// ---------------------------------------------------------------------------
// Load the face tracker
// ---------------------------------------------------------------------------
async function loadTracker() {
  try {
    const files = await FilesetResolver.forVisionTasks(new URL("./lib/wasm", location.href).href);
    const opts = (delegate) => ({
      baseOptions: { modelAssetPath: "./models/face_landmarker.task", delegate },
      runningMode: "VIDEO",
      numFaces: 1,
    });
    try {
      faceLandmarker = await FaceLandmarker.createFromOptions(files, opts("GPU"));
    } catch (e) {
      console.warn("GPU not available, using CPU", e);
      faceLandmarker = await FaceLandmarker.createFromOptions(files, opts("CPU"));
    }
    statusEl.textContent = "Ready. Tap Start.";
    btnStart.disabled = false;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Could not load the face tracker. Check the internet connection and reload the page.";
  }
}
loadTracker();

// ---------------------------------------------------------------------------
// WebGL face warp: moves small soft areas of the picture ("blobs")
// ---------------------------------------------------------------------------
const MAX_BLOBS = 32;
const glCanvas = document.createElement("canvas");
const gl = glCanvas.getContext("webgl", { preserveDrawingBuffer: true, premultipliedAlpha: false });
const frameCanvas = document.createElement("canvas");
const frameCtx = frameCanvas.getContext("2d");

const VERT = `
attribute vec2 a_pos;
varying vec2 v_uv;
void main() {
  v_uv = vec2(a_pos.x * 0.5 + 0.5, 0.5 - a_pos.y * 0.5);
  gl_Position = vec4(a_pos, 0.0, 1.0);
}`;
const FRAG = `
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_res;
uniform int u_n;
uniform vec4 u_a[${MAX_BLOBS}];
uniform vec4 u_b[${MAX_BLOBS}];
varying vec2 v_uv;
void main() {
  vec2 p = v_uv * u_res;
  vec2 d = vec2(0.0);
  for (int i = 0; i < ${MAX_BLOBS}; i++) {
    if (i >= u_n) break;
    vec2 q = p - u_a[i].xy;
    float r = u_a[i].z;
    d += exp(-dot(q, q) / (r * r)) * u_b[i].xy;
  }
  gl_FragColor = texture2D(u_tex, clamp((p - d) / u_res, 0.0, 1.0));
}`;

function compile(type, src) {
  const s = gl.createShader(type);
  gl.shaderSource(s, src);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw new Error(gl.getShaderInfoLog(s));
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, compile(gl.VERTEX_SHADER, VERT));
gl.attachShader(prog, compile(gl.FRAGMENT_SHADER, FRAG));
gl.linkProgram(prog);
gl.useProgram(prog);
const buf = gl.createBuffer();
gl.bindBuffer(gl.ARRAY_BUFFER, buf);
gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
const aPos = gl.getAttribLocation(prog, "a_pos");
gl.enableVertexAttribArray(aPos);
gl.vertexAttribPointer(aPos, 2, gl.FLOAT, false, 0, 0);
const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
const uRes = gl.getUniformLocation(prog, "u_res");
const uN = gl.getUniformLocation(prog, "u_n");
const uA = gl.getUniformLocation(prog, "u_a");
const uB = gl.getUniformLocation(prog, "u_b");
const blobA = new Float32Array(MAX_BLOBS * 4);
const blobB = new Float32Array(MAX_BLOBS * 4);

// Draws `source` (the photo), warped by `blobs`, into frameCanvas.
function renderWarp(source, blobs) {
  const w = source.width, h = source.height;
  if (glCanvas.width !== w || glCanvas.height !== h) {
    glCanvas.width = frameCanvas.width = w;
    glCanvas.height = frameCanvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  const n = Math.min(blobs.length, MAX_BLOBS);
  blobA.fill(0); blobB.fill(0);
  for (let i = 0; i < n; i++) {
    const b = blobs[i];
    blobA.set([b.x, b.y, b.r, 0], i * 4);
    blobB.set([b.dx, b.dy, 0, 0], i * 4);
  }
  gl.uniform2f(uRes, w, h);
  gl.uniform1i(uN, n);
  gl.uniform4fv(uA, blobA);
  gl.uniform4fv(uB, blobB);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  frameCtx.drawImage(glCanvas, 0, 0);
}

// ---------------------------------------------------------------------------
// Face geometry helpers (use the landmarks in `lm`)
// ---------------------------------------------------------------------------
const P = (i) => lm[i];

function faceFrame() {
  const top = P(10), chin = P(152);
  const dx = chin.x - top.x, dy = chin.y - top.y;
  const len = Math.hypot(dx, dy) || 1;
  const down = { x: dx / len, y: dy / len };
  const a = P(affected.eyeOuter), b = P(healthy.eyeOuter);
  const s = Math.hypot(a.x - b.x, a.y - b.y); // face scale: width between outer eye corners
  const out = { x: (a.x - b.x) / (s || 1), y: (a.y - b.y) / (s || 1) }; // toward the affected side
  return { down, out, s };
}

// Builds the list of warp blobs. `w` holds the strength (0..1) of each symptom.
function buildBlobs(w) {
  const { down, out, s } = faceFrame();
  const k = strength;
  const blobs = [];
  // a soft area at landmark i, radius r, moved `dn` down and `dout` toward the affected side (× face scale)
  const add = (i, r, dn, dout, amount) => {
    if (amount <= 0) return;
    const p = P(i);
    blobs.push({
      x: p.x, y: p.y, r: r * s,
      dx: (down.x * dn + out.x * dout) * s * amount * k,
      dy: (down.y * dn + out.y * dout) * s * amount * k,
    });
  };

  // 1. Eyebrow droops (the outer end droops most)
  add(affected.brow[1], 0.2, 0.075, 0, w.brow);
  add(affected.brow[0], 0.15, 0.04, 0, w.brow);

  // 2. Eye cannot close: lower lid sags, upper lid slightly raised
  add(affected.lowerLid, 0.09, 0.05, 0, w.eye);
  add(affected.lowerLidOuter, 0.07, 0.035, 0, w.eye);
  add(affected.lowerLidInner, 0.07, 0.03, 0, w.eye);
  add(affected.upperLid, 0.07, -0.035, 0, w.eye);

  // 3. Cheek sags, nasolabial fold flattens
  add(affected.cheek, 0.30, 0.06, 0, w.cheek);
  add(affected.fold, 0.22, 0.04, 0.025, w.cheek);
  add(affected.noseWing, 0.10, 0.025, 0, w.cheek);

  // 4. Mouth corner droops, mouth pulled toward the healthy side
  add(affected.mouthCorner, 0.20, 0.12, -0.02, w.mouth);
  add(affected.lowerLip, 0.13, 0.045, -0.01, w.mouth);
  add(affected.upperLip, 0.12, 0.035, -0.01, w.mouth);
  add(healthy.mouthCorner, 0.14, -0.012, -0.03, w.mouth);

  // 5. Drooling: a slight droop of the corner, the saliva is drawn on top
  add(affected.mouthCorner, 0.16, 0.035, 0, w.drool);

  return blobs;
}

// Where a landmark ends up after warping (so tears/drool follow the warped face)
function warpedPoint(i, blobs) {
  const p = P(i);
  let x = p.x, y = p.y;
  for (const b of blobs) {
    const qx = p.x - b.x, qy = p.y - b.y;
    const f = Math.exp(-(qx * qx + qy * qy) / (b.r * b.r));
    x += f * b.dx; y += f * b.dy;
  }
  return { x, y };
}

// ---------------------------------------------------------------------------
// Tears and drool (drawn on top of the warped photo)
// ---------------------------------------------------------------------------
function drawDrop(c, x, y, r, alpha) {
  const g = c.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r * 1.1);
  g.addColorStop(0, `rgba(255,255,255,${0.95 * alpha})`);
  g.addColorStop(0.35, `rgba(215,235,255,${0.55 * alpha})`);
  g.addColorStop(1, `rgba(160,200,240,${0.25 * alpha})`);
  c.fillStyle = g;
  c.beginPath();
  c.moveTo(x, y - r * 2.2);
  c.bezierCurveTo(x + r * 0.25, y - r * 1.3, x + r, y - r * 0.7, x + r, y);
  c.arc(x, y, r, 0, Math.PI);
  c.bezierCurveTo(x - r, y - r * 0.7, x - r * 0.25, y - r * 1.3, x, y - r * 2.2);
  c.fill();
}

function drawStream(c, x, y, len, width, alpha) {
  const g = c.createLinearGradient(x - width, 0, x + width, 0);
  g.addColorStop(0, `rgba(170,205,240,${0.25 * alpha})`);
  g.addColorStop(0.4, `rgba(255,255,255,${0.8 * alpha})`);
  g.addColorStop(1, `rgba(170,205,240,${0.3 * alpha})`);
  c.fillStyle = g;
  c.beginPath();
  c.moveTo(x - width, y);
  c.quadraticCurveTo(x - width * 0.4, y + len * 0.6, x - width * 0.3, y + len);
  c.lineTo(x + width * 0.3, y + len);
  c.quadraticCurveTo(x + width * 0.4, y + len * 0.6, x + width, y);
  c.closePath();
  c.fill();
}

// t = seconds since the effect started
function drawOverlays(c, w, blobs, t) {
  const { down, s } = faceFrame();
  const angle = Math.atan2(down.y, down.x) - Math.PI / 2;

  if (w.eye > 0) { // tears rolling down from the outer lower eyelid
    const p = warpedPoint(affected.lowerLidOuter, blobs);
    c.save();
    c.translate(p.x, p.y + 0.01 * s);
    c.rotate(angle);
    for (let n = 0; n < 2; n++) {
      const phase = ((t + n * 2.2) % 4.4) / 4.4;
      const y = phase * phase * 0.7 * s;
      const a = w.eye * Math.min(1, phase * 6) * (1 - phase);
      drawDrop(c, 0, y, 0.028 * s, a);
    }
    drawDrop(c, 0, 0.012 * s, 0.014 * s, w.eye * 0.8); // wet rim
    c.restore();
  }

  if (w.drool > 0) { // saliva from the affected mouth corner
    const p = warpedPoint(affected.mouthCorner, blobs);
    c.save();
    c.translate(p.x, p.y);
    c.rotate(angle);
    const len = 0.22 * s * Math.min(1, t / 3) * (0.9 + 0.1 * Math.sin(t * 2));
    drawStream(c, 0, 0, len, 0.018 * s, w.drool);
    const phase = (t % 3.2) / 3.2;
    const grow = Math.min(1, phase * 2.5);
    const fall = Math.max(0, phase - 0.4) / 0.6;
    drawDrop(c, 0, len + 0.02 * s + fall * fall * 0.9 * s, (0.012 + 0.016 * grow) * s,
             w.drool * (1 - fall * 0.8) * Math.min(1, t / 3));
    c.restore();
  }
}

// ---------------------------------------------------------------------------
// Steps of the experience
// ---------------------------------------------------------------------------
function buildSteps() {
  const steps = [
    { type: "countdown", dur: CONFIG.countdownSeconds },
    { type: "title", dur: CONFIG.titleSeconds },
  ];
  SYMPTOMS.forEach((sym, i) => {
    steps.push({ type: "photo", label: `Photo ${i + 1}` });
    steps.push({ type: "symptom", dur: CONFIG.symptomSeconds, index: i });
  });
  steps.push({ type: "photo", label: "Final photo" });
  steps.push({ type: "final", dur: CONFIG.finalSeconds });
  steps.push({ type: "thanks", dur: CONFIG.thanksSeconds });
  return steps;
}

let steps = [];
let stepIndex = 0;
let stepStart = 0;

function nextStep(now) {
  stepIndex++;
  stepStart = now;
  if (stepIndex >= steps.length) stopExperience();
}

const ease = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };

// ---------------------------------------------------------------------------
// Live face tracking
// ---------------------------------------------------------------------------
function track(now) {
  if (!faceLandmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = faceLandmarker.detectForVideo(video, now);
  const face = res.faceLandmarks && res.faceLandmarks[0];
  if (!face) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!liveLandmarks || now - lastFaceTime > 500) {
    liveLandmarks = face.map((p) => ({ x: p.x * vw, y: p.y * vh }));
  } else {
    const a = 0.6; // smoothing
    for (let i = 0; i < face.length; i++) {
      liveLandmarks[i].x += (face[i].x * vw - liveLandmarks[i].x) * a;
      liveLandmarks[i].y += (face[i].y * vh - liveLandmarks[i].y) * a;
    }
  }
  lastFaceTime = now;
}

const faceVisible = (now) => liveLandmarks && now - lastFaceTime < 400;

// ---------------------------------------------------------------------------
// Taking a photo
// ---------------------------------------------------------------------------
function takePhoto() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const c = document.createElement("canvas");
  c.width = vw; c.height = vh;
  c.getContext("2d").drawImage(video, 0, 0, vw, vh);
  const landmarks = liveLandmarks.map((p) => ({ x: p.x, y: p.y }));

  // crop around the face (portrait shape)
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  let ch = (maxY - minY) * 1.5, cw = ch * 0.8;
  cw = Math.max(cw, (maxX - minX) * 1.35); ch = cw / 0.8;
  cw = Math.min(cw, vw); ch = Math.min(ch, vh);
  const crop = {
    x: Math.max(0, Math.min(vw - cw, cx - cw / 2)),
    y: Math.max(0, Math.min(vh - ch, cy - ch / 2)),
    w: cw, h: ch,
  };
  photo = { canvas: c, landmarks, crop };
}

// Draws the cropped face from `src` into the box, mirrored like a selfie
function drawFace(src, crop, x, y, w, h) {
  ctx.save();
  ctx.translate(x + w, y);
  ctx.scale(-1, 1);
  ctx.drawImage(src, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h);
  ctx.restore();
}

// Renders the photo with the given symptom weights into frameCanvas
function renderPhoto(w, t) {
  lm = photo.landmarks;
  const blobs = buildBlobs(w);
  renderWarp(photo.canvas, blobs);
  drawOverlays(frameCtx, w, blobs, t);
}

// ---------------------------------------------------------------------------
// Drawing helpers
// ---------------------------------------------------------------------------
const FONT = `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

function background(w, h) {
  const g = ctx.createRadialGradient(w / 2, h * 0.4, 0, w / 2, h / 2, Math.max(w, h) * 0.7);
  g.addColorStop(0, "#1b2a4a");
  g.addColorStop(1, "#04060b");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function text(str, x, y, size, color = "#fff", weight = 700, alpha = 1) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${FONT}`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(str, x, y);
  ctx.globalAlpha = 1;
}

function wrapLines(str, maxW, size, weight) {
  ctx.font = `${weight} ${size}px ${FONT}`;
  const words = str.split(" ");
  const lines = [];
  let line = "";
  for (const wd of words) {
    const test = line ? line + " " + wd : wd;
    if (ctx.measureText(test).width > maxW && line) { lines.push(line); line = wd; } else line = test;
  }
  if (line) lines.push(line);
  return lines;
}

function roundRect(x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

// Two face pictures side by side (landscape) or on top of each other (portrait).
// Returns the area left over below for the caption.
function drawPair(w, h, top, bottom, labelA, labelB, drawA, drawB) {
  const u = Math.min(w, h);
  const labelH = u * 0.07;
  const areaH = bottom - top;
  const landscape = w >= h * 0.9;
  const aspect = photo.crop.w / photo.crop.h;
  let pw, ph, x1, y1, x2, y2;
  if (landscape) {
    const gap = w * 0.04;
    ph = areaH - labelH;
    pw = ph * aspect;
    if (pw * 2 + gap > w * 0.92) { pw = (w * 0.92 - gap) / 2; ph = pw / aspect; }
    x1 = w / 2 - gap / 2 - pw; x2 = w / 2 + gap / 2;
    y1 = y2 = top + (areaH - labelH - ph) / 2;
  } else {
    const gap = h * 0.02;
    ph = (areaH - gap - labelH * 2) / 2;
    pw = ph * aspect;
    if (pw > w * 0.9) { pw = w * 0.9; ph = pw / aspect; }
    x1 = x2 = (w - pw) / 2;
    y1 = top; y2 = top + ph + labelH + gap;
  }
  drawA(x1, y1, pw, ph);
  drawB(x2, y2, pw, ph);
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = Math.max(2, u * 0.004);
  ctx.strokeRect(x1, y1, pw, ph);
  ctx.strokeRect(x2, y2, pw, ph);
  text(labelA, x1 + pw / 2, y1 + ph + labelH * 0.5, u * 0.04, "#fff", 700);
  text(labelB, x2 + pw / 2, y2 + ph + labelH * 0.5, u * 0.04, "#ffd479", 700);
}

function caption(w, h, label, title, body, progress) {
  const u = Math.min(w, h);
  const boxW = Math.min(w * 0.9, u * 1.5);
  const titleSize = u * 0.05, bodySize = u * 0.03, labelSize = u * 0.026;
  const lines = wrapLines(body, boxW * 0.9, bodySize, 400);
  const boxH = labelSize * 1.6 + titleSize * 1.4 + lines.length * bodySize * 1.35 + u * 0.05;
  const bx = (w - boxW) / 2, by = h * 0.98 - boxH;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  roundRect(bx, by, boxW, boxH, u * 0.025);
  ctx.fill();
  let y = by + u * 0.02 + labelSize * 0.8;
  text(label, w / 2, y, labelSize, "#9fc3ff", 600);
  y += labelSize * 0.8 + titleSize * 0.75;
  text(title, w / 2, y, titleSize, "#ffd479", 800);
  y += titleSize * 0.65 + bodySize * 0.8;
  for (const ln of lines) { text(ln, w / 2, y, bodySize, "#fff", 400); y += bodySize * 1.35; }
  ctx.fillStyle = "rgba(255,255,255,0.2)";
  ctx.fillRect(bx + u * 0.02, by + boxH - u * 0.015, boxW - u * 0.04, u * 0.006);
  ctx.fillStyle = "#ffd479";
  ctx.fillRect(bx + u * 0.02, by + boxH - u * 0.015, (boxW - u * 0.04) * Math.min(1, progress), u * 0.006);
  return by;
}

// ---------------------------------------------------------------------------
// Main loop
// ---------------------------------------------------------------------------
function resize() {
  const dpr = Math.min(window.devicePixelRatio || 1, 2);
  canvas.width = Math.round(window.innerWidth * dpr);
  canvas.height = Math.round(window.innerHeight * dpr);
}
window.addEventListener("resize", resize);
resize();

function loop(now) {
  if (!running) return;
  requestAnimationFrame(loop);
  track(now);

  const step = steps[stepIndex];
  const local = (now - stepStart) / 1000;
  if (step.dur !== undefined && local >= step.dur) { nextStep(now); return; }

  const w = canvas.width, h = canvas.height, u = Math.min(w, h);
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, w, h);

  if (step.type === "countdown") {
    background(w, h);
    const n = Math.max(1, Math.ceil(step.dur - local));
    const f = local % 1;
    text(String(n), w / 2, h / 2, u * 0.35 * (1 + (1 - f) * 0.3), "#fff", 800, 1 - f * 0.6);
  } else if (step.type === "title") {
    background(w, h);
    const a = ease(local / 0.8) * ease((step.dur - local) / 0.6);
    text("Facial Nerve Palsy", w / 2, h * 0.45, u * 0.1, "#fff", 800, a);
    text("Bell's palsy", w / 2, h * 0.57, u * 0.05, "#ffd479", 500, a);
  } else if (step.type === "photo") {
    drawPhotoStep(now, local, step, w, h, u);
  } else if (step.type === "symptom") {
    const sym = SYMPTOMS[step.index];
    const weights = { brow: 0, eye: 0, cheek: 0, mouth: 0, drool: 0 };
    weights[sym.key] = ease((local - 0.8) / 2); // the change appears smoothly
    renderPhoto(weights, local);
    background(w, h);
    const capTop = caption(w, h, `Symptom ${step.index + 1} of ${SYMPTOMS.length}`, sym.title, sym.text, local / step.dur);
    drawPair(w, h, h * 0.03, capTop - u * 0.02, "Your face", sym.title,
      (x, y, pw, ph) => drawFace(photo.canvas, photo.crop, x, y, pw, ph),
      (x, y, pw, ph) => drawFace(frameCanvas, photo.crop, x, y, pw, ph));
    flash(local, w, h);
  } else if (step.type === "final") {
    renderPhoto(ALL, local + 2);
    background(w, h);
    text("All symptoms together", w / 2, h * 0.06, u * 0.05, "#9fc3ff", 700);
    drawPair(w, h, h * 0.12, h * 0.97, "Normal face", "Bell's palsy face",
      (x, y, pw, ph) => drawFace(photo.canvas, photo.crop, x, y, pw, ph),
      (x, y, pw, ph) => drawFace(frameCanvas, photo.crop, x, y, pw, ph));
    flash(local, w, h);
  } else if (step.type === "thanks") {
    background(w, h);
    const a = ease(local);
    text("Thank you", w / 2, h * 0.43, u * 0.11, "#fff", 800, a);
    text("for the experience", w / 2, h * 0.56, u * 0.06, "#ffd479", 500, a);
  }
}

// White camera flash right after a photo
function flash(local, w, h) {
  if (local > 0.5) return;
  ctx.fillStyle = `rgba(255,255,255,${1 - local / 0.5})`;
  ctx.fillRect(0, 0, w, h);
}

// Live camera with a countdown; the photo is taken when a face is visible
function drawPhotoStep(now, local, step, w, h, u) {
  if (video.readyState >= 2) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const sc = Math.max(w / vw, h / vh);
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1); // mirror, like a selfie
    ctx.drawImage(video, (w - vw * sc) / 2, (h - vh * sc) / 2, vw * sc, vh * sc);
    ctx.restore();
  }
  // face guide oval
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = u * 0.006;
  ctx.setLineDash([u * 0.02, u * 0.015]);
  ctx.beginPath();
  ctx.ellipse(w / 2, h * 0.5, u * 0.3, u * 0.4, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  ctx.fillStyle = "rgba(0,0,0,0.55)";
  roundRect(w / 2 - u * 0.45, h * 0.02, u * 0.9, u * 0.12, u * 0.02);
  ctx.fill();
  text(step.label, w / 2, h * 0.02 + u * 0.035, u * 0.03, "#9fc3ff", 600);
  const hasFace = faceVisible(now);
  text(hasFace ? "Look at the camera and keep still" : "Put your face inside the oval",
       w / 2, h * 0.02 + u * 0.08, u * 0.04, "#fff", 700);

  const left = CONFIG.photoCountdownSeconds - local;
  if (left > 0) {
    text(String(Math.ceil(left)), w / 2, h * 0.88, u * 0.14, "#ffd479", 800, 0.9);
  } else if (hasFace) {
    takePhoto();
    nextStep(now);
  }
}

// ---------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------
async function startExperience() {
  affected = SIDE[document.getElementById("side").value];
  healthy = affected === SIDE.left ? SIDE.right : SIDE.left;
  strength = parseFloat(document.getElementById("strength").value);

  try { await document.documentElement.requestFullscreen?.(); } catch (e) {}
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (e) {}

  statusEl.textContent = "Starting camera…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Camera not available. Please allow camera access and try again.";
    return;
  }
  video.srcObject = stream;
  try { await video.play(); } catch (e) {}

  liveLandmarks = null; photo = null;
  steps = buildSteps();
  stepIndex = 0;
  menu.classList.add("hidden");
  running = true;
  resize();
  stepStart = performance.now();
  requestAnimationFrame(loop);
}

function stopExperience() {
  running = false;
  if (stream) { stream.getTracks().forEach((tr) => tr.stop()); stream = null; }
  try { wakeLock?.release(); } catch (e) {}
  wakeLock = null;
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  menu.classList.remove("hidden");
  statusEl.textContent = "Ready for the next visitor. Tap Start.";
}

btnStart.addEventListener("click", startExperience);

// Double-tap (or double-click) to go back to the start screen
let lastTap = 0;
canvas.addEventListener("pointerdown", () => {
  const now = performance.now();
  if (running && now - lastTap < 400) stopExperience();
  lastTap = now;
});

// Debug hook used for automated testing
window.__palsy = { CONFIG, get step() { return running ? steps[stepIndex].type : "menu"; } };
