// Facial Nerve Palsy – tablet experience
// The front camera takes a photo of the visitor, then one side of the face in
// that photo is changed to show a symptom of Bell's palsy. A new photo is taken
// for every symptom. At the end: normal face next to the full Bell's palsy face.

import { FaceLandmarker, FilesetResolver } from "./lib/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// EASY SETTINGS – change these numbers to change the timing (in seconds)
// ---------------------------------------------------------------------------
const CONFIG = {
  countdownSeconds: 3,      // the "3 2 1" countdown at the start
  titleSeconds: 4,          // the "Facial Nerve Palsy" title
  photoGetReadySeconds: 4,  // "get ready" time before each photo
  photoHoldSeconds: 3,      // face must stay in a good position this long (3-2-1) before the photo
  symptomSeconds: 10,       // how long each changed photo is shown
  finalSeconds: 5,          // the final normal vs. Bell's palsy snapshot
  thanksSeconds: 15,        // thank-you screen, then back to the start
};

const DEBUG = new URLSearchParams(location.search).has("debug");

// Add ?quick to the web address to preview everything fast (5 s per symptom).
if (new URLSearchParams(location.search).has("quick")) {
  CONFIG.photoGetReadySeconds = 1;
  CONFIG.symptomSeconds = 5;
  CONFIG.thanksSeconds = 5;
}

const SYMPTOMS = [
  { key: "brow",  title: "Drooping eyebrow" },
  { key: "eye",   title: "Eye cannot close fully" },
  { key: "cheek", title: "Flattened cheek and smile line" },
  { key: "mouth", title: "Drooping mouth corner" },
  { key: "drool", title: "Drooling" },
];
const NONE = { brow: 0, eye: 0, cheek: 0, mouth: 0, drool: 0, close: 0 };
const ALL = { brow: 1, eye: 1, cheek: 1, mouth: 1, drool: 1, close: 0 };

// Face-mesh landmark numbers for the person's LEFT and RIGHT side.
const SIDE = {
  left: {
    brow: [300, 334], forehead: 299, upperLid: 386, lowerLid: 374,
    lowerLidOuter: 373, lowerLidInner: 380, eyeOuter: 263,
    mouthCorner: 291, lowerLip: 321, upperLip: 270,
    cheek: 280, fold: 425, noseWing: 358,
  },
  right: {
    brow: [70, 105], forehead: 69, upperLid: 159, lowerLid: 145,
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

let live = null;             // latest tracking: { lm, turn, blink, speed }
let lastFaceTime = -1e9;
let lastVideoTime = -1;
let lm = null;               // landmarks currently used for warping
let photo = null;            // { canvas, landmarks }

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
      outputFaceBlendshapes: true,
      outputFacialTransformationMatrixes: true,
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
// WebGL face warp: moves soft areas of the picture ("blobs") and smooths
// wrinkles/folds in other soft areas
// ---------------------------------------------------------------------------
const MAX_BLOBS = 32;
const MAX_SMOOTH = 8;
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
uniform int u_ns;
uniform vec4 u_s[${MAX_SMOOTH}];
uniform float u_blur;
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
  vec2 sp = p - d;
  vec4 c = texture2D(u_tex, clamp(sp / u_res, 0.0, 1.0));
  float m = 0.0;
  for (int i = 0; i < ${MAX_SMOOTH}; i++) {
    if (i >= u_ns) break;
    vec2 q = p - u_s[i].xy;
    float r = u_s[i].z;
    m = max(m, u_s[i].w * exp(-dot(q, q) / (r * r)));
  }
  if (m > 0.01) {
    vec4 acc = c;
    for (int k = 0; k < 12; k++) {
      float a = float(k) * 0.5236;
      vec2 dir = vec2(cos(a), sin(a));
      acc += texture2D(u_tex, clamp((sp + dir * u_blur * 0.5) / u_res, 0.0, 1.0));
      acc += texture2D(u_tex, clamp((sp + dir * u_blur) / u_res, 0.0, 1.0));
    }
    c = mix(c, acc / 25.0, m);
  }
  gl_FragColor = c;
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
const U = (name) => gl.getUniformLocation(prog, name);
const uRes = U("u_res"), uN = U("u_n"), uA = U("u_a"), uB = U("u_b");
const uNs = U("u_ns"), uS = U("u_s"), uBlur = U("u_blur");
const blobA = new Float32Array(MAX_BLOBS * 4);
const blobB = new Float32Array(MAX_BLOBS * 4);
const smoothS = new Float32Array(MAX_SMOOTH * 4);

// Draws `source` (the photo), changed by `fx`, into frameCanvas.
function renderWarp(source, fx) {
  const w = source.width, h = source.height;
  if (glCanvas.width !== w || glCanvas.height !== h) {
    glCanvas.width = frameCanvas.width = w;
    glCanvas.height = frameCanvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
  const n = Math.min(fx.blobs.length, MAX_BLOBS);
  blobA.fill(0); blobB.fill(0); smoothS.fill(0);
  for (let i = 0; i < n; i++) {
    const b = fx.blobs[i];
    blobA.set([b.x, b.y, b.r, 0], i * 4);
    blobB.set([b.dx, b.dy, 0, 0], i * 4);
  }
  const ns = Math.min(fx.smooth.length, MAX_SMOOTH);
  for (let i = 0; i < ns; i++) {
    const s = fx.smooth[i];
    smoothS.set([s.x, s.y, s.r, s.amount], i * 4);
  }
  gl.uniform2f(uRes, w, h);
  gl.uniform1i(uN, n);
  gl.uniform4fv(uA, blobA);
  gl.uniform4fv(uB, blobB);
  gl.uniform1i(uNs, ns);
  gl.uniform4fv(uS, smoothS);
  gl.uniform1f(uBlur, fx.blur);
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

// Builds the changes for the given symptom strengths `w` (0..1 each).
function buildEffects(w) {
  const { down, out, s } = faceFrame();
  const k = strength;
  const blobs = [], smooth = [];
  // move a soft area at landmark i (radius r) by `dn` down and `dout` toward the affected side
  // (all × face scale). `up` shifts the centre of the area upward.
  const add = (i, r, dn, dout, amount, up = 0) => {
    if (amount <= 0) return;
    const p = P(i);
    blobs.push({
      x: p.x - down.x * up * s, y: p.y - down.y * up * s, r: r * s,
      dx: (down.x * dn + out.x * dout) * s * amount * k,
      dy: (down.y * dn + out.y * dout) * s * amount * k,
    });
  };
  // smooth out wrinkles / folds around landmark i
  const flat = (i, r, amount, up = 0) => {
    if (amount <= 0) return;
    const p = P(i);
    smooth.push({ x: p.x - down.x * up * s, y: p.y - down.y * up * s, r: r * s, amount: Math.min(1, amount) });
  };

  // 1. Eyebrow droops (the outer end most), forehead smooth (no wrinkles)
  add(affected.brow[1], 0.28, 0.2, 0, w.brow, 0.06);
  add(affected.brow[0], 0.2, 0.11, 0, w.brow, 0.03);
  flat(affected.brow[1], 0.2, 0.6 * w.brow, 0.3);

  // 2. Eye cannot close: when trying to close both eyes only the healthy eye
  //    closes (w.close = 0..1); the affected lower lid sags a little
  add(affected.lowerLid, 0.09, 0.04, 0, w.eye, -0.05);
  if (w.close > 0) {
    const open = Math.hypot(P(healthy.upperLid).x - P(healthy.lowerLid).x,
                            P(healthy.upperLid).y - P(healthy.lowerLid).y) / s;
    const dn = open * 1.2 + 0.015;
    add(healthy.lowerLid, Math.max(0.09, dn * 1.1), -dn, 0, w.close, -0.04);
  }

  // 3. Cheek sags, the nose-to-mouth fold is flattened
  add(affected.fold, 0.28, 0.13, 0.03, w.cheek);
  add(affected.noseWing, 0.13, 0.05, 0, w.cheek);
  flat(affected.fold, 0.24, 0.9 * w.cheek);

  // 4. Mouth twisted toward the healthy side; the affected half hangs down
  add(13, 0.30, 0, -0.04, w.mouth);                          // whole mouth shifts
  add(affected.mouthCorner, 0.16, 0.15, -0.02, w.mouth);
  add(affected.lowerLip, 0.11, 0.09, 0, w.mouth);
  add(affected.upperLip, 0.11, 0.07, 0, w.mouth);
  add(14, 0.10, 0.04, 0, w.mouth);
  add(healthy.mouthCorner, 0.13, -0.04, -0.04, w.mouth);    // healthy corner pulled up and out
  flat(affected.fold, 0.2, 0.6 * w.mouth);

  // 5. Drooling: the corner sags a little, the saliva is drawn on top
  add(affected.mouthCorner, 0.15, 0.08, 0, w.drool);

  return { blobs, smooth, blur: 0.03 * s };
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
// Tears and drool (drawn on top of the changed photo)
// ---------------------------------------------------------------------------
function drawDrop(c, x, y, r, alpha) {
  const g = c.createRadialGradient(x - r * 0.35, y - r * 0.35, r * 0.1, x, y, r * 1.1);
  g.addColorStop(0, `rgba(255,255,255,${0.95 * alpha})`);
  g.addColorStop(0.35, `rgba(215,235,255,${0.6 * alpha})`);
  g.addColorStop(1, `rgba(150,195,240,${0.35 * alpha})`);
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
  g.addColorStop(0, `rgba(170,205,240,${0.35 * alpha})`);
  g.addColorStop(0.4, `rgba(255,255,255,${0.85 * alpha})`);
  g.addColorStop(1, `rgba(170,205,240,${0.4 * alpha})`);
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
      const y = phase * phase * 0.8 * s;
      const a = w.eye * Math.min(1, phase * 6) * (1 - phase);
      drawDrop(c, 0, y, 0.035 * s, a);
    }
    drawDrop(c, 0, 0.015 * s, 0.02 * s, w.eye * 0.9); // wet rim
    c.restore();
  }

  if (w.close > 0) { // eyelashes of the closed healthy eye
    const pts = [healthy === SIDE.left ? 362 : 133, healthy.lowerLidInner, healthy.lowerLid,
                 healthy.lowerLidOuter, healthy.eyeOuter].map((i) => warpedPoint(i, blobs));
    const lift = 0.02 * s;
    c.save();
    c.strokeStyle = `rgba(40,25,20,${0.75 * w.close})`;
    c.lineWidth = 0.018 * s;
    c.lineCap = "round";
    c.beginPath();
    c.moveTo(pts[0].x, pts[0].y - lift);
    for (let i = 1; i < pts.length; i++) c.lineTo(pts[i].x, pts[i].y - lift * (i === 4 ? 1 : 1.3));
    c.stroke();
    c.restore();
  }

  if (w.drool > 0) { // saliva from the affected mouth corner
    const p = warpedPoint(affected.mouthCorner, blobs);
    c.save();
    c.translate(p.x, p.y);
    c.rotate(angle);
    const len = 0.3 * s * Math.min(1, t / 2) * (0.9 + 0.1 * Math.sin(t * 2));
    drawStream(c, 0, 0, len, 0.03 * s, w.drool * 0.6);
    const phase = (t % 3.2) / 3.2;
    const grow = Math.min(1, phase * 2.5);
    const fall = Math.max(0, phase - 0.4) / 0.6;
    drawDrop(c, 0, len + 0.025 * s + fall * fall * 0.9 * s, (0.016 + 0.02 * grow) * s,
             w.drool * (1 - fall * 0.8) * Math.min(1, t / 2));
    c.restore();
  }
}

// Where to draw the yellow "look here" circle for each symptom
function highlightFor(key) {
  const { down, s } = faceFrame();
  const at = (i, r, dn = 0) => ({ x: P(i).x + down.x * dn * s, y: P(i).y + down.y * dn * s, r: r * s });
  if (key === "brow") return at(affected.brow[1], 0.32, -0.05);
  if (key === "eye") return at(affected.lowerLid, 0.24);
  if (key === "cheek") return at(affected.fold, 0.34, -0.05);
  if (key === "mouth") return at(affected.mouthCorner, 0.3);
  if (key === "drool") return at(affected.mouthCorner, 0.36, 0.15);
  return null;
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
    steps.push({ type: "photo", label: `Photo ${i + 1} of ${SYMPTOMS.length + 1}` });
    steps.push({ type: "symptom", dur: CONFIG.symptomSeconds, index: i });
  });
  steps.push({ type: "photo", label: `Photo ${SYMPTOMS.length + 1} of ${SYMPTOMS.length + 1}` });
  steps.push({ type: "final", dur: CONFIG.finalSeconds });
  steps.push({ type: "thanks", dur: CONFIG.thanksSeconds });
  return steps;
}

let steps = [];
let stepIndex = 0;
let stepStart = 0;
let goodSince = null;   // when the face got into a good position (photo step)
let lastProblem = null;

function nextStep(now) {
  stepIndex++;
  stepStart = now;
  goodSince = null;
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
  if (!face) { live = null; return; }
  const vw = video.videoWidth, vh = video.videoHeight;
  const pts = face.map((p) => ({ x: p.x * vw, y: p.y * vh }));

  // how far the head is turned away from the camera (degrees)
  let turn = 0;
  const m = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
  if (m) {
    const d = m.data;
    const n = Math.hypot(d[8], d[9], d[10]) || 1;
    turn = Math.acos(Math.min(1, Math.abs(d[10]) / n)) * 180 / Math.PI;
  }
  // are the eyes closed?
  let blink = 0;
  const bs = res.faceBlendshapes && res.faceBlendshapes[0];
  if (bs) for (const c of bs.categories) {
    if (c.categoryName === "eyeBlinkLeft" || c.categoryName === "eyeBlinkRight") blink = Math.max(blink, c.score);
  }
  // how fast the head moves (face heights per second)
  let speed = 0;
  if (live && now - lastFaceTime < 500) {
    const fh = Math.hypot(pts[152].x - pts[10].x, pts[152].y - pts[10].y) || 1;
    const dist = Math.hypot(pts[1].x - live.lm[1].x, pts[1].y - live.lm[1].y) / fh;
    const inst = dist / Math.max(0.016, (now - lastFaceTime) / 1000);
    speed = live.speed * 0.7 + inst * 0.3;
  }
  live = { lm: pts, turn, blink, speed };
  lastFaceTime = now;
}

const faceVisible = (now) => live && now - lastFaceTime < 400;

// ---------------------------------------------------------------------------
// Taking a photo
// ---------------------------------------------------------------------------
function takePhoto() {
  const vw = video.videoWidth, vh = video.videoHeight;
  const c = document.createElement("canvas");
  c.width = vw; c.height = vh;
  c.getContext("2d").drawImage(video, 0, 0, vw, vh);
  photo = { canvas: c, landmarks: live.lm.map((p) => ({ x: p.x, y: p.y })) };
}

// Part of the photo to show, with the given width/height shape, face filling it
function cropFor(aspect) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const p of photo.landmarks) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const fw = maxX - minX, fh = maxY - minY;
  let ch = fh / 0.8, cw = ch * aspect;
  if (cw < fw / 0.8) { cw = fw / 0.8; ch = cw / aspect; }
  const vw = photo.canvas.width, vh = photo.canvas.height;
  const fit = Math.min(1, vw / cw, vh / ch);
  cw *= fit; ch *= fit;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2 - fh * 0.04;
  return {
    x: Math.max(0, Math.min(vw - cw, cx - cw / 2)),
    y: Math.max(0, Math.min(vh - ch, cy - ch / 2)),
    w: cw, h: ch,
  };
}

// Draws the face from `src` into the box, mirrored like a selfie.
// Optionally draws a yellow circle around `ring` (photo coordinates).
function drawFace(src, x, y, w, h, ring, t) {
  const crop = cropFor(w / h);
  ctx.save();
  ctx.beginPath();
  roundRectPath(x, y, w, h, Math.min(w, h) * 0.04);
  ctx.clip();
  ctx.translate(x + w, y);
  ctx.scale(-1, 1);
  ctx.drawImage(src, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h);
  if (ring) {
    const sc = w / crop.w;
    const rx = (ring.x - crop.x) * sc, ry = (ring.y - crop.y) * sc, rr = ring.r * sc;
    const pulse = 1 + 0.06 * Math.sin(t * 4);
    ctx.strokeStyle = "rgba(255,212,121,0.95)";
    ctx.lineWidth = Math.max(3, w * 0.008);
    ctx.setLineDash([w * 0.025, w * 0.015]);
    ctx.beginPath();
    ctx.arc(rx, ry, rr * pulse, 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
}

// Renders the photo with the given symptom strengths into frameCanvas
function renderPhoto(w, t) {
  lm = photo.landmarks;
  const fx = buildEffects(w);
  renderWarp(photo.canvas, fx);
  drawOverlays(frameCtx, w, fx.blobs, t);
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

function text(str, x, y, size, color = "#fff", weight = 700, alpha = 1, maxW) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${FONT}`;
  if (maxW && ctx.measureText(str).width > maxW) {
    size *= maxW / ctx.measureText(str).width;
    ctx.font = `${weight} ${size}px ${FONT}`;
  }
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(str, x, y);
  ctx.globalAlpha = 1;
}

function roundRectPath(x, y, w, h, r) {
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function pill(str, cx, cy, size, color) {
  ctx.font = `700 ${size}px ${FONT}`;
  const maxW = canvas.width * 0.86;
  if (ctx.measureText(str).width > maxW) {
    size *= maxW / ctx.measureText(str).width;
    ctx.font = `700 ${size}px ${FONT}`;
  }
  const tw = ctx.measureText(str).width;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.beginPath();
  roundRectPath(cx - tw / 2 - size * 0.6, cy - size * 0.8, tw + size * 1.2, size * 1.6, size * 0.8);
  ctx.fill();
  text(str, cx, cy, size, color, 700);
}

// Two face pictures: on top of each other (portrait) or side by side (landscape)
function drawPair(w, h, top, bottom, labelA, labelB, drawA, drawB) {
  const u = Math.min(w, h);
  const gap = u * 0.03;
  const areaH = bottom - top;
  let boxes;
  if (h >= w) {
    const ph = (areaH - gap) / 2, pw = Math.min(w * 0.94, ph * 1.15);
    const x = (w - pw) / 2;
    boxes = [[x, top, pw, ph], [x, top + ph + gap, pw, ph]];
  } else {
    let pw = (w * 0.94 - gap) / 2, ph = Math.min(areaH, pw * 1.25);
    pw = Math.min(pw, ph / 0.8);
    const y = top + (areaH - ph) / 2;
    boxes = [[w / 2 - gap / 2 - pw, y, pw, ph], [w / 2 + gap / 2, y, pw, ph]];
  }
  const labelSize = u * 0.04;
  [[drawA, labelA, "#fff"], [drawB, labelB, "#ffd479"]].forEach(([draw, label, color], i) => {
    const [x, y, pw, ph] = boxes[i];
    draw(x, y, pw, ph);
    ctx.strokeStyle = i ? "rgba(255,212,121,0.9)" : "rgba(255,255,255,0.7)";
    ctx.lineWidth = Math.max(2, u * 0.005);
    ctx.beginPath();
    roundRectPath(x, y, pw, ph, Math.min(pw, ph) * 0.04);
    ctx.stroke();
    pill(label, x + pw / 2, y + ph - labelSize * 1.3, labelSize, color);
  });
}

function header(w, h, small, big) {
  const u = Math.min(w, h);
  text(small, w / 2, h * 0.035, u * 0.035, "#9fc3ff", 600);
  text(big, w / 2, h * 0.035 + u * 0.075, u * 0.07, "#ffd479", 800, 1, w * 0.92);
  return h * 0.035 + u * 0.13;
}

function progressBar(w, h, progress) {
  const u = Math.min(w, h);
  ctx.fillStyle = "rgba(255,255,255,0.2)";
  ctx.fillRect(w * 0.1, h - u * 0.03, w * 0.8, u * 0.008);
  ctx.fillStyle = "#ffd479";
  ctx.fillRect(w * 0.1, h - u * 0.03, w * 0.8 * Math.min(1, progress), u * 0.008);
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
    text("Facial Nerve Palsy", w / 2, h * 0.46, u * 0.1, "#fff", 800, a, w * 0.9);
    text("Bell's palsy", w / 2, h * 0.46 + u * 0.1, u * 0.05, "#ffd479", 500, a);
  } else if (step.type === "photo") {
    drawPhotoStep(now, local, step, w, h, u);
  } else if (step.type === "symptom") {
    const sym = SYMPTOMS[step.index];
    const weights = { ...NONE };
    weights[sym.key] = ease((local - 0.6) / 1.5); // the change appears smoothly
    if (sym.key === "eye") weights.close = blinkCycle(local);
    renderPhoto(weights, local);
    const ring = local > 1 ? highlightFor(sym.key) : null;
    background(w, h);
    const top = header(w, h, `Symptom ${step.index + 1} of ${SYMPTOMS.length}`, sym.title);
    drawPair(w, h, top, h - u * 0.06, "Your face", "Bell's palsy",
      (x, y, pw, ph) => drawFace(photo.canvas, x, y, pw, ph),
      (x, y, pw, ph) => drawFace(frameCanvas, x, y, pw, ph, ring, local));
    progressBar(w, h, local / step.dur);
    flash(local, w, h);
  } else if (step.type === "final") {
    renderPhoto(ALL, local + 2);
    background(w, h);
    const top = header(w, h, "Bell's palsy", "All symptoms together");
    drawPair(w, h, top, h - u * 0.03, "Normal face", "Bell's palsy face",
      (x, y, pw, ph) => drawFace(photo.canvas, x, y, pw, ph),
      (x, y, pw, ph) => drawFace(frameCanvas, x, y, pw, ph));
    flash(local, w, h);
  } else if (step.type === "thanks") {
    background(w, h);
    const a = ease(local);
    text("Thank you", w / 2, h * 0.45, u * 0.13, "#fff", 800, a);
    text("for the experience", w / 2, h * 0.45 + u * 0.12, u * 0.065, "#ffd479", 500, a);
  }
}

// "Try to close both eyes": the healthy eye closes every 3 seconds for 1.5 s
function blinkCycle(t) {
  if (t < 1.5) return 0;
  const p = (t - 1.5) % 3;
  return ease(p / 0.3) * ease((1.8 - p) / 0.3);
}

// White camera flash right after a photo
function flash(local, w, h) {
  if (local > 0.5) return;
  ctx.fillStyle = `rgba(255,255,255,${1 - local / 0.5})`;
  ctx.fillRect(0, 0, w, h);
}

// Checks if the face is well placed for a photo. Returns null if good, or a message.
function checkPosition(now, oval, map) {
  if (!faceVisible(now)) return "Stand in front of the tablet";
  const L = live.lm;
  const top = map(L[10]), chin = map(L[152]);
  const faceH = Math.hypot(chin.x - top.x, chin.y - top.y);
  const cx = (top.x + chin.x) / 2, cy = (top.y + chin.y) / 2;
  if (faceH < oval.ry * 1.2) return "Come a little closer";
  if (faceH > oval.ry * 1.95) return "Move back a little";
  // forehead-to-chin centre sits a bit below the oval centre (the oval also holds the hair)
  if (Math.abs(cx - oval.x) > oval.rx * 0.28 || Math.abs(cy - (oval.y + oval.ry * 0.1)) > oval.ry * 0.3)
    return "Move your face into the oval";
  const e1 = L[33], e2 = L[263];
  const roll = Math.abs(Math.atan2(e2.y - e1.y, e2.x - e1.x)) * 180 / Math.PI;
  if (Math.min(roll, 180 - roll) > 8) return "Keep your head straight";
  if (live.turn > 14) return "Look straight at the camera";
  if (live.blink > 0.5) return "Keep your eyes open";
  if (live.speed > 0.25) return "Keep still";
  return null;
}

// Live camera with face-position check; the photo is taken when the face
// has been in a good position for photoHoldSeconds.
function drawPhotoStep(now, local, step, w, h, u) {
  let map = (p) => p;
  if (video.readyState >= 2) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const sc = Math.max(w / vw, h / vh);
    const ox = (w - vw * sc) / 2, oy = (h - vh * sc) / 2;
    ctx.save();
    ctx.translate(w, 0);
    ctx.scale(-1, 1); // mirror, like a selfie
    ctx.drawImage(video, ox, oy, vw * sc, vh * sc);
    ctx.restore();
    map = (p) => ({ x: w - (ox + p.x * sc), y: oy + p.y * sc });
  }

  const oval = { x: w / 2, y: h * 0.46 };
  oval.ry = Math.min(h * 0.27, w * 0.45);
  oval.rx = oval.ry * 0.76;

  if (DEBUG && faceVisible(now)) { // show tracked points
    ctx.fillStyle = "#0f0";
    for (const p of live.lm) { const q = map(p); ctx.fillRect(q.x - 1, q.y - 1, 3, 3); }
  }
  const problem = checkPosition(now, oval, map);
  lastProblem = problem;
  const ready = local >= CONFIG.photoGetReadySeconds;
  if (problem || !ready) goodSince = null;
  else if (goodSince === null) goodSince = now;
  const held = goodSince === null ? 0 : (now - goodSince) / 1000;

  // darken outside the oval
  ctx.save();
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.beginPath();
  ctx.rect(0, 0, w, h);
  ctx.moveTo(oval.x + oval.rx, oval.y);
  ctx.ellipse(oval.x, oval.y, oval.rx, oval.ry, 0, 0, Math.PI * 2);
  ctx.fill("evenodd");
  ctx.restore();

  const good = !problem;
  ctx.strokeStyle = good ? "#3ddc84" : "#ff6b6b";
  ctx.lineWidth = u * 0.008;
  ctx.setLineDash(good ? [] : [u * 0.025, u * 0.018]);
  ctx.beginPath();
  ctx.ellipse(oval.x, oval.y, oval.rx, oval.ry, 0, 0, Math.PI * 2);
  ctx.stroke();
  ctx.setLineDash([]);

  text(step.label, w / 2, h * 0.04, u * 0.035, "#9fc3ff", 600);
  const msg = !ready ? (problem || "Get ready… look at the camera and smile")
            : problem || "Perfect! Smile and hold still";
  pill(msg, w / 2, oval.y + oval.ry + u * 0.08, u * 0.05, good ? "#3ddc84" : "#ffffff");

  if (goodSince !== null) {
    const left = CONFIG.photoHoldSeconds - held;
    if (left > 0) {
      text(String(Math.ceil(left)), w / 2, oval.y + oval.ry + u * 0.22, u * 0.14, "#ffd479", 800, 0.95);
    } else {
      takePhoto();
      nextStep(now);
    }
  } else if (!ready) {
    text("Get ready for the photo", w / 2, h * 0.04 + u * 0.06, u * 0.05, "#fff", 700);
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
  try { await screen.orientation?.lock?.("portrait"); } catch (e) {}
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

  live = null; photo = null; goodSince = null;
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
window.__palsy = {
  CONFIG,
  get step() { return running ? steps[stepIndex].type : "menu"; },
  get index() { return stepIndex; },
  // test helper: changed face for the given symptom strengths, as an image
  render(weights) {
    if (!live) return null;
    takePhoto();
    renderPhoto({ ...NONE, ...weights }, 2.2);
    const c = document.createElement("canvas");
    c.width = 500; c.height = 600;
    const crop = cropFor(500 / 600);
    c.getContext("2d").drawImage(frameCanvas, crop.x, crop.y, crop.w, crop.h, 0, 0, 500, 600);
    return c.toDataURL("image/png");
  },
  get problem() { return lastProblem; },
  get live() { return live && { turn: live.turn, blink: live.blink, speed: live.speed }; },
};
