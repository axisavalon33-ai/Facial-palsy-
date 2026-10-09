// Facial Nerve Palsy – live VR experience
// Tracks the visitor's face with the camera and warps one side of it to show
// the symptoms of Bell's palsy, one at a time, then all together.

import { FaceLandmarker, FilesetResolver } from "./lib/vision_bundle.mjs";

// ---------------------------------------------------------------------------
// EASY SETTINGS – change these numbers to change the timing (in seconds)
// ---------------------------------------------------------------------------
const CONFIG = {
  getReadySeconds: 6,      // VR only: time to slide the phone into the headset
  countdownSeconds: 3,     // the "3 2 1" countdown
  titleSeconds: 4,         // the "Facial Nerve Palsy" title
  symptomSeconds: 30,      // each single symptom
  allTogetherSeconds: 30,  // all symptoms at once (live)
  snapshotSeconds: 5,      // the normal vs. palsy snapshot
  thanksSeconds: 15,       // thank-you screen, then back to the menu
};

// Add ?quick to the web address to preview everything fast (5 s per symptom).
if (new URLSearchParams(location.search).has("quick")) {
  CONFIG.getReadySeconds = 2;
  CONFIG.symptomSeconds = 5;
  CONFIG.allTogetherSeconds = 5;
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
const btnVR = document.getElementById("startVR");
const btnScreen = document.getElementById("startScreen");

let faceLandmarker = null;
let stream = null;
let running = false;
let vrMode = true;
let mirror = false;
let affected = SIDE.left, healthy = SIDE.right;
let strength = 1;
let timeline = [];
let startTime = 0;
let wakeLock = null;

let landmarks = null;        // smoothed landmark pixels [{x,y}]
let lastFaceTime = -1e9;
let lastVideoTime = -1;
let snapshot = null;         // { normal, palsy } canvases

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
    statusEl.textContent = "Ready. Choose how you want to start.";
    btnVR.disabled = btnScreen.disabled = false;
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Could not load the face tracker. Check the internet connection and reload the page.";
  }
}
loadTracker();

// ---------------------------------------------------------------------------
// WebGL face warp: moves small soft areas of the picture ("blobs")
// ---------------------------------------------------------------------------
const MAX_BLOBS = 48;
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

// Draws the current video frame, warped by `blobs`, into frameCanvas.
function renderWarp(blobs) {
  const w = video.videoWidth, h = video.videoHeight;
  if (glCanvas.width !== w || glCanvas.height !== h) {
    glCanvas.width = frameCanvas.width = w;
    glCanvas.height = frameCanvas.height = h;
  }
  gl.viewport(0, 0, w, h);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, video);
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
// Face geometry helpers
// ---------------------------------------------------------------------------
const P = (i) => landmarks[i];

function faceFrame() {
  const top = P(10), chin = P(152);
  let dx = chin.x - top.x, dy = chin.y - top.y;
  const len = Math.hypot(dx, dy) || 1;
  const down = { x: dx / len, y: dy / len };
  const a = P(affected.eyeOuter), b = P(healthy.eyeOuter);
  const s = Math.hypot(a.x - b.x, a.y - b.y); // face scale: width between outer eye corners
  let ox = a.x - b.x, oy = a.y - b.y;
  const ol = Math.hypot(ox, oy) || 1;
  const out = { x: ox / ol, y: oy / ol }; // points toward the affected side
  return { down, out, s };
}

// Builds the list of warp blobs. `w` holds the strength (0..1) of each symptom.
function buildBlobs(w) {
  if (!landmarks) return [];
  const { down, out, s } = faceFrame();
  const k = strength;
  const blobs = [];
  // add a blob at landmark i, radius r, moved `dn` down and `dout` toward affected side (all × face scale)
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
// Tears and drool (drawn on top of the warped video)
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

// t = seconds since the effect started; amount = 0..1
function drawOverlays(c, w, blobs, t) {
  if (!landmarks) return;
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
// Timeline of the experience
// ---------------------------------------------------------------------------
function buildTimeline() {
  const tl = [];
  if (vrMode && CONFIG.getReadySeconds > 0) tl.push({ type: "getready", dur: CONFIG.getReadySeconds });
  tl.push({ type: "countdown", dur: CONFIG.countdownSeconds });
  tl.push({ type: "title", dur: CONFIG.titleSeconds });
  SYMPTOMS.forEach((sym, i) => tl.push({ type: "symptom", dur: CONFIG.symptomSeconds, index: i }));
  tl.push({ type: "all", dur: CONFIG.allTogetherSeconds });
  tl.push({ type: "snapshot", dur: CONFIG.snapshotSeconds });
  tl.push({ type: "thanks", dur: CONFIG.thanksSeconds });
  let t = 0;
  for (const st of tl) { st.start = t; t += st.dur; }
  return tl;
}

function stageAt(t) {
  for (const st of timeline) if (t < st.start + st.dur) return st;
  return null;
}

const ease = (x) => { x = Math.max(0, Math.min(1, x)); return x * x * (3 - 2 * x); };

function weightsFor(stage, local) {
  const w = { brow: 0, eye: 0, cheek: 0, mouth: 0, drool: 0 };
  if (stage.type === "symptom") {
    w[SYMPTOMS[stage.index].key] = ease(local / 2) * ease((stage.dur - local) / 0.8);
  } else if (stage.type === "all") {
    for (const k in w) w[k] = ease(local / 2);
  }
  return w;
}

// ---------------------------------------------------------------------------
// Face tracking every frame
// ---------------------------------------------------------------------------
function track(now) {
  if (!faceLandmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = faceLandmarker.detectForVideo(video, now);
  const face = res.faceLandmarks && res.faceLandmarks[0];
  if (!face) return;
  const vw = video.videoWidth, vh = video.videoHeight;
  if (!landmarks || now - lastFaceTime > 500) {
    landmarks = face.map((p) => ({ x: p.x * vw, y: p.y * vh }));
  } else {
    const a = 0.55; // smoothing: lower = smoother but slower
    for (let i = 0; i < face.length; i++) {
      landmarks[i].x += (face[i].x * vw - landmarks[i].x) * a;
      landmarks[i].y += (face[i].y * vh - landmarks[i].y) * a;
    }
  }
  lastFaceTime = now;
}

const faceVisible = (now) => landmarks && now - lastFaceTime < 600;

// ---------------------------------------------------------------------------
// Drawing helpers for the screen (one view, or two side-by-side views for VR)
// ---------------------------------------------------------------------------
function forEachEye(fn) {
  const W = canvas.width, H = canvas.height;
  if (vrMode) {
    const gap = Math.round(W * 0.01);
    const ew = (W - gap) / 2;
    eye(0, 0, ew, H, fn);
    eye(ew + gap, 0, ew, H, fn);
  } else {
    eye(0, 0, W, H, fn);
  }
}

function eye(x, y, w, h, fn) {
  ctx.save();
  ctx.beginPath();
  ctx.rect(x, y, w, h);
  ctx.clip();
  ctx.translate(x, y);
  fn(w, h);
  ctx.restore();
}

function drawCover(img, iw, ih, w, h, flip) {
  const sc = Math.max(w / iw, h / ih);
  const dw = iw * sc, dh = ih * sc;
  ctx.save();
  if (flip) { ctx.translate(w, 0); ctx.scale(-1, 1); }
  ctx.drawImage(img, (w - dw) / 2, (h - dh) / 2, dw, dh);
  ctx.restore();
}

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
  ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
  ctx.textAlign = "center";
  ctx.textBaseline = "middle";
  ctx.fillText(str, x, y);
  ctx.globalAlpha = 1;
}

function wrapLines(str, maxW, size, weight) {
  ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;
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

function caption(w, h, label, title, body, progress) {
  const u = Math.min(w, h);
  const boxW = Math.min(w * 0.82, u * 1.4);
  const titleSize = u * 0.05, bodySize = u * 0.03, labelSize = u * 0.026;
  const lines = wrapLines(body, boxW * 0.9, bodySize, 400);
  const boxH = labelSize * 1.6 + titleSize * 1.4 + lines.length * bodySize * 1.35 + u * 0.05;
  const bx = (w - boxW) / 2, by = h * 0.97 - boxH;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  roundRect(bx, by, boxW, boxH, u * 0.025);
  ctx.fill();
  let y = by + u * 0.02 + labelSize * 0.8;
  text(label, w / 2, y, labelSize, "#9fc3ff", 600);
  y += labelSize * 0.8 + titleSize * 0.75;
  text(title, w / 2, y, titleSize, "#ffd479", 800);
  y += titleSize * 0.65 + bodySize * 0.8;
  for (const ln of lines) { text(ln, w / 2, y, bodySize, "#fff", 400); y += bodySize * 1.35; }
  // progress bar
  ctx.fillStyle = "rgba(255,255,255,0.2)";
  ctx.fillRect(bx + u * 0.02, by + boxH - u * 0.015, boxW - u * 0.04, u * 0.006);
  ctx.fillStyle = "#ffd479";
  ctx.fillRect(bx + u * 0.02, by + boxH - u * 0.015, (boxW - u * 0.04) * progress, u * 0.006);
}

// ---------------------------------------------------------------------------
// Snapshot: same moment, normal face vs. Bell's palsy face
// ---------------------------------------------------------------------------
function takeSnapshot() {
  if (!landmarks || video.readyState < 2) { snapshot = null; return; }
  // crop around the face
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (const p of landmarks) {
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2;
  let ch = (maxY - minY) * 1.45, cw = ch * 0.78;
  cw = Math.max(cw, (maxX - minX) * 1.3); ch = cw / 0.78;
  const vw = video.videoWidth, vh = video.videoHeight;
  cw = Math.min(cw, vw); ch = Math.min(ch, vh);
  const crop = {
    x: Math.max(0, Math.min(vw - cw, cx - cw / 2)),
    y: Math.max(0, Math.min(vh - ch, cy - ch / 2)),
    w: cw, h: ch,
  };

  const make = (blobs, w) => {
    renderWarp(blobs);
    if (w) drawOverlays(frameCtx, w, blobs, 2.2);
    const c = document.createElement("canvas");
    c.width = Math.round(crop.w); c.height = Math.round(crop.h);
    const cc = c.getContext("2d");
    cc.fillStyle = "#000"; cc.fillRect(0, 0, c.width, c.height);
    if (mirror) { cc.translate(c.width, 0); cc.scale(-1, 1); }
    cc.drawImage(frameCanvas, crop.x, crop.y, crop.w, crop.h, 0, 0, c.width, c.height);
    return c;
  };
  const all = { brow: 1, eye: 1, cheek: 1, mouth: 1, drool: 1 };
  const normal = make([], null);
  const palsy = make(buildBlobs(all), all);
  snapshot = { normal, palsy };
}

function drawSnapshot(w, h, local) {
  background(w, h);
  const u = Math.min(w, h);
  text("Snapshot", w / 2, h * 0.1, u * 0.05, "#9fc3ff", 600);
  if (!snapshot) {
    text("No face was found for the snapshot", w / 2, h / 2, u * 0.05);
    return;
  }
  const fade = ease(local / 0.5);
  const gap = w * 0.04;
  const imgH = h * 0.66;
  let imgW = imgH * (snapshot.normal.width / snapshot.normal.height);
  if (imgW * 2 + gap > w * 0.94) { imgW = (w * 0.94 - gap) / 2; }
  const ih = imgW * (snapshot.normal.height / snapshot.normal.width);
  const y = h * 0.17;
  const x1 = w / 2 - gap / 2 - imgW, x2 = w / 2 + gap / 2;
  ctx.globalAlpha = fade;
  ctx.drawImage(snapshot.normal, x1, y, imgW, ih);
  ctx.drawImage(snapshot.palsy, x2, y, imgW, ih);
  ctx.globalAlpha = 1;
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = Math.max(2, u * 0.004);
  ctx.strokeRect(x1, y, imgW, ih);
  ctx.strokeRect(x2, y, imgW, ih);
  const ly = y + ih + u * 0.06;
  text("Normal face", x1 + imgW / 2, ly, u * 0.045, "#fff", 700, fade);
  text("Bell's palsy face", x2 + imgW / 2, ly, u * 0.045, "#ffd479", 700, fade);
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

let lastStage = null;

function loop(now) {
  if (!running) return;
  requestAnimationFrame(loop);
  track(now);

  const t = (now - startTime) / 1000;
  const stage = stageAt(t);
  if (!stage) { stopExperience(); return; }
  const local = t - stage.start;
  if (stage !== lastStage) {
    if (stage.type === "snapshot") takeSnapshot();
    lastStage = stage;
  }

  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);

  if (stage.type === "symptom" || stage.type === "all") {
    const hasFace = faceVisible(now);
    const w = weightsFor(stage, local);
    const blobs = hasFace ? buildBlobs(w) : [];
    if (video.readyState >= 2) {
      renderWarp(blobs);
      if (hasFace) drawOverlays(frameCtx, w, blobs, local);
    }
    const n = SYMPTOMS.length;
    forEachEye((ew, eh) => {
      if (video.readyState >= 2) drawCover(frameCanvas, frameCanvas.width, frameCanvas.height, ew, eh, mirror);
      const u = Math.min(ew, eh);
      if (stage.type === "symptom") {
        const s = SYMPTOMS[stage.index];
        caption(ew, eh, `Symptom ${stage.index + 1} of ${n}`, s.title, s.text, local / stage.dur);
      } else {
        caption(ew, eh, "Bell's palsy", "All symptoms together",
          "Bell's palsy is a sudden weakness of one side of the face caused by inflammation of the facial nerve (cranial nerve VII).",
          local / stage.dur);
      }
      if (!hasFace) {
        ctx.fillStyle = "rgba(0,0,0,0.55)";
        roundRect(ew * 0.2, eh * 0.08, ew * 0.6, u * 0.1, u * 0.02);
        ctx.fill();
        text(vrMode ? "Looking for a face… look at a mirror or a person" : "Looking for a face… look at the camera",
             ew / 2, eh * 0.08 + u * 0.05, u * 0.035, "#ffd479", 600);
      }
    });
    return;
  }

  forEachEye((ew, eh) => {
    const u = Math.min(ew, eh);
    if (stage.type === "getready") {
      background(ew, eh);
      text("Put the phone in the headset", ew / 2, eh * 0.42, u * 0.06);
      text(`Starting in ${Math.ceil(stage.dur - local)}…`, ew / 2, eh * 0.56, u * 0.045, "#9fc3ff", 500);
    } else if (stage.type === "countdown") {
      background(ew, eh);
      const n = Math.max(1, Math.ceil(stage.dur - local));
      const f = local % 1;
      const scale = 1 + (1 - f) * 0.35;
      text(String(n), ew / 2, eh / 2, u * 0.32 * scale, "#ffffff", 800, 1 - f * 0.6);
    } else if (stage.type === "title") {
      background(ew, eh);
      const a = ease(local / 0.8) * ease((stage.dur - local) / 0.6);
      text("Facial Nerve Palsy", ew / 2, eh * 0.45, u * 0.1, "#ffffff", 800, a);
      text("Bell's palsy", ew / 2, eh * 0.58, u * 0.05, "#ffd479", 500, a);
    } else if (stage.type === "snapshot") {
      drawSnapshot(ew, eh, local);
    } else if (stage.type === "thanks") {
      background(ew, eh);
      const a = ease(local / 1);
      text("Thank you", ew / 2, eh * 0.42, u * 0.11, "#ffffff", 800, a);
      text("for the experience", ew / 2, eh * 0.56, u * 0.06, "#ffd479", 500, a);
    }
  });
}

// ---------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------
async function startExperience(isVR) {
  vrMode = isVR;
  const camChoice = document.getElementById("camera").value;
  const facing = camChoice === "auto" ? (isVR ? "environment" : "user") : camChoice;
  mirror = facing === "user";
  affected = SIDE[document.getElementById("side").value];
  healthy = affected === SIDE.left ? SIDE.right : SIDE.left;
  strength = parseFloat(document.getElementById("strength").value);

  // Full screen + landscape for the headset (ignored where not supported)
  try { await document.documentElement.requestFullscreen?.(); } catch (e) {}
  if (isVR) { try { await screen.orientation?.lock?.("landscape"); } catch (e) {} }
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (e) {}

  statusEl.textContent = "Starting camera…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: { ideal: facing }, width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    console.error(e);
    statusEl.textContent = "Camera not available. Please allow camera access and try again.";
    return;
  }
  video.srcObject = stream;
  try { await video.play(); } catch (e) {}

  landmarks = null; snapshot = null; lastStage = null;
  timeline = buildTimeline();
  menu.classList.add("hidden");
  running = true;
  resize();
  startTime = performance.now();
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
  statusEl.textContent = "Ready for the next visitor.";
}

btnVR.addEventListener("click", () => startExperience(true));
btnScreen.addEventListener("click", () => startExperience(false));

// Double-tap (or double-click) to go back to the menu
let lastTap = 0;
canvas.addEventListener("pointerdown", () => {
  const now = performance.now();
  if (running && now - lastTap < 400) stopExperience();
  lastTap = now;
});

// Debug hook used for automated testing
window.__palsy = { CONFIG, startExperience, get stage() { return lastStage && lastStage.type; } };
