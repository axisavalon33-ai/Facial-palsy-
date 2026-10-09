// Facial Nerve Palsy – Bell's palsy simulation for a tablet (front camera)
//
// How it works
// 1. MediaPipe Face Landmarker finds 478 points on the face (468 face points
//    + 10 iris points) in every camera frame.
// 2. The points (plus a ring of fixed points around the face and the image
//    corners) are joined into triangles with a Delaunay triangulation.
// 3. Every frame each point gets a small displacement on the affected side of
//    the face. WebGL then draws the camera image with the moved triangles
//    (a triangle-mesh warp), so the picture bends smoothly with no edges.
// 4. Background, hair and neck use points that never move, so they stay as is.
//
// Experience: 3-2-1 → title → live palsy mirror with prompts → one photo →
// each symptom on the photo for 5 s → all symptoms together → thank you.

// ---------------------------------------------------------------------------
// EASY SETTINGS (seconds)
// ---------------------------------------------------------------------------
const CONFIG = {
  countdownSeconds: 3,     // "3 2 1"
  titleSeconds: 4,         // "Facial Nerve Palsy"
  promptSeconds: 6,        // each live prompt: smile / raise eyebrows / close eyes
  photoGetReadySeconds: 3, // time to get ready before the photo
  photoHoldSeconds: 3,     // face must stay in position this long (3-2-1)
  symptomSeconds: 5,       // each symptom on the photo
  finalSeconds: 5,         // normal face vs. Bell's palsy face
  thanksSeconds: 10,       // thank-you screen, then back to the start
  defaultGrade: 6,         // House-Brackmann grade on start (1-6)
  defaultSide: "left",     // affected side on start
  effectScale: 1.0,        // multiply all movements (staff fine-tuning)
};
const QUICK = new URLSearchParams(location.search).has("quick");
if (QUICK) { CONFIG.promptSeconds = 2; CONFIG.photoGetReadySeconds = 1; CONFIG.thanksSeconds = 4; }

const PROMPTS = ["Try to smile", "Raise your eyebrows", "Close your eyes tight"];

const SYMPTOMS = [
  { key: "brow",  title: "Drooping eyebrow" },
  { key: "eye",   title: "Eye cannot close fully" },
  { key: "cheek", title: "Flattened cheek and smile line" },
  { key: "mouth", title: "Drooping mouth corner" },
];
const ALL = { brow: 1, eye: 1, cheek: 1, mouth: 1 };

// House-Brackmann grades: name and how strong the simulation is (0..1)
const GRADES = [
  null,
  { name: "I – Normal", k: 0 },
  { name: "II – Mild dysfunction", k: 0.25 },
  { name: "III – Moderate dysfunction", k: 0.45 },
  { name: "IV – Moderately severe dysfunction", k: 0.65 },
  { name: "V – Severe dysfunction", k: 0.85 },
  { name: "VI – Total paralysis", k: 1 },
];

// ---------------------------------------------------------------------------
// Landmark numbers (MediaPipe face mesh). "left" = the person's own left.
// ---------------------------------------------------------------------------
const MIDLINE = [10, 168, 6, 1, 152]; // forehead → nose bridge → nose tip → chin
const FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109];
const ANCHORS = [168, 6, 197, 195, 133, 362]; // points that barely move with expressions

const SIDE = {
  left: {
    browInner: 336, browMid: 334, browOuter: 300, forehead: 299,
    upperLid: 386, lowerLid: 374, lowerLidOuter: 373, lowerLidInner: 380,
    eyeInner: 362, eyeOuter: 263,
    upper: [466, 388, 387, 386, 385, 384, 398], lower: [382, 381, 380, 374, 373, 390, 249],
    iris: [473, 474, 475, 476, 477],
    fold: 425, cheek: 280, noseWing: 358,
    mouthCorner: 291, upperLip: [269, 270], lowerLip: 321,
  },
  right: {
    browInner: 107, browMid: 105, browOuter: 70, forehead: 69,
    upperLid: 159, lowerLid: 145, lowerLidOuter: 144, lowerLidInner: 153,
    eyeInner: 133, eyeOuter: 33,
    upper: [246, 161, 160, 159, 158, 157, 173], lower: [7, 163, 144, 145, 153, 154, 155],
    iris: [468, 469, 470, 471, 472],
    fold: 205, cheek: 50, noseWing: 129,
    mouthCorner: 61, upperLip: [39, 40], lowerLip: 91,
  },
};

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------
const $ = (id) => document.getElementById(id);
const canvas = $("screen");
const ctx = canvas.getContext("2d");
const video = $("video");

let faceLandmarker = null;
let stream = null;
let running = false;
let wakeLock = null;
let sideName = CONFIG.defaultSide;
let grade = CONFIG.defaultGrade;
let showOriginal = false;

let live = null;          // smoothed tracking: { lm, turn, blend, speed }
let lastFaceTime = -1e9;
let lastVideoTime = -1;
let neutral = null;       // the person's relaxed face: { lm }
let eyeRef = null;        // a recent frame with the affected eye open
let photo = null;         // { canvas, lm, neutralLm }
let triangles = null;     // Delaunay triangle indices (Uint16Array)

const aff = () => SIDE[sideName];
const hea = () => SIDE[sideName === "left" ? "right" : "left"];
const sev = () => GRADES[grade].k * CONFIG.effectScale;

// ---------------------------------------------------------------------------
// Load MediaPipe from the CDN (with a local copy as a backup)
// ---------------------------------------------------------------------------
const CDN = "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.14";
const MODEL_CDN = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task";

async function createLandmarker(base, modelPath) {
  const { FaceLandmarker, FilesetResolver } = await import(`${base}/vision_bundle.mjs`);
  const files = await FilesetResolver.forVisionTasks(`${base}/wasm`);
  for (const delegate of ["GPU", "CPU"]) {
    try {
      return await FaceLandmarker.createFromOptions(files, {
        baseOptions: { modelAssetPath: modelPath, delegate },
        runningMode: "VIDEO",
        numFaces: 1,
        outputFaceBlendshapes: true,
        outputFacialTransformationMatrixes: true,
      });
    } catch (e) { console.warn(`${delegate} failed`, e); }
  }
  throw new Error("Face Landmarker could not start");
}

async function loadTracker() {
  const local = new URL("./lib", location.href).href;
  const tries = [[CDN, MODEL_CDN], [local, "./models/face_landmarker.task"]];
  for (const [base, model] of tries) {
    try {
      faceLandmarker = await createLandmarker(base, model);
      break;
    } catch (e) { console.warn("Loading from", base, "failed", e); }
  }
  if (faceLandmarker) {
    $("status").textContent = "Ready. Tap Start.";
    $("start").disabled = false;
  } else {
    $("status").textContent = "Could not load the face tracker. Check the internet connection and reload.";
  }
}
loadTracker();

// ---------------------------------------------------------------------------
// Small maths helpers
// ---------------------------------------------------------------------------
const clamp01 = (x) => Math.max(0, Math.min(1, x));
const ease = (x) => { x = clamp01(x); return x * x * (3 - 2 * x); };
const smoothstep = (a, b, x) => ease((x - a) / (b - a));
const dist = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);

// Best rotation+scale+shift that moves points `src` onto `dst` (2D Procrustes)
function similarity(src, dst) {
  const n = src.length;
  let sx = 0, sy = 0, dx = 0, dy = 0;
  for (let i = 0; i < n; i++) { sx += src[i].x; sy += src[i].y; dx += dst[i].x; dy += dst[i].y; }
  sx /= n; sy /= n; dx /= n; dy /= n;
  let a = 0, b = 0, d = 0;
  for (let i = 0; i < n; i++) {
    const px = src[i].x - sx, py = src[i].y - sy, qx = dst[i].x - dx, qy = dst[i].y - dy;
    a += px * qx + py * qy;
    b += px * qy - py * qx;
    d += px * px + py * py;
  }
  a /= d || 1; b /= d || 1;
  return (p) => ({ x: a * (p.x - sx) - b * (p.y - sy) + dx, y: b * (p.x - sx) + a * (p.y - sy) + dy });
}

// ---------------------------------------------------------------------------
// Delaunay triangulation (Bowyer–Watson). Runs once, on the first face seen.
// ---------------------------------------------------------------------------
function delaunay(pts) {
  const n = pts.length;
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of pts) {
    minX = Math.min(minX, p.x); minY = Math.min(minY, p.y);
    maxX = Math.max(maxX, p.x); maxY = Math.max(maxY, p.y);
  }
  const d = Math.max(maxX - minX, maxY - minY) * 20, mx = (minX + maxX) / 2, my = (minY + maxY) / 2;
  const P = pts.concat([{ x: mx - d, y: my - d }, { x: mx + d, y: my - d }, { x: mx, y: my + d }]);
  const tri = (a, b, c) => {
    const A = P[a], B = P[b], C = P[c];
    const D = 2 * (A.x * (B.y - C.y) + B.x * (C.y - A.y) + C.x * (A.y - B.y));
    const a2 = A.x * A.x + A.y * A.y, b2 = B.x * B.x + B.y * B.y, c2 = C.x * C.x + C.y * C.y;
    const ux = (a2 * (B.y - C.y) + b2 * (C.y - A.y) + c2 * (A.y - B.y)) / D;
    const uy = (a2 * (C.x - B.x) + b2 * (A.x - C.x) + c2 * (B.x - A.x)) / D;
    return { a, b, c, x: ux, y: uy, r2: (A.x - ux) ** 2 + (A.y - uy) ** 2 };
  };
  let tris = [tri(n, n + 1, n + 2)];
  for (let i = 0; i < n; i++) {
    const p = P[i];
    const keep = [], edges = new Map();
    for (const t of tris) {
      if ((p.x - t.x) ** 2 + (p.y - t.y) ** 2 < t.r2) {
        for (const [u, v] of [[t.a, t.b], [t.b, t.c], [t.c, t.a]]) {
          const key = u < v ? u * 4096 + v : v * 4096 + u;
          edges.set(key, edges.has(key) ? null : [u, v]);
        }
      } else keep.push(t);
    }
    for (const e of edges.values()) if (e) keep.push(tri(e[0], e[1], i));
    tris = keep;
  }
  const out = [];
  for (const t of tris) if (t.a < n && t.b < n && t.c < n) out.push(t.a, t.b, t.c);
  return new Uint16Array(out);
}

// All mesh points for a face: 478 landmarks + a ring outside the face + border.
// The ring and border points never move, which keeps hair/neck/background still.
const RING = FACE_OVAL.length;
function meshPoints(lm, w, h) {
  const c = lm[1];
  const pts = lm.slice(0, 478);
  for (const i of FACE_OVAL) {
    const p = lm[i];
    pts.push({ x: Math.max(0, Math.min(w, c.x + (p.x - c.x) * 1.3)),
               y: Math.max(0, Math.min(h, c.y + (p.y - c.y) * 1.3)) });
  }
  pts.push({ x: 0, y: 0 }, { x: w / 2, y: 0 }, { x: w, y: 0 }, { x: w, y: h / 2 },
           { x: w, y: h }, { x: w / 2, y: h }, { x: 0, y: h }, { x: 0, y: h / 2 });
  return pts;
}
const MESH_N = 478 + RING + 8;

// ---------------------------------------------------------------------------
// WebGL: draws the image through the moved triangle mesh
// ---------------------------------------------------------------------------
const glCanvas = document.createElement("canvas");
const gl = glCanvas.getContext("webgl", { preserveDrawingBuffer: true, premultipliedAlpha: false });

const VERT = `
attribute vec2 a_dst;   // where the point is drawn (pixels)
attribute vec2 a_src;   // where the colour is taken from (pixels)
attribute vec4 a_fx;    // smoothing, lower-lid shadow, colour change, opacity
uniform vec2 u_res;
varying vec2 v_uv;
varying vec4 v_fx;
void main() {
  v_uv = a_src / u_res;
  v_fx = a_fx;
  gl_Position = vec4(a_dst.x / u_res.x * 2.0 - 1.0, 1.0 - a_dst.y / u_res.y * 2.0, 0.0, 1.0);
}`;
const FRAG = `
precision highp float;
uniform sampler2D u_tex;
uniform vec2 u_res;
uniform float u_blur;
varying vec2 v_uv;
varying vec4 v_fx;
void main() {
  vec4 c = texture2D(u_tex, v_uv);
  // smoothing: soften wrinkles / the nasolabial fold with a small round blur
  if (v_fx.x > 0.01) {
    vec4 acc = c;
    for (int k = 0; k < 12; k++) {
      float a = float(k) * 0.5236;
      vec2 o = vec2(cos(a), sin(a)) * u_blur / u_res;
      acc += texture2D(u_tex, v_uv + o * 0.5) + texture2D(u_tex, v_uv + o);
    }
    c = mix(c, acc / 25.0, v_fx.x);
  }
  // very light shadow under the lower lid
  c.rgb *= 1.0 - v_fx.y;
  // slight colour change on the affected side (a little paler / flatter)
  float lum = dot(c.rgb, vec3(0.299, 0.587, 0.114));
  c.rgb = mix(c.rgb, vec3(lum) * 1.03, v_fx.z);
  gl_FragColor = vec4(c.rgb, v_fx.w);
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
const aDst = gl.getAttribLocation(prog, "a_dst");
const aSrc = gl.getAttribLocation(prog, "a_src");
const aFx = gl.getAttribLocation(prog, "a_fx");
const uRes = gl.getUniformLocation(prog, "u_res");
const uBlur = gl.getUniformLocation(prog, "u_blur");
const vbo = gl.createBuffer();
const ibo = gl.createBuffer();
const ibo2 = gl.createBuffer();
const VSIZE = 8; // floats per point: dst(2) src(2) fx(4)
const vdata = new Float32Array(MESH_N * VSIZE);
let patchCount = 0, patchSide = null;

function makeTexture() {
  const t = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, t);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  return t;
}
const texMain = makeTexture();
const texEye = makeTexture();
let texEyeDirty = false;

function upload(tex, source) {
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, source);
}

function bindVertices() {
  gl.bindBuffer(gl.ARRAY_BUFFER, vbo);
  gl.bufferData(gl.ARRAY_BUFFER, vdata, gl.DYNAMIC_DRAW);
  const F = 4;
  gl.enableVertexAttribArray(aDst); gl.vertexAttribPointer(aDst, 2, gl.FLOAT, false, VSIZE * F, 0);
  gl.enableVertexAttribArray(aSrc); gl.vertexAttribPointer(aSrc, 2, gl.FLOAT, false, VSIZE * F, 2 * F);
  gl.enableVertexAttribArray(aFx);  gl.vertexAttribPointer(aFx, 4, gl.FLOAT, false, VSIZE * F, 4 * F);
}

// ---------------------------------------------------------------------------
// THE PALSY WARP – works out how far every mesh point moves
// ---------------------------------------------------------------------------
// Distances below are in pixels for a face whose outer eye corners are
// REF_EYE px apart; they are scaled to the real face size and by severity.
const REF_EYE = 150;

// Handles: a point is moved by `down`/`out` pixels (out = toward the ear on
// the affected side). Nearby points follow with a Gaussian falloff (sigma).
function handles(S) {
  return [
    // BROW: the eyebrow sags (outer part most) – peripheral palsy
    // (the movement fades out before the upper eyelid, so the eye stays open)
    { group: "brow", i: S.browMid,   down: 11, out: 0, sigma: 35, stop: S.upperLid },
    { group: "brow", i: S.browOuter, down: 10, out: 0, sigma: 30, stop: S.upperLid },
    { group: "brow", i: S.browInner, down: 5,  out: 0, sigma: 25, stop: S.upperLid },
    // EYE: eye looks wider (upper lid slightly up) and the lower lid sags.
    // Small sigma so the iris is not touched (iris points are also pinned).
    { group: "eye", i: S.upperLid,      down: -2.5, out: 0, sigma: 12 },
    { group: "eye", i: S.lowerLid,      down: 6,    out: 0, sigma: 14 },
    { group: "eye", i: S.lowerLidOuter, down: 4,    out: 0, sigma: 12 },
    // CHEEK: mild sag; the nasolabial fold moves out a little (flatter)
    { group: "cheek", i: S.fold,     down: 6, out: 2, sigma: 35 },
    { group: "cheek", i: S.cheek,    down: 5, out: 0, sigma: 40 },
    { group: "cheek", i: S.noseWing, down: 2, out: 0, sigma: 20 },
    // MOUTH: corner pulled down 8-15 px and slightly outward;
    // the upper lip on that side is flattened (pushed down a little)
    { group: "mouth", i: S.mouthCorner, down: 15, out: 3, sigma: 30 },
    { group: "mouth", i: S.upperLip[0], down: 3,  out: 0, sigma: 14 },
    { group: "mouth", i: S.upperLip[1], down: 3,  out: 0, sigma: 14 },
    { group: "mouth", i: S.lowerLip,    down: 3,  out: 0, sigma: 15 },
  ];
}

const FACE_OVAL_SET = new Set(FACE_OVAL);

// Signed distance from p to the facial midline (positive = affected side)
function midlineSide(lm, p, out) {
  let best = Infinity, bx = 0, by = 0;
  for (let k = 0; k < MIDLINE.length - 1; k++) {
    const a = lm[MIDLINE[k]], b = lm[MIDLINE[k + 1]];
    const vx = b.x - a.x, vy = b.y - a.y;
    const t = clamp01(((p.x - a.x) * vx + (p.y - a.y) * vy) / (vx * vx + vy * vy || 1));
    const cx = a.x + vx * t, cy = a.y + vy * t;
    const d = (p.x - cx) ** 2 + (p.y - cy) ** 2;
    if (d < best) { best = d; bx = cx; by = cy; }
  }
  const s = Math.sqrt(best);
  return (p.x - bx) * out.x + (p.y - by) * out.y >= 0 ? s : -s;
}

// Fills vdata for the face `lm`.
//   w        : which symptom groups are on (0..1 each)
//   neutralLm: the relaxed face placed onto this frame (or null) – used to
//              make the affected side move less when smiling / raising brows
function buildMesh(lm, imgW, imgH, w, neutralLm) {
  const S = aff(), k = sev();
  const pts = meshPoints(lm, imgW, imgH);

  // face directions and size
  const down0 = { x: lm[152].x - lm[10].x, y: lm[152].y - lm[10].y };
  const dl = Math.hypot(down0.x, down0.y) || 1;
  const down = { x: down0.x / dl, y: down0.y / dl };
  const eyeD = dist(lm[S.eyeOuter], lm[hea().eyeOuter]);
  const out = { x: (lm[S.eyeOuter].x - lm[hea().eyeOuter].x) / eyeD, y: (lm[S.eyeOuter].y - lm[hea().eyeOuter].y) / eyeD };
  const px = eyeD / REF_EYE; // pixels-per-reference-pixel for this face

  const H = handles(S).filter((h) => w[h.group] > 0).map((h) => ({
    x: lm[h.i].x, y: lm[h.i].y, s2: 2 * (h.sigma * px) ** 2, group: h.group,
    dx: (down.x * h.down + out.x * h.out) * px * k * w[h.group],
    dy: (down.y * h.down + out.y * h.out) * px * k * w[h.group],
    // how far below the handle (along "down") the movement has faded to zero
    stopAt: h.stop === undefined ? Infinity
      : (lm[h.stop].x - lm[h.i].x) * down.x + (lm[h.stop].y - lm[h.i].y) * down.y,
  }));
  // sum of all Gaussian handles at point p
  const handleSum = (p, skipEye) => {
    let x = 0, y = 0;
    for (const h of H) {
      if (skipEye && h.group === "eye") continue;
      let f = Math.exp(-((p.x - h.x) ** 2 + (p.y - h.y) ** 2) / h.s2);
      if (h.stopAt !== Infinity) {
        const t = (p.x - h.x) * down.x + (p.y - h.y) * down.y;
        f *= 1 - smoothstep(0, h.stopAt, t);
      }
      x += f * h.dx; y += f * h.dy;
    }
    return { x, y };
  };
  // the iris moves as one piece (never stretched) and ignores the eyelid handles
  const irisMove = handleSum(lm[S.iris[0]], true);
  const gauss = (p, q, sigma) => Math.exp(-((p.x - q.x) ** 2 + (p.y - q.y) ** 2) / (2 * (sigma * px) ** 2));
  const pinned = new Set(S.iris);
  const lids = new Set([...S.upper, ...S.lower, S.eyeInner, S.eyeOuter]);
  const eyeLineY = (lm[S.eyeInner].y + lm[hea().eyeInner].y) / 2;
  const noseY = lm[1].y;
  const anyOn = Math.max(w.brow, w.eye, w.cheek, w.mouth);
  const shadowAt = { x: lm[S.lowerLid].x + down.x * 0.07 * eyeD, y: lm[S.lowerLid].y + down.y * 0.07 * eyeD };
  // turn the "move less" effect off when the head is turned (2D maths gets unreliable)
  const frontal = live ? 1 - smoothstep(10, 25, live.turn) : 1;

  for (let v = 0; v < MESH_N; v++) {
    const p = pts[v];
    let dx = 0, dy = 0, smooth = 0, shade = 0, tone = 0;
    if (pinned.has(v) && k > 0) {
      const side = smoothstep(0, 0.12 * eyeD, midlineSide(lm, p, out));
      dx = irisMove.x * side; dy = irisMove.y * side;
    } else if (v < 478 && !FACE_OVAL_SET.has(v) && k > 0) {
      // only real face points inside the face outline may move
      // the unaffected side stays untouched; fade in over a short distance
      const side = smoothstep(0, 0.12 * eyeD, midlineSide(lm, p, out));
      if (side > 0) {
        // 1) the static palsy shape: sum of Gaussian handles
        const m = handleSum(p, false);
        dx += m.x; dy += m.y;
        // 2) dynamic: the affected side follows expressions only partly.
        //    Brows don't move at all (keep 0%), mouth/cheek keep 20-40%.
        if (neutralLm && !lids.has(v)) {
          const isBrow = p.y < eyeLineY;
          const isMouth = p.y > noseY;
          const gw = isBrow ? w.brow : isMouth ? Math.max(w.mouth, w.cheek) : w.cheek;
          const retain = isBrow ? 1 - k : 1 - 0.7 * k;
          const n = neutralLm[v];
          dx += (n.x - p.x) * (1 - retain) * gw * frontal;
          dy += (n.y - p.y) * (1 - retain) * gw * frontal;
        }
        dx *= side; dy *= side;
        // 3) skin effects: smooth forehead wrinkles and the nasolabial fold
        smooth = Math.max(
          0.85 * w.brow * gauss(p, lm[S.forehead], 45),
          0.75 * w.cheek * gauss(p, lm[S.fold], 28)) * k * side;
        // 4) very subtle lower-lid shadow and colour change
        shade = 0.12 * w.eye * k * side * gauss(p, shadowAt, 16);
        tone = 0.05 * anyOn * k * side;
      }
    }
    const o = v * VSIZE;
    vdata[o] = p.x + dx; vdata[o + 1] = p.y + dy;   // drawn here
    vdata[o + 2] = p.x;  vdata[o + 3] = p.y;        // colour from here
    vdata[o + 4] = Math.min(1, smooth); vdata[o + 5] = shade; vdata[o + 6] = tone; vdata[o + 7] = 1;
  }
  return { eyeD, down };
}

// Renders `source` (video or photo canvas) with the palsy warp into glCanvas.
function renderFace(source, lm, w, neutralLm, eyeClose = 0) {
  const W = source.videoWidth || source.width, Hh = source.videoHeight || source.height;
  if (glCanvas.width !== W || glCanvas.height !== Hh) { glCanvas.width = W; glCanvas.height = Hh; }
  gl.viewport(0, 0, W, Hh);
  gl.clearColor(0, 0, 0, 1);
  gl.clear(gl.COLOR_BUFFER_BIT);
  upload(texMain, source);
  gl.uniform2f(uRes, W, Hh);
  if (!triangles || !lm) {
    // no face: show the picture as it is
    const quad = [[0, 0], [W, 0], [0, Hh], [W, Hh]];
    quad.forEach(([x, y], i) => vdata.set([x, y, x, y, 0, 0, 0, 1], i * VSIZE));
    bindVertices();
    gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
    return;
  }
  const { eyeD, down } = buildMesh(lm, W, Hh, w, neutralLm);
  bindVertices();
  gl.uniform1f(uBlur, 0.025 * eyeD);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
  gl.drawElements(gl.TRIANGLES, triangles.length, gl.UNSIGNED_SHORT, 0);

  // Live only: when the person closes their eyes, the affected eye stays
  // slightly open. We draw the eye area from a recent "eye open" frame on top.
  if (eyeClose > 0.01 && eyeRef && w.eye > 0) drawOpenEye(lm, eyeD, down, eyeClose * w.eye);
}

// Eye-region triangles (rebuilt when the side or reference frame changes)
function buildPatch() {
  const S = aff(), lm = eyeRef.lm;
  const c = { x: (lm[S.eyeInner].x + lm[S.eyeOuter].x) / 2, y: (lm[S.eyeInner].y + lm[S.eyeOuter].y) / 2 };
  const eyeD = dist(lm[S.eyeOuter], lm[hea().eyeOuter]);
  const inside = (i) => i < 478 && dist(lm[i], c) < 0.42 * eyeD;
  const idx = [];
  for (let t = 0; t < triangles.length; t += 3) {
    if (inside(triangles[t]) && inside(triangles[t + 1]) && inside(triangles[t + 2])) idx.push(triangles[t], triangles[t + 1], triangles[t + 2]);
  }
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo2);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
  patchCount = idx.length;
  patchSide = sideName;
}

function drawOpenEye(lm, eyeD, down, amount) {
  const S = aff();
  if (patchSide !== sideName || !patchCount) buildPatch();
  if (texEyeDirty) { upload(texEye, eyeRef.canvas); texEyeDirty = false; }
  const ref = eyeRef.lm;
  // place the old eye onto the current face
  const anchorIdx = [S.eyeInner, S.eyeOuter, 168, 6, S.browMid];
  const map = similarity(anchorIdx.map((i) => ref[i]), anchorIdx.map((i) => lm[i]));
  const c = { x: (lm[S.eyeInner].x + lm[S.eyeOuter].x) / 2, y: (lm[S.eyeInner].y + lm[S.eyeOuter].y) / 2 };
  const opening = dist(ref[S.upperLid], ref[S.lowerLid]);
  const upper = new Set(S.upper);
  const vis = smoothstep(0.35, 0.75, amount) * Math.min(1, sev() * 1.6);
  for (let v = 0; v < 478; v++) {
    const q = map(ref[v]);
    // "slightly open": the upper lid comes down 45% of the way
    if (upper.has(v)) { q.x += down.x * opening * 0.45 * vis; q.y += down.y * opening * 0.45 * vis; }
    const o = v * VSIZE;
    vdata[o] = q.x; vdata[o + 1] = q.y;
    vdata[o + 2] = ref[v].x; vdata[o + 3] = ref[v].y;
    // soft edge so the patch blends in
    const a = vis * (1 - smoothstep(0.18 * eyeD, 0.38 * eyeD, dist(q, c)));
    vdata[o + 4] = 0; vdata[o + 5] = 0.04 * vis; vdata[o + 6] = 0; vdata[o + 7] = a;
  }
  bindVertices();
  gl.bindTexture(gl.TEXTURE_2D, texEye);
  gl.enable(gl.BLEND);
  gl.blendFunc(gl.SRC_ALPHA, gl.ONE_MINUS_SRC_ALPHA);
  gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo2);
  gl.drawElements(gl.TRIANGLES, patchCount, gl.UNSIGNED_SHORT, 0);
  gl.disable(gl.BLEND);
  gl.bindTexture(gl.TEXTURE_2D, texMain);
}

// ---------------------------------------------------------------------------
// Face tracking with jitter smoothing (exponential moving average)
// ---------------------------------------------------------------------------
const EMA = 0.5; // 1 = no smoothing, lower = smoother

function blendScore(cats, ...names) {
  let m = 0;
  for (const c of cats) if (names.includes(c.categoryName)) m = Math.max(m, c.score);
  return m;
}

function eyeOpenness(lm, S) {
  return dist(lm[S.upperLid], lm[S.lowerLid]) / (dist(lm[S.eyeInner], lm[S.eyeOuter]) || 1);
}

function track(now) {
  if (!faceLandmarker || video.readyState < 2 || video.currentTime === lastVideoTime) return;
  lastVideoTime = video.currentTime;
  const res = faceLandmarker.detectForVideo(video, now);
  const face = res.faceLandmarks && res.faceLandmarks[0];
  if (!face) { if (now - lastFaceTime > 400) live = null; return; }
  const vw = video.videoWidth, vh = video.videoHeight;
  const raw = face.map((p) => ({ x: p.x * vw, y: p.y * vh }));

  // low-pass filter on the points
  let lm;
  if (live && now - lastFaceTime < 400) {
    lm = live.lm;
    for (let i = 0; i < raw.length; i++) {
      lm[i].x += (raw[i].x - lm[i].x) * EMA;
      lm[i].y += (raw[i].y - lm[i].y) * EMA;
    }
  } else lm = raw;

  // head turn (degrees away from facing the camera)
  let turn = 0;
  const m = res.facialTransformationMatrixes && res.facialTransformationMatrixes[0];
  if (m) {
    const d = m.data, n = Math.hypot(d[8], d[9], d[10]) || 1;
    turn = Math.acos(Math.min(1, Math.abs(d[10]) / n)) * 180 / Math.PI;
  }
  const cats = (res.faceBlendshapes && res.faceBlendshapes[0] && res.faceBlendshapes[0].categories) || [];
  const blend = {
    smile: blendScore(cats, "mouthSmileLeft", "mouthSmileRight"),
    brow: blendScore(cats, "browInnerUp", "browOuterUpLeft", "browOuterUpRight"),
    blink: blendScore(cats, "eyeBlinkLeft", "eyeBlinkRight"),
    jaw: blendScore(cats, "jawOpen"),
  };
  let speed = 0;
  if (live && now - lastFaceTime < 400) {
    const fh = dist(lm[10], lm[152]) || 1;
    const inst = dist(raw[1], live.prevNose) / fh / Math.max(0.016, (now - lastFaceTime) / 1000);
    speed = live.speed * 0.7 + inst * 0.3;
  }
  live = { lm, turn, blend, speed, prevNose: { ...raw[1] }, eyeClose: live ? live.eyeClose : 0 };
  lastFaceTime = now;

  if (!triangles) {
    triangles = delaunay(meshPoints(lm, vw, vh));
    gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ibo);
    gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, triangles, gl.STATIC_DRAW);
  }
  updateNeutral();
  updateEyeRef(now);
}

// Learn the relaxed face while the person is not making an expression
function updateNeutral() {
  const b = live.blend;
  if (b.smile > 0.25 || b.brow > 0.3 || b.blink > 0.35 || b.jaw > 0.2 || live.turn > 12) return;
  const lm = live.lm;
  if (!neutral) { neutral = { lm: lm.map((p) => ({ ...p })) }; return; }
  // bring the current face into the neutral face's position, then average
  const map = similarity(ANCHORS.map((i) => lm[i]), ANCHORS.map((i) => neutral.lm[i]));
  for (let i = 0; i < lm.length; i++) {
    const q = map(lm[i]);
    neutral.lm[i].x += (q.x - neutral.lm[i].x) * 0.05;
    neutral.lm[i].y += (q.y - neutral.lm[i].y) * 0.05;
  }
}

// The neutral face placed onto face `lm` (same head position / size)
function neutralOn(lm) {
  if (!neutral) return null;
  const map = similarity(ANCHORS.map((i) => neutral.lm[i]), ANCHORS.map((i) => lm[i]));
  return neutral.lm.map(map);
}

// Keep a recent frame where the affected eye is open
let eyeRefTime = 0;
function updateEyeRef(now) {
  const S = aff();
  const open = eyeOpenness(live.lm, S);
  live.eyeClose = eyeRef ? clamp01(1 - open / (eyeRef.open || 1)) : 0;
  if (live.blend.blink > 0.25 || open < 0.18 || live.turn > 15 || now - eyeRefTime < 300) return;
  if (eyeRef && open < eyeRef.open * 0.85 && now - eyeRefTime < 3000) return;
  if (!eyeRef) eyeRef = { canvas: document.createElement("canvas") };
  const c = eyeRef.canvas;
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  eyeRef.lm = live.lm.map((p) => ({ ...p }));
  eyeRef.open = open;
  eyeRefTime = now;
  texEyeDirty = true;
  patchSide = null; // rebuild the eye triangles for the new frame
}

const faceVisible = (now) => live && now - lastFaceTime < 400;

// ---------------------------------------------------------------------------
// Steps of the experience
// ---------------------------------------------------------------------------
let steps = [], stepIndex = 0, stepStart = 0, goodSince = null, lastProblem = null;

function buildSteps() {
  const s = [
    { type: "countdown", dur: CONFIG.countdownSeconds },
    { type: "title", dur: CONFIG.titleSeconds },
    { type: "live", dur: CONFIG.promptSeconds * PROMPTS.length },
    { type: "photo" },
  ];
  SYMPTOMS.forEach((sym, i) => s.push({ type: "symptom", dur: CONFIG.symptomSeconds, index: i }));
  s.push({ type: "final", dur: CONFIG.finalSeconds });
  s.push({ type: "thanks", dur: CONFIG.thanksSeconds });
  return s;
}

function goTo(index, now) {
  stepIndex = index;
  stepStart = now;
  goodSince = null;
  if (stepIndex >= steps.length) { stopExperience(); return; }
  $("controls").classList.toggle("hidden", steps[stepIndex].type !== "live");
}
const nextStep = (now) => goTo(stepIndex + 1, now);

// ---------------------------------------------------------------------------
// Drawing helpers (2D canvas on screen)
// ---------------------------------------------------------------------------
const FONT = `system-ui, -apple-system, "Segoe UI", Roboto, sans-serif`;

function background(w, h) {
  const g = ctx.createRadialGradient(w / 2, h * 0.4, 0, w / 2, h / 2, Math.max(w, h) * 0.7);
  g.addColorStop(0, "#1b2a4a");
  g.addColorStop(1, "#04060b");
  ctx.fillStyle = g;
  ctx.fillRect(0, 0, w, h);
}

function text(str, x, y, size, color = "#fff", weight = 700, alpha = 1, maxW = canvas.width * 0.92) {
  ctx.globalAlpha = alpha;
  ctx.fillStyle = color;
  ctx.font = `${weight} ${size}px ${FONT}`;
  const tw = ctx.measureText(str).width;
  if (tw > maxW) ctx.font = `${weight} ${size * maxW / tw}px ${FONT}`;
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

function pill(str, cx, cy, size, color = "#fff") {
  ctx.font = `700 ${size}px ${FONT}`;
  const maxW = canvas.width * 0.86;
  if (ctx.measureText(str).width > maxW) { size *= maxW / ctx.measureText(str).width; ctx.font = `700 ${size}px ${FONT}`; }
  const tw = ctx.measureText(str).width;
  ctx.fillStyle = "rgba(0,0,0,0.6)";
  ctx.beginPath();
  roundRectPath(cx - tw / 2 - size * 0.7, cy - size * 0.85, tw + size * 1.4, size * 1.7, size * 0.85);
  ctx.fill();
  text(str, cx, cy, size, color, 700);
}

// Video/canvas drawn to fill the box, mirrored like a selfie.
// Returns a function that maps image pixels to screen pixels.
function drawCover(src, sw, sh, x, y, w, h) {
  const sc = Math.max(w / sw, h / sh);
  const ox = (w - sw * sc) / 2, oy = (h - sh * sc) / 2;
  ctx.save();
  ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
  ctx.translate(x + w, y); ctx.scale(-1, 1);
  ctx.drawImage(src, ox, oy, sw * sc, sh * sc);
  ctx.restore();
  return (p) => ({ x: x + w - (ox + p.x * sc), y: y + oy + p.y * sc });
}

// Part of the photo around the face with the given shape
function faceCrop(lm, aspect, W, H) {
  let minX = 1e9, minY = 1e9, maxX = -1e9, maxY = -1e9;
  for (let i = 0; i < 468; i++) {
    const p = lm[i];
    minX = Math.min(minX, p.x); maxX = Math.max(maxX, p.x);
    minY = Math.min(minY, p.y); maxY = Math.max(maxY, p.y);
  }
  const fw = maxX - minX, fh = maxY - minY;
  let ch = fh / 0.78, cw = ch * aspect;
  if (cw < fw / 0.8) { cw = fw / 0.8; ch = cw / aspect; }
  const fit = Math.min(1, W / cw, H / ch);
  cw *= fit; ch *= fit;
  const cx = (minX + maxX) / 2, cy = (minY + maxY) / 2 - fh * 0.04;
  return { x: Math.max(0, Math.min(W - cw, cx - cw / 2)), y: Math.max(0, Math.min(H - ch, cy - ch / 2)), w: cw, h: ch };
}

// Draws the face (cropped from glCanvas or the photo) into a box, mirrored
function drawFaceBox(src, x, y, w, h, ring, t) {
  const crop = faceCrop(photo.lm, w / h, photo.canvas.width, photo.canvas.height);
  ctx.save();
  ctx.beginPath(); roundRectPath(x, y, w, h, Math.min(w, h) * 0.04); ctx.clip();
  ctx.translate(x + w, y); ctx.scale(-1, 1);
  ctx.drawImage(src, crop.x, crop.y, crop.w, crop.h, 0, 0, w, h);
  if (ring && ring.a > 0.01) { // yellow circle: "look here"
    const sc = w / crop.w;
    ctx.strokeStyle = `rgba(255,212,121,${0.95 * ring.a})`;
    ctx.lineWidth = Math.max(3, w * 0.008);
    ctx.setLineDash([w * 0.025, w * 0.015]);
    ctx.beginPath();
    ctx.arc((ring.x - crop.x) * sc, (ring.y - crop.y) * sc, ring.r * sc * (1 + 0.05 * Math.sin(t * 4)), 0, Math.PI * 2);
    ctx.stroke();
    ctx.setLineDash([]);
  }
  ctx.restore();
  ctx.strokeStyle = "rgba(255,255,255,0.6)";
  ctx.lineWidth = Math.max(2, Math.min(w, h) * 0.005);
  ctx.beginPath(); roundRectPath(x, y, w, h, Math.min(w, h) * 0.04); ctx.stroke();
}

function ringFor(key, lm) {
  const S = aff(), eyeD = dist(lm[S.eyeOuter], lm[hea().eyeOuter]);
  const at = (i, r) => ({ x: lm[i].x, y: lm[i].y, r: r * eyeD });
  return { brow: at(S.browMid, 0.3), eye: at(S.lowerLid, 0.22), cheek: at(S.fold, 0.32), mouth: at(S.mouthCorner, 0.28) }[key];
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
  const captionH = 34 * Math.min(window.devicePixelRatio || 1, 2);

  if (step.type === "countdown") {
    background(w, h);
    const n = Math.max(1, Math.ceil(step.dur - local)), f = local % 1;
    text(String(n), w / 2, h / 2, u * 0.35 * (1 + (1 - f) * 0.3), "#fff", 800, 1 - f * 0.6);
  } else if (step.type === "title") {
    background(w, h);
    const a = ease(local / 0.8) * ease((step.dur - local) / 0.6);
    text("Facial Nerve Palsy", w / 2, h * 0.46, u * 0.1, "#fff", 800, a);
    text("Bell's palsy", w / 2, h * 0.46 + u * 0.1, u * 0.05, "#ffd479", 500, a);
  } else if (step.type === "live") {
    drawLive(now, local, w, h, u);
  } else if (step.type === "photo") {
    drawPhotoStep(now, local, w, h, u);
  } else if (step.type === "symptom") {
    // one photo; the symptom fades in, stays, and fades out again
    const sym = SYMPTOMS[step.index];
    const amt = ease(local / 0.7) * ease((step.dur - local) / 0.7);
    const wts = { brow: 0, eye: 0, cheek: 0, mouth: 0 };
    wts[sym.key] = amt;
    renderFace(photo.canvas, photo.lm, wts, photo.neutralLm);
    background(w, h);
    text(`Symptom ${step.index + 1} of ${SYMPTOMS.length}`, w / 2, h * 0.04, u * 0.035, "#9fc3ff", 600);
    text(sym.title, w / 2, h * 0.04 + u * 0.07, u * 0.065, "#ffd479", 800);
    const top = h * 0.04 + u * 0.13, bottom = h - captionH - u * 0.03;
    const bh = bottom - top, bw = Math.min(w * 0.94, bh * 0.82);
    drawFaceBox(glCanvas, (w - bw) / 2, top, bw, bh, { ...ringFor(sym.key, photo.lm), a: amt }, local);
    if (stepIndex === steps.findIndex((s) => s.type === "symptom")) flash(local, w, h);
  } else if (step.type === "final") {
    renderFace(photo.canvas, photo.lm, ALL, photo.neutralLm);
    background(w, h);
    text("All symptoms together", w / 2, h * 0.05, u * 0.06, "#ffd479", 800);
    const top = h * 0.05 + u * 0.07, bottom = h - captionH - u * 0.02, gap = u * 0.025;
    const ph = (bottom - top - gap) / 2, pw = Math.min(w * 0.94, ph * 1.15), x = (w - pw) / 2;
    drawFaceBox(photo.canvas, x, top, pw, ph);
    pill("Normal face", w / 2, top + ph - u * 0.05, u * 0.04);
    drawFaceBox(glCanvas, x, top + ph + gap, pw, ph);
    pill("Bell's palsy face", w / 2, top + 2 * ph + gap - u * 0.05, u * 0.04, "#ffd479");
  } else if (step.type === "thanks") {
    background(w, h);
    const a = ease(local);
    text("Thank you", w / 2, h * 0.45, u * 0.13, "#fff", 800, a);
    text("for the experience", w / 2, h * 0.45 + u * 0.12, u * 0.065, "#ffd479", 500, a);
  }
}

// Live palsy mirror with prompts and controls
function drawLive(now, local, w, h, u) {
  if (video.readyState < 2) return;
  const has = faceVisible(now);
  const lm = has ? live.lm : null;
  if (showOriginal || !lm) renderFace(video, null, ALL, null);
  else renderFace(video, lm, ALL, neutralOn(lm), live.eyeClose || 0);
  drawCover(glCanvas, glCanvas.width, glCanvas.height, 0, 0, w, h);
  const i = Math.min(PROMPTS.length - 1, Math.floor(local / CONFIG.promptSeconds));
  pill(has ? PROMPTS[i] : "Look at the camera", w / 2, h * 0.07, u * 0.055, "#ffd479");
  if (showOriginal) pill("Original", w / 2, h * 0.07 + u * 0.1, u * 0.035);
}

function flash(local, w, h) {
  if (local > 0.5) return;
  ctx.fillStyle = `rgba(255,255,255,${1 - local / 0.5})`;
  ctx.fillRect(0, 0, w, h);
}

// Checks the face position for the photo. Returns null if good, else a message.
function checkPosition(now, oval, map) {
  if (!faceVisible(now)) return "Stand in front of the tablet";
  const L = live.lm;
  const top = map(L[10]), chin = map(L[152]);
  const faceH = dist(top, chin);
  const cx = (top.x + chin.x) / 2, cy = (top.y + chin.y) / 2;
  if (faceH < oval.ry * 1.2) return "Come a little closer";
  if (faceH > oval.ry * 1.95) return "Move back a little";
  if (Math.abs(cx - oval.x) > oval.rx * 0.28 || Math.abs(cy - (oval.y + oval.ry * 0.1)) > oval.ry * 0.3)
    return "Move your face into the oval";
  const roll = Math.abs(Math.atan2(L[263].y - L[33].y, L[263].x - L[33].x)) * 180 / Math.PI;
  if (Math.min(roll, 180 - roll) > 8) return "Keep your head straight";
  if (live.turn > 14) return "Look straight at the camera";
  if (live.blend.blink > 0.5) return "Keep your eyes open";
  if (live.speed > 0.25) return "Keep still";
  return null;
}

// Live camera (original) with an oval; the photo is taken once the face has
// been in a good position for photoHoldSeconds.
function drawPhotoStep(now, local, w, h, u) {
  if (video.readyState < 2) return;
  const map = drawCover(video, video.videoWidth, video.videoHeight, 0, 0, w, h);
  const oval = { x: w / 2, y: h * 0.45 };
  oval.ry = Math.min(h * 0.27, w * 0.45);
  oval.rx = oval.ry * 0.76;

  const problem = checkPosition(now, oval, map);
  lastProblem = problem;
  const ready = local >= CONFIG.photoGetReadySeconds;
  if (problem || !ready) goodSince = null;
  else if (goodSince === null) goodSince = now;

  ctx.save(); // darken around the oval
  ctx.fillStyle = "rgba(0,0,0,0.45)";
  ctx.beginPath(); ctx.rect(0, 0, w, h);
  ctx.moveTo(oval.x + oval.rx, oval.y);
  ctx.ellipse(oval.x, oval.y, oval.rx, oval.ry, 0, 0, Math.PI * 2);
  ctx.fill("evenodd");
  ctx.restore();
  ctx.strokeStyle = problem ? "#ff6b6b" : "#3ddc84";
  ctx.lineWidth = u * 0.008;
  ctx.setLineDash(problem ? [u * 0.025, u * 0.018] : []);
  ctx.beginPath(); ctx.ellipse(oval.x, oval.y, oval.rx, oval.ry, 0, 0, Math.PI * 2); ctx.stroke();
  ctx.setLineDash([]);

  text("Photo", w / 2, h * 0.04, u * 0.04, "#9fc3ff", 600);
  pill(problem || (ready ? "Perfect! Smile and hold still" : "Get ready… look at the camera and smile"),
       w / 2, oval.y + oval.ry + u * 0.08, u * 0.05, problem ? "#fff" : "#3ddc84");

  if (goodSince !== null) {
    const left = CONFIG.photoHoldSeconds - (now - goodSince) / 1000;
    if (left > 0) text(String(Math.ceil(left)), w / 2, oval.y + oval.ry + u * 0.22, u * 0.14, "#ffd479", 800);
    else { takePhoto(); nextStep(now); }
  }
}

function takePhoto() {
  const c = document.createElement("canvas");
  c.width = video.videoWidth; c.height = video.videoHeight;
  c.getContext("2d").drawImage(video, 0, 0);
  const lm = live.lm.map((p) => ({ ...p }));
  photo = { canvas: c, lm, neutralLm: neutralOn(lm) };
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function updateControls() {
  $("sideLeft").classList.toggle("on", sideName === "left");
  $("sideRight").classList.toggle("on", sideName === "right");
  $("severity").value = grade;
  $("sevLabel").textContent = "House-Brackmann grade " + GRADES[grade].name;
  $("original").classList.toggle("on", showOriginal);
  $("original").textContent = showOriginal ? "Show palsy" : "Show original";
}
$("sideLeft").onclick = () => { sideName = "left"; eyeRef = null; updateControls(); };
$("sideRight").onclick = () => { sideName = "right"; eyeRef = null; updateControls(); };
$("severity").oninput = (e) => { grade = +e.target.value; updateControls(); };
$("original").onclick = () => { showOriginal = !showOriginal; updateControls(); };
$("photoBtn").onclick = () => { if (running) goTo(steps.findIndex((s) => s.type === "photo"), performance.now()); };
updateControls();

// ---------------------------------------------------------------------------
// Start / stop
// ---------------------------------------------------------------------------
async function startExperience() {
  try { await document.documentElement.requestFullscreen?.(); } catch (e) {}
  try { await screen.orientation?.lock?.("portrait"); } catch (e) {}
  try { wakeLock = await navigator.wakeLock?.request("screen"); } catch (e) {}
  $("status").textContent = "Starting camera…";
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: false,
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 } },
    });
  } catch (e) {
    console.error(e);
    $("status").textContent = "Camera not available. Please allow camera access and try again.";
    return;
  }
  video.srcObject = stream;
  try { await video.play(); } catch (e) {}
  live = null; neutral = null; eyeRef = null; photo = null; triangles = null; showOriginal = false;
  updateControls();
  steps = buildSteps();
  $("menu").classList.add("hidden");
  running = true;
  resize();
  goTo(0, performance.now());
  requestAnimationFrame(loop);
}

function stopExperience() {
  running = false;
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  try { wakeLock?.release(); } catch (e) {}
  wakeLock = null;
  $("controls").classList.add("hidden");
  ctx.fillStyle = "#000";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  $("menu").classList.remove("hidden");
  $("status").textContent = "Ready for the next visitor. Tap Start.";
}

$("start").addEventListener("click", startExperience);

// Double-tap the picture to go back to the start screen
let lastTap = 0;
canvas.addEventListener("pointerdown", () => {
  const now = performance.now();
  if (running && now - lastTap < 400) stopExperience();
  lastTap = now;
});

// Hook used for automated testing
window.__palsy = {
  get step() { return running ? steps[stepIndex].type : "menu"; },
  get index() { return stepIndex; },
  get problem() { return lastProblem; },
  get live() { return live && { turn: live.turn, blend: live.blend, tris: triangles && triangles.length / 3 }; },
  setGrade(g) { grade = g; updateControls(); },
  // test only: pretend the relaxed face has the mouth corners lower and narrower
  fakeNeutral() {
    const lm = live.lm, e = dist(lm[33], lm[263]);
    neutral = { lm: lm.map((p) => ({ ...p })) };
    for (const c of [61, 291]) {
      const C = lm[c], dir = c === 61 ? 1 : -1;
      neutral.lm.forEach((q, i) => {
        const f = Math.exp(-(dist(lm[i], C) ** 2) / (2 * (0.2 * e) ** 2));
        q.x += dir * 0.06 * e * f; q.y += 0.07 * e * f;
      });
    }
  },
  // render the current camera frame as a photo with the given symptoms → PNG
  render(w) {
    takePhoto();
    renderFace(photo.canvas, photo.lm, { brow: 0, eye: 0, cheek: 0, mouth: 0, ...w }, photo.neutralLm);
    const c = document.createElement("canvas");
    c.width = 500; c.height = 600;
    const crop = faceCrop(photo.lm, 500 / 600, photo.canvas.width, photo.canvas.height);
    c.getContext("2d").drawImage(glCanvas, crop.x, crop.y, crop.w, crop.h, 0, 0, 500, 600);
    return c.toDataURL("image/png");
  },
};
