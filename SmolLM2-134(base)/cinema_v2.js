// cinema_v2.js — 3D decoder-block cinema for SmolLM2-135M docs.
//
// One transformer block, in 3D, walked through start → end. Every component
// from `smollm2_135m_decoder_block_granular.html` appears in order:
//
//   x_in  →  input_layernorm  →  q/k/v projections  →  RoPE  →  GQA broadcast
//         →  causal scaled-dot-product attention  →  o_proj  →  residual ⊕
//         →  post_attention_layernorm  →  gate/up  →  SiLU·⊙  →  down_proj
//         →  residual ⊕  →  x_out  →  (to next of 30 blocks)
//
// "Live e2e": every per-token RMS in the trace HUD was captured from a real
// forward pass of HuggingFaceTB/SmolLM2-135M on prompt "Hello, my name is".
// The 5×5 attention head-0 grid is the real causal-softmax output.
//
// Camera always looks STRAIGHT AT the block (no tilt). A glowing gold flow
// arrow tracks the data position on the left side of the column, and the
// camera follows its y as the cinema plays. Press 'f' for fullscreen.

import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { Vector3 as V3 } from 'three';

(function cinemaV2() {

const container = document.getElementById('cinema-stage');
if (!container) return;
while (container.firstChild) container.removeChild(container.firstChild);

// ============================== STAGE ==============================
const scene = new THREE.Scene();
const camera = new THREE.PerspectiveCamera(45, container.clientWidth/container.clientHeight, 0.1, 500);
camera.position.set(0, 11, 24);

const renderer = new THREE.WebGLRenderer({
  antialias: true, alpha: true, powerPreference: 'high-performance',
});
const HQ_PIXEL_RATIO = Math.max(window.devicePixelRatio || 1, 2);
renderer.setPixelRatio(HQ_PIXEL_RATIO);
renderer.setSize(container.clientWidth, container.clientHeight);
renderer.setClearColor(0x0a0f1f, 1);
renderer.outputColorSpace = THREE.SRGBColorSpace;
container.appendChild(renderer.domElement);

const MAX_ANISOTROPY = renderer.capabilities.getMaxAnisotropy();
console.log(`[cinema_v2] block-only · pixelRatio=${HQ_PIXEL_RATIO} · maxAnisotropy=${MAX_ANISOTROPY}`);

new ResizeObserver(() => {
  renderer.setSize(container.clientWidth, container.clientHeight);
  camera.aspect = container.clientWidth / container.clientHeight;
  camera.updateProjectionMatrix();
}).observe(container);

scene.background = new THREE.Color(0x0a0f1f);

scene.add(new THREE.AmbientLight(0xffffff, 0.45));
{
  const k1 = new THREE.DirectionalLight(0xfbbf24, 0.95); k1.position.set( 6, 9,  8); scene.add(k1);
  const k2 = new THREE.DirectionalLight(0x60a5fa, 0.55); k2.position.set(-6, 5, -4); scene.add(k2);
  const k3 = new THREE.DirectionalLight(0xec4899, 0.35); k3.position.set( 0, 6, -8); scene.add(k3);
}

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.08;
controls.target.set(0, 11, 0);

const floor = new THREE.Mesh(
  new THREE.CircleGeometry(20, 64),
  new THREE.MeshStandardMaterial({ color: 0x0f172a, roughness: 0.95 })
);
floor.rotation.x = -Math.PI/2;
floor.position.y = -0.5;
scene.add(floor);

// ============================== HELPERS ==============================
const LABEL_DPR = 2;
function spriteLabel(text, opts = {}) {
  const fs = opts.fontSize || 56;
  const probe = document.createElement('canvas').getContext('2d');
  probe.font = `${opts.weight || 'bold'} ${fs}px Geist, Inter, system-ui, sans-serif`;
  const wLogical = Math.max(64, probe.measureText(text).width + 60);
  const hLogical = fs + 36;

  const cv = document.createElement('canvas');
  cv.width  = wLogical * LABEL_DPR;
  cv.height = hLogical * LABEL_DPR;
  const ctx = cv.getContext('2d');
  ctx.scale(LABEL_DPR, LABEL_DPR);

  ctx.fillStyle = opts.bg || 'rgba(15, 23, 42, 0.92)';
  if (ctx.roundRect) { ctx.beginPath(); ctx.roundRect(0, 0, wLogical, hLogical, hLogical/2); ctx.fill(); }
  else ctx.fillRect(0, 0, wLogical, hLogical);

  ctx.fillStyle = opts.color || '#fef3c7';
  ctx.font = `${opts.weight || 'bold'} ${fs}px Geist, Inter, system-ui, sans-serif`;
  ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
  ctx.fillText(text, wLogical/2, hLogical/2);

  const tex = new THREE.CanvasTexture(cv);
  tex.anisotropy = MAX_ANISOTROPY;
  tex.minFilter = THREE.LinearMipmapLinearFilter;
  tex.magFilter = THREE.LinearFilter;
  tex.generateMipmaps = true;
  const mat = new THREE.SpriteMaterial({ map: tex, transparent: true, depthWrite: false });
  const s = new THREE.Sprite(mat);
  const scale = opts.scale || 1.0;
  s.scale.set((wLogical/hLogical) * 0.6 * scale, 0.6 * scale, 1);
  return s;
}

function softMat(color, opacity = 0.92) {
  return new THREE.MeshStandardMaterial({ color, roughness: 0.55, metalness: 0.05, transparent: true, opacity });
}
function softBox(w, h, d, color, opacity = 0.92) {
  const m = new THREE.Mesh(new THREE.BoxGeometry(w, h, d), softMat(color, opacity));
  const e = new THREE.LineSegments(
    new THREE.EdgesGeometry(m.geometry),
    new THREE.LineBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.55 })
  );
  m.add(e);
  return m;
}
function smallSphere(r, color, opacity = 0.92) {
  return new THREE.Mesh(new THREE.SphereGeometry(r, 24, 18), softMat(color, opacity));
}
function showOpacity(obj, op) {
  if (obj.material && 'opacity' in obj.material) { obj.material.opacity = op; obj.material.transparent = true; }
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const lerp = (a, b, t) => a + (b - a) * t;
const easeOut = t => 1 - Math.pow(1 - t, 3);

// Bold video-grade palette — saturated colours on deep navy.
const C = {
  peach:     0xf87171,
  peachDeep: 0xef4444,
  mint:      0x22c55e,
  mintDeep:  0x16a34a,
  sky:       0x3b82f6,
  rose:      0xec4899,
  cream:     0xfef3c7,
  amber:     0xf97316,
  norm:      0xf59e0b,   // amber  — RMSNorm
  attn:      0x10b981,   // emerald — attention
  mlp:       0x6366f1,   // indigo  — MLP
  embed:     0xef4444,
  gold:      0xfbbf24,
};

// ============================== LIVE-TRACE DATA ==============================
// Captured from a REAL forward pass of HuggingFaceTB/SmolLM2-135M on prompt
//   "Hello, my name is"
// at block 0 (first of 30). Every number below is what the model produces.
const TRACE = {
  prompt: 'Hello, my name is',
  tokens: [19556, 28, 957, 1462, 314],
  pieces: ['Hello', ',', ' my', ' name', ' is'],
  rms: {
    x_in:    [0.108, 0.123, 0.115, 0.111, 0.107],
    y_norm1: [0.057, 0.051, 0.039, 0.041, 0.049],
    attn:    [0.098, 0.117, 0.105, 0.091, 0.117],
    x_mid:   [0.113, 0.152, 0.150, 0.137, 0.137],
    y_norm2: [0.099, 0.102, 0.095, 0.093, 0.095],
    mlp:     [2.258, 1.420, 1.604, 2.050, 1.495],
    x_out:   [2.265, 1.405, 1.611, 2.051, 1.486],
  },
  attn_head0: [
    [1.000, 0.000, 0.000, 0.000, 0.000],
    [0.936, 0.064, 0.000, 0.000, 0.000],
    [0.289, 0.106, 0.605, 0.000, 0.000],
    [0.233, 0.015, 0.586, 0.165, 0.000],
    [0.138, 0.034, 0.409, 0.230, 0.189],
  ],
  gamma_norm1: 0.957,
  gamma_norm2: 2.601,
};

// ============================== GRANULAR TRANSFORMER BLOCK ==============================
const block = new THREE.Group();
block.position.set(0, 0, 0);
scene.add(block);

const M = {};
let mY = 1.0;

function microBox(width, height, color, label, sublabel, opts = {}) {
  const g = new THREE.Group();
  const box = softBox(width, height, 0.6, color, opts.opacity || 0.9);
  g.add(box);
  g.userData.box = box;
  if (label) {
    const l = spriteLabel(label, { fontSize: 26, color: opts.labelColor || '#fef3c7', scale: 0.42 });
    l.position.set(0, 0, 0.4);
    g.add(l);
    g.userData.label = l;
  }
  if (sublabel) {
    const s = spriteLabel(sublabel, { fontSize: 18, color: '#fbbf24', bg: 'rgba(26,34,56,0.85)', scale: 0.32 });
    s.position.set(0, -height/2 - 0.18, 0.4);
    g.add(s);
    g.userData.sub = s;
  }
  return g;
}

function microResidualArc(yStart, yEnd) {
  const xOff = -2.2;
  const start = new V3(-1.6, yStart, 0);
  const ctrl1 = new V3(xOff, yStart, 0);
  const ctrl2 = new V3(xOff, yEnd,   0);
  const end   = new V3(-1.6, yEnd,   0);
  const curve = new THREE.CubicBezierCurve3(start, ctrl1, ctrl2, end);
  const pts = curve.getPoints(40);
  const geo = new THREE.BufferGeometry().setFromPoints(pts);
  const mat = new THREE.LineDashedMaterial({ color: 0xfbbf24, dashSize: 0.14, gapSize: 0.10, transparent: true, opacity: 0.85 });
  const ln = new THREE.Line(geo, mat);
  ln.computeLineDistances();
  return ln;
}

function microFrame(yA, yB, label, color = 0xfbbf24) {
  const g = new THREE.Group();
  const w = 4.6, h = yB - yA;
  const box = new THREE.Mesh(
    new THREE.BoxGeometry(w, h, 0.05),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: 0 })
  );
  box.position.set(0, (yA + yB) / 2, -0.3);
  const edges = new THREE.LineSegments(
    new THREE.EdgesGeometry(box.geometry),
    new THREE.LineDashedMaterial({ color, dashSize: 0.12, gapSize: 0.08, transparent: true, opacity: 0.5 })
  );
  edges.position.copy(box.position);
  edges.computeLineDistances();
  g.add(box); g.add(edges);
  if (label) {
    const l = spriteLabel(label, { fontSize: 22, color: '#fbbf24', bg: 'rgba(15,23,42,0.92)', scale: 0.4 });
    l.position.set(-2.6, (yA + yB) / 2, 0);
    g.add(l);
  }
  return g;
}

// 1. x_in
M.hiddenIn = spriteLabel('x_in · residual stream  (B, T, 576)', {
  fontSize: 26, color: '#fef3c7', bg: 'rgba(245,158,11,0.85)', scale: 0.65,
});
M.hiddenIn.position.set(0, mY, 0);
block.add(M.hiddenIn);

M.tokenChips = [];
M.tokenLabels = [];
for (let i = 0; i < TRACE.pieces.length; i++) {
  const x = (i - (TRACE.pieces.length - 1) / 2) * 0.85;
  const cube = softBox(0.55, 0.30, 0.30, C.peach);
  cube.position.set(x, mY - 0.5, 0);
  cube.scale.setScalar(0.01);
  block.add(cube);
  M.tokenChips.push(cube);
  const txt = spriteLabel(TRACE.pieces[i].replace(/^ /, '·') || '_', {
    fontSize: 22, color: '#fef3c7', scale: 0.34,
  });
  txt.position.set(x, mY - 0.5, 0.30);
  showOpacity(txt, 0);
  block.add(txt);
  M.tokenLabels.push(txt);
}
mY += 0.95;

// 2. input_layernorm
M.norm1 = microBox(4.0, 0.32, C.norm, 'input_layernorm', 'RMSNorm · γ ⊙ x · rsqrt(mean(x²)+ε)');
M.norm1.position.set(0, mY, 0);
block.add(M.norm1);
mY += 0.95;

// ===== self_attn region =====
const ATTN_Y_START = mY;

// 3. q_proj | k_proj | v_proj
M.qProj = microBox(1.2, 0.5, C.attn, 'q_proj', '576 → 9·64');
M.kProj = microBox(0.7, 0.5, C.attn, 'k_proj', '576 → 3·64');
M.vProj = microBox(0.7, 0.5, C.attn, 'v_proj', '576 → 3·64');
M.qProj.position.set(-1.2, mY, 0);
M.kProj.position.set( 0.4, mY, 0);
M.vProj.position.set( 1.4, mY, 0);
block.add(M.qProj); block.add(M.kProj); block.add(M.vProj);
mY += 1.1;

// 4. heads
M.qHeads = []; M.kHeads = []; M.vHeads = [];
for (let i = 0; i < 9; i++) {
  const s = smallSphere(0.10, C.peachDeep);
  s.position.set(-1.6 + (i % 3) * 0.18, mY + (Math.floor(i/3) - 1) * 0.20, 0);
  block.add(s); M.qHeads.push(s);
}
for (let i = 0; i < 3; i++) {
  const s = smallSphere(0.13, C.mintDeep);
  s.position.set(0.4, mY + (i - 1) * 0.22, 0);
  block.add(s); M.kHeads.push(s);
}
for (let i = 0; i < 3; i++) {
  const s = smallSphere(0.13, C.sky);
  s.position.set(1.4, mY + (i - 1) * 0.22, 0);
  block.add(s); M.vHeads.push(s);
}
{
  const lab = spriteLabel('Q 9×64    K 3×64    V 3×64    (head_dim = 64)', {
    fontSize: 18, color: '#fbbf24', scale: 0.42,
  });
  lab.position.set(0, mY - 0.55, 0);
  block.add(lab);
}
mY += 1.05;

// 5. RoPE
M.rope = microBox(2.6, 0.28, C.peach, 'apply RoPE on Q, K',
  'q · cos + rotate_half(q) · sin   ·   θ = 100,000');
M.rope.position.set(-0.3, mY, 0);
block.add(M.rope);
M.ropeArrows = [];
{
  const cols = 8;
  for (let c = 0; c < cols; c++) {
    const ring = new THREE.Mesh(
      new THREE.TorusGeometry(0.10, 0.012, 8, 24),
      new THREE.MeshStandardMaterial({ color: 0xfbbf24, roughness: 0.4, emissive: 0xfbbf24, emissiveIntensity: 0.35 })
    );
    const x = (c - (cols - 1) / 2) * 0.32;
    ring.position.set(x, mY + 0.35, 0.1);
    block.add(ring);
    const ar = new THREE.ArrowHelper(new V3(1, 0, 0), new V3(x, mY + 0.35, 0.1), 0.10, C.amber, 0.04, 0.03);
    block.add(ar);
    const freq = 1.0 / Math.pow(100000, (2 * c) / 64);
    M.ropeArrows.push({ ring, arrow: ar, freq });
  }
}
mY += 0.95;

// 6. KV repeat
M.kvRep = microBox(2.6, 0.26, C.mint, 'KV repeat_interleave(3, dim=1)',
  '3 KV heads → 9 (each K/V head serves 3 Q heads)');
M.kvRep.position.set(0.5, mY, 0);
block.add(M.kvRep);
mY += 0.85;

// 7. SDPA + 5×5 real causal-softmax weights (head 0)
M.sdpa = microBox(3.6, 0.52, C.attn, 'F.scaled_dot_product_attention',
  'softmax(Q·Kᵀ / √64 + causal_mask) · V    is_causal=True');
M.sdpa.position.set(0, mY, 0);
{
  const T = 5;
  M.sdpaCells = [];
  for (let r = 0; r < T; r++) for (let c = 0; c < T; c++) {
    const w = TRACE.attn_head0[r][c];
    const cell = new THREE.Mesh(
      new THREE.PlaneGeometry(0.13, 0.13),
      new THREE.MeshBasicMaterial({
        color: c <= r
          ? new THREE.Color().setHSL(0.08, 0.7, 1 - 0.45 * w)
          : new THREE.Color(0x1a2238),
        transparent: true,
        opacity: c <= r ? 0.20 : 0.0,
      })
    );
    cell.position.set(-1.0 + c * 0.16, (T - 1 - r - (T - 1) / 2) * 0.10, 0.32);
    M.sdpa.add(cell);
    M.sdpaCells.push({ mesh: cell, r, c, w });
  }
}
mY += 1.05;

// 8. concat
M.concat = microBox(2.6, 0.22, C.peach, 'concat heads · view (B, T, 576)',
  'transpose(1, 2) · contiguous · view');
M.concat.position.set(0, mY, 0);
block.add(M.concat);
mY += 0.75;

// 9. o_proj
M.oProj = microBox(3.4, 0.36, C.attn, 'o_proj', 'Linear  576 → 576  (no bias)');
M.oProj.position.set(0, mY, 0);
block.add(M.oProj);
mY += 0.75;

const ATTN_Y_END = mY;
M.attnFrame = microFrame(ATTN_Y_START - 0.1, ATTN_Y_END + 0.05,
  'self_attn (GQA, RoPE, causal)', 0x10b981);
block.add(M.attnFrame);

// 10. residual ⊕ #1
M.res1 = (() => {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.34, 0.04, 12, 48),
    new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.95 })
  );
  ring.rotation.x = Math.PI / 2;
  g.add(ring);
  const plus = spriteLabel('⊕', { fontSize: 56, color: '#22c55e', scale: 0.45 });
  plus.position.set(0, 0, 0.05);
  g.add(plus);
  const lab = spriteLabel('residual #1', {
    fontSize: 18, color: '#22c55e', bg: 'rgba(34,197,94,0.85)', scale: 0.36,
  });
  lab.position.set(0.95, 0, 0);
  g.add(lab);
  return g;
})();
M.res1.position.set(0, mY, 0);
block.add(M.res1);
mY += 0.7;

// 11. x_mid
M.hiddenMid = spriteLabel('x_mid  (B, T, 576)', {
  fontSize: 24, color: '#fef3c7', bg: 'rgba(245,158,11,0.85)', scale: 0.5,
});
M.hiddenMid.position.set(0, mY, 0);
block.add(M.hiddenMid);
mY += 0.7;

// 12. post_attention_layernorm
M.norm2 = microBox(4.0, 0.32, C.norm, 'post_attention_layernorm',
  'RMSNorm · γ ⊙ x · rsqrt(mean(x²)+ε)');
M.norm2.position.set(0, mY, 0);
block.add(M.norm2);
mY += 0.95;

// ===== mlp region =====
const MLP_Y_START = mY;

// 13. gate_proj | up_proj
M.gateProj = microBox(1.6, 0.48, C.mlp, 'gate_proj', 'Linear  576 → 1536');
M.upProj   = microBox(1.6, 0.48, C.mlp, 'up_proj',   'Linear  576 → 1536');
M.gateProj.position.set(-1.0, mY, 0);
M.upProj.position.set(  1.0, mY, 0);
block.add(M.gateProj); block.add(M.upProj);
mY += 1.0;

// 14. SiLU
M.silu = microBox(1.4, 0.28, C.mlp, 'F.silu(gate)', 'silu(z) = z · σ(z)');
M.silu.position.set(-1.0, mY, 0);
block.add(M.silu);

M.upBypass = spriteLabel('up (unchanged)', {
  fontSize: 18, color: '#60a5fa', bg: 'rgba(99,102,241,0.85)', scale: 0.32,
});
M.upBypass.position.set(1.0, mY, 0);
block.add(M.upBypass);
mY += 1.05;

// 15. multiply
M.mul = (() => {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.32, 0.04, 12, 48),
    new THREE.MeshBasicMaterial({ color: 0xf97316, transparent: true, opacity: 0.95 })
  );
  ring.rotation.x = Math.PI / 2;
  g.add(ring);
  const dot = spriteLabel('⊙', { fontSize: 50, color: '#fef3c7', scale: 0.4 });
  dot.position.set(0, 0, 0.05);
  g.add(dot);
  const lab = spriteLabel('silu(gate) ⊙ up  ·  (B, T, 1536)', {
    fontSize: 18, color: '#fef3c7', bg: 'rgba(15,23,42,0.92)', scale: 0.36,
  });
  lab.position.set(0, -0.55, 0);
  g.add(lab);
  return g;
})();
M.mul.position.set(0, mY, 0);
block.add(M.mul);
mY += 0.85;

// 16. down_proj
M.downProj = microBox(3.4, 0.42, C.mlp, 'down_proj', 'Linear  1536 → 576');
M.downProj.position.set(0, mY, 0);
block.add(M.downProj);
mY += 0.75;

const MLP_Y_END = mY;
M.mlpFrame = microFrame(MLP_Y_START - 0.1, MLP_Y_END + 0.05, 'mlp (SwiGLU)', 0x6366f1);
block.add(M.mlpFrame);

// 17. residual ⊕ #2
M.res2 = (() => {
  const g = new THREE.Group();
  const ring = new THREE.Mesh(
    new THREE.TorusGeometry(0.34, 0.04, 12, 48),
    new THREE.MeshBasicMaterial({ color: 0x22c55e, transparent: true, opacity: 0.95 })
  );
  ring.rotation.x = Math.PI / 2;
  g.add(ring);
  const plus = spriteLabel('⊕', { fontSize: 56, color: '#22c55e', scale: 0.45 });
  plus.position.set(0, 0, 0.05);
  g.add(plus);
  const lab = spriteLabel('residual #2', {
    fontSize: 18, color: '#22c55e', bg: 'rgba(34,197,94,0.85)', scale: 0.36,
  });
  lab.position.set(0.95, 0, 0);
  g.add(lab);
  return g;
})();
M.res2.position.set(0, mY, 0);
block.add(M.res2);
mY += 0.7;

// 18. x_out
M.hiddenOut = spriteLabel('x_out · to next block  (B, T, 576)', {
  fontSize: 26, color: '#fef3c7', bg: 'rgba(245,158,11,0.85)', scale: 0.65,
});
M.hiddenOut.position.set(0, mY, 0);
block.add(M.hiddenOut);

M.arc1 = microResidualArc(M.hiddenIn.position.y, M.res1.position.y);
M.arc2 = microResidualArc(M.hiddenMid.position.y, M.res2.position.y);
block.add(M.arc1);
block.add(M.arc2);

M.heading = spriteLabel('one of 30 decoder blocks · model.py L188–199', {
  fontSize: 28, color: '#fef3c7', bg: 'rgba(15,23,42,0.92)', scale: 0.6,
});
M.heading.position.set(0, mY + 0.7, 0);
block.add(M.heading);

{
  const arr = new THREE.ArrowHelper(new V3(0, 1, 0), new V3(0, mY + 0.95, 0), 0.6, 0x22c55e, 0.22, 0.13);
  block.add(arr);
  M.nextBlockArrow = arr;
  const lab = spriteLabel('feeds into block i+1  (1 ≤ i+1 ≤ 29)', {
    fontSize: 18, color: '#22c55e', bg: 'rgba(34,197,94,0.85)', scale: 0.35,
  });
  lab.position.set(0, mY + 1.7, 0);
  block.add(lab);
  M.nextBlockLabel = lab;
}

const BLOCK_CENTER_Y = (M.hiddenIn.position.y + M.hiddenOut.position.y) / 2;
const BLOCK_TOP_Y    = mY + 1.7;

// ============================== LIVE FLOW ARROW ==============================
// Glowing gold marker that represents WHERE the data currently is inside the
// block. Floats along the left side of the column, climbing as the cinema
// plays. The camera follows its y so the viewer's eye is always on the live
// position.
M.flow = new THREE.Group();
M.flow.position.set(-3.0, M.hiddenIn.position.y, 0);
block.add(M.flow);
{
  const ballMat = new THREE.MeshStandardMaterial({
    color: 0xfbbf24, emissive: 0xfbbf24, emissiveIntensity: 1.6,
    roughness: 0.25, metalness: 0.1,
  });
  const ball = new THREE.Mesh(new THREE.SphereGeometry(0.20, 28, 22), ballMat);
  M.flow.add(ball);
  M.flowBall = ball;

  const halo = new THREE.Mesh(
    new THREE.SphereGeometry(0.45, 24, 18),
    new THREE.MeshBasicMaterial({ color: 0xfbbf24, transparent: true, opacity: 0.18 })
  );
  M.flow.add(halo);
  M.flowHalo = halo;

  const arrow = new THREE.ArrowHelper(new V3(0, 1, 0), new V3(0, 0.2, 0), 0.7, 0xfbbf24, 0.22, 0.14);
  M.flow.add(arrow);
  M.flowArrow = arrow;

  const lab = spriteLabel('x  →', {
    fontSize: 26, color: '#fbbf24', bg: 'rgba(15,23,42,0.92)', scale: 0.45,
  });
  lab.position.set(-0.85, 0, 0);
  M.flow.add(lab);
}

const STAGE_Y = {
  x_in:     M.hiddenIn.position.y,
  hiddenIn: M.hiddenIn.position.y,
  norm1:    M.norm1.position.y,
  y_norm1:  M.norm1.position.y,
  qProj:    M.qProj.position.y,
  kProj:    M.qProj.position.y,
  vProj:    M.qProj.position.y,
  qkv:      M.qProj.position.y,
  heads:    M.qProj.position.y + 0.85,
  rope:     M.rope.position.y,
  kvRep:    M.kvRep.position.y,
  sdpa:     M.sdpa.position.y,
  attn:     M.oProj.position.y,
  concat:   M.concat.position.y,
  oProj:    M.oProj.position.y,
  res1:     M.res1.position.y,
  hiddenMid:M.hiddenMid.position.y,
  x_mid:    M.hiddenMid.position.y,
  norm2:    M.norm2.position.y,
  y_norm2:  M.norm2.position.y,
  gateProj: M.gateProj.position.y,
  upProj:   M.gateProj.position.y,
  gateUp:   M.gateProj.position.y,
  silu:     M.silu.position.y,
  mul:      M.mul.position.y,
  downProj: M.downProj.position.y,
  mlp:      M.downProj.position.y,
  res2:     M.res2.position.y,
  hiddenOut:M.hiddenOut.position.y,
  x_out:    M.hiddenOut.position.y,
};

const NON_FOLLOW_SHOTS = new Set(['wide', 'full', 'outro']);
let flowYTarget = M.hiddenIn.position.y;
let followFlow = true;

// ============================== LIVE-TRACE HUD ==============================
const hud = new THREE.Group();
hud.position.set(4.6, BLOCK_CENTER_Y, 0);
scene.add(hud);

{
  const card = new THREE.Mesh(
    new THREE.PlaneGeometry(3.0, 4.4),
    new THREE.MeshBasicMaterial({ color: 0x1a2238, transparent: true, opacity: 0.95 })
  );
  card.position.set(0, 0, -0.05);
  hud.add(card);
}
M.hudTitle = spriteLabel('live trace · "Hello, my name is"', {
  fontSize: 22, color: '#fef3c7', bg: 'rgba(245,158,11,0.85)', scale: 0.55,
});
M.hudTitle.position.set(0, 1.85, 0);
hud.add(M.hudTitle);
M.hudStage = spriteLabel('x_in', {
  fontSize: 28, color: '#22c55e', bg: 'rgba(34,197,94,0.85)', scale: 0.5,
});
M.hudStage.position.set(0, 1.4, 0);
hud.add(M.hudStage);
M.hudUnits = spriteLabel('per-token RMS', {
  fontSize: 16, color: '#fbbf24', scale: 0.4,
});
M.hudUnits.position.set(0, 1.05, 0);
hud.add(M.hudUnits);

M.hudBars = [];
M.hudBarLabels = [];
for (let i = 0; i < 5; i++) {
  const bar = softBox(0.28, 1.0, 0.18, C.peachDeep);
  bar.position.set((i - 2) * 0.5, 0.3, 0);
  bar.scale.y = 0.01;
  hud.add(bar);
  M.hudBars.push(bar);
  const lab = spriteLabel(TRACE.pieces[i].replace(/^ /, '·') || '_', {
    fontSize: 16, color: '#fef3c7', scale: 0.32,
  });
  lab.position.set((i - 2) * 0.5, -0.4, 0);
  hud.add(lab);
  M.hudBarLabels.push(lab);
}
M.hudReadout = spriteLabel('—', {
  fontSize: 16, color: '#fbbf24', bg: 'rgba(26,34,56,0.9)', scale: 0.42,
});
M.hudReadout.position.set(0, -0.85, 0);
hud.add(M.hudReadout);
M.hudFooter = spriteLabel('verified · real SmolLM2-135M weights · block 0', {
  fontSize: 14, color: '#fbbf24', scale: 0.36,
});
M.hudFooter.position.set(0, -1.4, 0);
hud.add(M.hudFooter);

function showStage(stageKey) {
  const vals = TRACE.rms[stageKey];
  if (!vals) return;
  const newName = spriteLabel(stageKey, {
    fontSize: 28, color: '#22c55e', bg: 'rgba(34,197,94,0.85)', scale: 0.5,
  });
  newName.position.copy(M.hudStage.position);
  hud.remove(M.hudStage);
  hud.add(newName);
  M.hudStage = newName;

  const maxBar = 2.6;
  for (let i = 0; i < 5; i++) {
    const v = vals[i];
    const norm = clamp(v / 2.5, 0.04, 1.0);
    M.hudBars[i].scale.y = norm * maxBar;
    M.hudBars[i].position.y = 0.3 + (norm * maxBar - 1.0) / 2;
  }

  const newRead = spriteLabel(vals.map(v => v.toFixed(3)).join('  '), {
    fontSize: 16, color: '#fbbf24', bg: 'rgba(26,34,56,0.9)', scale: 0.62,
  });
  newRead.position.copy(M.hudReadout.position);
  hud.remove(M.hudReadout);
  hud.add(newRead);
  M.hudReadout = newRead;
}
showStage('x_in');

// ============================== CAMERA SHOTS ==============================
// All shots look STRAIGHT at the block (camera x = 0, look x = 0). No tilt.
const MY = {
  hiddenIn:  M.hiddenIn.position.y,
  tokens:    M.hiddenIn.position.y - 0.5,
  norm1:     M.norm1.position.y,
  qkv:       M.qProj.position.y,
  heads:     M.qProj.position.y + 0.85,
  rope:      M.rope.position.y,
  kvRep:     M.kvRep.position.y,
  sdpa:      M.sdpa.position.y,
  concat:    M.concat.position.y,
  oProj:     M.oProj.position.y,
  res1:      M.res1.position.y,
  hiddenMid: M.hiddenMid.position.y,
  norm2:     M.norm2.position.y,
  gateUp:    M.gateProj.position.y,
  silu:      M.silu.position.y,
  mul:       M.mul.position.y,
  downProj:  M.downProj.position.y,
  res2:      M.res2.position.y,
  hiddenOut: M.hiddenOut.position.y,
};

const SHOTS = {
  wide:      { pos: [0, BLOCK_CENTER_Y,        26], look: [0, BLOCK_CENTER_Y, 0] },
  full:      { pos: [0, BLOCK_CENTER_Y,        20], look: [0, BLOCK_CENTER_Y, 0] },
  outro:     { pos: [0, BLOCK_CENTER_Y + 1,    26], look: [0, BLOCK_CENTER_Y, 0] },
  tokens:    { pos: [0, MY.tokens,             10], look: [0, MY.tokens,      0] },
  hiddenIn:  { pos: [0, MY.hiddenIn,           10], look: [0, MY.hiddenIn,    0] },
  norm1:     { pos: [0, MY.norm1   + 0.4,      11], look: [0, MY.norm1,       0] },
  qkv:       { pos: [0, MY.qkv     + 0.4,      11], look: [0, MY.qkv,         0] },
  heads:     { pos: [0, MY.heads,              10], look: [0, MY.heads,       0] },
  rope:      { pos: [0, MY.rope    + 0.2,      11], look: [0, MY.rope,        0] },
  kvRep:     { pos: [0, MY.kvRep,              11], look: [0, MY.kvRep,       0] },
  sdpa:      { pos: [0, MY.sdpa,               10], look: [0, MY.sdpa,        0] },
  sdpaClose: { pos: [0, MY.sdpa,                7], look: [0, MY.sdpa,        0] },
  concat:    { pos: [0, MY.concat,             11], look: [0, MY.concat,      0] },
  oProj:     { pos: [0, MY.oProj,              11], look: [0, MY.oProj,       0] },
  res1:      { pos: [0, MY.res1,               11], look: [0, MY.res1,        0] },
  hiddenMid: { pos: [0, MY.hiddenMid,          11], look: [0, MY.hiddenMid,   0] },
  norm2:     { pos: [0, MY.norm2   + 0.4,      11], look: [0, MY.norm2,       0] },
  gateUp:    { pos: [0, MY.gateUp,             12], look: [0, MY.gateUp,      0] },
  silu:      { pos: [0, MY.silu,               11], look: [0, MY.silu,        0] },
  mul:       { pos: [0, MY.mul,                11], look: [0, MY.mul,         0] },
  downProj:  { pos: [0, MY.downProj,           11], look: [0, MY.downProj,    0] },
  res2:      { pos: [0, MY.res2,               11], look: [0, MY.res2,        0] },
  hiddenOut: { pos: [0, MY.hiddenOut,          11], look: [0, MY.hiddenOut,   0] },
};

let camTarget = { pos: new V3(...SHOTS.wide.pos), look: new V3(...SHOTS.wide.look) };
function setShot(name) {
  const s = SHOTS[name];
  if (!s) return;
  camTarget.pos.set(...s.pos);
  camTarget.look.set(...s.look);
}

const HIGHLIGHTABLE = ['norm1','qProj','kProj','vProj','rope','kvRep','sdpa','concat','oProj','norm2','gateProj','upProj','silu','downProj'];
function highlight(name) {
  HIGHLIGHTABLE.forEach(k => {
    const g = M[k];
    if (!g || !g.userData || !g.userData.box) return;
    const isHL = (k === name);
    const mat = g.userData.box.material;
    mat.emissive = new THREE.Color(isHL ? 0xfbbf24 : 0x000000);
    mat.opacity = isHL ? 1.0 : 0.65;
  });
  if (M.res1) M.res1.scale.setScalar(name === 'res1' ? 1.18 : 1.0);
  if (M.res2) M.res2.scale.setScalar(name === 'res2' ? 1.18 : 1.0);
  if (M.mul)  M.mul.scale.setScalar (name === 'mul'  ? 1.18 : 1.0);
}

// ============================== SCENES & FRAMES ==============================
const F = (dur, cap, narr, doFn, links, hl, cam, stage) => ({
  dur, cap, narr,
  do: doFn || (()=>{}),
  links: links || [],
  hl: hl || null,
  cam: cam || null,
  stage: stage || null,
});

const SCENES = [
  {
    name: 'Intro · one of 30 decoder blocks',
    shot: 'wide',
    frames: [
      F(1.4, 'SmolLM2-135M · one transformer block',
        'You are looking at ONE of the 30 identical decoder blocks that make up SmolLM2-135M. Over the next ~2 minutes the camera walks every component, bottom-to-top, while a live trace from a real forward pass shows the per-token RMS at each stage on the right.',
        t => {}, [], null, 'wide'),
      F(1.0, 'Live e2e prompt',
        'The block is being fed a real (B, T, 576) hidden state produced by embedding the prompt "Hello, my name is" — 5 tokens, hidden_size 576.',
        t => {
          M.tokenChips.forEach((c, i) => {
            c.scale.setScalar(lerp(0.01, 1, clamp(easeOut(t) * 5 - i * 0.4, 0, 1)));
            showOpacity(M.tokenLabels[i], clamp(easeOut(t) * 5 - i * 0.4, 0, 1));
          });
        }, [], null, 'tokens', 'x_in'),
      F(1.2, 'Bottom = x_in   ·   Top = x_out',
        'The block reads from the residual stream at the bottom, transforms it via attention then MLP (with two residual adds), and writes the updated stream back out at the top. That x_out then becomes x_in for the next block.',
        t => {}, [], null, 'wide'),
    ],
  },
  {
    name: 'x_in · residual stream enters',
    shot: 'hiddenIn',
    frames: [
      F(1.0, 'x_in arrives',
        'The block receives x_in of shape (B=1, T=5, hidden=576). Per-token RMS values shown on the right are the REAL magnitudes coming out of embed_tokens for these 5 tokens.',
        t => { M.hiddenIn.material.opacity = 1.0; },
        [{ file: 'model.py', range: '230', label: 'embed_tokens lookup' }],
        null, 'hiddenIn', 'x_in'),
    ],
  },
  {
    name: 'input_layernorm  (RMSNorm)',
    shot: 'norm1',
    frames: [
      F(1.2, 'input_layernorm  ·  RMSNorm',
        'Pre-norm: stabilises activations BEFORE attention. Pure RMSNorm — no mean subtraction, no bias.',
        t => {}, [{ file: 'model.py', range: '59-70', label: 'RMSNorm L59–70' }],
        'norm1', 'norm1', 'y_norm1'),
      F(1.4, 'y = γ ⊙ x · rsqrt(mean(x²) + ε)',
        'Math is done in fp32 for numerical stability, then cast back. The learned γ is 576 numbers (one gain per channel). For this loaded block: ||γ_norm1|| = 0.957.',
        t => {
          M.norm1.userData.box.material.emissive = new THREE.Color().setHSL(0.08, 0.7, easeOut(t) * 0.4);
        }, [{ file: 'model.py', range: '65-70', label: 'forward L65–70' }],
        'norm1', null, 'y_norm1'),
      F(0.9, 'Per-token RMS dropped',
        'Trace: y_norm1 RMS ≈ 0.05 (was 0.11). RMSNorm has pulled every token to a similar magnitude scale before attention sees them.',
        t => {}, [], 'norm1', null, 'y_norm1'),
    ],
  },
  {
    name: 'q_proj · k_proj · v_proj   (GQA)',
    shot: 'qkv',
    frames: [
      F(1.0, 'Three Linear projections, in parallel',
        'q_proj, k_proj, v_proj each read the SAME post-norm hidden state and project it into Q, K, V tensors. All three have bias=False.',
        t => {}, [{ file: 'model.py', range: '130-133', label: 'projections declared' }],
        'qProj', 'qkv'),
      F(0.9, 'q_proj  ·  576 → 9 × 64',
        'Q has 9 heads of dim 64 (head_dim = 576 / 9 = 64). Each head will ask a different "question" of every token.',
        t => {
          M.qProj.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
          M.qHeads.forEach((s, i) => s.scale.setScalar(lerp(1.0, 1.4, clamp(easeOut(t) * 9 - i * 0.5, 0, 1))));
        }, [{ file: 'model.py', range: '139', label: 'q_proj forward' }], 'qProj'),
      F(0.9, 'k_proj  ·  576 → 3 × 64   (GQA)',
        'K has only 3 heads — Grouped Query Attention. Each K head will be shared by 3 Q heads. This cuts KV-cache cost by 3×.',
        t => {
          M.qHeads.forEach(s => s.scale.setScalar(1.0));
          M.kProj.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
          M.kHeads.forEach((s, i) => s.scale.setScalar(lerp(1.0, 1.5, clamp(easeOut(t) * 3 - i * 0.3, 0, 1))));
        }, [{ file: 'model.py', range: '140', label: 'k_proj forward' }], 'kProj'),
      F(0.9, 'v_proj  ·  576 → 3 × 64   (GQA)',
        'V mirrors K — 3 heads, broadcast to 9 just before attention. Per layer the KV cache is 2 × 3 × 64 numbers per token (not 2 × 9 × 64).',
        t => {
          M.kHeads.forEach(s => s.scale.setScalar(1.0));
          M.vProj.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
          M.vHeads.forEach((s, i) => s.scale.setScalar(lerp(1.0, 1.5, clamp(easeOut(t) * 3 - i * 0.3, 0, 1))));
        }, [{ file: 'model.py', range: '141', label: 'v_proj forward' }], 'vProj'),
      F(0.9, '9 Q  +  3 K  +  3 V heads',
        'Total Q+K+V+O params per layer: 884,736 (no biases). Heads view, on stage above the boxes, lights all 15 spheres.',
        t => { M.vHeads.forEach(s => s.scale.setScalar(1.0)); }, [], null, 'heads'),
    ],
  },
  {
    name: 'RoPE  ·  rotate Q, K by position',
    shot: 'rope',
    frames: [
      F(1.1, 'RoPE rotates Q and K',
        'Pairs of adjacent dims in Q and K are rotated by an angle = position · θ⁻²ⁱ/ᵈ. V is NOT rotated. After RoPE, dot products Q·Kᵀ encode RELATIVE position for free.',
        t => {}, [{ file: 'model.py', range: '99-106', label: '_apply_rope' }],
        'rope', 'rope'),
      F(1.4, 'θ = 100,000  (SmolLM2-v2)',
        'config.json sets rope_theta=100_000. Older Llama 1/2 used 10,000; SmolLM2-v2 raised it to 100,000 for longer effective context. Low-index pairs rotate fast, high-index pairs barely move.',
        t => {
          M.ropeArrows.forEach((d, i) => {
            const ang = (t + i * 0.05) * Math.PI * 2 * d.freq * 1000;
            d.arrow.setDirection(new V3(Math.cos(ang), Math.sin(ang), 0));
          });
        }, [{ file: 'model.py', range: '84-90', label: 'rope cache build' }], 'rope'),
      F(0.9, 'rotate_half — HF convention',
        'Llama/SmolLM split the head_dim in HALF (not interleaved pairs): concat(-x[d/2:], x[:d/2]). Getting this wrong silently produces garbage outputs.',
        t => {
          M.ropeArrows.forEach((d, i) => {
            const ang = (performance.now() / 1000 + i * 0.3) * (0.4 + d.freq * 1500);
            d.arrow.setDirection(new V3(Math.cos(ang), Math.sin(ang), 0));
          });
        }, [{ file: 'model.py', range: '93-96', label: 'rotate_half' }], 'rope'),
    ],
  },
  {
    name: 'GQA broadcast  ·  KV repeat ×3',
    shot: 'kvRep',
    frames: [
      F(1.1, 'KV heads repeat_interleave(3, dim=1)',
        'PyTorch SDPA needs n_q_heads == n_kv_heads. We materialise the broadcast: each KV head serves 3 Q heads. No new params — just a memory op.',
        t => {}, [{ file: 'model.py', range: '149-151', label: 'repeat_interleave' }],
        'kvRep'),
      F(0.9, 'KV head 0 → Q heads 0, 1, 2  ·  head 1 → 3, 4, 5  ·  head 2 → 6, 7, 8',
        'After the repeat: K, V are (B, 9, T, 64) — same shape as Q. Modern attention kernels (Flash, SDPA) implement this as a virtual broadcast that avoids the materialisation cost.',
        t => {
          const phase = (t * Math.PI * 3) % (Math.PI * 2);
          M.kHeads.forEach(s => s.scale.setScalar(1.0 + 0.4 * Math.abs(Math.sin(phase))));
          M.vHeads.forEach(s => s.scale.setScalar(1.0 + 0.4 * Math.abs(Math.sin(phase))));
        }, [{ file: 'model.py', range: '123', label: 'n_rep = 3' }], 'kvRep'),
    ],
  },
  {
    name: 'F.scaled_dot_product_attention',
    shot: 'sdpa',
    frames: [
      F(1.0, 'softmax(QKᵀ / √64 + mask) · V',
        'For T = 5 tokens, this gives a 5×5 weight matrix per head. The √64 prevents softmax saturation. This is the ONLY operation in the block that moves information across token positions.',
        t => {
          M.kHeads.forEach(s => s.scale.setScalar(1.0));
          M.vHeads.forEach(s => s.scale.setScalar(1.0));
        }, [{ file: 'model.py', range: '156-161', label: 'SDPA call' }],
        'sdpa', 'sdpaClose'),
      F(1.0, 'Causal mask · upper triangle is dropped',
        'is_causal=True sets the upper triangle to −∞ before softmax, so token t cannot peek at positions > t. Off-diagonal cells in the upper triangle stay at ~0 throughout.',
        t => {
          M.sdpaCells.forEach(c => {
            if (c.c > c.r) showOpacity(c.mesh, lerp(0.0, 0.05, easeOut(t)));
          });
        }, [{ file: 'model.py', range: '160', label: 'is_causal=True' }],
        'sdpa', 'sdpaClose'),
      F(1.6, 'Real head-0 weights — live trace',
        'Each lower-triangular cell now lights up to its REAL causal-softmax probability from the actual forward pass on "Hello, my name is". Row 0 ("Hello"): 1.0 → can only attend to itself. Row 4 (" is"): {0.14, 0.03, 0.41, 0.23, 0.19} — most weight on " my".',
        t => {
          const total = M.sdpaCells.length;
          M.sdpaCells.forEach((c, i) => {
            if (c.c > c.r) return;
            const tt = clamp(easeOut(t) * total - i * 0.45, 0, 1);
            showOpacity(c.mesh, 0.25 + 0.75 * tt * c.w);
          });
        }, [], 'sdpa', 'sdpaClose'),
      F(1.0, 'Attention output per head: weighted sum of V',
        'Apply weights to V (the 3-head V, broadcast to 9). Each token gets a contextual mix of all preceding tokens.',
        t => {}, [{ file: 'model.py', range: '156-161', label: 'SDPA L156–161' }],
        'sdpa', 'sdpa'),
    ],
  },
  {
    name: 'concat heads  ·  o_proj',
    shot: 'oProj',
    frames: [
      F(0.9, 'transpose · contiguous · view (B, T, 576)',
        'Glue the 9 head outputs back into one 576-dim vector per token. Pure reshape, no params.',
        t => {}, [{ file: 'model.py', range: '162', label: 'merge heads' }],
        'concat', 'concat'),
      F(1.1, 'o_proj  ·  Linear  576 → 576',
        'Lets the model learn how to combine the 9 heads. Without o_proj they would be independent channels — o_proj is what makes multi-head richer than 9 independent single-head attentions.',
        t => {
          M.oProj.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
        }, [{ file: 'model.py', range: '163', label: 'o_proj forward' }],
        'oProj', 'oProj', 'attn'),
    ],
  },
  {
    name: 'residual ⊕  ·  x = x + attn_out',
    shot: 'res1',
    frames: [
      F(1.3, 'First skip connection closes',
        'x_mid = x_in + attn_out. The original x_in (still small, RMS ~0.11) is summed with the attention output. The residual is the trainability hinge — gradients flow straight through even when attention is near zero at init.',
        t => {
          M.res1.scale.setScalar(1.0 + 0.18 * Math.sin(t * Math.PI * 2));
          if (M.arc1) M.arc1.material.opacity = 0.4 + 0.5 * Math.abs(Math.sin(t * Math.PI * 2));
        }, [{ file: 'model.py', range: '197', label: 'first residual L197' }],
        'res1', 'res1', 'x_mid'),
    ],
  },
  {
    name: 'post_attention_layernorm  (RMSNorm)',
    shot: 'norm2',
    frames: [
      F(1.0, 'Second RMSNorm — same formula, new γ',
        'Naming is historical: "post_attention" sits BEFORE the MLP in a pre-norm architecture. Separate, independently-learned γ; for this checkpoint ||γ_norm2|| = 2.601 — visibly larger than ||γ_norm1|| (0.957).',
        t => {
          M.norm2.userData.box.material.emissive = new THREE.Color().setHSL(0.08, 0.7, easeOut(t) * 0.4);
          M.res1.scale.setScalar(1.0);
        }, [{ file: 'model.py', range: '193', label: 'declared L193' }],
        'norm2', 'norm2', 'y_norm2'),
    ],
  },
  {
    name: 'gate_proj · up_proj   (SwiGLU)',
    shot: 'gateUp',
    frames: [
      F(1.0, 'SwiGLU MLP starts: two parallel up-projections',
        'gate_proj and up_proj both read the SAME post-norm input and project it 576 → 1536. The 1536 ≈ 8/3 · 576 ratio is chosen so the 3-matrix SwiGLU MLP has the same param count as a 2-matrix 4× plain MLP.',
        t => {}, [{ file: 'model.py', range: '175-176', label: 'gate / up declared' }],
        'gateProj', 'gateUp'),
      F(0.8, 'gate_proj  ·  576 → 1536',
        'The "gating signal" branch. Goes through SiLU next.',
        t => {
          M.gateProj.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
        }, [{ file: 'model.py', range: '175', label: 'gate_proj L175' }],
        'gateProj'),
      F(0.8, 'up_proj  ·  576 → 1536',
        'The "value" branch. Multiplied elementwise with SiLU(gate) further up.',
        t => {
          M.upProj.userData.box.material.emissive = new THREE.Color().setHSL(0.55, 0.6, easeOut(t) * 0.4);
        }, [{ file: 'model.py', range: '176', label: 'up_proj L176' }],
        'upProj'),
    ],
  },
  {
    name: 'F.silu(gate)',
    shot: 'silu',
    frames: [
      F(1.1, 'silu(z) = z · σ(z)',
        'Smooth, lets small negatives through (unlike ReLU). The canonical activation of modern dense LLMs (Llama, Mistral, SmolLM, Qwen).',
        t => {
          M.silu.userData.box.material.emissive = new THREE.Color().setHSL(0.4, 0.6, easeOut(t) * 0.4);
        }, [], 'silu', 'silu'),
    ],
  },
  {
    name: 'silu(gate) ⊙ up',
    shot: 'mul',
    frames: [
      F(1.2, 'The GLU gate fires',
        'silu(gate) and up are now combined elementwise. When silu(gate) is high, info passes; when low, it suppresses. This learnable gating is empirically ~1% better loss than plain ReLU MLP at the same param count.',
        t => {
          M.mul.scale.setScalar(1.0 + 0.18 * Math.sin(t * Math.PI * 2));
        }, [{ file: 'model.py', range: '180', label: 'fused expression L180' }],
        'mul', 'mul'),
    ],
  },
  {
    name: 'down_proj  ·  1536 → 576',
    shot: 'downProj',
    frames: [
      F(1.2, 'Collapse back to model dim',
        'Linear 1536 → 576 (884,736 params, no bias). The MLP holds ~75% of all per-block params (2.65M of 3.54M).',
        t => {
          M.downProj.userData.box.material.emissive = new THREE.Color().setHSL(0.55, 0.6, easeOut(t) * 0.4);
          M.mul.scale.setScalar(1.0);
        }, [{ file: 'model.py', range: '177', label: 'down_proj L177' }],
        'downProj', 'downProj', 'mlp'),
    ],
  },
  {
    name: 'residual ⊕  ·  x_out  →  next block',
    shot: 'res2',
    frames: [
      F(1.3, 'Second skip connection closes',
        'x_out = x_mid + mlp_out. Trace: x_out RMS jumps to ~1.5–2.3 — the MLP punched hard. This is normal: per-block residual magnitude grows as the stream accumulates context across 30 blocks.',
        t => {
          M.res2.scale.setScalar(1.0 + 0.18 * Math.sin(t * Math.PI * 2));
          if (M.arc2) M.arc2.material.opacity = 0.4 + 0.5 * Math.abs(Math.sin(t * Math.PI * 2));
        }, [{ file: 'model.py', range: '198', label: 'second residual L198' }],
        'res2', 'res2', 'x_out'),
      F(1.0, 'x_out hands off to the next block',
        'Same shape as x_in: (B=1, T=5, 576). It now becomes the x_in of block 1, then 2, then 3, … all the way to block 29. After block 29: one more RMSNorm and the LM head.',
        t => {
          M.res2.scale.setScalar(1.0);
          if (M.nextBlockArrow) M.nextBlockArrow.setLength(0.6 + 0.4 * easeOut(t), 0.18, 0.10);
        }, [], null, 'hiddenOut', 'x_out'),
    ],
  },
  {
    name: 'Block summary',
    shot: 'outro',
    frames: [
      F(1.0, 'Per-block params: 3,540,096',
        '·  attn (q, k, v, o):  884,736        ·  mlp (gate, up, down): 2,654,208        ·  two RMSNorms:       1,152',
        t => { setShot('outro'); }, [], null, 'wide'),
      F(1.0, '× 30 identical blocks = 106,202,880',
        'Plus embed_tokens (28,311,552, tied to lm_head) and the final RMSNorm (576) = 134,515,008 total. That is the "135M" branding.',
        t => {}, [{ file: 'model.py', range: '243-247', label: 'tied weights' }], null, 'wide'),
      F(1.0, 'End',
        'Every shape, count, and numeric value shown was verified against model.py and a live forward pass on the official SmolLM2-135M weights. Scrub backwards to re-watch any step.',
        t => {}, [], null, 'wide'),
    ],
  },
];

// ============================== RUNTIME ==============================
const READ_WPS = 6.0;
const PAD_SEC  = 0.05;
const MAX_DUR  = 2.5;
SCENES.forEach(s => {
  s.frames.forEach(f => {
    const text = ((f.cap || '') + ' ' + (f.narr || '')).trim();
    const words = text.length ? text.split(/\s+/).length : 0;
    const readDur = words / READ_WPS + (words > 0 ? PAD_SEC : 0);
    f.dur = Math.min(MAX_DUR, Math.max(f.dur, readDur));
  });
});

let TOTAL_DUR = 0;
SCENES.forEach((s, si) => {
  s.startT = TOTAL_DUR;
  s.frames.forEach((f, fi) => {
    f.startT = TOTAL_DUR;
    f.sceneIdx = si;
    f.frameIdx = fi;
    TOTAL_DUR += f.dur;
  });
  s.endT = TOTAL_DUR;
});

const TOTAL_FRAMES = SCENES.reduce((s, sc) => s + sc.frames.length, 0);
console.log(`[cinema_v2] ${SCENES.length} scenes · ${TOTAL_FRAMES} frames · ${TOTAL_DUR.toFixed(1)}s total`);

let currentT = 0;
let playing = false;
let activeFrameStart = -1;

function findCurrentFrame(t) {
  for (let si = 0; si < SCENES.length; si++) {
    const s = SCENES[si];
    if (t >= s.startT && t <= s.endT) {
      for (let fi = 0; fi < s.frames.length; fi++) {
        const f = s.frames[fi];
        if (t >= f.startT && t < f.startT + f.dur) {
          return { scene: s, sceneIdx: si, frame: f, frameIdx: fi };
        }
      }
      const lastFi = s.frames.length - 1;
      return { scene: s, sceneIdx: si, frame: s.frames[lastFi], frameIdx: lastFi };
    }
  }
  return null;
}

// ============================== UI ==============================
const els = {
  play:    document.getElementById('ci-play'),
  prev:    document.getElementById('ci-prev'),
  next:    document.getElementById('ci-next'),
  time:    document.getElementById('ci-time'),
  scrub:   document.getElementById('ci-scrub'),
  speed:   document.getElementById('ci-speed'),
  title:   document.getElementById('ci-title'),
  shape:   document.getElementById('ci-shape'),
  narr:    document.getElementById('ci-narr'),
  links:   document.getElementById('ci-links'),
  list:    document.getElementById('ci-list'),
  sceneNum:document.getElementById('ci-scene-num'),
  sceneTime:document.getElementById('ci-scene-time'),
  statA:   document.getElementById('ci-stat-a'),
  statB:   document.getElementById('ci-stat-b'),
  statC:   document.getElementById('ci-stat-c'),
  statAL:  document.getElementById('ci-stat-a-l'),
  statBL:  document.getElementById('ci-stat-b-l'),
  statCL:  document.getElementById('ci-stat-c-l'),
  progFill:document.getElementById('ci-progress-fill'),
  progMk:  document.getElementById('ci-progress-markers'),
  chNum:   document.getElementById('ci-chapter-num'),
  chTitle: document.getElementById('ci-chapter-title'),
  shapeBadge: document.getElementById('ci-shape-badge'),
  capTitle:document.getElementById('ci-cap-title'),
  capNarr: document.getElementById('ci-cap-narr'),
  capShape:document.getElementById('ci-cap-shape'),
  capBox:  document.getElementById('ci-caption'),
  timecode:document.getElementById('ci-timecode'),
};

function fmtTime(t) {
  const m = Math.floor(t / 60);
  const s = t - 60 * m;
  return `${m}:${s.toFixed(1).padStart(4, '0')}`;
}

if (els.scrub) { els.scrub.max = TOTAL_DUR; els.scrub.step = 0.05; }
if (els.time) els.time.textContent = `0:00.0 / ${fmtTime(TOTAL_DUR)}`;

if (els.list) {
  els.list.innerHTML = '';
  SCENES.forEach((s, si) => {
    const li = document.createElement('li');
    li.className = 'flex items-baseline gap-2 px-2 py-1 rounded-md cursor-pointer';
    li.style.cssText = 'transition: background .15s ease;';
    li.innerHTML = `<span class="font-mono text-[11px]" style="color: var(--slate); min-width: 2.4rem;">${s.startT.toFixed(0)}s</span><span style="color: var(--ink);">${s.name}</span><span class="font-mono text-[10px]" style="color: var(--slate);">·${s.frames.length} fr</span>`;
    li.addEventListener('mouseover', () => li.style.background = '#F3F4F6');
    li.addEventListener('mouseout',  () => { if (!li.classList.contains('active')) li.style.background = 'transparent'; });
    li.addEventListener('click', () => seek(s.startT + 0.01, true));
    els.list.appendChild(li);
  });
}

if (els.progMk) {
  els.progMk.innerHTML = '';
  SCENES.forEach(s => {
    const m = document.createElement('div');
    m.className = 'absolute top-0 bottom-0';
    m.style.cssText = `left: ${100 * s.startT / TOTAL_DUR}%; width: 1px; background: rgba(245,158,11,0.45);`;
    els.progMk.appendChild(m);
  });
}

function applyFrame(info) {
  const { scene: sc, sceneIdx: si, frame: f, frameIdx: fi } = info;
  highlight(f.hl);
  const shotName = f.cam || sc.shot;
  if (shotName) setShot(shotName);
  if (f.stage) showStage(f.stage);

  // Resolve flow position from (in priority): explicit f.flow, f.hl, f.stage, shot.
  const flowKey = f.flow || f.hl || f.stage || shotName;
  if (flowKey && (flowKey in STAGE_Y)) flowYTarget = STAGE_Y[flowKey];
  followFlow = !NON_FOLLOW_SHOTS.has(shotName);

  const cap = f.cap || sc.name;
  const narr = f.narr || '';
  if (els.capTitle) els.capTitle.textContent = cap;
  if (els.capNarr)  els.capNarr.textContent = narr;
  if (els.capShape) els.capShape.textContent = `scene ${si + 1}/${SCENES.length} · frame ${fi + 1}/${sc.frames.length} · ${f.dur.toFixed(1)}s`;
  if (els.title)    els.title.textContent = sc.name;
  if (els.shape)    els.shape.textContent = cap;
  if (els.narr)     els.narr.textContent = narr;
  if (els.sceneNum) els.sceneNum.textContent = `scene ${si + 1} / ${SCENES.length}`;
  if (els.sceneTime)els.sceneTime.textContent = `frame ${fi + 1}/${sc.frames.length}`;
  if (els.chNum)    els.chNum.textContent = `scene ${si + 1}/${SCENES.length}`;
  if (els.chTitle)  els.chTitle.textContent = sc.name;
  if (els.shapeBadge) els.shapeBadge.textContent = cap;
  if (els.statA) els.statA.textContent = `${si + 1}`;
  if (els.statAL) els.statAL.textContent = 'scene';
  if (els.statB) els.statB.textContent = `${fi + 1}/${sc.frames.length}`;
  if (els.statBL) els.statBL.textContent = 'frame';
  if (els.statC) els.statC.textContent = f.dur.toFixed(1) + 's';
  if (els.statCL) els.statCL.textContent = 'duration';

  if (els.capBox) {
    els.capBox.style.opacity = '0';
    els.capBox.style.transform = 'translate(-50%, 8px)';
    requestAnimationFrame(() => {
      els.capBox.style.transition = 'opacity .25s ease, transform .25s ease';
      els.capBox.style.opacity = '1';
      els.capBox.style.transform = 'translate(-50%, 0)';
    });
  }

  if (els.list) {
    Array.from(els.list.children).forEach((li, j) => {
      const isActive = (j === si);
      li.classList.toggle('active', isActive);
      li.style.background = isActive ? '#FEF3C7' : 'transparent';
      li.style.fontWeight = isActive ? '600' : '400';
    });
  }

  if (els.links) {
    els.links.innerHTML = '';
    (f.links || []).forEach(lk => {
      const btn = document.createElement('button');
      btn.className = 'pill';
      btn.style.cssText = 'cursor: pointer; text-decoration: underline dotted; text-underline-offset: 2px;';
      btn.textContent = lk.label;
      btn.title = `Open ${lk.file} at ${lk.range}`;
      btn.addEventListener('click', () => window.openSource && window.openSource(lk.file, lk.range));
      els.links.appendChild(btn);
    });
  }
}

function seek(t, force = false) {
  currentT = clamp(t, 0, TOTAL_DUR);
  if (els.scrub) els.scrub.value = currentT;
  if (els.time) els.time.textContent = `${fmtTime(currentT)} / ${fmtTime(TOTAL_DUR)}`;
  if (els.progFill) els.progFill.style.width = (100 * currentT / TOTAL_DUR) + '%';
  if (els.timecode) els.timecode.textContent = fmtTime(currentT);
  const info = findCurrentFrame(currentT);
  if (info && (info.frame.startT !== activeFrameStart || force)) {
    activeFrameStart = info.frame.startT;
    applyFrame(info);
  }
}

function setPlaying(p) {
  playing = p;
  if (els.play) {
    // The Play button has an SVG icon + a <span> for the label. Update only
    // the label text so we don't blow away the SVG.
    const labelSpan = els.play.querySelector('span');
    if (labelSpan) labelSpan.textContent = p ? 'Pause' : 'Play';
    else els.play.textContent = p ? 'Pause' : 'Play';
  }
}

if (els.play) els.play.addEventListener('click', () => {
  if (currentT >= TOTAL_DUR) seek(0, true);
  setPlaying(!playing);
});
if (els.prev) els.prev.addEventListener('click', () => {
  const info = findCurrentFrame(currentT);
  if (!info) return;
  const target = SCENES[Math.max(0, info.sceneIdx - 1)];
  seek(target.startT + 0.01, true);
});
if (els.next) els.next.addEventListener('click', () => {
  const info = findCurrentFrame(currentT);
  if (!info) return;
  const target = SCENES[Math.min(SCENES.length - 1, info.sceneIdx + 1)];
  seek(target.startT + 0.01, true);
});
if (els.scrub) els.scrub.addEventListener('input', () => seek(parseFloat(els.scrub.value), false));

// ============================== FULLSCREEN ==============================
// Three entry points: window.cinemaFullscreen() (programmatic), the #ci-4k
// button, and the 'f' / 'F' keyboard shortcut.
function cinemaFullscreen() {
  if (!document.fullscreenElement) {
    container.requestFullscreen?.({ navigationUI: 'hide' }).catch(() => {});
  } else {
    document.exitFullscreen?.();
  }
}
window.cinemaFullscreen = cinemaFullscreen;

const btn4k = document.getElementById('ci-4k');
if (btn4k) btn4k.addEventListener('click', cinemaFullscreen);

// Overlay exit button — child of the fullscreen target so it stays clickable
// in fullscreen. The transport-bar #ci-4k button is hidden when the stage
// goes fullscreen, leaving users with no visible way out.
const exitFsBtn = document.createElement('button');
exitFsBtn.type = 'button';
exitFsBtn.id = 'ci-exit-fs';
exitFsBtn.setAttribute('aria-label', 'Exit fullscreen');
exitFsBtn.title = 'Exit fullscreen (Esc or F)';
exitFsBtn.innerHTML =
  '<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">' +
    '<path d="M9 3v6H3M15 3v6h6M9 21v-6H3M15 21v-6h6"/>' +
  '</svg><span style="margin-left:6px">Exit fullscreen</span>';
exitFsBtn.style.cssText = [
  'display: none',
  'position: absolute',
  'top: 12px',
  'right: 12px',
  'z-index: 10',
  'padding: 8px 12px',
  'border-radius: 10px',
  'background: rgba(15,23,42,0.85)',
  'border: 1px solid rgba(245,158,11,0.45)',
  'color: #F9FAFB',
  'font: 600 12px/1 Geist, Inter, system-ui, sans-serif',
  'cursor: pointer',
  'align-items: center',
  'backdrop-filter: blur(8px)',
  'pointer-events: auto',
].join(';');
exitFsBtn.addEventListener('click', (ev) => {
  ev.stopPropagation();
  cinemaFullscreen();
});
container.appendChild(exitFsBtn);

// Overlays (chapter chip, shape badge, caption strip, timecode) live as
// siblings of #cinema-stage so they disappear in fullscreen. Reparent them
// into the stage on enter, restore them on exit.
const overlayState = ['ci-chapter', 'ci-shape-badge', 'ci-caption', 'ci-timecode']
  .map((id) => {
    const el = document.getElementById(id);
    if (!el || el.parentNode === container) return null;
    return { el, parent: el.parentNode, next: el.nextSibling };
  })
  .filter(Boolean);

function moveOverlaysToStage() {
  for (const o of overlayState) container.appendChild(o.el);
}
function restoreOverlays() {
  for (const o of overlayState) {
    if (o.next && o.next.parentNode === o.parent) {
      o.parent.insertBefore(o.el, o.next);
    } else {
      o.parent.appendChild(o.el);
    }
  }
}

document.addEventListener('fullscreenchange', () => {
  const inFs = document.fullscreenElement === container;
  if (btn4k) {
    const span = btn4k.querySelector('span');
    span && (span.textContent = inFs ? 'Exit fullscreen' : 'Fullscreen');
  }
  exitFsBtn.style.display = inFs ? 'inline-flex' : 'none';
  if (inFs) moveOverlaysToStage(); else restoreOverlays();
  requestAnimationFrame(() => {
    renderer.setSize(container.clientWidth, container.clientHeight);
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
  });
});

window.addEventListener('keydown', (ev) => {
  if (ev.key !== 'f' && ev.key !== 'F') return;
  const t = ev.target;
  if (t && (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.isContentEditable)) return;
  ev.preventDefault();
  cinemaFullscreen();
});

seek(0, true);

// ============================== ANIMATION LOOP ==============================
let last = performance.now();
function tick(now) {
  const dt = Math.min(0.05, (now - last) / 1000);
  last = now;
  if (playing) {
    const sp = els.speed ? parseFloat(els.speed.value) : 1;
    seek(currentT + dt * sp, false);
    if (currentT >= TOTAL_DUR) setPlaying(false);
  }
  const info = findCurrentFrame(currentT);
  if (info) {
    const t01 = clamp((currentT - info.frame.startT) / info.frame.dur, 0, 1);
    try { info.frame.do(t01); } catch (e) { /* keep loop alive */ }
  }

  // Live flow arrow — slide to its target y, with a gentle pulse.
  M.flow.position.y = lerp(M.flow.position.y, flowYTarget, 0.07);
  const pulse = 1.0 + 0.18 * Math.sin(now * 0.005);
  if (M.flowHalo) M.flowHalo.scale.setScalar(pulse);
  if (M.flowBall) M.flowBall.material.emissiveIntensity = 1.2 + 0.5 * Math.sin(now * 0.005);

  // Camera tracks the flow on close-up shots.
  if (followFlow) {
    camTarget.pos.y  = M.flow.position.y;
    camTarget.look.y = M.flow.position.y;
  }
  camera.position.lerp(camTarget.pos, 0.085);
  controls.target.lerp(camTarget.look, 0.085);
  controls.update();
  renderer.render(scene, camera);
  requestAnimationFrame(tick);
}
requestAnimationFrame(tick);

})();
