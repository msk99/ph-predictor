// pH Predictor PWA — Core Application
// Uses ONNX Runtime Web (WASM) for in-browser inference

'use strict';

// ---- Configuration ----
const MODEL_PATH = './model/model.onnx';
const IMAGE_SIZE = 224;
const SAT_THRESHOLD = 35;
const CROP_PADDING = 0.08;
const PH_MIN = 5;
const PH_MAX = 10;

// ---- State ----
let session = null;
let modelReady = false;

// ---- DOM Elements ----
const statusDot = document.getElementById('statusDot');
const statusText = document.getElementById('statusText');
const progressContainer = document.getElementById('progressContainer');
const progressFill = document.getElementById('progressFill');
const progressText = document.getElementById('progressText');
const btnCamera = document.getElementById('btnCamera');
const btnUpload = document.getElementById('btnUpload');
const cameraInput = document.getElementById('cameraInput');
const fileInput = document.getElementById('fileInput');
const previewArea = document.getElementById('previewArea');
const previewOriginal = document.getElementById('previewOriginal');
const previewCropped = document.getElementById('previewCropped');
const spinner = document.getElementById('spinner');
const resultSection = document.getElementById('resultSection');
const resultValue = document.getElementById('resultValue');
const phMarker = document.getElementById('phMarker');
const resultTime = document.getElementById('resultTime');
const previewCroppedLabel = document.getElementById('previewCroppedLabel');
const btnAdjust = document.getElementById('btnAdjust');
const cropEditor = document.getElementById('cropEditor');
const cropStage = document.getElementById('cropStage');
const cropCanvas = document.getElementById('cropCanvas');
const cropBoxEl = document.getElementById('cropBox');
const btnCropReset = document.getElementById('btnCropReset');
const btnCropCancel = document.getElementById('btnCropCancel');
const btnCropConfirm = document.getElementById('btnCropConfirm');

// ---- Model Loading ----
async function loadModel() {
  try {
    statusText.textContent = 'Downloading model (first time only)...';
    progressContainer.style.display = 'block';

    // Fetch with progress tracking
    const response = await fetch(MODEL_PATH);
    if (!response.ok) throw new Error(`HTTP ${response.status}`);

    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength) : 80 * 1024 * 1024;
    const reader = response.body.getReader();
    const chunks = [];
    let received = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      chunks.push(value);
      received += value.length;
      const pct = Math.min(100, Math.round((received / total) * 100));
      progressFill.style.width = pct + '%';
      progressText.textContent = `${(received / 1024 / 1024).toFixed(1)} / ${(total / 1024 / 1024).toFixed(1)} MB`;
    }

    const modelData = new Uint8Array(received);
    let offset = 0;
    for (const chunk of chunks) {
      modelData.set(chunk, offset);
      offset += chunk.length;
    }

    // Create ONNX Runtime session
    statusText.textContent = 'Initialising model...';
    session = await ort.InferenceSession.create(modelData.buffer, {
      executionProviders: ['wasm'],
      graphOptimizationLevel: 'all',
    });

    modelReady = true;
    statusDot.className = 'status-dot ready';
    statusText.textContent = 'Ready \u2014 take a photo or upload an image';
    progressContainer.style.display = 'none';
    btnCamera.disabled = false;
    btnUpload.disabled = false;

    console.log('ONNX model loaded. Inputs:', session.inputNames, 'Outputs:', session.outputNames);
  } catch (err) {
    statusDot.className = 'status-dot error';
    statusText.textContent = 'Failed to load model: ' + err.message;
    console.error('Model load error:', err);
  }
}

// ---- Image Processing (Auto-crop) ----

function rgbToHsv(r, g, b) {
  r /= 255; g /= 255; b /= 255;
  const max = Math.max(r, g, b), min = Math.min(r, g, b);
  const d = max - min;
  let s = max === 0 ? 0 : d / max;
  return s * 255; // Only need saturation
}

function autoCropStrips(canvas) {
  const w = canvas.width, h = canvas.height;

  // Downscale for processing
  const scale = 500 / Math.max(w, h);
  const sw = Math.round(w * scale), sh = Math.round(h * scale);

  const smallCanvas = document.createElement('canvas');
  smallCanvas.width = sw;
  smallCanvas.height = sh;
  const sCtx = smallCanvas.getContext('2d', { willReadFrequently: true });
  sCtx.drawImage(canvas, 0, 0, sw, sh);
  const pixels = sCtx.getImageData(0, 0, sw, sh).data;

  // Build saturation mask
  const mask = new Uint8Array(sw * sh);
  for (let i = 0; i < sw * sh; i++) {
    const sat = rgbToHsv(pixels[i * 4], pixels[i * 4 + 1], pixels[i * 4 + 2]);
    mask[i] = sat > SAT_THRESHOLD ? 1 : 0;
  }

  // Morphological closing (dilate then erode) to connect the two strips
  const kernelSize = Math.max(3, Math.floor(Math.max(sw, sh) / 15));
  const halfK = Math.floor(kernelSize / 2);

  function dilate(src) {
    const dst = new Uint8Array(sw * sh);
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        let found = false;
        for (let dy = -halfK; dy <= halfK && !found; dy++) {
          for (let dx = -halfK; dx <= halfK && !found; dx++) {
            const ny = y + dy, nx = x + dx;
            if (ny >= 0 && ny < sh && nx >= 0 && nx < sw && src[ny * sw + nx]) {
              found = true;
            }
          }
        }
        dst[y * sw + x] = found ? 1 : 0;
      }
    }
    return dst;
  }

  function erode(src) {
    const dst = new Uint8Array(sw * sh);
    for (let y = 0; y < sh; y++) {
      for (let x = 0; x < sw; x++) {
        let allSet = true;
        for (let dy = -halfK; dy <= halfK && allSet; dy++) {
          for (let dx = -halfK; dx <= halfK && allSet; dx++) {
            const ny = y + dy, nx = x + dx;
            // Out-of-bounds treated as 0 (not set)
            if (ny < 0 || ny >= sh || nx < 0 || nx >= sw || !src[ny * sw + nx]) {
              allSet = false;
            }
          }
        }
        dst[y * sw + x] = allSet ? 1 : 0;
      }
    }
    return dst;
  }

  // Closing = dilate twice, then erode twice (matches Python: iterations=2)
  let closed = dilate(dilate(mask));
  closed = erode(erode(closed));

  // Fill holes (flood fill from edges, then invert)
  const filled = new Uint8Array(closed);
  const visited = new Uint8Array(sw * sh);
  const queue = [];
  // Seed from all edges
  for (let x = 0; x < sw; x++) {
    if (!closed[x]) queue.push(x);                           // top row
    if (!closed[(sh - 1) * sw + x]) queue.push((sh - 1) * sw + x); // bottom row
  }
  for (let y = 0; y < sh; y++) {
    if (!closed[y * sw]) queue.push(y * sw);                  // left col
    if (!closed[y * sw + sw - 1]) queue.push(y * sw + sw - 1); // right col
  }
  for (const idx of queue) visited[idx] = 1;
  while (queue.length > 0) {
    const ci = queue.pop();
    const cy = Math.floor(ci / sw), cx = ci % sw;
    const neighbors = [
      cy > 0 ? (cy - 1) * sw + cx : -1,
      cy < sh - 1 ? (cy + 1) * sw + cx : -1,
      cx > 0 ? cy * sw + (cx - 1) : -1,
      cx < sw - 1 ? cy * sw + (cx + 1) : -1,
    ];
    for (const ni of neighbors) {
      if (ni >= 0 && !visited[ni] && !closed[ni]) {
        visited[ni] = 1;
        queue.push(ni);
      }
    }
  }
  // Any unvisited background pixel is inside a hole — fill it
  for (let i = 0; i < sw * sh; i++) {
    if (!closed[i] && !visited[i]) filled[i] = 1;
  }
  closed = filled;

  // Connected component labeling (flood fill) — track bounding box & centroid per component
  const labels = new Int32Array(sw * sh);
  let nextLabel = 1;
  const components = {}; // label → { size, ySum, rmin, rmax, cmin, cmax }

  for (let y = 0; y < sh; y++) {
    for (let x = 0; x < sw; x++) {
      const idx = y * sw + x;
      if (closed[idx] && !labels[idx]) {
        const label = nextLabel++;
        let size = 0, ySum = 0;
        let crmin = sh, crmax = 0, ccmin = sw, ccmax = 0;
        const queue = [idx];
        labels[idx] = label;

        while (queue.length > 0) {
          const ci = queue.pop();
          size++;
          const cy = Math.floor(ci / sw), cx = ci % sw;
          ySum += cy;
          if (cy < crmin) crmin = cy;
          if (cy > crmax) crmax = cy;
          if (cx < ccmin) ccmin = cx;
          if (cx > ccmax) ccmax = cx;
          const neighbors = [
            cy > 0 ? (cy - 1) * sw + cx : -1,
            cy < sh - 1 ? (cy + 1) * sw + cx : -1,
            cx > 0 ? cy * sw + (cx - 1) : -1,
            cx < sw - 1 ? cy * sw + (cx + 1) : -1,
          ];
          for (const ni of neighbors) {
            if (ni >= 0 && closed[ni] && !labels[ni]) {
              labels[ni] = label;
              queue.push(ni);
            }
          }
        }
        components[label] = { size, yCenter: ySum / size, rmin: crmin, rmax: crmax, cmin: ccmin, cmax: ccmax };
      }
    }
  }

  // Filter components: ignore tiny ones (<1% of image) and ones whose
  // centroid is in the bottom 20% (likely background clutter, not strips)
  const minSize = sw * sh * 0.01;
  const maxYCenter = sh * 0.80;
  let bestLabel = 0, bestSize = 0;
  for (const [label, comp] of Object.entries(components)) {
    if (comp.size >= minSize && comp.yCenter <= maxYCenter && comp.size > bestSize) {
      bestSize = comp.size;
      bestLabel = parseInt(label);
    }
  }

  // Fallback: if no component passed filtering, use largest overall
  if (!bestLabel) {
    for (const [label, comp] of Object.entries(components)) {
      if (comp.size > bestSize) {
        bestSize = comp.size;
        bestLabel = parseInt(label);
      }
    }
  }

  if (!bestLabel) return { canvas, box: { x: 0, y: 0, w, h } };

  // Use bounding box from the selected component
  let rmin = components[bestLabel].rmin;
  let rmax = components[bestLabel].rmax;
  let cmin = components[bestLabel].cmin;
  let cmax = components[bestLabel].cmax;

  // Also merge any nearby components that overlap horizontally (the two strips)
  for (const [label, comp] of Object.entries(components)) {
    const lab = parseInt(label);
    if (lab === bestLabel || comp.size < minSize) continue;
    // Check if this component overlaps horizontally with the best one and is vertically close
    const hOverlap = comp.cmin <= cmax && comp.cmax >= cmin;
    const vGap = Math.max(0, Math.min(comp.rmin, rmin) === comp.rmin ? rmin - comp.rmax : comp.rmin - rmax);
    if (hOverlap && vGap < sh * 0.3) {
      rmin = Math.min(rmin, comp.rmin);
      rmax = Math.max(rmax, comp.rmax);
      cmin = Math.min(cmin, comp.cmin);
      cmax = Math.max(cmax, comp.cmax);
    }
  }

  if (rmax <= rmin || cmax <= cmin) return { canvas, box: { x: 0, y: 0, w, h } };

  // Scale to original coords
  rmin = Math.floor(rmin / scale);
  rmax = Math.floor(rmax / scale);
  cmin = Math.floor(cmin / scale);
  cmax = Math.floor(cmax / scale);

  // Padding
  const pad = Math.floor(CROP_PADDING * Math.min(h, w));
  rmin = Math.max(0, rmin - pad);
  rmax = Math.min(h - 1, rmax + pad);
  cmin = Math.max(0, cmin - pad);
  cmax = Math.min(w - 1, cmax + pad);

  const cropW = cmax - cmin + 1, cropH = rmax - rmin + 1;
  const cropCanvas = document.createElement('canvas');
  cropCanvas.width = cropW;
  cropCanvas.height = cropH;
  cropCanvas.getContext('2d').drawImage(canvas, cmin, rmin, cropW, cropH, 0, 0, cropW, cropH);
  return { canvas: cropCanvas, box: { x: cmin, y: rmin, w: cropW, h: cropH } };
}

function cropFromBox(sourceCanvas, box) {
  const c = document.createElement('canvas');
  c.width = Math.max(1, Math.round(box.w));
  c.height = Math.max(1, Math.round(box.h));
  c.getContext('2d').drawImage(
    sourceCanvas,
    Math.round(box.x), Math.round(box.y), Math.round(box.w), Math.round(box.h),
    0, 0, c.width, c.height
  );
  return c;
}

// ---- Preprocessing for Xception ----

function preprocessForXception(canvas) {
  const resized = document.createElement('canvas');
  resized.width = IMAGE_SIZE;
  resized.height = IMAGE_SIZE;
  const ctx = resized.getContext('2d');
  ctx.drawImage(canvas, 0, 0, IMAGE_SIZE, IMAGE_SIZE);

  const pixels = ctx.getImageData(0, 0, IMAGE_SIZE, IMAGE_SIZE).data;

  // Xception preprocess_input: (x / 127.5) - 1.0
  // ONNX expects NHWC layout (same as Keras)
  const input = new Float32Array(1 * IMAGE_SIZE * IMAGE_SIZE * 3);
  for (let i = 0; i < IMAGE_SIZE * IMAGE_SIZE; i++) {
    input[i * 3 + 0] = pixels[i * 4 + 0] / 127.5 - 1.0;
    input[i * 3 + 1] = pixels[i * 4 + 1] / 127.5 - 1.0;
    input[i * 3 + 2] = pixels[i * 4 + 2] / 127.5 - 1.0;
  }
  return input;
}

// ---- ONNX Inference ----

async function runInference(inputData) {
  const inputName = session.inputNames[0];
  const tensor = new ort.Tensor('float32', inputData, [1, IMAGE_SIZE, IMAGE_SIZE, 3]);
  const feeds = {};
  feeds[inputName] = tensor;

  const results = await session.run(feeds);
  const outputName = session.outputNames[0];
  const output = results[outputName];
  return output.data[0];
}

// ---- Main Prediction Pipeline ----

// Shared state so the crop editor can re-crop without re-loading the image
const cropState = {
  sourceCanvas: null,
  autoBox: null,
  currentBox: null,
};

async function runFromCroppedCanvas(cropped, label) {
  const startTime = performance.now();
  previewCropped.src = cropped.toDataURL('image/jpeg', 0.8);
  previewArea.style.display = 'block';

  const inputData = preprocessForXception(cropped);
  const pH = await runInference(inputData);
  const elapsed = performance.now() - startTime;

  resultValue.textContent = pH.toFixed(2);
  resultSection.style.display = 'block';

  const phClamped = Math.max(PH_MIN - 0.5, Math.min(PH_MAX + 0.5, pH));
  const markerPos = ((phClamped - (PH_MIN - 1)) / (PH_MAX - PH_MIN + 2)) * 100;
  phMarker.style.left = markerPos + '%';

  resultTime.textContent = `Inference: ${elapsed.toFixed(0)} ms${label ? ' \u2014 ' + label : ''}`;
}

async function predictFromImage(imgElement) {
  const canvas = document.createElement('canvas');
  canvas.width = imgElement.naturalWidth || imgElement.width;
  canvas.height = imgElement.naturalHeight || imgElement.height;
  canvas.getContext('2d').drawImage(imgElement, 0, 0);

  previewOriginal.src = canvas.toDataURL('image/jpeg', 0.8);

  const { canvas: cropped, box } = autoCropStrips(canvas);
  cropState.sourceCanvas = canvas;
  cropState.autoBox = box;
  cropState.currentBox = { ...box };

  btnAdjust.style.display = 'inline-flex';
  await runFromCroppedCanvas(cropped, 'auto');
}

// ---- Event Handlers ----

function handleImageSelect(file) {
  if (!file || !modelReady) return;

  spinner.style.display = 'block';
  resultSection.style.display = 'none';
  previewArea.style.display = 'none';
  cropEditor.hidden = true;
  btnAdjust.style.display = 'none';
  previewCroppedLabel.textContent = 'Auto-cropped';

  const img = new Image();
  img.onload = async () => {
    try {
      await predictFromImage(img);
    } catch (err) {
      statusDot.className = 'status-dot error';
      statusText.textContent = 'Prediction failed: ' + err.message;
      console.error(err);
    } finally {
      spinner.style.display = 'none';
    }
  };
  img.onerror = () => {
    spinner.style.display = 'none';
    statusText.textContent = 'Failed to load image';
  };
  img.src = URL.createObjectURL(file);
}

btnCamera.addEventListener('click', () => cameraInput.click());
btnUpload.addEventListener('click', () => fileInput.click());
cameraInput.addEventListener('change', e => {
  handleImageSelect(e.target.files[0]);
  e.target.value = ''; // Allow re-selecting same file
});
fileInput.addEventListener('change', e => {
  handleImageSelect(e.target.files[0]);
  e.target.value = '';
});

// ---- Manual Crop Editor ----

const editorState = {
  box: null,        // {x,y,w,h} in source-canvas pixels
  stageW: 0,        // displayed stage width in CSS pixels
  stageH: 0,        // displayed stage height in CSS pixels
  scale: 1,         // source px per stage px
  pointers: new Map(), // pointerId -> {x,y} in source px
  drag: null,       // { type: 'move'|'nw'|'ne'|'sw'|'se', startBox, startPt }
  pinch: null,      // { startBox, startDist, centerPx } for 2-finger pinch
};

const MIN_BOX = 40; // minimum box size in source px

function clampBox(b, sw, sh) {
  const w = Math.max(MIN_BOX, Math.min(b.w, sw));
  const h = Math.max(MIN_BOX, Math.min(b.h, sh));
  const x = Math.max(0, Math.min(b.x, sw - w));
  const y = Math.max(0, Math.min(b.y, sh - h));
  return { x, y, w, h };
}

function layoutStage() {
  const src = cropState.sourceCanvas;
  if (!src) return;
  // Fit source image into the available container width, preserving aspect ratio.
  const containerW = cropEditor.clientWidth - 32; // minus padding
  const maxH = Math.min(window.innerHeight * 0.6, 520);
  const aspect = src.height / src.width;
  let stageW = containerW;
  let stageH = stageW * aspect;
  if (stageH > maxH) {
    stageH = maxH;
    stageW = stageH / aspect;
  }
  editorState.stageW = stageW;
  editorState.stageH = stageH;
  editorState.scale = src.width / stageW; // source px per stage px
  cropStage.style.width = stageW + 'px';
  cropStage.style.height = stageH + 'px';
  cropCanvas.width = src.width;
  cropCanvas.height = src.height;
  cropCanvas.getContext('2d').drawImage(src, 0, 0);
  cropCanvas.style.width = stageW + 'px';
  cropCanvas.style.height = stageH + 'px';
  drawBox();
}

function drawBox() {
  const b = editorState.box;
  if (!b) return;
  const s = editorState.scale;
  cropBoxEl.style.left = (b.x / s) + 'px';
  cropBoxEl.style.top = (b.y / s) + 'px';
  cropBoxEl.style.width = (b.w / s) + 'px';
  cropBoxEl.style.height = (b.h / s) + 'px';
}

function clientToSource(evt) {
  const rect = cropStage.getBoundingClientRect();
  const sx = (evt.clientX - rect.left) * editorState.scale;
  const sy = (evt.clientY - rect.top) * editorState.scale;
  return { x: sx, y: sy };
}

function openCropEditor() {
  if (!cropState.sourceCanvas) return;
  editorState.box = clampBox(
    { ...cropState.currentBox },
    cropState.sourceCanvas.width,
    cropState.sourceCanvas.height
  );
  cropEditor.hidden = false;
  btnAdjust.style.display = 'none';
  // Layout after the editor is visible so clientWidth is correct
  requestAnimationFrame(layoutStage);
}

function closeCropEditor() {
  cropEditor.hidden = true;
  btnAdjust.style.display = 'inline-flex';
  editorState.pointers.clear();
  editorState.drag = null;
  editorState.pinch = null;
}

function onPointerDown(evt) {
  evt.preventDefault();
  cropStage.setPointerCapture(evt.pointerId);
  const pt = clientToSource(evt);
  editorState.pointers.set(evt.pointerId, pt);

  if (editorState.pointers.size === 2) {
    // Enter pinch mode
    const pts = [...editorState.pointers.values()];
    const dx = pts[1].x - pts[0].x, dy = pts[1].y - pts[0].y;
    editorState.pinch = {
      startBox: { ...editorState.box },
      startDist: Math.hypot(dx, dy) || 1,
    };
    editorState.drag = null;
    return;
  }

  // Single pointer — figure out what got hit
  const target = evt.target;
  const handleAttr = target && target.dataset ? target.dataset.handle : null;
  if (handleAttr) {
    editorState.drag = { type: handleAttr, startBox: { ...editorState.box }, startPt: pt };
  } else {
    const b = editorState.box;
    const inside = pt.x >= b.x && pt.x <= b.x + b.w && pt.y >= b.y && pt.y <= b.y + b.h;
    editorState.drag = { type: inside ? 'move' : 'move', startBox: { ...b }, startPt: pt };
    if (!inside) {
      // If user taps outside, recentre the box on the tap
      editorState.box = clampBox(
        { x: pt.x - b.w / 2, y: pt.y - b.h / 2, w: b.w, h: b.h },
        cropState.sourceCanvas.width,
        cropState.sourceCanvas.height
      );
      editorState.drag.startBox = { ...editorState.box };
      drawBox();
    }
  }
}

function onPointerMove(evt) {
  if (!editorState.pointers.has(evt.pointerId)) return;
  evt.preventDefault();
  const pt = clientToSource(evt);
  editorState.pointers.set(evt.pointerId, pt);

  const sw = cropState.sourceCanvas.width;
  const sh = cropState.sourceCanvas.height;

  if (editorState.pinch && editorState.pointers.size >= 2) {
    const pts = [...editorState.pointers.values()];
    const dx = pts[1].x - pts[0].x, dy = pts[1].y - pts[0].y;
    const dist = Math.hypot(dx, dy) || 1;
    const ratio = dist / editorState.pinch.startDist;
    const sb = editorState.pinch.startBox;
    const cx = sb.x + sb.w / 2, cy = sb.y + sb.h / 2;
    const newW = sb.w / ratio, newH = sb.h / ratio; // pinch out = zoom in = smaller crop box
    editorState.box = clampBox({ x: cx - newW / 2, y: cy - newH / 2, w: newW, h: newH }, sw, sh);
    drawBox();
    return;
  }

  if (!editorState.drag) return;
  const { type, startBox, startPt } = editorState.drag;
  const dx = pt.x - startPt.x, dy = pt.y - startPt.y;
  let nb;
  if (type === 'move') {
    nb = { x: startBox.x + dx, y: startBox.y + dy, w: startBox.w, h: startBox.h };
  } else {
    let x = startBox.x, y = startBox.y, w = startBox.w, h = startBox.h;
    if (type.includes('w')) { x = startBox.x + dx; w = startBox.w - dx; }
    if (type.includes('e')) { w = startBox.w + dx; }
    if (type.includes('n')) { y = startBox.y + dy; h = startBox.h - dy; }
    if (type.includes('s')) { h = startBox.h + dy; }
    if (w < MIN_BOX) { if (type.includes('w')) x = startBox.x + startBox.w - MIN_BOX; w = MIN_BOX; }
    if (h < MIN_BOX) { if (type.includes('n')) y = startBox.y + startBox.h - MIN_BOX; h = MIN_BOX; }
    nb = { x, y, w, h };
  }
  editorState.box = clampBox(nb, sw, sh);
  drawBox();
}

function onPointerUp(evt) {
  editorState.pointers.delete(evt.pointerId);
  if (editorState.pointers.size < 2) editorState.pinch = null;
  if (editorState.pointers.size === 0) editorState.drag = null;
  try { cropStage.releasePointerCapture(evt.pointerId); } catch (_) {}
}

function onWheel(evt) {
  evt.preventDefault();
  const b = editorState.box;
  const sw = cropState.sourceCanvas.width;
  const sh = cropState.sourceCanvas.height;
  // Zoom around the pointer so the point under the cursor stays put
  const pt = clientToSource(evt);
  const factor = evt.deltaY < 0 ? 0.9 : 1.1; // wheel up = zoom in = shrink box
  const newW = b.w * factor, newH = b.h * factor;
  const tx = (pt.x - b.x) / b.w;
  const ty = (pt.y - b.y) / b.h;
  const nx = pt.x - tx * newW;
  const ny = pt.y - ty * newH;
  editorState.box = clampBox({ x: nx, y: ny, w: newW, h: newH }, sw, sh);
  drawBox();
}

cropStage.addEventListener('pointerdown', onPointerDown);
cropStage.addEventListener('pointermove', onPointerMove);
cropStage.addEventListener('pointerup', onPointerUp);
cropStage.addEventListener('pointercancel', onPointerUp);
cropStage.addEventListener('wheel', onWheel, { passive: false });
window.addEventListener('resize', () => { if (!cropEditor.hidden) layoutStage(); });

btnAdjust.addEventListener('click', openCropEditor);
btnCropCancel.addEventListener('click', closeCropEditor);
btnCropReset.addEventListener('click', () => {
  if (!cropState.autoBox) return;
  editorState.box = clampBox(
    { ...cropState.autoBox },
    cropState.sourceCanvas.width,
    cropState.sourceCanvas.height
  );
  drawBox();
});
btnCropConfirm.addEventListener('click', async () => {
  if (!editorState.box || !cropState.sourceCanvas) return;
  btnCropConfirm.disabled = true;
  btnCropCancel.disabled = true;
  spinner.style.display = 'block';
  try {
    cropState.currentBox = { ...editorState.box };
    const cropped = cropFromBox(cropState.sourceCanvas, editorState.box);
    previewCroppedLabel.textContent = 'Manual crop';
    await runFromCroppedCanvas(cropped, 'manual crop');
    closeCropEditor();
  } catch (err) {
    console.error(err);
    statusText.textContent = 'Prediction failed: ' + err.message;
  } finally {
    btnCropConfirm.disabled = false;
    btnCropCancel.disabled = false;
    spinner.style.display = 'none';
  }
});

// ---- Service Worker ----
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js')
    .then(reg => console.log('SW registered:', reg.scope))
    .catch(err => console.log('SW registration failed:', err));
}

// ---- Initialize ----
loadModel();
