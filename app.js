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

  if (!bestLabel) return canvas;

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

  if (rmax <= rmin || cmax <= cmin) return canvas;

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
  return cropCanvas;
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

async function predictFromImage(imgElement) {
  const startTime = performance.now();

  // Draw to canvas
  const canvas = document.createElement('canvas');
  canvas.width = imgElement.naturalWidth || imgElement.width;
  canvas.height = imgElement.naturalHeight || imgElement.height;
  canvas.getContext('2d').drawImage(imgElement, 0, 0);

  // Show original
  previewOriginal.src = canvas.toDataURL('image/jpeg', 0.8);

  // Auto-crop
  const cropped = autoCropStrips(canvas);
  previewCropped.src = cropped.toDataURL('image/jpeg', 0.8);
  previewArea.style.display = 'block';

  // Preprocess & predict
  const inputData = preprocessForXception(cropped);
  const pH = await runInference(inputData);
  const elapsed = performance.now() - startTime;

  // Display
  resultValue.textContent = pH.toFixed(2);
  resultSection.style.display = 'block';

  const phClamped = Math.max(PH_MIN - 0.5, Math.min(PH_MAX + 0.5, pH));
  const markerPos = ((phClamped - (PH_MIN - 1)) / (PH_MAX - PH_MIN + 2)) * 100;
  phMarker.style.left = markerPos + '%';

  resultTime.textContent = `Inference: ${elapsed.toFixed(0)} ms`;
}

// ---- Event Handlers ----

function handleImageSelect(file) {
  if (!file || !modelReady) return;

  spinner.style.display = 'block';
  resultSection.style.display = 'none';
  previewArea.style.display = 'none';

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

// ---- Service Worker ----
if ('serviceWorker' in navigator) {
  navigator.serviceWorker.register('./sw.js')
    .then(reg => console.log('SW registered:', reg.scope))
    .catch(err => console.log('SW registration failed:', err));
}

// ---- Initialize ----
loadModel();
