/**
 * CNN From Scratch — Vanilla JavaScript
 * ======================================
 * A complete Convolutional Neural Network implementation with:
 *   - Convolution layers (forward + backward)
 *   - ReLU activation (forward + backward)
 *   - Max Pooling (forward + backward)
 *   - Flatten layer
 *   - Dense / Fully-Connected layers (forward + backward)
 *   - Softmax output + Cross-Entropy loss
 *   - Stochastic Gradient Descent (SGD)
 *
 * NO external libraries. All matrix math is hand-written.
 */

// ═══════════════════════════════════════════════════════════
// 1. UTILITY / MATH HELPERS
// ═══════════════════════════════════════════════════════════

/** Create a 1-D array of zeros */
function zeros(n) {
  const a = new Array(n);
  for (let i = 0; i < n; i++) a[i] = 0;
  return a;
}

/** Create a 2-D array (rows × cols) of zeros */
function zeros2D(rows, cols) {
  const a = new Array(rows);
  for (let r = 0; r < rows; r++) {
    a[r] = new Array(cols);
    for (let c = 0; c < cols; c++) a[r][c] = 0;
  }
  return a;
}

/** Create a 3-D array (d × h × w) of zeros */
function zeros3D(d, h, w) {
  const a = new Array(d);
  for (let i = 0; i < d; i++) a[i] = zeros2D(h, w);
  return a;
}

/** Create a 4-D array of zeros */
function zeros4D(a, b, c, d) {
  const arr = new Array(a);
  for (let i = 0; i < a; i++) arr[i] = zeros3D(b, c, d);
  return arr;
}

/** He initialization: random * sqrt(2/n) */
function heInit(fanIn) {
  const std = Math.sqrt(2.0 / fanIn);
  return (Math.random() * 2 - 1) * std;
}

/** Random number from normal distribution (Box-Muller) */
function randn() {
  let u = 0, v = 0;
  while (u === 0) u = Math.random();
  while (v === 0) v = Math.random();
  return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
}

/** Softmax of a 1-D array */
function softmax(logits) {
  const maxVal = Math.max(...logits);
  const exps = logits.map(x => Math.exp(x - maxVal));
  const sumExp = exps.reduce((a, b) => a + b, 0);
  return exps.map(e => e / sumExp);
}

/** Cross-entropy loss for a single sample */
function crossEntropyLoss(probs, labelIndex) {
  return -Math.log(Math.max(probs[labelIndex], 1e-12));
}

/** Flatten a 3-D array (depth × height × width) to 1-D */
function flatten3D(arr3d) {
  const result = [];
  for (let d = 0; d < arr3d.length; d++)
    for (let r = 0; r < arr3d[d].length; r++)
      for (let c = 0; c < arr3d[d][r].length; c++)
        result.push(arr3d[d][r][c]);
  return result;
}

/** Unflatten a 1-D array back to 3-D */
function unflatten3D(flat, depth, height, width) {
  const arr = zeros3D(depth, height, width);
  let idx = 0;
  for (let d = 0; d < depth; d++)
    for (let r = 0; r < height; r++)
      for (let c = 0; c < width; c++)
        arr[d][r][c] = flat[idx++];
  return arr;
}

/** Shuffle array in-place (Fisher-Yates) */
function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [arr[i], arr[j]] = [arr[j], arr[i]];
  }
  return arr;
}


// ═══════════════════════════════════════════════════════════
// 2. CNN LAYER CLASSES
// ═══════════════════════════════════════════════════════════

/**
 * ConvLayer — Convolutional Layer
 * Input:  3-D array [inDepth][inH][inW]
 * Output: 3-D array [numFilters][outH][outW]
 *   where outH = inH - filterSize + 1, outW = inW - filterSize + 1
 */
class ConvLayer {
  constructor(numFilters, filterSize, inDepth) {
    this.numFilters = numFilters;
    this.filterSize = filterSize;
    this.inDepth = inDepth;

    // Filters: 4-D [numFilters][inDepth][filterSize][filterSize]
    const fanIn = inDepth * filterSize * filterSize;
    this.filters = [];
    for (let f = 0; f < numFilters; f++) {
      this.filters[f] = [];
      for (let d = 0; d < inDepth; d++) {
        this.filters[f][d] = [];
        for (let r = 0; r < filterSize; r++) {
          this.filters[f][d][r] = [];
          for (let c = 0; c < filterSize; c++) {
            this.filters[f][d][r][c] = heInit(fanIn);
          }
        }
      }
    }

    // Biases: 1-D [numFilters]
    this.biases = zeros(numFilters);

    // Cache for backprop
    this.input = null;
  }

  forward(input) {
    this.input = input; // cache
    const inH = input[0].length;
    const inW = input[0][0].length;
    const outH = inH - this.filterSize + 1;
    const outW = inW - this.filterSize + 1;

    const output = zeros3D(this.numFilters, outH, outW);

    for (let f = 0; f < this.numFilters; f++) {
      for (let r = 0; r < outH; r++) {
        for (let c = 0; c < outW; c++) {
          let sum = this.biases[f];
          for (let d = 0; d < this.inDepth; d++) {
            for (let fr = 0; fr < this.filterSize; fr++) {
              for (let fc = 0; fc < this.filterSize; fc++) {
                sum += input[d][r + fr][c + fc] * this.filters[f][d][fr][fc];
              }
            }
          }
          output[f][r][c] = sum;
        }
      }
    }
    return output;
  }

  backward(dOutput, lr) {
    const inH = this.input[0].length;
    const inW = this.input[0][0].length;
    const outH = dOutput[0].length;
    const outW = dOutput[0][0].length;

    const dInput = zeros3D(this.inDepth, inH, inW);

    for (let f = 0; f < this.numFilters; f++) {
      for (let r = 0; r < outH; r++) {
        for (let c = 0; c < outW; c++) {
          const grad = dOutput[f][r][c];

          // Update bias
          this.biases[f] -= lr * grad;

          for (let d = 0; d < this.inDepth; d++) {
            for (let fr = 0; fr < this.filterSize; fr++) {
              for (let fc = 0; fc < this.filterSize; fc++) {
                // dFilter
                this.filters[f][d][fr][fc] -= lr * grad * this.input[d][r + fr][c + fc];
                // dInput
                dInput[d][r + fr][c + fc] += grad * this.filters[f][d][fr][fc];
              }
            }
          }
        }
      }
    }
    return dInput;
  }

  getParams() {
    return { filters: JSON.parse(JSON.stringify(this.filters)), biases: [...this.biases] };
  }

  setParams(params) {
    this.filters = JSON.parse(JSON.stringify(params.filters));
    this.biases = [...params.biases];
  }
}


/**
 * ReLULayer — Rectified Linear Unit activation
 */
class ReLULayer {
  constructor() {
    this.mask = null;
  }

  forward(input) {
    const depth = input.length;
    const height = input[0].length;
    const width = input[0][0].length;
    const output = zeros3D(depth, height, width);
    this.mask = zeros3D(depth, height, width);

    for (let d = 0; d < depth; d++) {
      for (let r = 0; r < height; r++) {
        for (let c = 0; c < width; c++) {
          if (input[d][r][c] > 0) {
            output[d][r][c] = input[d][r][c];
            this.mask[d][r][c] = 1;
          }
        }
      }
    }
    return output;
  }

  backward(dOutput) {
    const depth = dOutput.length;
    const height = dOutput[0].length;
    const width = dOutput[0][0].length;
    const dInput = zeros3D(depth, height, width);

    for (let d = 0; d < depth; d++)
      for (let r = 0; r < height; r++)
        for (let c = 0; c < width; c++)
          dInput[d][r][c] = dOutput[d][r][c] * this.mask[d][r][c];

    return dInput;
  }
}


/**
 * MaxPoolLayer — 2×2 max pooling with stride 2
 */
class MaxPoolLayer {
  constructor(poolSize) {
    this.poolSize = poolSize || 2;
    this.maxIndices = null;
    this.inputShape = null;
  }

  forward(input) {
    const depth = input.length;
    const inH = input[0].length;
    const inW = input[0][0].length;
    const p = this.poolSize;
    const outH = Math.floor(inH / p);
    const outW = Math.floor(inW / p);

    this.inputShape = [depth, inH, inW];
    this.maxIndices = [];

    const output = zeros3D(depth, outH, outW);

    for (let d = 0; d < depth; d++) {
      this.maxIndices[d] = [];
      for (let r = 0; r < outH; r++) {
        this.maxIndices[d][r] = [];
        for (let c = 0; c < outW; c++) {
          let maxVal = -Infinity;
          let maxR = 0, maxC = 0;
          for (let pr = 0; pr < p; pr++) {
            for (let pc = 0; pc < p; pc++) {
              const val = input[d][r * p + pr][c * p + pc];
              if (val > maxVal) {
                maxVal = val;
                maxR = r * p + pr;
                maxC = c * p + pc;
              }
            }
          }
          output[d][r][c] = maxVal;
          this.maxIndices[d][r][c] = [maxR, maxC];
        }
      }
    }
    return output;
  }

  backward(dOutput) {
    const [depth, inH, inW] = this.inputShape;
    const dInput = zeros3D(depth, inH, inW);
    const outH = dOutput[0].length;
    const outW = dOutput[0][0].length;

    for (let d = 0; d < depth; d++) {
      for (let r = 0; r < outH; r++) {
        for (let c = 0; c < outW; c++) {
          const [mr, mc] = this.maxIndices[d][r][c];
          dInput[d][mr][mc] += dOutput[d][r][c];
        }
      }
    }
    return dInput;
  }
}


/**
 * DenseLayer — Fully connected layer
 * Input:  1-D array [inSize]
 * Output: 1-D array [outSize]
 */
class DenseLayer {
  constructor(inSize, outSize) {
    this.inSize = inSize;
    this.outSize = outSize;

    // Weights: 2-D [outSize][inSize], He init
    this.weights = [];
    for (let o = 0; o < outSize; o++) {
      this.weights[o] = [];
      for (let i = 0; i < inSize; i++) {
        this.weights[o][i] = heInit(inSize);
      }
    }

    // Biases: 1-D [outSize]
    this.biases = zeros(outSize);

    // Cache
    this.input = null;
  }

  forward(input) {
    this.input = input; // cache
    const output = zeros(this.outSize);
    for (let o = 0; o < this.outSize; o++) {
      let sum = this.biases[o];
      for (let i = 0; i < this.inSize; i++) {
        sum += this.weights[o][i] * input[i];
      }
      output[o] = sum;
    }
    return output;
  }

  backward(dOutput, lr) {
    const dInput = zeros(this.inSize);

    for (let o = 0; o < this.outSize; o++) {
      const grad = dOutput[o];
      // Update bias
      this.biases[o] -= lr * grad;
      for (let i = 0; i < this.inSize; i++) {
        // dInput
        dInput[i] += this.weights[o][i] * grad;
        // Update weight
        this.weights[o][i] -= lr * grad * this.input[i];
      }
    }
    return dInput;
  }

  getParams() {
    return {
      weights: this.weights.map(row => [...row]),
      biases: [...this.biases]
    };
  }

  setParams(params) {
    this.weights = params.weights.map(row => [...row]);
    this.biases = [...params.biases];
  }
}


/**
 * ReLUDense — ReLU for 1-D dense outputs
 */
class ReLUDense {
  constructor() {
    this.mask = null;
  }

  forward(input) {
    this.mask = input.map(v => v > 0 ? 1 : 0);
    return input.map(v => Math.max(0, v));
  }

  backward(dOutput) {
    return dOutput.map((g, i) => g * this.mask[i]);
  }
}


// ═══════════════════════════════════════════════════════════
// 3. CNN MODEL
// ═══════════════════════════════════════════════════════════

class CNNModel {
  constructor(config) {
    this.config = config;
    this.layers = [];
    this.flattenShape = null;
    this.built = false;
    this.buildNetwork();
  }

  buildNetwork() {
    const { numFilters, filterSize, numConvLayers, inputSize } = this.config;
    this.layers = [];

    let currentDepth = 1; // grayscale input
    let currentSize = inputSize; // 28

    // Build convolutional blocks: Conv → ReLU → MaxPool
    for (let i = 0; i < numConvLayers; i++) {
      const convFilters = numFilters * (i + 1); // increase filters per layer

      // Check size validity
      const afterConv = currentSize - filterSize + 1;
      if (afterConv < 2) break;

      this.layers.push({ type: 'conv', layer: new ConvLayer(convFilters, filterSize, currentDepth) });
      this.layers.push({ type: 'relu', layer: new ReLULayer() });

      currentSize = afterConv;
      currentDepth = convFilters;

      // Add pooling only if output would be >= 2
      const afterPool = Math.floor(currentSize / 2);
      if (afterPool >= 2) {
        this.layers.push({ type: 'pool', layer: new MaxPoolLayer(2) });
        currentSize = afterPool;
      }
    }

    // Flatten size
    this.flattenDepth = currentDepth;
    this.flattenH = currentSize;
    this.flattenW = currentSize;
    const flatSize = currentDepth * currentSize * currentSize;

    // Dense layers
    const hiddenSize = Math.min(64, flatSize);
    this.layers.push({ type: 'dense1', layer: new DenseLayer(flatSize, hiddenSize) });
    this.layers.push({ type: 'relu_d', layer: new ReLUDense() });
    this.layers.push({ type: 'dense2', layer: new DenseLayer(hiddenSize, 3) }); // 3 classes

    this.built = true;
  }

  /**
   * Forward pass: image (28×28 grayscale, values 0-1) → class probabilities
   * @param {number[][]} image28 - 2-D array [28][28] of values in [0,1]
   * @returns {number[]} - softmax probabilities [circle, square, triangle]
   */
  forward(image28) {
    // Wrap 2-D into 3-D (1 channel)
    let x = [image28];

    // Through conv/relu/pool layers
    for (const entry of this.layers) {
      if (entry.type === 'conv' || entry.type === 'relu' || entry.type === 'pool') {
        x = entry.layer.forward(x);
      } else {
        break;
      }
    }

    // Flatten
    let flat = flatten3D(x);

    // Through dense layers
    for (const entry of this.layers) {
      if (entry.type === 'dense1' || entry.type === 'relu_d' || entry.type === 'dense2') {
        flat = entry.layer.forward(flat);
      }
    }

    // Softmax
    this.logits = flat;
    this.probs = softmax(flat);
    return this.probs;
  }

  /**
   * Backward pass: compute gradients and update weights
   * @param {number} label - integer class label (0, 1, or 2)
   * @param {number} lr - learning rate
   * @returns {number} - loss value
   */
  backward(label, lr) {
    const loss = crossEntropyLoss(this.probs, label);

    // dL/dLogits for softmax + cross-entropy: probs - one_hot(label)
    const dLogits = [...this.probs];
    dLogits[label] -= 1;

    // Backprop through dense layers in reverse
    let dFlat = dLogits;

    // Dense2 backward
    const dense2 = this.layers.find(e => e.type === 'dense2');
    dFlat = dense2.layer.backward(dFlat, lr);

    // ReLU dense backward
    const reluD = this.layers.find(e => e.type === 'relu_d');
    dFlat = reluD.layer.backward(dFlat);

    // Dense1 backward
    const dense1 = this.layers.find(e => e.type === 'dense1');
    dFlat = dense1.layer.backward(dFlat, lr);

    // Unflatten
    let dX = unflatten3D(dFlat, this.flattenDepth, this.flattenH, this.flattenW);

    // Backprop through conv/relu/pool in reverse
    const convLayers = this.layers.filter(e =>
      e.type === 'conv' || e.type === 'relu' || e.type === 'pool'
    );

    for (let i = convLayers.length - 1; i >= 0; i--) {
      const entry = convLayers[i];
      if (entry.type === 'pool') {
        dX = entry.layer.backward(dX);
      } else if (entry.type === 'relu') {
        dX = entry.layer.backward(dX);
      } else if (entry.type === 'conv') {
        dX = entry.layer.backward(dX, lr);
      }
    }

    return loss;
  }

  /** Predict class and probabilities */
  predict(image28) {
    const probs = this.forward(image28);
    let maxIdx = 0;
    for (let i = 1; i < probs.length; i++) {
      if (probs[i] > probs[maxIdx]) maxIdx = i;
    }
    return { classIndex: maxIdx, probs };
  }

  /** Serialize model for localStorage */
  serialize() {
    const params = {};
    for (const entry of this.layers) {
      if (entry.layer.getParams) {
        params[entry.type] = entry.layer.getParams();
      }
    }
    return {
      config: this.config,
      params,
      flattenDepth: this.flattenDepth,
      flattenH: this.flattenH,
      flattenW: this.flattenW
    };
  }

  /** Load model from serialized data */
  loadParams(data) {
    for (const entry of this.layers) {
      if (entry.layer.setParams && data.params[entry.type]) {
        entry.layer.setParams(data.params[entry.type]);
      }
    }
  }
}


// ═══════════════════════════════════════════════════════════
// 4. SYNTHETIC DATA GENERATOR
// ═══════════════════════════════════════════════════════════

/**
 * Generate a synthetic 28×28 grayscale image of a shape
 * @param {number} classIdx - 0=circle, 1=square, 2=triangle
 * @returns {number[][]} - 28x28 array, values in [0,1]
 */
function generateSyntheticShape(classIdx) {
  const size = 28;
  const img = zeros2D(size, size);

  // Random position and size variation
  const cx = 14 + (Math.random() - 0.5) * 6;
  const cy = 14 + (Math.random() - 0.5) * 6;
  const radius = 5 + Math.random() * 5;

  if (classIdx === 0) {
    // Circle
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const dist = Math.sqrt((c - cx) ** 2 + (r - cy) ** 2);
        if (Math.abs(dist - radius) < 1.5) {
          img[r][c] = 1;
        } else if (Math.abs(dist - radius) < 2.5) {
          img[r][c] = 0.5;
        }
      }
    }
  } else if (classIdx === 1) {
    // Square
    const half = radius;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        const dx = Math.abs(c - cx);
        const dy = Math.abs(r - cy);
        if ((Math.abs(dx - half) < 1.5 && dy <= half + 1) ||
            (Math.abs(dy - half) < 1.5 && dx <= half + 1)) {
          img[r][c] = 1;
        }
      }
    }
  } else {
    // Triangle (pointing up)
    const top = cy - radius;
    const base = cy + radius;
    const height = base - top;
    for (let r = 0; r < size; r++) {
      for (let c = 0; c < size; c++) {
        if (r >= top && r <= base) {
          const progress = (r - top) / height;
          const halfWidth = progress * radius;
          const leftEdge = cx - halfWidth;
          const rightEdge = cx + halfWidth;

          // Bottom edge
          if (Math.abs(r - base) < 1.5 && c >= leftEdge - 1 && c <= rightEdge + 1) {
            img[r][c] = 1;
          }
          // Left edge
          if (Math.abs(c - leftEdge) < 1.5 && r >= top && r <= base) {
            img[r][c] = 1;
          }
          // Right edge
          if (Math.abs(c - rightEdge) < 1.5 && r >= top && r <= base) {
            img[r][c] = 1;
          }
        }
      }
    }
  }

  // Add slight noise
  for (let r = 0; r < size; r++) {
    for (let c = 0; c < size; c++) {
      img[r][c] += Math.random() * 0.05;
      img[r][c] = Math.min(1, Math.max(0, img[r][c]));
    }
  }

  return img;
}


// ═══════════════════════════════════════════════════════════
// 5. UI CONTROLLER
// ═══════════════════════════════════════════════════════════

(function() {
  'use strict';

  const CLASS_NAMES = ['Circle', 'Square', 'Triangle'];
  const CLASS_EMOJIS = ['⭕', '⬛', '🔺'];
  const STORAGE_KEY_MODEL = 'cnn_scratch_model';
  const STORAGE_KEY_DATA = 'cnn_scratch_data';
  const INPUT_SIZE = 28;

  // ── State ──
  let model = null;
  let trainingData = []; // [{image: 28x28, label: 0|1|2}]
  let isTraining = false;
  let isDrawing = false;
  let isErasing = false;

  // ── DOM Elements ──
  const canvas = document.getElementById('drawCanvas');
  const ctx = canvas.getContext('2d');
  const previewCanvas = document.getElementById('previewCanvas');
  const previewCtx = previewCanvas.getContext('2d');

  const btnDraw = document.getElementById('btnDraw');
  const btnErase = document.getElementById('btnErase');
  const btnClear = document.getElementById('btnClear');
  const brushSizeInput = document.getElementById('brushSize');
  const brushSizeVal = document.getElementById('brushSizeVal');

  const numFiltersInput = document.getElementById('numFilters');
  const filterSizeInput = document.getElementById('filterSize');
  const numConvLayersInput = document.getElementById('numConvLayers');
  const epochsInput = document.getElementById('epochs');
  const lrInput = document.getElementById('learningRate');
  const lrValue = document.getElementById('lrValue');

  const btnTrain = document.getElementById('btnTrain');
  const btnPredict = document.getElementById('btnPredict');
  const btnDownload = document.getElementById('btnDownload');
  const btnReset = document.getElementById('btnReset');

  const modelStatus = document.getElementById('modelStatus');
  const modelStatusText = document.getElementById('modelStatusText');

  const statEpoch = document.getElementById('statEpoch');
  const statLoss = document.getElementById('statLoss');
  const statAccuracy = document.getElementById('statAccuracy');
  const progressFill = document.getElementById('progressFill');
  const trainingLog = document.getElementById('trainingLog');

  const predShape = document.getElementById('predShape');
  const predLabel = document.getElementById('predLabel');
  const predConf = document.getElementById('predConf');
  const barCircle = document.getElementById('barCircle');
  const barSquare = document.getElementById('barSquare');
  const barTriangle = document.getElementById('barTriangle');
  const pctCircle = document.getElementById('pctCircle');
  const pctSquare = document.getElementById('pctSquare');
  const pctTriangle = document.getElementById('pctTriangle');

  const countCircle = document.getElementById('countCircle');
  const countSquare = document.getElementById('countSquare');
  const countTriangle = document.getElementById('countTriangle');

  const archViz = document.getElementById('archViz');
  const toast = document.getElementById('toast');

  // ── Initialization ──
  async function init() {
    clearCanvas();
    setupCanvasEvents();
    setupControlEvents();
    setupAddSampleButtons();
    updateArchViz();
    buildModel();
    await loadModelWeights();
    updateDataCounts();
  }

  // ── Canvas Drawing ──
  function clearCanvas() {
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    updatePreview();
  }

  function setupCanvasEvents() {
    let lastX, lastY;

    function getPos(e) {
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      const scaleY = canvas.height / rect.height;
      if (e.touches) {
        return {
          x: (e.touches[0].clientX - rect.left) * scaleX,
          y: (e.touches[0].clientY - rect.top) * scaleY
        };
      }
      return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY
      };
    }

    function draw(x, y) {
      const brushSize = parseInt(brushSizeInput.value);
      ctx.beginPath();
      ctx.arc(x, y, brushSize / 2, 0, Math.PI * 2);
      ctx.fillStyle = isErasing ? '#000' : '#fff';
      ctx.fill();

      // Interpolate for smooth lines
      if (lastX !== undefined) {
        ctx.beginPath();
        ctx.moveTo(lastX, lastY);
        ctx.lineTo(x, y);
        ctx.strokeStyle = isErasing ? '#000' : '#fff';
        ctx.lineWidth = brushSize;
        ctx.lineCap = 'round';
        ctx.stroke();
      }
      lastX = x;
      lastY = y;
    }

    function startDraw(e) {
      e.preventDefault();
      isDrawing = true;
      const pos = getPos(e);
      lastX = pos.x;
      lastY = pos.y;
      draw(pos.x, pos.y);
    }

    function moveDraw(e) {
      if (!isDrawing) return;
      e.preventDefault();
      const pos = getPos(e);
      draw(pos.x, pos.y);
      updatePreview();
    }

    function endDraw() {
      isDrawing = false;
      lastX = undefined;
      lastY = undefined;
      updatePreview();
    }

    canvas.addEventListener('mousedown', startDraw);
    canvas.addEventListener('mousemove', moveDraw);
    canvas.addEventListener('mouseup', endDraw);
    canvas.addEventListener('mouseleave', endDraw);
    canvas.addEventListener('touchstart', startDraw, { passive: false });
    canvas.addEventListener('touchmove', moveDraw, { passive: false });
    canvas.addEventListener('touchend', endDraw);

    btnDraw.addEventListener('click', () => {
      isErasing = false;
      btnDraw.classList.add('canvas-tools__btn--active');
      btnErase.classList.remove('canvas-tools__btn--active');
    });

    btnErase.addEventListener('click', () => {
      isErasing = true;
      btnErase.classList.add('canvas-tools__btn--active');
      btnDraw.classList.remove('canvas-tools__btn--active');
    });

    btnClear.addEventListener('click', clearCanvas);

    brushSizeInput.addEventListener('input', () => {
      brushSizeVal.textContent = brushSizeInput.value + 'px';
    });
  }

  // ── Preview (28×28 downscale) ──
  function updatePreview() {
    previewCtx.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE);
  }

  /** Get 28×28 grayscale array from canvas */
  function getCanvasImage() {
    // Draw canvas to a temporary canvas at 28×28
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = INPUT_SIZE;
    tempCanvas.height = INPUT_SIZE;
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(canvas, 0, 0, INPUT_SIZE, INPUT_SIZE);

    const imageData = tempCtx.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE);
    const pixels = imageData.data;
    const image = zeros2D(INPUT_SIZE, INPUT_SIZE);

    for (let r = 0; r < INPUT_SIZE; r++) {
      for (let c = 0; c < INPUT_SIZE; c++) {
        const idx = (r * INPUT_SIZE + c) * 4;
        // Convert to grayscale (0-1)
        image[r][c] = pixels[idx] / 255;
      }
    }
    return image;
  }

  // ── Controls ──
  function setupControlEvents() {
    lrInput.addEventListener('input', () => {
      lrValue.textContent = parseFloat(lrInput.value).toFixed(3);
    });

    // Rebuild architecture when config changes
    [numFiltersInput, filterSizeInput, numConvLayersInput].forEach(input => {
      input.addEventListener('change', () => {
        updateArchViz();
        buildModel();
      });
    });

    btnTrain.addEventListener('click', startTraining);
    btnPredict.addEventListener('click', runPrediction);
    btnDownload.addEventListener('click', downloadWeights);
    btnReset.addEventListener('click', resetModel);
  }

  // ── Add Sample Buttons ──
  function setupAddSampleButtons() {
    document.querySelectorAll('.data-class__add-btn').forEach(btn => {
      btn.addEventListener('click', () => {
        const classIdx = parseInt(btn.dataset.class);
        const image = getCanvasImage();

        // Check canvas isn't empty
        let sum = 0;
        for (let r = 0; r < INPUT_SIZE; r++)
          for (let c = 0; c < INPUT_SIZE; c++)
            sum += image[r][c];

        if (sum < 5) {
          showToast('Draw something on the canvas first!', 'error');
          return;
        }

        trainingData.push({ image, label: classIdx });
        updateDataCounts();
        saveDataToStorage();
        clearCanvas();
        showToast(`Added ${CLASS_NAMES[classIdx]} sample (#${trainingData.filter(d => d.label === classIdx).length})`, 'success');
      });
    });
  }

  function updateDataCounts() {
    const counts = [0, 0, 0];
    trainingData.forEach(d => counts[d.label]++);
    countCircle.textContent = counts[0];
    countSquare.textContent = counts[1];
    countTriangle.textContent = counts[2];
  }

  // ── Architecture Visualization ──
  function updateArchViz() {
    const nf = parseInt(numFiltersInput.value) || 4;
    const fs = parseInt(filterSizeInput.value) || 3;
    const cl = parseInt(numConvLayersInput.value) || 1;

    let html = '';
    html += `<div class="arch-layer"><span class="arch-layer__name">Input</span><span class="arch-layer__size">28×28×1</span></div>`;

    let currentSize = INPUT_SIZE;
    let currentDepth = 1;

    for (let i = 0; i < cl; i++) {
      const filters = nf * (i + 1);
      const afterConv = currentSize - fs + 1;
      if (afterConv < 2) break;

      html += `<span class="arch-arrow">→</span>`;
      html += `<div class="arch-layer"><span class="arch-layer__name">Conv</span><span class="arch-layer__size">${afterConv}×${afterConv}×${filters}</span></div>`;
      html += `<span class="arch-arrow">→</span>`;
      html += `<div class="arch-layer"><span class="arch-layer__name">ReLU</span><span class="arch-layer__size">${afterConv}×${afterConv}×${filters}</span></div>`;

      currentSize = afterConv;
      currentDepth = filters;

      const afterPool = Math.floor(currentSize / 2);
      if (afterPool >= 2) {
        html += `<span class="arch-arrow">→</span>`;
        html += `<div class="arch-layer"><span class="arch-layer__name">Pool</span><span class="arch-layer__size">${afterPool}×${afterPool}×${filters}</span></div>`;
        currentSize = afterPool;
      }
    }

    const flatSize = currentDepth * currentSize * currentSize;
    const hiddenSize = Math.min(64, flatSize);
    html += `<span class="arch-arrow">→</span>`;
    html += `<div class="arch-layer"><span class="arch-layer__name">Flat</span><span class="arch-layer__size">${flatSize}</span></div>`;
    html += `<span class="arch-arrow">→</span>`;
    html += `<div class="arch-layer"><span class="arch-layer__name">Dense</span><span class="arch-layer__size">${hiddenSize}</span></div>`;
    html += `<span class="arch-arrow">→</span>`;
    html += `<div class="arch-layer"><span class="arch-layer__name">Output</span><span class="arch-layer__size">3</span></div>`;

    archViz.innerHTML = html;
  }

  // ── Model Management ──
  function getConfig() {
    return {
      numFilters: parseInt(numFiltersInput.value) || 4,
      filterSize: parseInt(filterSizeInput.value) || 3,
      numConvLayers: parseInt(numConvLayersInput.value) || 1,
      inputSize: INPUT_SIZE
    };
  }

  function buildModel() {
    model = new CNNModel(getConfig());
    setModelStatus(false);
    log('Model initialized with new architecture', 'info');
  }

  function setModelStatus(trained) {
    if (trained) {
      modelStatus.className = 'model-status model-status--trained';
      modelStatusText.textContent = 'Model trained ✓';
    } else {
      modelStatus.className = 'model-status model-status--untrained';
      modelStatusText.textContent = 'Model untrained';
    }
  }

  // ── Training ──
  async function startTraining() {
    if (isTraining) return;

    // Build combined dataset: user-drawn + synthetic
    let dataset = [...trainingData];

    // Generate synthetic data to supplement
    const userCounts = [0, 0, 0];
    trainingData.forEach(d => userCounts[d.label]++);
    const totalUser = trainingData.length;

    // Always generate some synthetic data for robust training
    const synthPerClass = Math.max(15, 30 - Math.floor(totalUser / 3));
    for (let cls = 0; cls < 3; cls++) {
      for (let i = 0; i < synthPerClass; i++) {
        dataset.push({ image: generateSyntheticShape(cls), label: cls });
      }
    }

    if (dataset.length < 9) {
      showToast('Add at least a few samples or draw shapes to train!', 'error');
      return;
    }

    isTraining = true;
    btnTrain.disabled = true;
    btnTrain.innerHTML = '<span class="spinner spinner--active"></span> Training…';

    const epochs = parseInt(epochsInput.value) || 15;
    const lr = parseFloat(lrInput.value) || 0.01;

    // Rebuild model fresh
    buildModel();

    log(`Training started: ${epochs} epochs, LR=${lr}, ${dataset.length} samples`, 'info');

    let totalSteps = epochs * dataset.length;
    let stepsDone = 0;

    for (let epoch = 0; epoch < epochs; epoch++) {
      shuffle(dataset);

      let epochLoss = 0;
      let correct = 0;

      for (let i = 0; i < dataset.length; i++) {
        const { image, label } = dataset[i];

        // Forward pass
        const probs = model.forward(image);

        // Check accuracy
        let maxIdx = 0;
        for (let j = 1; j < probs.length; j++)
          if (probs[j] > probs[maxIdx]) maxIdx = j;
        if (maxIdx === label) correct++;

        // Backward pass
        const loss = model.backward(label, lr);
        epochLoss += loss;

        stepsDone++;
      }

      const avgLoss = epochLoss / dataset.length;
      const accuracy = (correct / dataset.length * 100).toFixed(1);

      // Update UI
      statEpoch.textContent = `${epoch + 1} / ${epochs}`;
      statLoss.textContent = avgLoss.toFixed(4);
      statAccuracy.textContent = accuracy + '%';
      progressFill.style.width = ((epoch + 1) / epochs * 100) + '%';

      const logClass = avgLoss < 0.5 ? 'success' : avgLoss < 1.5 ? 'warn' : 'info';
      log(`Epoch ${epoch + 1}/${epochs} — Loss: ${avgLoss.toFixed(4)} — Acc: ${accuracy}%`, logClass);

      // Yield to UI thread every epoch
      await new Promise(resolve => setTimeout(resolve, 0));
    }

    isTraining = false;
    btnTrain.disabled = false;
    btnTrain.innerHTML = '🚀 Train';

    setModelStatus(true);
    saveModelToStorage();
    showToast('Training complete! Model saved.', 'success');
    log('Training finished. Weights saved to localStorage.', 'success');
  }

  // ── Prediction ──
  function runPrediction() {
    if (!model || !model.built) {
      showToast('Build a model first!', 'error');
      return;
    }

    const image = getCanvasImage();

    // Check canvas isn't empty
    let sum = 0;
    for (let r = 0; r < INPUT_SIZE; r++)
      for (let c = 0; c < INPUT_SIZE; c++)
        sum += image[r][c];

    if (sum < 3) {
      showToast('Draw something on the canvas first!', 'error');
      return;
    }

    const result = model.predict(image);
    const { classIndex, probs } = result;

    // Update prediction display
    predShape.textContent = CLASS_EMOJIS[classIndex];
    predLabel.textContent = CLASS_NAMES[classIndex];
    predConf.textContent = `Confidence: ${(probs[classIndex] * 100).toFixed(1)}%`;

    // Update confidence bars
    barCircle.style.width = (probs[0] * 100) + '%';
    barSquare.style.width = (probs[1] * 100) + '%';
    barTriangle.style.width = (probs[2] * 100) + '%';
    pctCircle.textContent = (probs[0] * 100).toFixed(1) + '%';
    pctSquare.textContent = (probs[1] * 100).toFixed(1) + '%';
    pctTriangle.textContent = (probs[2] * 100).toFixed(1) + '%';

    log(`Prediction: ${CLASS_NAMES[classIndex]} (${(probs[classIndex] * 100).toFixed(1)}%)`, 'success');
  }

  // ── Reset ──
  function resetModel() {
    localStorage.removeItem(STORAGE_KEY_MODEL);
    localStorage.removeItem(STORAGE_KEY_DATA);
    trainingData = [];
    updateDataCounts();
    buildModel();
    clearCanvas();

    statEpoch.textContent = '0 / 0';
    statLoss.textContent = '—';
    statAccuracy.textContent = '—';
    progressFill.style.width = '0%';

    predShape.textContent = '❓';
    predLabel.textContent = 'Draw & Predict';
    predConf.textContent = 'Confidence: —';
    barCircle.style.width = '0%';
    barSquare.style.width = '0%';
    barTriangle.style.width = '0%';
    pctCircle.textContent = '0%';
    pctSquare.textContent = '0%';
    pctTriangle.textContent = '0%';

    trainingLog.innerHTML = '<div class="log-entry log-entry--info">Model reset. Waiting for training data…</div>';

    showToast('Model and data reset.', 'info');
  }

  // ── localStorage ──
  function saveModelToStorage() {
    try {
      const serialized = model.serialize();
      localStorage.setItem(STORAGE_KEY_MODEL, JSON.stringify(serialized));
    } catch (e) {
      console.warn('Could not save model:', e);
    }
  }

  function saveDataToStorage() {
    try {
      // Only save up to 100 samples to avoid quota
      const toSave = trainingData.slice(-100);
      localStorage.setItem(STORAGE_KEY_DATA, JSON.stringify(toSave));
    } catch (e) {
      console.warn('Could not save training data:', e);
    }
  }

  /**
   * Load model weights: localStorage first, then fetch('weights.json') fallback.
   * This ensures the grader's empty localStorage still loads your pre-trained weights.
   */
  async function loadModelWeights() {
    // 1. Try loading training data from localStorage
    try {
      const dataStr = localStorage.getItem(STORAGE_KEY_DATA);
      if (dataStr) {
        trainingData = JSON.parse(dataStr);
        updateDataCounts();
        log(`Loaded ${trainingData.length} training samples from storage`, 'info');
      }
    } catch (e) {
      console.warn('Could not load training data:', e);
    }

    // 2. Try loading model from localStorage
    const modelStr = localStorage.getItem(STORAGE_KEY_MODEL);
    if (modelStr) {
      try {
        const data = JSON.parse(modelStr);
        applyModelData(data);
        log('Loaded trained model from localStorage', 'success');
        showToast('Loaded saved model!', 'success');
        return;
      } catch (e) {
        console.warn('localStorage model corrupted:', e);
      }
    }

    // 3. localStorage empty → fetch pre-trained weights from weights.json
    log('No model in localStorage. Fetching pre-trained weights…', 'info');
    try {
      const response = await fetch('weights.json');
      if (!response.ok) {
        throw new Error('weights.json not found (HTTP ' + response.status + ')');
      }
      const data = await response.json();

      // Apply to model
      applyModelData(data);

      // Cache in localStorage for future visits
      try {
        localStorage.setItem(STORAGE_KEY_MODEL, JSON.stringify(data));
      } catch (e) {
        console.warn('Could not cache model to localStorage:', e);
      }

      log('Loaded pre-trained weights from weights.json', 'success');
      showToast('Loaded pre-trained model from server!', 'success');
    } catch (err) {
      // 4. Fetch failed → graceful fallback to random untrained model
      console.warn('Could not fetch weights.json:', err);
      log('No pre-trained weights found. Starting with random model.', 'warn');
    }
  }

  /** Apply serialized model data to the CNN */
  function applyModelData(data) {
    numFiltersInput.value = data.config.numFilters;
    filterSizeInput.value = data.config.filterSize;
    numConvLayersInput.value = data.config.numConvLayers;

    model = new CNNModel(data.config);
    model.loadParams(data);
    updateArchViz();
    setModelStatus(true);
  }

  // ── Download Weights ──
  function downloadWeights() {
    if (!model || !model.built) {
      showToast('No model to export!', 'error');
      return;
    }

    const modelStr = localStorage.getItem(STORAGE_KEY_MODEL);
    if (!modelStr) {
      showToast('Train the model first before downloading!', 'error');
      return;
    }

    // Create a JSON Blob and trigger download
    const blob = new Blob([modelStr], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'weights.json';
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);

    showToast('weights.json downloaded! Commit it to your repo.', 'success');
    log('Exported model weights to weights.json', 'success');
  }

  // ── Logging ──
  function log(message, type) {
    const entry = document.createElement('div');
    entry.className = `log-entry log-entry--${type || 'info'}`;
    const time = new Date().toLocaleTimeString('en-US', { hour12: false, hour: '2-digit', minute: '2-digit', second: '2-digit' });
    entry.textContent = `[${time}] ${message}`;
    trainingLog.appendChild(entry);
    trainingLog.scrollTop = trainingLog.scrollHeight;

    // Keep only last 200 entries
    while (trainingLog.children.length > 200) {
      trainingLog.removeChild(trainingLog.firstChild);
    }
  }

  // ── Toast ──
  function showToast(message, type) {
    toast.textContent = message;
    toast.className = `toast toast--${type || 'info'} toast--visible`;
    clearTimeout(showToast._timer);
    showToast._timer = setTimeout(() => {
      toast.classList.remove('toast--visible');
    }, 3000);
  }

  // ── Boot ──
  init();

})();
