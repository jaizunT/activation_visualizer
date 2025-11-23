import type { Edge, Node } from 'reactflow';

export type Activation = 'ReLU' | 'Tanh' | 'Sigmoid';
export type Architecture = 'mlp' | 'cnn' | 'rnn' | 'transformer';

export type InitMode = 'random' | 'constant';

export interface BackpropConfig {
  architecture: Architecture;
  layers: number;
  hiddenDim: number;
  activation: Activation;
  inputDim: number;
  heads?: number;
  initMode?: InitMode;
  initValue?: number;
  inputValue?: number;
  paramOverrides?: Record<string, number[]>;
  inputVector?: number[];
}

export interface ParamInfo {
  shape: number[];
  grad_mean: number;
  grad_std: number;
  value_sample: number[];
}

export interface LayerDetails {
  in_shape: number[] | string;
  out_shape: number[];
  forward_mean: number;
  params: Record<string, ParamInfo>;
  attention_pattern?: number[][];
  attention_pattern_heads?: number[][][];
  attention_heads?: number;
  input_sample?: number[];
  output_sample?: number[];
}

export type FrontendNode = Node<{
  label: string;
  details: LayerDetails;
}>;

export type FrontendEdge = Edge;

export interface BackpropResult {
  nodes: FrontendNode[];
  edges: FrontendEdge[];
  loss: number;
}

class Value {
  data: number;
  grad = 0;
  prev: Set<Value>;
  op: string;
  label: string;
  shape: number[];
  params: Record<string, { val: Value; shape: number[] }>;
  id: string;
  _backward: () => void;

  constructor(data: number, children: Value[] = [], op = '', label = '') {
    this.data = data;
    this.prev = new Set(children);
    this.op = op;
    this.label = label;
    this.shape = [1, 1];
    this.params = {};
    this.id = Math.random().toString(36).slice(2);
    this._backward = () => {};
  }

  add(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data + o.data, [this, o], 'Add');
    out._backward = () => {
      this.grad += 1.0 * out.grad;
      o.grad += 1.0 * out.grad;
    };
    return out;
  }

  sigmoid(): Value {
    const s = 1 / (1 + Math.exp(-this.data));
    const out = new Value(s, [this], 'Sigmoid');
    out._backward = () => {
      this.grad += s * (1 - s) * out.grad;
    };
    return out;
  }

  mul(other: Value | number): Value {
    const o = other instanceof Value ? other : new Value(other);
    const out = new Value(this.data * o.data, [this, o], 'Mul');
    out._backward = () => {
      this.grad += o.data * out.grad;
      o.grad += this.data * out.grad;
    };
    return out;
  }

  relu(): Value {
    const out = new Value(this.data < 0 ? 0 : this.data, [this], 'ReLU');
    out._backward = () => {
      this.grad += (out.data > 0 ? 1 : 0) * out.grad;
    };
    return out;
  }

  tanh(): Value {
    const t = Math.tanh(this.data);
    const out = new Value(t, [this], 'Tanh');
    out._backward = () => {
      this.grad += (1 - t * t) * out.grad;
    };
    return out;
  }

  linear(inDim: number, outDim: number, init?: { mode: InitMode; value: number }): Value {
    const mode = init?.mode ?? 'random';
    const base = init?.value ?? 0;

    const wVal = mode === 'constant' ? base : (Math.random() - 0.5) * 0.1;
    const bVal = mode === 'constant' ? base : 0.0;

    const W = new Value(wVal, [], 'Weight', 'W');
    const b = new Value(bVal, [], 'Bias', 'b');

    const wx = this.mul(W);
    const out = wx.add(b);

    out.op = 'Linear';
    out.shape = [1, outDim];
    out.params = {
      W: { val: W, shape: [inDim, outDim] },
      b: { val: b, shape: [outDim] },
    };

    return out;
  }

  backward() {
    const topo: Value[] = [];
    const visited = new Set<Value>();

    const buildTopo = (v: Value) => {
      if (!visited.has(v)) {
        visited.add(v);
        v.prev.forEach((child) => buildTopo(child));
        topo.push(v);
      }
    };

    buildTopo(this);
    this.grad = 1.0;
    for (const node of topo.reverse()) {
      node._backward();
    }
  }
}

// --- Helpers for synthetic stats ---

function makeParam(shape: number[]): ParamInfo {
  return {
    shape,
    grad_mean: Math.random() * 0.1,
    grad_std: 0,
    value_sample: [],
  };
}

function computeTinyConvStats(
  initMode?: InitMode,
  initValue?: number,
  overrideW?: number,
  overrideB?: number,
): {
  forwardMean: number;
  wGradMean: number;
  bGradMean: number;
  wSample: number;
  bSample: number;
  xSample: number[];
  ySample: number[];
} {
  const mode = initMode ?? 'random';
  const base = initValue ?? 0;

  const xs: Value[] = [];
  for (let i = 0; i < 9; i++) {
    xs.push(new Value(0.5, [], 'Input', 'x'));
  }

  const ws: Value[] = [];
  for (let i = 0; i < 9; i++) {
    const defaultW = mode === 'constant' ? base : (Math.random() - 0.5) * 0.1;
    const wVal = overrideW !== undefined ? overrideW : defaultW;
    ws.push(new Value(wVal, [], 'Weight', 'W'));
  }

  const defaultB = mode === 'constant' ? base : 0.0;
  const bVal = overrideB !== undefined ? overrideB : defaultB;
  const b = new Value(bVal, [], 'Bias', 'b');

  let y = new Value(0.0);
  for (let i = 0; i < 9; i++) {
    const term = xs[i].mul(ws[i]);
    y = y.add(term);
  }
  y = y.add(b);

  const target = new Value(1.0);
  const diff = y.add(target.mul(-1.0));
  const loss = diff.mul(diff);
  loss.backward();

  const wGradMean = ws.reduce((s, w) => s + Math.abs(w.grad), 0) / ws.length;
  const bGradMean = Math.abs(b.grad);

  const xSample = xs.map((v) => v.data);
  const ySample = [y.data];

  return {
    forwardMean: y.data,
    wGradMean,
    bGradMean,
    wSample: ws[0]?.data ?? 0,
    bSample: b.data,
    xSample,
    ySample,
  };
}

function computeTinyRNNStats(
  initMode?: InitMode,
  initValue?: number,
  overrideWx?: number,
  overrideWh?: number,
  overrideB?: number,
): {
  forwardMean: number;
  wXGradMean: number;
  wHGradMean: number;
  bGradMean: number;
  wXSample: number;
  wHSample: number;
  bSample: number;
  xSample: number[];
  hSample: number[];
  ySample: number;
  lossSample: number;
} {
  const mode = initMode ?? 'random';
  const base = initValue ?? 0;

  const T = 3;
  const xs: Value[] = [];
  for (let t = 0; t < T; t++) {
    xs.push(new Value(0.5, [], 'Input', 'x_t'));
  }

  const defaultWx = mode === 'constant' ? base : (Math.random() - 0.5) * 0.1;
  const defaultWh = mode === 'constant' ? base : (Math.random() - 0.5) * 0.1;
  const defaultB = mode === 'constant' ? base : 0.0;

  const wXVal = overrideWx !== undefined ? overrideWx : defaultWx;
  const wHVal = overrideWh !== undefined ? overrideWh : defaultWh;
  const bVal = overrideB !== undefined ? overrideB : defaultB;

  const W_x = new Value(wXVal, [], 'Weight', 'W_x');
  const W_h = new Value(wHVal, [], 'Weight', 'W_h');
  const b = new Value(bVal, [], 'Bias', 'b');

  let hPrev = new Value(0.0, [], 'Hidden', 'h_0');
  const hs: Value[] = [];
  for (let t = 0; t < T; t++) {
    const lin = hPrev.mul(W_h).add(xs[t].mul(W_x)).add(b);
    const h = lin.tanh();
    hs.push(h);
    hPrev = h;
  }

  const y = hs[hs.length - 1];
  const target = new Value(1.0);
  const diff = y.add(target.mul(-1.0));
  const loss = diff.mul(diff);
  loss.backward();

  const forwardMean = hs.reduce((s, h) => s + h.data, 0) / hs.length;

  const xSample = xs.map((v) => v.data);
  const hSample = hs.map((v) => v.data);

  return {
    forwardMean,
    wXGradMean: Math.abs(W_x.grad),
    wHGradMean: Math.abs(W_h.grad),
    bGradMean: Math.abs(b.grad),
    wXSample: W_x.data,
    wHSample: W_h.data,
    bSample: b.data,
    xSample,
    hSample,
    ySample: y.data,
    lossSample: loss.data,
  };
}

function computeTinySelfAttnStats(
  initMode?: InitMode,
  initValue?: number,
  overrideWq?: number,
  overrideWk?: number,
  overrideWv?: number,
  overrideWo?: number,
): {
  forwardMean: number;
  wqGrad: number;
  wkGrad: number;
  wvGrad: number;
  woGrad: number;
  wqSample: number;
  wkSample: number;
  wvSample: number;
  woSample: number;
  inputSample: number[];
  outputSample: number[];
} {
  const mode = initMode ?? 'random';
  const base = initValue ?? 0;

  const x = new Value(0.5, [], 'Input', 'x');

  const makeW = (label: string, override?: number) =>
    new Value(
      override !== undefined
        ? override
        : mode === 'constant'
        ? base
        : (Math.random() - 0.5) * 0.1,
      [],
      'Weight',
      label,
    );

  const W_q = makeW('W_q', overrideWq);
  const W_k = makeW('W_k', overrideWk);
  const W_v = makeW('W_v', overrideWv);
  const W_o = makeW('W_o', overrideWo);

  const q = x.mul(W_q);
  const k = x.mul(W_k);
  const v = x.mul(W_v);

  const score = q.mul(k);
  const attn = score.sigmoid();
  const y = attn.mul(v).mul(W_o);

  const target = new Value(1.0);
  const diff = y.add(target.mul(-1.0));
  const loss = diff.mul(diff);
  loss.backward();

  return {
    forwardMean: y.data,
    wqGrad: Math.abs(W_q.grad),
    wkGrad: Math.abs(W_k.grad),
    wvGrad: Math.abs(W_v.grad),
    woGrad: Math.abs(W_o.grad),
    wqSample: W_q.data,
    wkSample: W_k.data,
    wvSample: W_v.data,
    woSample: W_o.data,
    inputSample: [x.data],
    outputSample: [y.data],
  };
}

function computeTinyFFNStats(
  initMode?: InitMode,
  initValue?: number,
  overrideW1?: number,
  overrideB1?: number,
  overrideW2?: number,
  overrideB2?: number,
): {
  forwardMean: number;
  w1Grad: number;
  b1Grad: number;
  w2Grad: number;
  b2Grad: number;
  w1Sample: number;
  b1Sample: number;
  w2Sample: number;
  b2Sample: number;
  inputSample: number[];
  hiddenSample: number[];
  outputSample: number[];
} {
  const mode = initMode ?? 'random';
  const base = initValue ?? 0;

  const x = new Value(0.5, [], 'Input', 'x');

  const makeW = (label: string, override?: number) =>
    new Value(
      override !== undefined
        ? override
        : mode === 'constant'
        ? base
        : (Math.random() - 0.5) * 0.1,
      [],
      'Weight',
      label,
    );
  const makeB = (label: string, override?: number) =>
    new Value(override !== undefined ? override : mode === 'constant' ? base : 0.0, [], 'Bias', label);

  const W1 = makeW('W1', overrideW1);
  const b1 = makeB('b1', overrideB1);
  const W2 = makeW('W2', overrideW2);
  const b2 = makeB('b2', overrideB2);

  const h = x.mul(W1).add(b1).tanh();
  const y = h.mul(W2).add(b2);

  const target = new Value(1.0);
  const diff = y.add(target.mul(-1.0));
  const loss = diff.mul(diff);
  loss.backward();

  return {
    forwardMean: y.data,
    w1Grad: Math.abs(W1.grad),
    b1Grad: Math.abs(b1.grad),
    w2Grad: Math.abs(W2.grad),
    b2Grad: Math.abs(b2.grad),
    w1Sample: W1.data,
    b1Sample: b1.data,
    w2Sample: W2.data,
    b2Sample: b2.data,
    inputSample: [x.data],
    hiddenSample: [h.data],
    outputSample: [y.data],
  };
}

function makeAttentionPattern(T: number, heads: number): { aggregate: number[][]; perHead: number[][][] } {
  const H = Math.max(1, heads | 0);
  const perHead: number[][][] = Array.from({ length: H }, () => []);
  const aggregate: number[][] = [];

  for (let i = 0; i < T; i++) {
    const headRows: number[][] = [];
    for (let h = 0; h < H; h++) {
      const row: number[] = [];
      for (let j = 0; j < T; j++) {
        row.push(Math.random() * 2 - 1);
      }
      const max = Math.max(...row);
      const exps = row.map((v) => Math.exp(v - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      const normalized = exps.map((e) => e / sum);
      headRows.push(normalized);
      perHead[h].push(normalized);
    }

    const aggRow: number[] = [];
    for (let j = 0; j < T; j++) {
      let s = 0;
      for (let h = 0; h < H; h++) {
        s += headRows[h][j];
      }
      aggRow.push(s / H);
    }
    aggregate.push(aggRow);
  }

  return { aggregate, perHead };
}

function addSequentialNode(
  nodes: FrontendNode[],
  edges: FrontendEdge[],
  index: number,
  label: string,
  inShape: number[] | string,
  outShape: number[],
  params: Record<string, ParamInfo>,
  forwardMean: number,
  inputSample?: number[],
  outputSample?: number[],
) {
  const nodeId = `layer-${index}`;
  const details: LayerDetails = {
    in_shape: inShape,
    out_shape: outShape,
    forward_mean: forwardMean,
    params,
    input_sample: inputSample,
    output_sample: outputSample,
  };

  nodes.push({
    id: nodeId,
    type: 'customLayer',
    data: { label, details },
    position: { x: 0, y: 0 },
  } as FrontendNode);

  if (index > 0) {
    const prevId = `layer-${index - 1}`;
    edges.push({
      id: `e-${prevId}-${nodeId}`,
      source: prevId,
      target: nodeId,
      animated: true,
      style: { stroke: '#94a3b8', strokeWidth: 2 },
    } as FrontendEdge);
  }
}

// --- MLP architecture using real backprop with vector/matrix params ---

type MlpLayerKind = 'input' | 'linear' | 'activation' | 'output' | 'loss';

type MlpLayerMeta = {
  label: string;
  kind: MlpLayerKind;
  vec: Value[];
  inVec?: Value[];
  inShape: number[];
  outShape: number[];
  W?: Value[];
  b?: Value[];
  wShape?: number[];
  bShape?: number[];
};

function makeInitArray(
  length: number,
  override: number[] | undefined,
  mode: InitMode,
  base: number,
  kind: 'weight' | 'bias',
): number[] {
  if (override && override.length > 0) {
    if (override.length === length) {
      return [...override];
    }
    const vals: number[] = [];
    for (let i = 0; i < length; i++) {
      vals.push(override[i % override.length]);
    }
    return vals;
  }

  if (mode === 'constant') {
    return Array.from({ length }, () => base);
  }
  // Random mode: Normal(0, 1) for both weights and biases
  const randNormal = () => {
    let u = 0;
    let v = 0;
    while (u === 0) u = Math.random();
    while (v === 0) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  };

  return Array.from({ length }, () => randNormal());
}

function makeLinearLayerFromVector(
  inVec: Value[],
  inDim: number,
  outDim: number,
  nodeId: string,
  paramOverrides: Record<string, number[]> | undefined,
  initMode: InitMode | undefined,
  initValue: number | undefined,
): { W: Value[]; b: Value[]; outVec: Value[] } {
  const mode = initMode ?? 'random';
  const base = initValue ?? 0;

  const keyW = `${nodeId}:W`;
  const keyB = `${nodeId}:b`;
  const overrideW = paramOverrides ? paramOverrides[keyW] : undefined;
  const overrideB = paramOverrides ? paramOverrides[keyB] : undefined;

  const totalW = inDim * outDim;
  const totalB = outDim;

  const wInit = makeInitArray(totalW, overrideW, mode, base, 'weight');
  const bInit = makeInitArray(totalB, overrideB, mode, base, 'bias');

  const W: Value[] = wInit.map((w) => new Value(w, [], 'Weight', 'W'));
  const b: Value[] = bInit.map((bv) => new Value(bv, [], 'Bias', 'b'));

  const outVec: Value[] = [];
  for (let j = 0; j < outDim; j++) {
    let sum = new Value(0.0);
    for (let i = 0; i < inDim; i++) {
      const wVal = W[i * outDim + j];
      const term = inVec[i].mul(wVal);
      sum = sum.add(term);
    }
    const z = sum.add(b[j]);
    outVec.push(z);
  }

  return { W, b, outVec };
}

// MLP with explicit vector/matrix weights and full backprop
function runMLP(config: BackpropConfig): BackpropResult {
  const { layers, hiddenDim, activation, inputDim } = config;

  const inDim0 = Math.max(1, inputDim || 1);

  const baseInput = typeof config.inputValue === 'number' ? config.inputValue : 0.5;
  let xData: number[];
  if (config.inputVector && config.inputVector.length) {
    const raw = config.inputVector;
    if (raw.length >= inDim0) {
      xData = raw.slice(0, inDim0);
    } else {
      const pad = Array.from({ length: inDim0 - raw.length }, () => baseInput);
      xData = [...raw, ...pad];
    }
  } else {
    xData = Array.from({ length: inDim0 }, () => baseInput);
  }

  const xVec: Value[] = xData.map((v, i) => new Value(v, [], 'Input', `x_${i}`));

  const layersMeta: MlpLayerMeta[] = [];

  // Input layer
  layersMeta.push({
    label: 'Input',
    kind: 'input',
    vec: xVec,
    inShape: [1, inDim0],
    outShape: [1, inDim0],
  });

  let currentVec = xVec;
  let currentDim = inDim0;

  for (let i = 0; i < layers; i++) {
    const linearNodeId = `layer-${layersMeta.length}`;
    const { W, b, outVec } = makeLinearLayerFromVector(
      currentVec,
      currentDim,
      hiddenDim,
      linearNodeId,
      config.paramOverrides,
      config.initMode,
      config.initValue,
    );

    layersMeta.push({
      label: `Linear ${i + 1}`,
      kind: 'linear',
      vec: outVec,
      inVec: currentVec,
      inShape: [1, currentDim],
      outShape: [1, hiddenDim],
      W,
      b,
      wShape: [currentDim, hiddenDim],
      bShape: [hiddenDim],
    });

    let actVec: Value[];
    if (activation === 'ReLU') {
      actVec = outVec.map((v) => v.relu());
    } else if (activation === 'Tanh') {
      actVec = outVec.map((v) => v.tanh());
    } else {
      actVec = outVec.map((v) => v.sigmoid());
    }

    layersMeta.push({
      label: `${activation} ${i + 1}`,
      kind: 'activation',
      vec: actVec,
      inVec: outVec,
      inShape: [1, hiddenDim],
      outShape: [1, hiddenDim],
    });

    currentVec = actVec;
    currentDim = hiddenDim;
  }

  // Output layer (maps to scalar)
  const outputNodeId = `layer-${layersMeta.length}`;
  const { W: Wout, b: Bout, outVec: yVec } = makeLinearLayerFromVector(
    currentVec,
    currentDim,
    1,
    outputNodeId,
    config.paramOverrides,
    config.initMode,
    config.initValue,
  );

  layersMeta.push({
    label: 'Output',
    kind: 'output',
    vec: yVec,
    inVec: currentVec,
    inShape: [1, currentDim],
    outShape: [1, 1],
    W: Wout,
    b: Bout,
    wShape: [currentDim, 1],
    bShape: [1],
  });

  const y = yVec[0];
  const target = new Value(1.0);
  const diff = y.add(target.mul(-1.0));
  const loss = diff.mul(diff);
  loss.op = 'MSELoss';
  loss.label = 'Loss';

  layersMeta.push({
    label: 'Loss',
    kind: 'loss',
    vec: [loss],
    inVec: [y],
    inShape: [1, 1],
    outShape: [1, 1],
  });

  loss.backward();

  const nodes: FrontendNode[] = [];
  const edges: FrontendEdge[] = [];

  layersMeta.forEach((layer, index) => {
    const nodeId = `layer-${index}`;

    const forwardMean =
      layer.vec.length > 0
        ? layer.vec.reduce((s, v) => s + v.data, 0) / layer.vec.length
        : 0;

    const params: Record<string, ParamInfo> = {};

    if (layer.W && layer.b && layer.wShape && layer.bShape) {
      const flatten = (arr: Value[]) => arr.map((v) => v.data);
      const gradMean = (arr: Value[]) =>
        arr.length > 0
          ? arr.reduce((s, v) => s + Math.abs(v.grad), 0) / arr.length
          : 0;

      params.W = {
        shape: layer.wShape,
        grad_mean: gradMean(layer.W),
        grad_std: 0,
        value_sample: flatten(layer.W),
      };

      params.b = {
        shape: layer.bShape,
        grad_mean: gradMean(layer.b),
        grad_std: 0,
        value_sample: flatten(layer.b),
      };
    }

    const inputSample = layer.inVec && layer.inVec.length > 0 ? layer.inVec.map((v) => v.data) : undefined;
    const outputSample = layer.vec && layer.vec.length > 0 ? layer.vec.map((v) => v.data) : undefined;

    const details: LayerDetails = {
      in_shape: layer.inShape,
      out_shape: layer.outShape,
      forward_mean: forwardMean,
      params,
      input_sample: inputSample,
      output_sample: outputSample,
    };

    nodes.push({
      id: nodeId,
      type: 'customLayer',
      data: { label: layer.label, details },
      position: { x: 0, y: 0 },
    } as FrontendNode);

    if (index > 0) {
      const prevId = `layer-${index - 1}`;
      edges.push({
        id: `e-${prevId}-${nodeId}`,
        source: prevId,
        target: nodeId,
        animated: true,
        style: { stroke: '#94a3b8', strokeWidth: 2 },
      } as FrontendEdge);
    }
  });

  return { nodes, edges, loss: loss.data };
}

// --- CNN: conceptual conv blocks ---

function runCNN(config: BackpropConfig): BackpropResult {
  const { layers, hiddenDim, inputDim, activation, initMode, initValue, paramOverrides } = config;
  const nodes: FrontendNode[] = [];
  const edges: FrontendEdge[] = [];

  // Limit CNN spatial size to at most 8x8 for visualization
  const size = Math.max(4, Math.min(8, inputDim || 4));
  let C = 1;
  let H = size;
  let W = size;
  let index = 0;

  // Build a simple input sample from the global inputVector / inputValue
  const totalHW = H * W;
  const baseInput = typeof config.inputValue === 'number' ? config.inputValue : 0.5;
  let imgSample: number[];
  if (config.inputVector && config.inputVector.length) {
    const raw = config.inputVector;
    if (raw.length >= totalHW) {
      imgSample = raw.slice(0, totalHW);
    } else {
      const pad = Array.from({ length: totalHW - raw.length }, () => baseInput);
      imgSample = [...raw, ...pad];
    }
  } else {
    imgSample = Array.from({ length: totalHW }, () => baseInput);
  }

  // Treat the flattened image as the output of the Input layer
  addSequentialNode(nodes, edges, index++, 'Input', [C, H, W], [C, H, W], {}, 0, undefined, imgSample);

  let lastFeatureSample: number[] | undefined;

  for (let i = 0; i < layers; i++) {
    const C_out = Math.max(2, Math.min(hiddenDim, C * 2));
    const convNodeId = `layer-${index}`;
    const keyW = `${convNodeId}:W`;
    const keyB = `${convNodeId}:b`;
    const overrides = paramOverrides ?? {};
    const overrideWArr = overrides[keyW];
    const overrideBArr = overrides[keyB];

    const agg = (arr: number[] | undefined): number | undefined => {
      if (!arr || !arr.length) return undefined;
      const sum = arr.reduce((a, b) => a + b, 0);
      return sum / arr.length;
    };

    const overrideW = agg(overrideWArr);
    const overrideB = agg(overrideBArr);

    const convStats = computeTinyConvStats(initMode, initValue, overrideW, overrideB);
    const convParams: Record<string, ParamInfo> = {
      W: {
        shape: [C_out, C, 3, 3],
        grad_mean: convStats.wGradMean,
        grad_std: 0,
        value_sample: [convStats.wSample],
      },
      b: {
        shape: [C_out],
        grad_mean: convStats.bGradMean,
        grad_std: 0,
        value_sample: [convStats.bSample],
      },
    };
    addSequentialNode(
      nodes,
      edges,
      index++,
      `Conv ${i + 1}`,
      [C, H, W],
      [C_out, H, W],
      convParams,
      convStats.forwardMean,
      convStats.xSample,
      convStats.ySample,
    );

    const actLabel = `${activation} ${i + 1}`;

    // Derive tiny activation samples from conv outputs
    const actIn = convStats.ySample;
    let actOut: number[];
    if (activation === 'ReLU') {
      actOut = actIn.map((v) => (v < 0 ? 0 : v));
    } else if (activation === 'Tanh') {
      actOut = actIn.map((v) => Math.tanh(v));
    } else {
      actOut = actIn.map((v) => 1 / (1 + Math.exp(-v)));
    }

    addSequentialNode(
      nodes,
      edges,
      index++,
      actLabel,
      [C_out, H, W],
      [C_out, H, W],
      {},
      convStats.forwardMean,
      actIn,
      actOut,
    );

    lastFeatureSample = actOut;
    C = C_out;
  }

  // Global average pool + flatten + output
  const featureIn = lastFeatureSample && lastFeatureSample.length ? lastFeatureSample : imgSample;
  const gapVal = featureIn.length
    ? featureIn.reduce((a, b) => a + b, 0) / featureIn.length
    : 0;
  const gapOut = [gapVal];

  addSequentialNode(
    nodes,
    edges,
    index++,
    'GlobalAvgPool',
    [C, H, W],
    [C, 1, 1],
    {},
    gapVal,
    featureIn,
    gapOut,
  );

  addSequentialNode(
    nodes,
    edges,
    index++,
    'Flatten',
    [C, 1, 1],
    [C],
    {},
    gapVal,
    gapOut,
    gapOut,
  );

  const outParams: Record<string, ParamInfo> = {
    W: makeParam([C, 1]),
    b: makeParam([1]),
  };

  // For the final output, reuse the pooled feature as a tiny scalar example
  addSequentialNode(nodes, edges, index++, 'Output', [C], [1], outParams, gapVal, gapOut, gapOut);

  const lossVal = gapVal * gapVal;
  addSequentialNode(nodes, edges, index++, 'Loss', [1], [1], {}, lossVal, gapOut, [lossVal]);

  return { nodes, edges, loss: lossVal };
}

// --- RNN: conceptual sequence model ---

function runRNN(config: BackpropConfig): BackpropResult {
  const { layers, hiddenDim, inputDim, initMode, initValue, paramOverrides } = config;
  const nodes: FrontendNode[] = [];
  const edges: FrontendEdge[] = [];

  const T = Math.max(2, Math.min(8, layers * 2));
  let index = 0;
  const baseStats = computeTinyRNNStats(initMode, initValue);

  addSequentialNode(
    nodes,
    edges,
    index++,
    'Input Seq',
    [T, inputDim],
    [T, inputDim],
    {},
    baseStats.forwardMean,
    undefined,
    baseStats.xSample,
  );

  let inDim = inputDim;
  let lastStats = baseStats;
  for (let i = 0; i < layers; i++) {
    const rnnNodeId = `layer-${index}`;
    const overrides = paramOverrides ?? {};
    const keyWx = `${rnnNodeId}:W_x`;
    const keyWh = `${rnnNodeId}:W_h`;
    const keyB = `${rnnNodeId}:b`;

    const agg = (arr: number[] | undefined): number | undefined => {
      if (!arr || !arr.length) return undefined;
      const sum = arr.reduce((a, b) => a + b, 0);
      return sum / arr.length;
    };

    const overrideWx = agg(overrides[keyWx]);
    const overrideWh = agg(overrides[keyWh]);
    const overrideB = agg(overrides[keyB]);

    const stats = computeTinyRNNStats(initMode, initValue, overrideWx, overrideWh, overrideB);
    lastStats = stats;

    const params: Record<string, ParamInfo> = {
      W_x: {
        shape: [inDim, hiddenDim],
        grad_mean: stats.wXGradMean,
        grad_std: 0,
        value_sample: [stats.wXSample],
      },
      W_h: {
        shape: [hiddenDim, hiddenDim],
        grad_mean: stats.wHGradMean,
        grad_std: 0,
        value_sample: [stats.wHSample],
      },
      b: {
        shape: [hiddenDim],
        grad_mean: stats.bGradMean,
        grad_std: 0,
        value_sample: [stats.bSample],
      },
    };
    addSequentialNode(
      nodes,
      edges,
      index++,
      `RNN ${i + 1}`,
      [T, inDim],
      [T, hiddenDim],
      params,
      stats.forwardMean,
      stats.xSample,
      stats.hSample,
    );
    inDim = hiddenDim;
  }

  const finalH = lastStats.hSample.length ? lastStats.hSample[lastStats.hSample.length - 1] : 0;
  addSequentialNode(
    nodes,
    edges,
    index++,
    'Final h_T',
    [T, hiddenDim],
    [hiddenDim],
    {},
    finalH,
    lastStats.hSample,
    [finalH],
  );

  addSequentialNode(
    nodes,
    edges,
    index++,
    'Loss',
    [hiddenDim],
    [1],
    {},
    lastStats.lossSample,
    [finalH],
    [lastStats.lossSample],
  );

  return { nodes, edges, loss: lastStats.lossSample };
}

// --- Transformer: conceptual encoder stack ---

function runTransformer(config: BackpropConfig): BackpropResult {
  const { layers, hiddenDim, inputDim, initMode, initValue, paramOverrides } = config;
  const nodes: FrontendNode[] = [];
  const edges: FrontendEdge[] = [];

  const T = Math.max(2, Math.min(8, 4));
  const dModel = hiddenDim || inputDim || 8;
  const heads = config.heads ?? 4;
  let index = 0;

  const baseInput = typeof config.inputValue === 'number' ? config.inputValue : 0.5;
  const tokenSample = [baseInput];
  const posSample = [baseInput];

  addSequentialNode(
    nodes,
    edges,
    index++,
    'Token Embeddings',
    [T, dModel],
    [T, dModel],
    {},
    baseInput,
    undefined,
    tokenSample,
  );
  addSequentialNode(
    nodes,
    edges,
    index++,
    'Positional Enc',
    [T, dModel],
    [T, dModel],
    {},
    baseInput,
    tokenSample,
    posSample,
  );

  for (let i = 0; i < layers; i++) {
    const blockInputIndex = index - 1;
    const overrides = paramOverrides ?? {};

    const selfNodeId = `layer-${index}`;
    const keyWq = `${selfNodeId}:W_q`;
    const keyWk = `${selfNodeId}:W_k`;
    const keyWv = `${selfNodeId}:W_v`;
    const keyWo = `${selfNodeId}:W_o`;

    const agg = (arr: number[] | undefined): number | undefined => {
      if (!arr || !arr.length) return undefined;
      const sum = arr.reduce((a, b) => a + b, 0);
      return sum / arr.length;
    };

    const overrideWq = agg(overrides[keyWq]);
    const overrideWk = agg(overrides[keyWk]);
    const overrideWv = agg(overrides[keyWv]);
    const overrideWo = agg(overrides[keyWo]);

    const attnStats = computeTinySelfAttnStats(
      initMode,
      initValue,
      overrideWq,
      overrideWk,
      overrideWv,
      overrideWo,
    );
    const attnParams: Record<string, ParamInfo> = {
      W_q: {
        shape: [dModel, dModel],
        grad_mean: attnStats.wqGrad,
        grad_std: 0,
        value_sample: [attnStats.wqSample],
      },
      W_k: {
        shape: [dModel, dModel],
        grad_mean: attnStats.wkGrad,
        grad_std: 0,
        value_sample: [attnStats.wkSample],
      },
      W_v: {
        shape: [dModel, dModel],
        grad_mean: attnStats.wvGrad,
        grad_std: 0,
        value_sample: [attnStats.wvSample],
      },
      W_o: {
        shape: [dModel, dModel],
        grad_mean: attnStats.woGrad,
        grad_std: 0,
        value_sample: [attnStats.woSample],
      },
    };
    const selfIdx = index;
    addSequentialNode(
      nodes,
      edges,
      selfIdx,
      `Self-Attn ${i + 1}`,
      [T, dModel],
      [T, dModel],
      attnParams,
      attnStats.forwardMean,
      attnStats.inputSample,
      attnStats.outputSample,
    );
    index++;

    const selfAttnNode = nodes[nodes.length - 1];
    if (selfAttnNode && selfAttnNode.data) {
      const patterns = makeAttentionPattern(T, heads);
      selfAttnNode.data.details.attention_pattern = patterns.aggregate;
      selfAttnNode.data.details.attention_pattern_heads = patterns.perHead;
      selfAttnNode.data.details.attention_heads = heads;
    }

    const ffnNodeId = `layer-${index}`;
    const keyW1 = `${ffnNodeId}:W1`;
    const keyB1 = `${ffnNodeId}:b1`;
    const keyW2 = `${ffnNodeId}:W2`;
    const keyB2 = `${ffnNodeId}:b2`;

    const overrideW1 = agg(overrides[keyW1]);
    const overrideB1 = agg(overrides[keyB1]);
    const overrideW2 = agg(overrides[keyW2]);
    const overrideB2 = agg(overrides[keyB2]);

    const ffnStats = computeTinyFFNStats(initMode, initValue, overrideW1, overrideB1, overrideW2, overrideB2);
    const ffnParams: Record<string, ParamInfo> = {
      W1: {
        shape: [dModel, 4 * dModel],
        grad_mean: ffnStats.w1Grad,
        grad_std: 0,
        value_sample: [ffnStats.w1Sample],
      },
      b1: {
        shape: [4 * dModel],
        grad_mean: ffnStats.b1Grad,
        grad_std: 0,
        value_sample: [ffnStats.b1Sample],
      },
      W2: {
        shape: [4 * dModel, dModel],
        grad_mean: ffnStats.w2Grad,
        grad_std: 0,
        value_sample: [ffnStats.w2Sample],
      },
      b2: {
        shape: [dModel],
        grad_mean: ffnStats.b2Grad,
        grad_std: 0,
        value_sample: [ffnStats.b2Sample],
      },
    };
    const ffnIdx = index;
    addSequentialNode(
      nodes,
      edges,
      ffnIdx,
      `FFN ${i + 1}`,
      [T, dModel],
      [T, dModel],
      ffnParams,
      ffnStats.forwardMean,
      ffnStats.inputSample,
      ffnStats.outputSample,
    );
    var lastBlockOutput: number[] | undefined = ffnStats.outputSample;
    index++;

    const blockInputId = `layer-${blockInputIndex}`;
    const ffnId = `layer-${ffnIdx}`;
    edges.push({
      id: `res-${blockInputId}-${ffnId}`,
      source: blockInputId,
      target: ffnId,
      animated: false,
      style: { stroke: '#22c55e', strokeWidth: 2, strokeDasharray: '4 2' },
    } as FrontendEdge);
  }

  const encoderSample = lastBlockOutput && lastBlockOutput.length ? lastBlockOutput : posSample;
  const encoderMean = encoderSample.reduce((a, b) => a + b, 0) / encoderSample.length;

  addSequentialNode(
    nodes,
    edges,
    index++,
    'Encoder Output',
    [T, dModel],
    [T, dModel],
    {},
    encoderMean,
    encoderSample,
    encoderSample,
  );

  const lossVal = encoderMean * encoderMean;
  addSequentialNode(nodes, edges, index++, 'Loss', [T, dModel], [1], {}, lossVal, encoderSample, [lossVal]);

  return { nodes, edges, loss: lossVal };
}

export function runBackpropSimulation(config: BackpropConfig): BackpropResult {
  switch (config.architecture) {
    case 'cnn':
      return runCNN(config);
    case 'rnn':
      return runRNN(config);
    case 'transformer':
      return runTransformer(config);
    case 'mlp':
    default:
      return runMLP(config);
  }
}
