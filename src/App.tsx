import { useState, useCallback, useEffect, useMemo } from 'react';
import ReactFlow, {
  Background,
  Controls,
  useNodesState,
  useEdgesState,
  BackgroundVariant,
} from 'reactflow';
import 'reactflow/dist/style.css';
import { Play, Activity } from 'lucide-react';

import BackpropNode from './components/BackpropNode';
import AiAssistantPanel from './components/AiAssistantPanel';
import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css';
import { getLayoutedElements } from './utils/layout';
import {
  runBackpropSimulation,
  type Activation,
  type Architecture,
  type InitMode,
  type LayerDetails,
  type ParamInfo,
} from './engine';

const nodeTypes = { customLayer: BackpropNode };

type ParamChip = {
  id: string;
  nodeId: string;
  layerLabel: string;
  paramName: string;
};

type BlockTemplate = {
  type: string;
  label: string;
};

type VectorViewMode = 'numbers' | 'blocks';

export type AiProvider = 'openai' | 'anthropic' | 'google';

export type AiMessage = {
  role: 'user' | 'assistant';
  content: string;
};

function valueToColor(v: number): string {
  const clamp = (x: number) => Math.max(-1, Math.min(1, x));
  const x = clamp(v);

  const neutral = { r: 15, g: 23, b: 42 }; // #0f172a
  const pos = { r: 34, g: 197, b: 94 }; // #22c55e
  const neg = { r: 248, g: 113, b: 113 }; // #f87171

  const mix = (a: typeof neutral, b: typeof neutral, t: number) => ({
    r: a.r + (b.r - a.r) * t,
    g: a.g + (b.g - a.g) * t,
    b: a.b + (b.b - a.b) * t,
  });

  if (x >= 0) {
    const c = mix(neutral, pos, x);
    return `rgb(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)})`;
  }
  const c = mix(neutral, neg, -x);
  return `rgb(${Math.round(c.r)}, ${Math.round(c.g)}, ${Math.round(c.b)})`;
}

const BLOCK_TEMPLATES: BlockTemplate[] = [
  { type: 'Linear', label: 'Linear' },
  { type: 'ReLU', label: 'ReLU' },
  { type: 'Conv', label: 'Conv' },
  { type: 'Self-Attn', label: 'Self-Attn' },
  { type: 'FFN', label: 'FFN' },
  { type: 'LayerNorm', label: 'LayerNorm' },
  { type: 'BatchNorm', label: 'BatchNorm' },
  { type: 'Loss', label: 'Loss' },
];

function buildChainRuleForParam(param: ParamChip, architecture: Architecture): string {
  const p = param.paramName;
  const layer = param.layerLabel;

  if (layer.startsWith('Self-Attn')) {
    if (p === 'W_q' || p === 'W_k' || p === 'W_v') {
      return `\\frac{\\partial L}{\\partial ${p}} = \\sum_t \\frac{\\partial L}{\\partial A_t} \\cdot \\frac{\\partial A_t}{\\partial Q_t} \\cdot \\frac{\\partial Q_t}{\\partial ${p}}`;
    }
    if (p === 'W_o') {
      return `\\frac{\\partial L}{\\partial ${p}} = \\sum_t \\frac{\\partial L}{\\partial y_t} \\cdot \\frac{\\partial y_t}{\\partial O_t} \\cdot \\frac{\\partial O_t}{\\partial ${p}}`;
    }
  }

  if (layer.startsWith('RNN')) {
    if (p === 'W_x' || p === 'W_h') {
      return `\\frac{\\partial L}{\\partial ${p}} = \\sum_t \\frac{\\partial L}{\\partial h_t} \\cdot \\frac{\\partial h_t}{\\partial ${p}}`;
    }
  }

  if (layer.startsWith('Conv') && p === 'W') {
    return `\\frac{\\partial L}{\\partial W} = \\sum_{i,j} \\frac{\\partial L}{\\partial y_{i,j}} \\cdot \\frac{\\partial y_{i,j}}{\\partial W}`;
  }

  if ((layer.startsWith('LayerNorm') || layer.startsWith('BatchNorm')) && p === 'gamma') {
    return `\\frac{\\partial L}{\\partial \\gamma} = \\sum_i \\frac{\\partial L}{\\partial y_i} \\cdot \\hat{x}_i`;
  }

  if ((layer.startsWith('LayerNorm') || layer.startsWith('BatchNorm')) && p === 'beta') {
    return `\\frac{\\partial L}{\\partial \\beta} = \\sum_i \\frac{\\partial L}{\\partial y_i}`;
  }

  if (layer.startsWith('Linear') || layer === 'Output') {
    return `\\frac{\\partial L}{\\partial ${p}} = \\frac{\\partial L}{\\partial y} \\cdot \\frac{\\partial y}{\\partial z} \\cdot \\frac{\\partial z}{\\partial ${p}}`;
  }

  return `\\frac{\\partial L}{\\partial ${p}} = \\sum_i \\frac{\\partial L}{\\partial h_i} \\cdot \\frac{\\partial h_i}{\\partial ${p}}`;
}

function buildBackSignalForLayer(layerLabel: string): string {
  if (
    layerLabel.startsWith('Input') ||
    layerLabel.startsWith('Input Seq') ||
    layerLabel.startsWith('Token') ||
    layerLabel.startsWith('Positional')
  ) {
    return '';
  }

  if (layerLabel.startsWith('RNN')) {
    return '\\frac{\\partial L}{\\partial h_{t-1}}';
  }
  if (layerLabel.startsWith('Self-Attn')) {
    return '\\left(\\frac{\\partial L}{\\partial Q_t},\\; \\frac{\\partial L}{\\partial K_t},\\; \\frac{\\partial L}{\\partial V_t}\\right)';
  }
  if (layerLabel.startsWith('Conv')) {
    return '\\frac{\\partial L}{\\partial x_{c,h,w}}';
  }
  if (layerLabel.startsWith('Linear') || layerLabel === 'Output') {
    return '\\frac{\\partial L}{\\partial x}';
  }
  return '\\frac{\\partial L}{\\partial x}';
}

function buildSequentialEdges(nodes: any[]): any[] {
  if (!nodes || nodes.length <= 1) return [];
  return nodes.slice(1).map((node, i) => ({
    id: `e-${nodes[i].id}-${node.id}`,
    source: nodes[i].id,
    target: node.id,
    animated: true,
    style: { stroke: '#94a3b8', strokeWidth: 2 },
  }));
}

export default function App() {
  const [nodes, setNodes, onNodesChange] = useNodesState([]);
  const [edges, setEdges, onEdgesChange] = useEdgesState([]);

  const [layers, setLayers] = useState(2);
  const [hiddenDim, setHiddenDim] = useState(16);
  const [inputDim, setInputDim] = useState(10);
  const [seqLen, setSeqLen] = useState(4);
  const [loading, setLoading] = useState(false);
  const [activation, setActivation] = useState<Activation>('ReLU');
  const [architecture, setArchitecture] = useState<Architecture>('mlp');
  const [attnHeads, setAttnHeads] = useState(4);
  const [initMode, setInitMode] = useState<InitMode>('random');
  const [initValue, setInitValue] = useState(0);
  const [inputValue, setInputValue] = useState(0.5);
  const [inputVector, setInputVector] = useState<number[]>(() =>
    Array.from({ length: 10 }, () => 0.5),
  );

  const [viewMode, setViewMode] = useState<VectorViewMode>('numbers');
  const [isBlockPanelOpen, setIsBlockPanelOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [pinnedNodeId, setPinnedNodeId] = useState<string | null>(null);
  const [activeParam, setActiveParam] = useState<ParamChip | null>(null);
  const [activeNodeParams, setActiveNodeParams] = useState<ParamChip[] | null>(null);
  const [activeLayerLabel, setActiveLayerLabel] = useState<string | null>(null);
  const [activeLayerDetails, setActiveLayerDetails] = useState<LayerDetails | null>(null);
  const [isAddingResidual, setIsAddingResidual] = useState(false);
  const [residualSourceId, setResidualSourceId] = useState<string | null>(null);
  const [paramOverrides, setParamOverrides] = useState<Record<string, number[]>>({});
  const [frozenParams, setFrozenParams] = useState<Record<string, number[]>>({});
  const [editingParam, setEditingParam] = useState<ParamChip | null>(null);
  const [editingParamValue, setEditingParamValue] = useState<number | null>(null);
  const [editingParamValues, setEditingParamValues] = useState<number[] | null>(null);
  const [editingParamShape, setEditingParamShape] = useState<number[] | null>(null);
  const [hoveredInputIndex, setHoveredInputIndex] = useState<number | null>(null);
  const [pinnedInputIndex, setPinnedInputIndex] = useState<number | null>(null);
  const [hoveredParamIndex, setHoveredParamIndex] = useState<number | null>(null);
  const [pinnedParamIndex, setPinnedParamIndex] = useState<number | null>(null);
  const [rnnStep, setRnnStep] = useState(0);
  const [rnnH0, setRnnH0] = useState<number[]>([]);
  const [activeH0Index, setActiveH0Index] = useState<number | null>(null);
  const [isAiOpen, setIsAiOpen] = useState(false);
  const [aiProvider, setAiProvider] = useState<AiProvider | ''>('');
  const [aiApiKey, setAiApiKey] = useState('');
  const [aiModel, setAiModel] = useState('');
  const [aiMessages, setAiMessages] = useState<AiMessage[]>([]);
  const [aiLoading, setAiLoading] = useState(false);

  const activeInputIndex = pinnedInputIndex ?? hoveredInputIndex;
  const activeParamIndex = pinnedParamIndex ?? hoveredParamIndex;

  const effectiveInputDim = useMemo(() => {
    if (architecture === 'cnn') {
      const size = Math.max(4, Math.min(8, inputDim || 4));
      return size * size;
    }
    if (architecture === 'rnn') {
      const T = Math.max(1, Math.min(4, seqLen || 1));
      const d = inputDim || 1;
      return T * d;
    }
    return inputDim;
  }, [architecture, inputDim, seqLen]);

  const runSimulation = useCallback(() => {
    setLoading(true);
    try {
      const mergedOverrides: Record<string, number[]> = {
        ...frozenParams,
        ...paramOverrides,
      };

      const { nodes: rawNodes, edges: rawEdges } = runBackpropSimulation({
        architecture,
        layers,
        hiddenDim,
        activation,
        inputDim,
        heads: attnHeads,
        initMode,
        initValue,
        inputValue,
        inputVector,
        paramOverrides: mergedOverrides,
        seqLen,
        rnnH0,
      });

      // If we don't yet have a frozen parameter snapshot, capture one from this run.
      setFrozenParams((prev) => {
        if (Object.keys(prev).length > 0) return prev;

        const next: Record<string, number[]> = {};
        for (const node of rawNodes) {
          const data = node.data as { details?: LayerDetails } | undefined;
          const details = data?.details;
          if (!details || !details.params) continue;
          const params = details.params as Record<string, ParamInfo>;
          Object.entries(params).forEach(([paramName, info]) => {
            if (!info.value_sample || !info.value_sample.length) return;
            const key = `${node.id}:${paramName}`;
            if (next[key] === undefined) {
              next[key] = info.value_sample.slice();
            }
          });
        }

        if (!Object.keys(next).length) return prev;
        return next;
      });

      const applyParamOverrides = (nodeList: typeof rawNodes) => {
        if (!nodeList || nodeList.length === 0) return nodeList;
        return nodeList.map((node) => {
          const data = node.data as {
            label?: string;
            details?: LayerDetails;
          };
          if (!data || !data.details || !data.details.params) return node;
          const params = data.details.params as Record<string, ParamInfo>;
          let changed = false;
          const nextParams: Record<string, ParamInfo> = { ...params };
          Object.keys(nextParams).forEach((paramName) => {
            const key = `${node.id}:${paramName}`;
            const overrideVals = mergedOverrides[key];
            if (overrideVals && overrideVals.length) {
              nextParams[paramName] = {
                ...nextParams[paramName],
                value_sample: overrideVals,
              };
              changed = true;
            }
          });
          if (!changed) return node;
          return {
            ...node,
            data: {
              ...data,
              details: {
                ...data.details,
                params: nextParams,
              },
            },
          };
        });
      };

      setNodes((prevNodes) => {
        if (prevNodes.length === 0 || prevNodes.length !== rawNodes.length) {
          const layouted = getLayoutedElements(rawNodes, rawEdges);
          setEdges(layouted.edges);
          return applyParamOverrides(layouted.nodes);
        }

        const byId = new Map(rawNodes.map((n) => [n.id, n]));
        const merged = prevNodes.map((oldNode) => {
          const fresh = byId.get(oldNode.id);
          if (!fresh) return oldNode;
          return {
            ...fresh,
            position: oldNode.position,
            dragging: false,
            selected: oldNode.selected,
          };
        });
        setEdges(rawEdges);
        return applyParamOverrides(merged);
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [
    architecture,
    layers,
    hiddenDim,
    activation,
    inputDim,
    attnHeads,
    initMode,
    initValue,
    inputValue,
    inputVector,
    setNodes,
    setEdges,
    paramOverrides,
    frozenParams,
    seqLen,
    rnnH0,
  ]);

  useEffect(() => {
    runSimulation();
  }, [
    runSimulation,
    architecture,
    layers,
    hiddenDim,
    activation,
    inputDim,
    attnHeads,
    initMode,
    initValue,
    inputValue,
    inputVector,
    paramOverrides,
    seqLen,
    rnnH0,
  ]);

  // When switching to CNN, set sensible defaults for kernel size and input dim.
  useEffect(() => {
    if (architecture !== 'cnn') return;

    const defaultKernel = 3;
    const defaultDim = 8;

    setHiddenDim(defaultKernel);
    setInputDim(defaultDim);

    const dimForVector = defaultDim * defaultDim;
    setInputVector((prev) => {
      const next = Array.from({ length: dimForVector }, (_, i) => prev[i] ?? 0.5);
      const mean =
        next.length > 0 ? next.reduce((a, b) => a + b, 0) / next.length : 0;
      setInputValue(mean);
      return next;
    });
  }, [architecture]);

  useEffect(() => {
    if (architecture !== 'rnn') return;
    const T = Math.max(1, Math.min(4, seqLen || 1));
    const id = window.setInterval(() => {
      setRnnStep((s) => ((s + 1) % T + T) % T);
    }, 900);
    return () => window.clearInterval(id);
  }, [architecture, seqLen]);

  useEffect(() => {
    if (architecture !== 'rnn') {
      setRnnH0([]);
      return;
    }
    const H = Math.max(1, hiddenDim || 1);
    setRnnH0((prev) => {
      const next = Array.from({ length: H }, (_, i) => prev[i] ?? 0);
      return next;
    });
  }, [architecture, hiddenDim]);

  useEffect(() => {
    if (architecture !== 'rnn') return;
    setNodes((prev) =>
      prev.map((node) => {
        const data = node.data as {
          label?: string;
          details?: LayerDetails;
          rnnStep?: number;
        };
        return {
          ...node,
          data: {
            ...data,
            rnnStep,
          },
        };
      }),
    );
  }, [architecture, rnnStep, setNodes]);

  const highlightPathTo = useCallback(
    (nodeId: string | null) => {
      setEdges((prevEdges) => {
        if (!nodeId) {
          return prevEdges.map((e) => ({
            ...e,
            style: { ...(e.style || {}), stroke: '#94a3b8', strokeWidth: 2 },
          }));
        }

        const active = new Set<string>();
        const visitForward = (sourceId: string) => {
          prevEdges.forEach((e) => {
            if (e.source === sourceId) {
              active.add(e.id as string);
              visitForward(e.target as string);
            }
          });
        };
        visitForward(nodeId);

        return prevEdges.map((e) => ({
          ...e,
          style: {
            ...(e.style || {}),
            stroke: active.has(e.id as string) ? '#f97316' : '#94a3b8',
            strokeWidth: active.has(e.id as string) ? 3 : 2,
          },
        }));
      });
    },
    [setEdges],
  );

  const allParams: ParamChip[] = useMemo(
    () =>
      nodes.flatMap((node) => {
        const data = node.data as {
          label?: string;
          details?: { params?: Record<string, unknown> };
        };
        if (!data || !data.details || !data.details.params) return [];
        return Object.keys(data.details.params).map((paramName) => ({
          id: `${node.id}-${paramName}`,
          nodeId: node.id,
          layerLabel: data.label ?? node.id,
          paramName,
        }));
      }),
    [nodes],
  );

  const activeChainRules = useMemo(
    () => {
      const source: ParamChip[] = activeParam ? [activeParam] : activeNodeParams || [];
      return source.map((chip) => ({
        chip,
        latex: buildChainRuleForParam(chip, architecture),
      }));
    },
    [activeParam, activeNodeParams, architecture],
  );

  const effectiveLayerLabel = activeLayerLabel
    ? activeLayerLabel
    : pinnedNodeId
    ? (() => {
        const node = nodes.find((n) => n.id === pinnedNodeId);
        const data = node?.data as { label?: string } | undefined;
        return data?.label ?? null;
      })()
    : null;

  const overlayBackSignal = useMemo(
    () => (effectiveLayerLabel ? buildBackSignalForLayer(effectiveLayerLabel) : null),
    [effectiveLayerLabel],
  );

  const overlayActivations = useMemo(() => {
    if (!activeLayerDetails) return null;

    const cap = 16;

    const isInputLike =
      effectiveLayerLabel &&
      (effectiveLayerLabel.startsWith('Input') ||
        effectiveLayerLabel.startsWith('Input Seq') ||
        effectiveLayerLabel.startsWith('Token') ||
        effectiveLayerLabel.startsWith('Positional'));

    if (isInputLike) {
      const srcDimBase = inputDim || inputVector.length || 1;
      const srcDim =
        architecture === 'cnn' ? effectiveInputDim || inputVector.length || srcDimBase : srcDimBase;
      const L = architecture === 'cnn' ? srcDim : Math.min(srcDim, cap);
      const baseVec =
        inputVector.length >= L
          ? inputVector.slice(0, L)
          : Array.from({ length: L }, () => inputValue);

      // For input layers we only really use outVec; x(in) row is hidden in the UI.
      return { inVec: baseVec, outVec: baseVec };
    }

    if (architecture === 'rnn' && effectiveLayerLabel === 'y_t') {
      const rnnNodes = nodes.filter((n) => {
        const data = n.data as { label?: string; details?: LayerDetails } | undefined;
        const lbl = data?.label ?? '';
        return lbl.startsWith('RNN');
      });

      if (rnnNodes.length) {
        const last = rnnNodes[rnnNodes.length - 1];
        const lastDetails = (last.data as { details?: LayerDetails }).details;

        if (lastDetails && Array.isArray(lastDetails.in_shape) && Array.isArray(lastDetails.out_shape)) {
          let T = typeof lastDetails.in_shape[0] === 'number' ? lastDetails.in_shape[0] : 0;
          let d = typeof lastDetails.in_shape[1] === 'number' ? lastDetails.in_shape[1] : 0;
          let H = typeof lastDetails.out_shape[1] === 'number' ? lastDetails.out_shape[1] : 0;

          if (!T) {
            T = Math.max(1, Math.min(4, seqLen || 1));
          }
          if (!d) {
            d = Math.max(1, Math.min(16, inputDim || 1));
          }
          if (!H) {
            H = Math.max(1, Math.min(16, hiddenDim || 1));
          }

          const tActive = ((rnnStep % T) + T) % T;
          const flatIn = (lastDetails.input_sample || []) as number[];
          const flatH = (lastDetails.output_sample || []) as number[];

          const getRow = (flatArr: number[], rowLen: number, t: number): number[] => {
            if (rowLen <= 0) return [];
            if (!flatArr.length) {
              return Array.from({ length: rowLen }, () => 0);
            }
            const start = t * rowLen;
            const end = start + rowLen;
            if (end <= flatArr.length) {
              return flatArr.slice(start, end);
            }
            const row: number[] = [];
            for (let j = 0; j < rowLen; j++) {
              const idx = (start + j) % flatArr.length;
              row.push(flatArr[idx]);
            }
            return row;
          };

          const xRow = getRow(flatIn, d, tActive);
          const hRow = getRow(flatH, H, tActive);

          const params = (lastDetails.params || {}) as Record<string, ParamInfo>;
          const Wy = params.W_y;
          const Uy = params.U_y;
          const By = params.b_y;

          const computeY = (): number[] => {
            if (
              Wy && Uy && By &&
              Array.isArray(Wy.shape) && Wy.shape.length === 2 &&
              Array.isArray(Uy.shape) && Uy.shape.length === 2 &&
              Array.isArray(By.shape) && By.shape.length === 1 &&
              Wy.value_sample.length &&
              Uy.value_sample.length &&
              By.value_sample.length &&
              Wy.shape[0] === H &&
              Wy.shape[1] === H &&
              Uy.shape[0] === d &&
              Uy.shape[1] === H &&
              By.shape[0] === H
            ) {
              const y: number[] = [];
              const wyVals = Wy.value_sample;
              const uyVals = Uy.value_sample;
              const byVals = By.value_sample;
              for (let j = 0; j < H; j++) {
                let sum = 0;
                for (let k = 0; k < H; k++) {
                  const w = wyVals[k * H + j] ?? 0;
                  sum += (hRow[k] ?? 0) * w;
                }
                for (let i = 0; i < d; i++) {
                  const w = uyVals[i * H + j] ?? 0;
                  sum += (xRow[i] ?? 0) * w;
                }
                sum += byVals[j] ?? 0;
                y.push(sum);
              }
              return y;
            }
            return hRow;
          };

          const yRow = computeY();

          return {
            inVec: yRow.slice(0, cap),
            outVec: yRow.slice(0, cap),
          };
        }
      }
    }

    if (
      architecture === 'rnn' &&
      effectiveLayerLabel &&
      effectiveLayerLabel.startsWith('RNN') &&
      activeLayerDetails
    ) {
      const inShape = activeLayerDetails.in_shape as number[] | string;
      const outShape = activeLayerDetails.out_shape as number[] | string;

      let T = 0;
      let d = 0;
      let H = 0;

      if (Array.isArray(inShape) && inShape.length >= 2) {
        T = typeof inShape[0] === 'number' ? inShape[0] : 0;
        d = typeof inShape[1] === 'number' ? inShape[1] : 0;
      }
      if (Array.isArray(outShape) && outShape.length >= 2) {
        if (!T) {
          T = typeof outShape[0] === 'number' ? outShape[0] : 0;
        }
        H = typeof outShape[1] === 'number' ? outShape[1] : 0;
      }

      if (!T) {
        T = Math.max(1, Math.min(4, seqLen || 1));
      }
      if (!d) {
        d = Math.max(1, Math.min(16, inputDim || 1));
      }
      if (!H) {
        H = Math.max(1, Math.min(16, hiddenDim || 1));
      }

      const tActive = ((rnnStep % T) + T) % T;

      const flatIn = (activeLayerDetails.input_sample && activeLayerDetails.input_sample.length
        ? activeLayerDetails.input_sample
        : activeLayerDetails.output_sample || []) as number[];
      const flatOut = (activeLayerDetails.output_sample && activeLayerDetails.output_sample.length
        ? activeLayerDetails.output_sample
        : activeLayerDetails.input_sample || []) as number[];

      const getRow = (flat: number[], rowLen: number, t: number): number[] => {
        if (rowLen <= 0) return [];
        if (!flat.length) {
          return Array.from({ length: rowLen }, () => 0);
        }
        const start = t * rowLen;
        const end = start + rowLen;
        if (end <= flat.length) {
          return flat.slice(start, end);
        }
        const row: number[] = [];
        for (let j = 0; j < rowLen; j++) {
          const idx = (start + j) % flat.length;
          row.push(flat[idx]);
        }
        return row;
      };

      const xRow = getRow(flatIn, d, tActive);
      const hRow = getRow(flatOut, H, tActive);

      return {
        inVec: xRow.slice(0, cap),
        outVec: hRow.slice(0, cap),
      };
    }

    const inSample = activeLayerDetails.input_sample;
    const outSample = activeLayerDetails.output_sample;

    if ((inSample && inSample.length) || (outSample && outSample.length)) {
      const inSource = inSample && inSample.length ? inSample : outSample ?? [];
      const outSource = outSample && outSample.length ? outSample : inSample ?? [];

      const inVec = inSource.slice(0, cap);
      const outVec = outSource.slice(0, cap);

      return { inVec, outVec };
    }

    const getDim = (shape: number[] | string | undefined): number => {
      if (!shape || typeof shape === 'string') return 4;
      if (Array.isArray(shape) && shape.length >= 2) {
        return shape[shape.length - 1] ?? shape[0] ?? 4;
      }
      if (Array.isArray(shape) && shape.length === 1) return shape[0] ?? 4;
      return 4;
    };

    const lenIn = getDim(activeLayerDetails.in_shape as any);
    const lenOut = getDim(activeLayerDetails.out_shape as any);

    const makeVec = (len: number, center: number): number[] => {
      const L = Math.min(len || 4, cap);
      const res: number[] = [];
      for (let i = 0; i < L; i++) {
        const t = L === 1 ? 0 : (i / (L - 1)) * 2 - 1;
        res.push(center + t * 0.2);
      }
      return res;
    };

    const isInputLayer =
      architecture === 'mlp' && effectiveLayerLabel && effectiveLayerLabel.startsWith('Input');

    let inVec: number[];
    let outVec: number[];

    if (isInputLayer) {
      const L = Math.min(inputVector.length || lenOut || 4, cap);
      const base = inputVector.slice(0, L);
      outVec = base.length === L ? base : makeVec(lenOut, activeLayerDetails.forward_mean ?? 0);
      inVec = makeVec(lenIn, 0);
    } else {
      inVec = makeVec(lenIn, 0);
      outVec = makeVec(lenOut, activeLayerDetails.forward_mean ?? 0);
      if (effectiveLayerLabel && effectiveLayerLabel.startsWith('ReLU')) {
        outVec = outVec.map((v) => (v < 0 ? 0 : v));
      }
    }

    return { inVec, outVec };
  }, [
    activeLayerDetails,
    effectiveLayerLabel,
    architecture,
    inputVector,
    inputDim,
    inputValue,
    nodes,
    seqLen,
    hiddenDim,
    rnnStep,
  ]);

  const renderVector = useCallback(
    (vec: number[]) => {
      if (viewMode === 'numbers') {
        return <span>[ {vec.map((v) => v.toFixed(2)).join(', ')} ]</span>;
      }
      return (
        <div className="flex gap-[2px]">
          {vec.map((v, idx) => {
            return (
              <div
                key={idx}
                className="w-3 h-3 rounded-sm"
                style={{ backgroundColor: valueToColor(v) }}
              />
            );
          })}
        </div>
      );
    },
    [viewMode],
  );

  const ioVectors = useMemo(() => {
    if (!nodes.length) return null;

    const getDim = (shape: number[] | string | undefined): number => {
      if (!shape || typeof shape === 'string') return 4;
      if (Array.isArray(shape) && shape.length >= 2) {
        return shape[shape.length - 1] ?? shape[0] ?? 4;
      }
      if (Array.isArray(shape) && shape.length === 1) return shape[0] ?? 4;
      return 4;
    };

    const makeVec = (len: number, center: number): number[] => {
      const cap = 16;
      const L = Math.min(len || 4, cap);
      const res: number[] = [];
      for (let i = 0; i < L; i++) {
        const t = L === 1 ? 0 : (i / (L - 1)) * 2 - 1;
        res.push(center + t * 0.2);
      }
      return res;
    };

    const getLabel = (node: any): string => {
      const data = node.data as { label?: string };
      return data?.label ?? '';
    };

    const inputNode =
      nodes.find((n) => {
        const label = getLabel(n);
        return (
          label.startsWith('Input') ||
          label.startsWith('Token') ||
          label.startsWith('Positional')
        );
      }) ?? nodes[0];

    const outputNode =
      [...nodes].reverse().find((n) => {
        const label = getLabel(n);
        if (label === 'Loss') return false;
        if (label === 'Output') return true;
        if (label === 'y_t') return true;
        if (label.startsWith('Encoder Output')) return true;
        return false;
      }) ?? nodes[nodes.length - 1];

    const inDetails = (inputNode.data as any)?.details as LayerDetails | undefined;
    const outDetails = (outputNode.data as any)?.details as LayerDetails | undefined;

    if (!inDetails || !outDetails) return null;

    const lenIn = getDim(inDetails.out_shape as any);
    const lenOut = getDim(outDetails.out_shape as any);
    const cap = 16;

    const isMLPInput = architecture === 'mlp' && getLabel(inputNode).startsWith('Input');

    const inputLabel = getLabel(inputNode);
    const isInputLikeNode =
      inputLabel.startsWith('Input') ||
      inputLabel.startsWith('Input Seq') ||
      inputLabel.startsWith('Token') ||
      inputLabel.startsWith('Positional');

    let inVec: number[];
    if (isInputLikeNode) {
      const srcDimBase = inputDim || inputVector.length || lenIn || 1;
      const srcDim =
        architecture === 'cnn' ? effectiveInputDim || inputVector.length || srcDimBase : srcDimBase;
      const L = architecture === 'cnn' ? srcDim : Math.min(srcDim, cap);
      const src =
        inputVector.length >= L
          ? inputVector
          : Array.from({ length: srcDim }, () => inputValue);
      inVec = src.slice(0, L);
    } else if (inDetails.output_sample && inDetails.output_sample.length) {
      inVec = inDetails.output_sample.slice(0, cap);
    } else if (isMLPInput) {
      const L = Math.min(inputVector.length || lenIn || 4, cap);
      const base = inputVector.slice(0, L);
      const inCenter = inDetails.forward_mean ?? 0;
      inVec = base.length === L ? base : makeVec(lenIn, inCenter);
    } else {
      const inCenter = inDetails.forward_mean ?? 0;
      inVec = makeVec(lenIn, inCenter);
    }

    let outVec: number[];
    if (outDetails.output_sample && outDetails.output_sample.length) {
      outVec = outDetails.output_sample.slice(0, cap);
    } else {
      const outCenter = outDetails.forward_mean ?? 0;
      outVec = makeVec(lenOut, outCenter);
    }

    return {
      inputLabel: getLabel(inputNode) || 'Input',
      outputLabel: getLabel(outputNode) || 'Output',
      inVec,
      outVec,
      inShape: inDetails.out_shape,
      outShape: outDetails.out_shape,
    } as {
      inputLabel: string;
      outputLabel: string;
      inVec: number[];
      outVec: number[];
      inShape?: number[] | string;
      outShape?: number[] | string;
    };
  }, [nodes, architecture, inputVector]);

  const handleInputEntryChange = useCallback(
    (index: number, value: number) => {
      const clamped = Math.max(-1, Math.min(1, value));
      setInputVector((prev) => {
        const dim = effectiveInputDim || prev.length || 1;
        const next = Array.from({ length: dim }, (_, i) =>
          i === index ? clamped : prev[i] ?? 0,
        );
        const mean =
          next.length > 0 ? next.reduce((a, b) => a + b, 0) / next.length : 0;
        setInputValue(mean);
        return next;
      });
    },
    [inputDim, effectiveInputDim],
  );

  const handleParamValueChange = useCallback(
    (value: number) => {
      if (!editingParam || !editingParamValues) return;
      setEditingParamValue(value);

      const count = editingParamValues.length || 1;
      const filled = Array.from({ length: count }, () => value);
      setEditingParamValues(filled);

      const key = `${editingParam.nodeId}:${editingParam.paramName}`;
      setParamOverrides((prev) => ({
        ...prev,
        [key]: filled,
      }));

      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          if (node.id !== editingParam.nodeId) return node;
          const data = node.data as {
            label?: string;
            details?: LayerDetails;
          };
          if (!data || !data.details || !data.details.params) return node;
          const params = data.details.params as Record<string, ParamInfo>;
          if (!params[editingParam.paramName]) return node;
          const nextParams: Record<string, ParamInfo> = {
            ...params,
            [editingParam.paramName]: {
              ...params[editingParam.paramName],
              value_sample: filled,
            },
          };
          return {
            ...node,
            data: {
              ...data,
              details: {
                ...data.details,
                params: nextParams,
              },
            },
          };
        }),
      );
    },
    [editingParam, editingParamValues, setNodes],
  );

  const handleAiAsk = useCallback(
    async (question: string) => {
      const q = question.trim();
      if (!q || !aiProvider || !aiApiKey) return;

      setAiMessages((prev) => [...prev, { role: 'user', content: q }]);
      setAiLoading(true);
      try {
        const contextParts: string[] = [];
        contextParts.push('You answer questions about the math and theory of neural networks.');
        contextParts.push(`Current architecture: ${architecture}.`);
        if (activeLayerLabel) {
          contextParts.push(`Current layer or block: ${activeLayerLabel}.`);
        }
        const describeShape = (shape: number[] | string | undefined): string => {
          if (!shape) return '?';
          if (typeof shape === 'string') return shape;
          if (!Array.isArray(shape)) return '?';
          return `[${shape.join(', ')}]`;
        };

        const layersSummary = nodes
          .map((node, idx) => {
            const data = node.data as {
              label?: string;
              details?: LayerDetails;
            } | undefined;
            if (!data) return null;
            const label = data.label ?? `Layer ${idx + 1}`;
            const details = data.details;
            if (!details) {
              return `${idx + 1}. ${label}`;
            }
            const inS = describeShape(details.in_shape as any);
            const outS = describeShape(details.out_shape as any);
            return `${idx + 1}. ${label}: ${inS} -> ${outS}`;
          })
          .filter((x): x is string => Boolean(x))
          .join(' ');

        if (layersSummary) {
          contextParts.push('Current architecture sequence with shapes (in -> out):');
          contextParts.push(layersSummary);
        }
        const systemPrompt = contextParts.join(' ');

        let answer = '';

        if (aiProvider === 'openai') {
          const body = {
            model: aiModel || 'gpt-4.1-mini',
            messages: [
              { role: 'system', content: systemPrompt },
              { role: 'user', content: q },
            ],
          };
          const resp = await fetch('https://api.openai.com/v1/chat/completions', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              Authorization: `Bearer ${aiApiKey}`,
            },
            body: JSON.stringify(body),
          });
          const data = await resp.json();
          answer =
            data.choices?.[0]?.message?.content?.trim() ??
            'The model did not return any content.';
        } else if (aiProvider === 'anthropic') {
          const body = {
            model: aiModel || 'claude-3-5-sonnet-20241022',
            max_tokens: 512,
            system: systemPrompt,
            messages: [
              {
                role: 'user',
                content: q,
              },
            ],
          };
          const resp = await fetch('https://api.anthropic.com/v1/messages', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
              'x-api-key': aiApiKey,
              'anthropic-version': '2023-06-01',
            },
            body: JSON.stringify(body),
          });
          const data = await resp.json();
          if (Array.isArray(data.content)) {
            answer = data.content
              .map((part: any) => part.text || '')
              .join(' ')
              .trim();
          }
          if (!answer) {
            answer = 'The model did not return any content.';
          }
        } else if (aiProvider === 'google') {
          const modelId = aiModel || 'gemini-1.5-flash';
          const url = `https://generativelanguage.googleapis.com/v1beta/models/${modelId}:generateContent?key=${encodeURIComponent(
            aiApiKey,
          )}`;
          const body = {
            contents: [
              {
                parts: [
                  { text: systemPrompt },
                  { text: `User question: ${q}` },
                ],
              },
            ],
          };
          const resp = await fetch(url, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
          });
          const data = await resp.json();
          if (Array.isArray(data.candidates) && data.candidates.length > 0) {
            const parts = data.candidates[0].content?.parts;
            if (Array.isArray(parts)) {
              answer = parts
                .map((p: any) => p.text || '')
                .join(' ')
                .trim();
            }
          }
          if (!answer) {
            answer = 'The model did not return any content.';
          }
        }

        setAiMessages((prev) => [...prev, { role: 'assistant', content: answer }]);
      } catch (error: any) {
        const message =
          error?.message || 'Error calling the model. Check your API key and model.';
        setAiMessages((prev) => [
          ...prev,
          { role: 'assistant', content: message },
        ]);
      } finally {
        setAiLoading(false);
      }
    },
    [aiProvider, aiApiKey, aiModel, architecture, activeLayerLabel, nodes],
  );

  const handleParamElementChange = useCallback(
    (index: number, value: number) => {
      if (!editingParam || !editingParamValues) return;

      const clampedIndex = Math.max(0, Math.min(index, editingParamValues.length - 1));
      const nextValues = editingParamValues.map((v, i) => (i === clampedIndex ? value : v));
      setEditingParamValues(nextValues);
      setEditingParamValue(value);

      const key = `${editingParam.nodeId}:${editingParam.paramName}`;
      setParamOverrides((prev) => ({
        ...prev,
        [key]: nextValues,
      }));

      setNodes((prevNodes) =>
        prevNodes.map((node) => {
          if (node.id !== editingParam.nodeId) return node;
          const data = node.data as {
            label?: string;
            details?: LayerDetails;
          };
          if (!data || !data.details || !data.details.params) return node;
          const params = data.details.params as Record<string, ParamInfo>;
          if (!params[editingParam.paramName]) return node;
          const nextParams: Record<string, ParamInfo> = {
            ...params,
            [editingParam.paramName]: {
              ...params[editingParam.paramName],
              value_sample: nextValues,
            },
          };
          return {
            ...node,
            data: {
              ...data,
              details: {
                ...data.details,
                params: nextParams,
              },
            },
          };
        }),
      );
    },
    [editingParam, editingParamValues, setNodes],
  );

  const handleBlockDragStart = useCallback((event: any, template: BlockTemplate) => {
    if (!event?.dataTransfer) return;
    event.dataTransfer.setData('application/x-block-template', JSON.stringify(template));
    event.dataTransfer.effectAllowed = 'move';
  }, []);

  const handleCanvasDrop = useCallback(
    (event: any) => {
      event.preventDefault();
      if (!event?.dataTransfer) return;
      const raw = event.dataTransfer.getData('application/x-block-template');
      if (!raw) return;

      let template: BlockTemplate;
      try {
        template = JSON.parse(raw);
      } catch {
        return;
      }

      const rect = event.currentTarget.getBoundingClientRect();
      const ratio = rect.width > 0 ? (event.clientX - rect.left) / rect.width : 0;
      const clamped = Math.max(0, Math.min(0.999, ratio));

      setNodes((prevNodes) => {
        const id = `custom-${Math.random().toString(36).slice(2)}`;
        const details: {
          in_shape: any;
          out_shape: number[];
          forward_mean: number;
          params: Record<string, { shape: number[]; grad_mean: number; grad_std: number; value_sample: number[] }>;
        } = {
          in_shape: '-',
          out_shape: [1, 1],
          forward_mean: 0,
          params: {},
        };

        if (template.type === 'LayerNorm' || template.type === 'BatchNorm') {
          details.params = {
            gamma: {
              shape: [1],
              grad_mean: Math.random() * 0.1,
              grad_std: 0,
              value_sample: [],
            },
            beta: {
              shape: [1],
              grad_mean: Math.random() * 0.1,
              grad_std: 0,
              value_sample: [],
            },
          };
        }

        const newNode = {
          id,
          type: 'customLayer',
          position: { x: 0, y: 0 },
          data: {
            label: template.label,
            details,
          },
        };

        const nextNodes = [...prevNodes];
        const insertIndex = nextNodes.length ? Math.round(clamped * nextNodes.length) : 0;
        nextNodes.splice(insertIndex, 0, newNode);

        const baseEdges = buildSequentialEdges(nextNodes);
        const layouted = getLayoutedElements(nextNodes, baseEdges);
        setEdges(layouted.edges);
        return layouted.nodes;
      });
    },
    [setNodes, setEdges],
  );

  const insertBlock = useCallback(
    (where: 'before' | 'after' | 'end', template: BlockTemplate) => {
      setNodes((prevNodes) => {
        const id = `custom-${Math.random().toString(36).slice(2)}`;
        const details: {
          in_shape: any;
          out_shape: number[];
          forward_mean: number;
          params: Record<string, { shape: number[]; grad_mean: number; grad_std: number; value_sample: number[] }>;
        } = {
          in_shape: '-',
          out_shape: [1, 1],
          forward_mean: 0,
          params: {},
        };

        if (template.type === 'LayerNorm' || template.type === 'BatchNorm') {
          details.params = {
            gamma: {
              shape: [1],
              grad_mean: Math.random() * 0.1,
              grad_std: 0,
              value_sample: [],
            },
            beta: {
              shape: [1],
              grad_mean: Math.random() * 0.1,
              grad_std: 0,
              value_sample: [],
            },
          };
        }

        const newNode = {
          id,
          type: 'customLayer',
          position: { x: 0, y: 0 },
          data: {
            label: template.label,
            details,
          },
        };

        const nextNodes = [...prevNodes];

        if (where === 'end' || !selectedNodeId) {
          nextNodes.push(newNode);
        } else {
          const targetIndex = nextNodes.findIndex((n) => n.id === selectedNodeId);
          if (targetIndex === -1) {
            nextNodes.push(newNode);
          } else if (where === 'before') {
            nextNodes.splice(targetIndex, 0, newNode);
          } else {
            nextNodes.splice(targetIndex + 1, 0, newNode);
          }
        }

        const baseEdges = buildSequentialEdges(nextNodes);
        const layouted = getLayoutedElements(nextNodes, baseEdges);
        setEdges(layouted.edges);
        return layouted.nodes;
      });
    },
    [selectedNodeId, setNodes, setEdges],
  );

  return (
    <div className="relative h-screen w-screen bg-slate-950 text-white flex flex-col">
      <div className="h-16 border-b border-slate-800 bg-slate-900 flex items-center px-6 justify-between z-10">
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2">
            <Activity className="text-blue-500" />
            <h1 className="font-bold text-xl tracking-tight">Activation Visualizer</h1>
          </div>
          <button
            type="button"
            onClick={() => setIsBlockPanelOpen((open) => !open)}
            className="bg-slate-800 hover:bg-slate-700 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-600"
          >
            {isBlockPanelOpen ? 'Close Blocks' : 'Blocks'}
          </button>
          <button
            type="button"
            onClick={() => setIsAiOpen((open) => !open)}
            className="bg-slate-800 hover:bg-slate-700 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-600"
          >
            {isAiOpen ? 'Hide AI' : 'AI Assistant'}
          </button>
        </div>

        <div className="flex items-center gap-4 bg-slate-800 p-1 rounded-lg">
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">Hidden Layers:</span>
            <input
              type="number"
              value={layers}
              onChange={(e) => setLayers(Number(e.target.value))}
              className="w-12 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
            />
          </div>
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">
              {architecture === 'cnn' ? 'Kernel (K):' : 'Dim:'}
            </span>
            <input
              type="number"
              value={hiddenDim}
              onChange={(e) => setHiddenDim(Number(e.target.value))}
              className="w-12 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
            />
          </div>
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">
              {architecture === 'cnn'
                ? 'Input Dim (H=W):'
                : architecture === 'rnn'
                ? 'Input Dim (d):'
                : 'Input Dim:'}
            </span>
            <input
              type="number"
              value={inputDim}
              min={architecture === 'cnn' ? 4 : 1}
              max={architecture === 'cnn' ? 8 : 16}
              onChange={(e) => {
                const raw = Number(e.target.value) || 1;
                const isCNN = architecture === 'cnn';
                const isRNN = architecture === 'rnn';
                const minDim = isCNN ? 4 : 1;
                const maxDim = isCNN ? 8 : 16;
                const dim = Math.max(minDim, Math.min(maxDim, raw));
                setInputDim(dim);

                const T = isRNN ? Math.max(1, Math.min(4, seqLen || 1)) : 1;
                const dimForVector = isCNN ? dim * dim : isRNN ? T * dim : dim;
                setInputVector((prev) => {
                  const next = Array.from({ length: dimForVector }, (_, i) => prev[i] ?? 0.5);
                  const mean =
                    next.length > 0
                      ? next.reduce((a, b) => a + b, 0) / next.length
                      : 0;
                  setInputValue(mean);
                  return next;
                });
              }}
              className="w-16 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
            />
          </div>
          {architecture === 'rnn' && (
            <div className="flex items-center gap-2 px-3">
              <span className="text-xs text-slate-400">Seq Len:</span>
              <input
                type="number"
                value={seqLen}
                min={1}
                max={4}
                onChange={(e) => {
                  const raw = Number(e.target.value) || 1;
                  const T = Math.max(1, Math.min(4, raw));
                  setSeqLen(T);

                  const d = inputDim || 1;
                  const dimForVector = T * d;
                  setInputVector((prev) => {
                    const next = Array.from({ length: dimForVector }, (_, i) => prev[i] ?? 0.5);
                    const mean =
                      next.length > 0
                        ? next.reduce((a, b) => a + b, 0) / next.length
                        : 0;
                    setInputValue(mean);
                    return next;
                  });
                }}
                className="w-16 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
              />
            </div>
          )}
          {architecture === 'rnn' && (
            <div className="flex items-center gap-2 px-3">
              <span className="text-xs text-slate-400">h0:</span>
              <button
                type="button"
                onClick={() => {
                  const H = Math.max(1, hiddenDim || 1);
                  const rand = () => Math.random() * 2 - 1;
                  const vec = Array.from({ length: H }, () => rand());
                  setRnnH0(vec);
                  setParamOverrides((prev) => {
                    const next = { ...prev };
                    Object.keys(next).forEach((key) => {
                      if (key.endsWith(':h0')) {
                        delete next[key];
                      }
                    });
                    return next;
                  });
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white px-2 py-1 rounded-md text-[11px] border border-slate-500"
              >
                Randomize all h0 (same)
              </button>
              <button
                type="button"
                onClick={() => {
                  const H = Math.max(1, hiddenDim || 1);
                  const L = Math.max(1, layers || 1);
                  const rand = () => Math.random() * 2 - 1;
                  setParamOverrides((prev) => {
                    const next = { ...prev };
                    for (let layerIdx = 0; layerIdx < L; layerIdx++) {
                      const nodeIndex = 1 + layerIdx;
                      const key = `layer-${nodeIndex}:h0`;
                      next[key] = Array.from({ length: H }, () => rand());
                    }
                    return next;
                  });
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white px-2 py-1 rounded-md text-[11px] border border-slate-500"
              >
                Randomize all h0 (different)
              </button>
              <button
                type="button"
                onClick={() => {
                  const H = Math.max(1, hiddenDim || 1);
                  const zeros = Array.from({ length: H }, () => 0);
                  setRnnH0(zeros);
                  setParamOverrides((prev) => {
                    const next = { ...prev };
                    Object.keys(next).forEach((key) => {
                      if (key.endsWith(':h0')) {
                        delete next[key];
                      }
                    });
                    return next;
                  });
                }}
                className="bg-slate-700 hover:bg-slate-600 text-white px-2 py-1 rounded-md text-[11px] border border-slate-500"
              >
                Set all h0 = 0
              </button>
            </div>
          )}
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">Activation:</span>
            <select
              value={activation}
              onChange={(e) => setActivation(e.target.value as Activation)}
              className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs"
            >
              <option value="ReLU">ReLU</option>
              <option value="Tanh">Tanh</option>
              <option value="Sigmoid">Sigmoid</option>
            </select>
          </div>
          {architecture === 'transformer' && (
            <div className="flex items-center gap-2 px-3">
              <span className="text-xs text-slate-400">Heads:</span>
              <select
                value={attnHeads}
                onChange={(e) => setAttnHeads(Number(e.target.value) || 1)}
                className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs"
              >
                <option value={1}>1 (single)</option>
                <option value={4}>4 (multi-head)</option>
              </select>
            </div>
          )}
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">Architecture:</span>
            <select
              value={architecture}
              onChange={(e) => setArchitecture(e.target.value as Architecture)}
              className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs"
            >
              <option value="mlp">MLP</option>
              <option value="cnn">CNN</option>
              <option value="rnn">RNN</option>
              <option value="transformer">Transformer</option>
            </select>
          </div>
          <button
            type="button"
            onClick={() => {
              setInitMode('random');
              setParamOverrides({});
              setFrozenParams({});
              setEditingParam(null);
              setEditingParamValue(null);
            }}
            className="bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-500"
          >
            Randomize params
          </button>
          <button
            type="button"
            onClick={() => {
              const dim = effectiveInputDim || inputDim || inputVector.length || 1;
              const rand = () => Math.random() * 2 - 1; // [-1,1]
              setInputVector(() => {
                const next = Array.from({ length: dim }, () => rand());
                const mean =
                  next.length > 0
                    ? next.reduce((a, b) => a + b, 0) / next.length
                    : 0;
                setInputValue(mean);
                return next;
              });
            }}
            className="bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-500"
          >
            Randomize x
          </button>
        </div>
      </div>

      {allParams.length > 0 && (
        <div className="h-10 border-b border-slate-800 bg-slate-900/80 flex items-center px-6 gap-2 overflow-x-auto text-[11px] font-mono">
          <span className="uppercase tracking-wider text-slate-500 mr-2 flex-shrink-0">
            Params
          </span>
          {allParams.map((p) => (
            <button
              key={p.id}
              onMouseEnter={() => {
                highlightPathTo(p.nodeId);
                setActiveParam(p);
                setActiveNodeParams(null);
                setActiveLayerLabel(p.layerLabel);
                const node = nodes.find((n) => n.id === p.nodeId);
                if (node && (node.data as any)?.details) {
                  setActiveLayerDetails((node.data as any).details as LayerDetails);
                }
                setNodes((prev) =>
                  prev.map((n) => ({
                    ...n,
                    selected: n.id === p.nodeId,
                  })),
                );
              }}
              onMouseLeave={() => {
                if (pinnedNodeId) {
                  const pinned = nodes.find((n) => n.id === pinnedNodeId);
                  const data = pinned?.data as { label?: string; details?: LayerDetails } | undefined;

                  if (pinned) {
                    highlightPathTo(pinnedNodeId);
                  } else {
                    highlightPathTo(null);
                  }

                  setActiveParam(null);
                  setActiveLayerLabel(data?.label ?? null);
                  setActiveLayerDetails(data?.details ?? null);

                  if (data && data.details && data.details.params) {
                    const chips: ParamChip[] = Object.keys(data.details.params).map((paramName) => ({
                      id: `${pinnedNodeId}-${paramName}`,
                      nodeId: pinnedNodeId,
                      layerLabel: data.label ?? pinnedNodeId,
                      paramName,
                    }));
                    setActiveNodeParams(chips);
                  } else {
                    setActiveNodeParams(null);
                  }
                } else {
                  highlightPathTo(null);
                  setActiveParam(null);
                  setActiveNodeParams(null);
                  setActiveLayerLabel(null);
                  setActiveLayerDetails(null);
                  setNodes((prev) =>
                    prev.map((n) => ({
                      ...n,
                      selected: false,
                    })),
                  );
                }
              }}
              onClick={() => {
                if (
                  editingParam &&
                  editingParam.nodeId === p.nodeId &&
                  editingParam.paramName === p.paramName
                ) {
                  setEditingParam(null);
                  setEditingParamValue(null);
                  setEditingParamValues(null);
                  setEditingParamShape(null);
                  setHoveredParamIndex(null);
                  setPinnedParamIndex(null);
                  return;
                }

                const node = nodes.find((n) => n.id === p.nodeId);
                if (!node || !(node.data as any)?.details) {
                  setEditingParam(p);
                  setEditingParamValue(0);
                  setEditingParamValues([0]);
                  setEditingParamShape([1]);
                  return;
                }

                const details = (node.data as any).details as LayerDetails;
                const params = details.params as Record<string, ParamInfo>;
                const info = params && params[p.paramName];

                let shape: number[] = [];
                let flatVals: number[] = [];

                if (info) {
                  shape = Array.isArray(info.shape) ? info.shape : [];
                  const total = shape.length
                    ? shape.reduce((a, b) => a * Math.max(1, b), 1)
                    : info.value_sample && info.value_sample.length
                    ? info.value_sample.length
                    : 1;
                  const hasSample = info.value_sample && info.value_sample.length;

                  if (hasSample && info.value_sample!.length >= total) {
                    flatVals = info.value_sample!.slice(0, total);
                  } else {
                    const rand = () => Math.random() * 2 - 1; // ~Uniform(-1,1)
                    flatVals = Array.from({ length: total }, () => rand());
                  }
                } else {
                  shape = [1];
                  flatVals = [0];
                }

                const initial = flatVals[0] ?? 0;

                setEditingParam(p);
                setEditingParamShape(shape);
                setEditingParamValues(flatVals);
                setEditingParamValue(initial);

                const key = `${p.nodeId}:${p.paramName}`;
                setParamOverrides((prev) => ({
                  ...prev,
                  [key]: flatVals,
                }));
              }}
              className="px-2 py-0.5 rounded-full border border-slate-700 bg-slate-800/70 text-slate-200 hover:border-orange-400 hover:text-orange-300 whitespace-nowrap"
            >
              {p.paramName}
              <span className="text-slate-500">{` (${p.layerLabel})`}</span>
            </button>
          ))}
        </div>
      )}

      {ioVectors && (
        <div className="border-b border-slate-800 bg-slate-900/80 px-6 py-2 text-[11px] font-mono flex items-center justify-between gap-6">
          <div className="flex items-center gap-2">
            <span className="uppercase tracking-wider text-slate-500 flex-shrink-0">
              {ioVectors.inputLabel} vector
            </span>
            {architecture === 'cnn' &&
            viewMode === 'blocks' &&
            ioVectors.inShape &&
            Array.isArray(ioVectors.inShape) &&
            (ioVectors.inShape as number[]).length >= 2 ? (
              (() => {
                const vec = ioVectors.inVec;
                const total = Math.max(1, vec.length);
                const shape = ioVectors.inShape as number[];
                let H = 1;
                let W = total;
                if (shape.length >= 3) {
                  H = shape[shape.length - 2] ?? total;
                  W = shape[shape.length - 1] ?? 1;
                } else if (shape.length === 2) {
                  H = shape[0] ?? total;
                  W = shape[1] ?? 1;
                }
                H = Math.max(1, Math.min(8, H));
                W = Math.max(1, Math.min(8, W));

                const cells: JSX.Element[] = [];
                for (let r = 0; r < H; r++) {
                  for (let c = 0; c < W; c++) {
                    const idx = r * W + c;
                    const v = vec.length ? vec[idx % vec.length] : 0;
                    cells.push(
                      <div
                        key={idx}
                        className="w-3 h-3 rounded-sm"
                        style={{ backgroundColor: valueToColor(v) }}
                      />,
                    );
                  }
                }

                return (
                  <div
                    className="grid gap-[2px]"
                    style={{ gridTemplateColumns: `repeat(${W}, minmax(0, 1fr))` }}
                  >
                    {cells}
                  </div>
                );
              })()
            ) : (
              renderVector(ioVectors.inVec)
            )}
          </div>
          <div className="flex items-center gap-2">
            <span className="uppercase tracking-wider text-slate-500 flex-shrink-0">
              {ioVectors.outputLabel} vector
            </span>
            {renderVector(ioVectors.outVec)}
          </div>
          <div className="flex items-center gap-1">
            <span className="text-[10px] text-slate-500">View:</span>
            <button
              type="button"
              onClick={() => setViewMode('numbers')}
              className={`px-2 py-0.5 rounded-full border text-[10px] ${
                viewMode === 'numbers'
                  ? 'bg-slate-200 text-slate-900 border-slate-300'
                  : 'bg-slate-800 text-slate-200 border-slate-600'
              }`}
            >
              123
            </button>
            <button
              type="button"
              onClick={() => setViewMode('blocks')}
              className={`px-2 py-0.5 rounded-full border text-[10px] ${
                viewMode === 'blocks'
                  ? 'bg-slate-200 text-slate-900 border-slate-300'
                  : 'bg-slate-800 text-slate-200 border-slate-600'
              }`}
            >
              ▢
            </button>
          </div>
        </div>
      )}

      <AiAssistantPanel
        isOpen={isAiOpen}
        onToggle={() => setIsAiOpen((open) => !open)}
        provider={aiProvider}
        onProviderChange={setAiProvider}
        apiKey={aiApiKey}
        onApiKeyChange={setAiApiKey}
        model={aiModel}
        onModelChange={setAiModel}
        messages={aiMessages}
        onAsk={handleAiAsk}
        loading={aiLoading}
      />

      {(activeLayerLabel || pinnedNodeId || activeParam || (activeNodeParams && activeNodeParams.length > 0)) &&
        (activeChainRules.length > 0 || overlayBackSignal || overlayActivations) && (
        <div
          className={`pointer-events-none absolute top-40 z-30 ${
            isBlockPanelOpen ? 'left-72' : 'left-4'
          }`}
        >
          <div className="pointer-events-auto max-w-[60vw] max-h-[60vh] rounded-lg border border-slate-700 bg-slate-900/95 px-4 py-2 text-xs shadow-xl flex flex-col gap-1 justify-start overflow-y-auto">
            <span className="uppercase tracking-wider text-slate-500 text-[10px] flex-shrink-0 text-left">
              More info
            </span>
            {effectiveLayerLabel && (
              <div className="text-[10px] text-slate-400 font-mono">
                Layer: <span className="text-slate-200">{effectiveLayerLabel}</span>
              </div>
            )}
            {overlayActivations && (
              <div className="mt-1 space-y-1 text-[10px] text-slate-300 font-mono">
                {! (
                  effectiveLayerLabel &&
                  (effectiveLayerLabel.startsWith('Input') ||
                    effectiveLayerLabel.startsWith('Input Seq') ||
                    effectiveLayerLabel.startsWith('Token') ||
                    effectiveLayerLabel.startsWith('Positional'))
                ) && (
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">x (in):</span>
                    {architecture === 'cnn' &&
                    viewMode === 'blocks' &&
                    activeLayerDetails &&
                    Array.isArray(activeLayerDetails.in_shape) &&
                    activeLayerDetails.in_shape.length >= 2 ? (
                      (() => {
                        const vec = overlayActivations.inVec;
                        const total = Math.max(1, vec.length);
                        const shape = activeLayerDetails.in_shape as number[];
                        let H = 1;
                        let W = total;
                        if (shape.length >= 3) {
                          H = shape[shape.length - 2] ?? total;
                          W = shape[shape.length - 1] ?? 1;
                        } else if (shape.length === 2) {
                          H = shape[0] ?? total;
                          W = shape[1] ?? 1;
                        }
                        H = Math.max(1, Math.min(8, H));
                        W = Math.max(1, Math.min(8, W));

                        const cells: JSX.Element[] = [];
                        for (let r = 0; r < H; r++) {
                          for (let c = 0; c < W; c++) {
                            const idx = r * W + c;
                            const v = vec.length ? vec[idx % vec.length] : 0;
                            cells.push(
                              <div
                                key={idx}
                                className="w-3 h-3 rounded-sm"
                                style={{ backgroundColor: valueToColor(v) }}
                              />,
                            );
                          }
                        }
                        return (
                          <div
                            className="grid gap-[2px]"
                            style={{ gridTemplateColumns: `repeat(${W}, minmax(0, 1fr))` }}
                          >
                            {cells}
                          </div>
                        );
                      })()
                    ) : (
                      renderVector(overlayActivations.inVec)
                    )}
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">
                    {effectiveLayerLabel &&
                    (effectiveLayerLabel.startsWith('Input') ||
                      effectiveLayerLabel.startsWith('Input Seq') ||
                      effectiveLayerLabel.startsWith('Token') ||
                      effectiveLayerLabel.startsWith('Positional'))
                      ? 'x:'
                      : 'y (out):'}
                  </span>
                  {architecture === 'cnn' &&
                  viewMode === 'blocks' &&
                  activeLayerDetails &&
                  Array.isArray(activeLayerDetails.out_shape) &&
                  activeLayerDetails.out_shape.length >= 2 ? (
                    (() => {
                      const vec = overlayActivations.outVec;
                      const total = Math.max(1, vec.length);
                      const shape = activeLayerDetails.out_shape as number[];
                      let H = 1;
                      let W = total;
                      if (shape.length >= 3) {
                        H = shape[shape.length - 2] ?? total;
                        W = shape[shape.length - 1] ?? 1;
                      } else if (shape.length === 2) {
                        H = shape[0] ?? total;
                        W = shape[1] ?? 1;
                      }
                      H = Math.max(1, Math.min(8, H));
                      W = Math.max(1, Math.min(8, W));

                      const cells: JSX.Element[] = [];
                      for (let r = 0; r < H; r++) {
                        for (let c = 0; c < W; c++) {
                          const idx = r * W + c;
                          const v = vec.length ? vec[idx % vec.length] : 0;
                          cells.push(
                            <div
                              key={idx}
                              className="w-3 h-3 rounded-sm"
                              style={{ backgroundColor: valueToColor(v) }}
                            />,
                          );
                        }
                      }
                      return (
                        <div
                          className="grid gap-[2px]"
                          style={{ gridTemplateColumns: `repeat(${W}, minmax(0, 1fr))` }}
                        >
                          {cells}
                        </div>
                      );
                    })()
                  ) : (
                    renderVector(overlayActivations.outVec)
                  )}
                </div>

                {architecture === 'rnn' &&
                  effectiveLayerLabel &&
                  effectiveLayerLabel.startsWith('RNN') &&
                  activeLayerDetails && (
                    <div className="mt-2 rounded border border-slate-700 bg-slate-900/80 p-2">
                      <div className="mb-1 flex justify-between items-center">
                        <span className="text-[9px] text-slate-400 font-mono">RNN timesteps</span>
                        {(() => {
                          const shape = activeLayerDetails.in_shape as number[] | string;
                          let T = 0;
                          if (Array.isArray(shape) && shape.length >= 2) {
                            T = typeof shape[0] === 'number' ? shape[0] : 0;
                          }
                          if (!T) {
                            T = Math.max(1, Math.min(4, seqLen || 1));
                          }
                          const tActive = ((rnnStep % T) + T) % T;
                          return (
                            <span className="text-[9px] text-slate-500 font-mono">
                              t = {tActive + 1} / {T}
                            </span>
                          );
                        })()}
                      </div>
                      {(() => {
                        const inShape = activeLayerDetails.in_shape as number[] | string;
                        const outShape = activeLayerDetails.out_shape as number[] | string;

                        let T = 0;
                        let d = 0;
                        let H = 0;

                        if (Array.isArray(inShape) && inShape.length >= 2) {
                          T = typeof inShape[0] === 'number' ? inShape[0] : 0;
                          d = typeof inShape[1] === 'number' ? inShape[1] : 0;
                        }
                        if (Array.isArray(outShape) && outShape.length >= 2) {
                          if (!T) {
                            T = typeof outShape[0] === 'number' ? outShape[0] : 0;
                          }
                          H = typeof outShape[1] === 'number' ? outShape[1] : 0;
                        }

                        if (!T) {
                          T = Math.max(1, Math.min(4, seqLen || 1));
                        }
                        if (!d) {
                          d = Math.max(1, Math.min(16, inputDim || 1));
                        }
                        if (!H) {
                          H = Math.max(1, Math.min(16, hiddenDim || 1));
                        }

                        const tActive = ((rnnStep % T) + T) % T;

                        const flatIn = (activeLayerDetails.input_sample && activeLayerDetails.input_sample.length
                          ? activeLayerDetails.input_sample
                          : activeLayerDetails.output_sample || []) as number[];
                        const flatOut = (activeLayerDetails.output_sample && activeLayerDetails.output_sample.length
                          ? activeLayerDetails.output_sample
                          : activeLayerDetails.input_sample || []) as number[];

                        const getRow = (flat: number[], rowLen: number, t: number): number[] => {
                          if (rowLen <= 0) return [];
                          if (!flat.length) {
                            return Array.from({ length: rowLen }, () => 0);
                          }
                          const start = t * rowLen;
                          const end = start + rowLen;
                          if (end <= flat.length) {
                            return flat.slice(start, end);
                          }
                          const row: number[] = [];
                          for (let j = 0; j < rowLen; j++) {
                            const idx = (start + j) % flat.length;
                            row.push(flat[idx]);
                          }
                          return row;
                        };

                        const formatVec = (row: number[]): string => {
                          if (!row.length) return '[ ]';
                          const maxCols = 6;
                          const parts = row.slice(0, maxCols).map((v) => v.toFixed(2));
                          const suffix = row.length > maxCols ? ', …' : '';
                          return `[ ${parts.join(', ')}${suffix} ]`;
                        };

                        const rows: JSX.Element[] = [];
                        rows.push(
                          <div
                            key="hdr"
                            className="grid grid-cols-4 gap-1 text-[9px] text-slate-400 font-mono mb-1"
                          >
                            <span className="text-center">t</span>
                            <span className="text-center">x_t</span>
                            <span className="text-center">h_t</span>
                            <span className="text-center">y_t</span>
                          </div>,
                        );

                        const computeYRow = (t: number, xRow: number[], hRow: number[]): number[] => {
                          const params = (activeLayerDetails.params || {}) as Record<string, ParamInfo>;
                          const Wy = params.W_y;
                          const Uy = params.U_y;
                          const By = params.b_y;

                          if (
                            Wy && Uy && By &&
                            Array.isArray(Wy.shape) && Wy.shape.length === 2 &&
                            Array.isArray(Uy.shape) && Uy.shape.length === 2 &&
                            Array.isArray(By.shape) && By.shape.length === 1 &&
                            Wy.value_sample.length &&
                            Uy.value_sample.length &&
                            By.value_sample.length &&
                            Wy.shape[0] === H &&
                            Wy.shape[1] === H &&
                            Uy.shape[0] === d &&
                            Uy.shape[1] === H &&
                            By.shape[0] === H
                          ) {
                            const y: number[] = [];
                            const wyVals = Wy.value_sample;
                            const uyVals = Uy.value_sample;
                            const byVals = By.value_sample;
                            for (let j = 0; j < H; j++) {
                              let sum = 0;
                              for (let k = 0; k < H; k++) {
                                const w = wyVals[k * H + j] ?? 0;
                                sum += (hRow[k] ?? 0) * w;
                              }
                              for (let i = 0; i < d; i++) {
                                const w = uyVals[i * H + j] ?? 0;
                                sum += (xRow[i] ?? 0) * w;
                              }
                              sum += byVals[j] ?? 0;
                              y.push(sum);
                            }
                            return y;
                          }

                          // Fallback: simple combination if we can't read real weights
                          const alpha = 0.7;
                          const beta = 0.3;
                          const yFallback: number[] = [];
                          for (let j = 0; j < H; j++) {
                            const hVal = hRow[j] ?? 0;
                            const xVal = d > 0 ? xRow[j % d] ?? 0 : 0;
                            yFallback.push(alpha * hVal + beta * xVal);
                          }
                          return yFallback;
                        };

                        for (let t = 0; t < T; t++) {
                          const isActiveRow = t === tActive;
                          const xRow = getRow(flatIn, d, t);
                          const hRow = getRow(flatOut, H, t);

                          const yRow = computeYRow(t, xRow, hRow);

                          rows.push(
                            <div
                              key={t}
                              className={`grid grid-cols-4 gap-1 text-[9px] font-mono px-1 py-0.5 rounded ${
                                isActiveRow
                                  ? 'bg-amber-500/10 text-amber-100'
                                  : 'opacity-40 text-slate-400'
                              }`}
                            >
                              <span className="text-center">{t + 1}</span>
                              {viewMode === 'numbers' ? (
                                <>
                                  <span className="text-center">{formatVec(xRow)}</span>
                                  <span className="text-center">{formatVec(hRow)}</span>
                                  <span className="text-center">{formatVec(yRow)}</span>
                                </>
                              ) : (
                                <>
                                  <div className="flex justify-center gap-[1px]">
                                    {xRow
                                      .slice(0, Math.min(12, xRow.length || 1))
                                      .map((v, idx) => (
                                        <div
                                          key={idx}
                                          className="w-3 h-3 rounded-sm"
                                          style={{ backgroundColor: valueToColor(v) }}
                                        />
                                      ))}
                                    {xRow.length > 12 && (
                                      <span className="text-[8px] text-slate-500 ml-1">…</span>
                                    )}
                                  </div>
                                  <div className="flex justify-center gap-[1px]">
                                    {hRow
                                      .slice(0, Math.min(12, hRow.length || 1))
                                      .map((v, idx) => (
                                        <div
                                          key={idx}
                                          className="w-3 h-3 rounded-sm"
                                          style={{ backgroundColor: valueToColor(v) }}
                                        />
                                      ))}
                                    {hRow.length > 12 && (
                                      <span className="text-[8px] text-slate-500 ml-1">…</span>
                                    )}
                                  </div>
                                  <div className="flex justify-center gap-[1px]">
                                    {yRow
                                      .slice(0, Math.min(12, yRow.length || 1))
                                      .map((v, idx) => (
                                        <div
                                          key={idx}
                                          className="w-3 h-3 rounded-sm"
                                          style={{ backgroundColor: valueToColor(v) }}
                                        />
                                      ))}
                                    {yRow.length > 12 && (
                                      <span className="text-[8px] text-slate-500 ml-1">…</span>
                                    )}
                                  </div>
                                </>
                              )}
                            </div>,
                          );
                        }

                        return <div className="mt-1 space-y-0.5">{rows}</div>;
                      })()}
                    </div>
                  )}

                {effectiveLayerLabel &&
                  (effectiveLayerLabel.startsWith('Input') ||
                    effectiveLayerLabel.startsWith('Input Seq') ||
                    effectiveLayerLabel.startsWith('Token') ||
                    effectiveLayerLabel.startsWith('Positional')) && (
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-500">
                        {architecture === 'rnn'
                          ? `Input seq (T = ${Math.max(
                              1,
                              Math.min(4, seqLen || 1),
                            )}, d = ${inputDim || effectiveInputDim || 1})`
                          : `Input vector (d = ${effectiveInputDim ?? inputDim})`}
                      </span>
                      <span className="text-slate-400 font-mono text-[10px]">
                        mean = {inputValue.toFixed(3)}
                      </span>
                    </div>

                    <div className="max-h-32 overflow-y-auto pr-1">
                      {(() => {
                        const dim = effectiveInputDim || inputDim || 1;
                        const total = dim;
                        let rows = 1;
                        let cols = total;

                        if (architecture === 'cnn') {
                          const size = Math.max(1, Math.floor(Math.sqrt(total)));
                          rows = size;
                          cols = size;
                        } else if (architecture === 'rnn') {
                          const T = Math.max(1, Math.min(4, seqLen || 1));
                          const d = Math.max(1, Math.min(16, inputDim || 1));
                          rows = T;
                          cols = d;
                        }

                        const cells: JSX.Element[] = [];
                        for (let r = 0; r < rows; r++) {
                          for (let c = 0; c < cols; c++) {
                            const i = r * cols + c;
                            if (i >= total) break;
                            const v = inputVector[i] ?? 0;
                            const isActive = activeInputIndex === i;
                            const isInactiveRnnRow =
                              architecture === 'rnn' && rows > 1
                                ? r !== ((rnnStep % rows) + rows) % rows
                                : false;
                            cells.push(
                              <button
                                key={i}
                                type="button"
                                className={`w-4 h-4 rounded-sm border ${
                                  isActive
                                    ? 'ring-2 ring-blue-400 ring-offset-1 ring-offset-slate-900 border-transparent'
                                    : `border-slate-600${isInactiveRnnRow ? ' opacity-40' : ''}`
                                }`}
                                style={{ backgroundColor: valueToColor(v) }}
                                onMouseEnter={() => setHoveredInputIndex(i)}
                                onMouseLeave={() =>
                                  setHoveredInputIndex((prev) => (prev === i ? null : prev))
                                }
                                onClick={() =>
                                  setPinnedInputIndex((prev) => (prev === i ? null : i))
                                }
                              />,
                            );
                          }
                        }

                        return (
                          <div
                            className="grid gap-[4px]"
                            style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
                          >
                            {cells}
                          </div>
                        );
                      })()}
                    </div>
                    {activeInputIndex !== null &&
                      activeInputIndex < (effectiveInputDim || inputDim) && (
                        <div className="mt-2 flex items-center gap-2">
                          <span className="w-10 text-slate-500 font-mono">x{activeInputIndex}</span>
                          <input
                            type="range"
                            min={-1}
                            max={1}
                            step={0.01}
                            value={inputVector[activeInputIndex] ?? 0}
                            onChange={(e) =>
                              handleInputEntryChange(activeInputIndex, Number(e.target.value))
                            }
                            className="flex-1"
                          />
                          <span className="w-14 text-right text-slate-100 font-mono">
                            {(inputVector[activeInputIndex] ?? 0).toFixed(2)}
                          </span>
                        </div>
                      )}
                  </div>
                )}

                {overlayBackSignal && (
                  <div className="mt-2 flex items-center gap-3 justify-start text-slate-100">
                    <span className="uppercase tracking-wider text-slate-400 text-[10px]">
                      backprop to prev
                    </span>
                    <span className="flex items-center gap-1 font-mono text-sm">
                      <span className="text-slate-400">←</span>
                      <span className="text-[13px]">
                        <Latex>{`$$ ${overlayBackSignal} $$`}</Latex>
                      </span>
                    </span>
                  </div>
                )}
              </div>
            )}
            {activeChainRules.map(({ chip, latex }) => (
              <div key={chip.id} className="flex items-center gap-2 justify-start">
                <span className="text-[10px] text-slate-400 font-mono flex-shrink-0">
                  {chip.paramName} ({chip.layerLabel})
                </span>
                <div className="text-[11px] text-slate-200 overflow-x-auto">
                  <Latex>{`$$ ${latex} $$`}</Latex>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {editingParam && (
        <div className="absolute top-20 right-6 z-30">
          <div className="rounded-lg border border-slate-700 bg-slate-900/95 px-5 py-4 text-xs shadow-xl max-w-2xl w-[520px] max-h-[80vh] overflow-y-auto">
            <div className="flex items-center justify-between mb-2">
              <span className="uppercase tracking-wider text-slate-500 text-[10px]">
                Edit parameter
              </span>
              <button
                type="button"
                onClick={() => {
                  setEditingParam(null);
                  setEditingParamValue(null);
                }}
                className="text-slate-400 hover:text-slate-200 text-[11px]"
              >
                ×
              </button>
            </div>
            <div className="mb-2 text-[11px] font-mono text-slate-200">
              {editingParam.paramName}{' '}
              <span className="text-slate-500">({editingParam.layerLabel})</span>
            </div>
            <div className="mb-3 space-y-1">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-slate-400">Broadcast all entries</span>
                {editingParamShape && (
                  <span className="text-[10px] text-slate-500 font-mono">
                    shape: [{editingParamShape.join('×')}]
                  </span>
                )}
              </div>
              <div className="flex items-center gap-2">
                <input
                  type="range"
                  min={-2}
                  max={2}
                  step={0.01}
                  value={editingParamValue ?? 0}
                  onChange={(e) => handleParamValueChange(Number(e.target.value))}
                  className="flex-1"
                />
                <span className="w-16 text-right font-mono text-slate-100">
                  {(editingParamValue ?? 0).toFixed(2)}
                </span>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[10px] text-slate-400">Manual:</span>
                <input
                  type="number"
                  step={0.01}
                  value={editingParamValue ?? 0}
                  onChange={(e) => handleParamValueChange(Number(e.target.value) || 0)}
                  className="w-24 bg-slate-800 border border-slate-600 rounded px-1 text-[11px] text-slate-100"
                />
              </div>
            </div>

            {editingParamValues && (
              <div className="mt-2 border-t border-slate-800 pt-2 space-y-1">
                <div className="flex items-center justify-between mb-1">
                  <span className="text-[10px] text-slate-400">Entries</span>
                  <span className="text-[10px] text-slate-500 font-mono">
                    {editingParamValues.length} values
                  </span>
                </div>
                <div className="max-h-56 overflow-y-auto pr-1">
                  {(() => {
                    const values = editingParamValues;
                    const shape = editingParamShape;
                    let rows = 1;
                    let cols = values.length || 1;

                    if (shape && shape.length > 0) {
                      const last = shape[shape.length - 1] || cols;
                      const rest = shape
                        .slice(0, -1)
                        .reduce((acc, d) => acc * Math.max(1, d || 1), 1);
                      rows = Math.max(1, rest);
                      cols = Math.max(1, last);
                    }

                    const cells: JSX.Element[] = [];
                    for (let r = 0; r < rows; r++) {
                      for (let c = 0; c < cols; c++) {
                        const idx = r * cols + c;
                        if (idx >= values.length) break;
                        const v = values[idx];
                        const isActive = activeParamIndex === idx;
                        cells.push(
                          <button
                            key={idx}
                            type="button"
                            className={`flex flex-col items-center justify-center gap-0.5 rounded px-1 py-0.5 border text-left text-slate-50 text-[10px] ${
                              isActive
                                ? 'ring-2 ring-blue-400 ring-offset-[1px] ring-offset-slate-900 border-transparent'
                                : 'border-slate-700'
                            }`}
                            style={{ backgroundColor: valueToColor(v) }}
                            onMouseEnter={() => setHoveredParamIndex(idx)}
                            onMouseLeave={() =>
                              setHoveredParamIndex((prev) => (prev === idx ? null : prev))
                            }
                            onClick={() =>
                              setPinnedParamIndex((prev) => (prev === idx ? null : idx))
                            }
                          >
                            <span className="text-[8px] text-slate-900/80 font-mono">{idx}</span>
                            <span className="text-[10px] font-mono leading-tight overflow-hidden text-ellipsis max-w-[3ch]">
                              {(() => {
                                const abs = Math.abs(v);
                                if (abs < 1e-3 && v !== 0) return v.toExponential(1); // tiny
                                if (abs >= 100) return v.toFixed(0); // large magnitudes
                                return v.toFixed(2); // default truncated view
                              })()}
                            </span>
                          </button>,
                        );
                      }
                    }

                    return (
                      <div
                        className="grid gap-1"
                        style={{ gridTemplateColumns: `repeat(${cols}, minmax(0, 1fr))` }}
                      >
                        {cells}
                      </div>
                    );
                  })()}
                </div>
                {activeParamIndex !== null &&
                  activeParamIndex >= 0 &&
                  activeParamIndex < editingParamValues.length && (
                    <div className="mt-2 space-y-1">
                      <div className="flex items-center justify-between">
                        <span className="text-[10px] text-slate-400 font-mono">
                          entry[{activeParamIndex}]
                        </span>
                        <span className="text-[10px] text-slate-300 font-mono">
                          {editingParamValues[activeParamIndex].toFixed(3)}
                        </span>
                      </div>
                      <input
                        type="range"
                        min={-2}
                        max={2}
                        step={0.01}
                        value={editingParamValues[activeParamIndex]}
                        onChange={(e) =>
                          handleParamElementChange(
                            activeParamIndex,
                            Number(e.target.value) || 0,
                          )
                        }
                        className="w-full"
                      />
                    </div>
                  )}
              </div>
            )}
          </div>
        </div>
      )}

      {isBlockPanelOpen && (
        <div className="absolute left-0 top-16 bottom-0 w-64 border-r border-slate-800 bg-slate-900/95 z-20 flex flex-col">
          <div className="px-4 py-3 border-b border-slate-800">
            <p className="text-xs font-semibold text-slate-100">Blocks</p>
            <p className="text-[11px] text-slate-400 mt-1">
              Click a node to select it, then choose where to insert a block.
            </p>
          </div>
          <div className="flex-1 overflow-y-auto px-3 py-3 space-y-2">
            {BLOCK_TEMPLATES.map((tpl) => (
              <div
                key={tpl.type}
                className="border border-slate-700 rounded-md p-2 bg-slate-800/60"
                draggable
                onDragStart={(e) => handleBlockDragStart(e, tpl)}
              >
                <div className="text-xs font-mono text-slate-100 mb-2">{tpl.label}</div>
                <div className="flex flex-wrap gap-1">
                  <button
                    type="button"
                    onClick={() => insertBlock('before', tpl)}
                    className="px-2 py-0.5 rounded-full bg-slate-700 text-[10px] text-slate-100 hover:bg-slate-600"
                  >
                    Before
                  </button>
                  <button
                    type="button"
                    onClick={() => insertBlock('after', tpl)}
                    className="px-2 py-0.5 rounded-full bg-slate-700 text-[10px] text-slate-100 hover:bg-slate-600"
                  >
                    After
                  </button>
                  <button
                    type="button"
                    onClick={() => insertBlock('end', tpl)}
                    className="px-2 py-0.5 rounded-full bg-slate-700 text-[10px] text-slate-100 hover:bg-slate-600"
                  >
                    End
                  </button>
                </div>
              </div>
            ))}
          </div>
          <div className="px-4 py-2 border-t border-slate-800 text-[10px] text-slate-500 space-y-1">
            <div>Selected node: {selectedNodeId ?? 'none'}</div>
            <button
              type="button"
              disabled={!selectedNodeId}
              onClick={() => {
                if (!selectedNodeId) return;
                setNodes((prevNodes) => {
                  const nextNodes = prevNodes.filter((n) => n.id !== selectedNodeId);

                  // Rebuild main sequential edges
                  const baseEdges = buildSequentialEdges(nextNodes);

                  // Preserve residual edges that do not touch the deleted node
                  setEdges((prevEdges) => {
                    const residuals = prevEdges.filter(
                      (e) =>
                        !String(e.id).startsWith('e-') &&
                        e.source !== selectedNodeId &&
                        e.target !== selectedNodeId,
                    );
                    const layouted = getLayoutedElements(nextNodes, baseEdges);
                    return [...layouted.edges, ...residuals];
                  });

                  const layoutedNodes = getLayoutedElements(nextNodes, baseEdges).nodes;
                  return layoutedNodes;
                });
                setSelectedNodeId(null);
              }}
              className="w-full px-2 py-1 rounded-md border text-[10px] font-medium bg-rose-900/60 border-rose-500/70 text-rose-100 disabled:opacity-40"
            >
              Delete selected block
            </button>
            <button
              type="button"
              onClick={() => {
                setIsAddingResidual((v) => !v);
                setResidualSourceId(null);
              }}
              className={`w-full px-2 py-1 rounded-md border text-[10px] font-medium ${
                isAddingResidual
                  ? 'bg-emerald-600/70 border-emerald-400 text-white'
                  : 'bg-slate-800 border-slate-600 text-slate-200'
              }`}
            >
              {isAddingResidual ? 'Click two nodes to connect (click again to cancel)' : 'Add residual connection'}
            </button>
            {isAddingResidual && (
              <div className="text-[9px] text-slate-400">
                {residualSourceId
                  ? `Source: ${residualSourceId}. Click target node in graph.`
                  : 'Click first node in graph to use as source.'}
              </div>
            )}
          </div>
        </div>
      )}

      <div
        className="flex-1"
        onDrop={handleCanvasDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          nodeTypes={nodeTypes}
          onNodeClick={(_, node) => {
            if (isAddingResidual) {
              if (!residualSourceId) {
                setResidualSourceId(node.id);
              } else if (residualSourceId !== node.id) {
                const sourceId = residualSourceId;
                const targetId = node.id;
                setEdges((prev) => [
                  ...prev,
                  {
                    id: `res-user-${sourceId}-${targetId}-${Math.random().toString(36).slice(2)}`,
                    source: sourceId,
                    target: targetId,
                    animated: false,
                    style: { stroke: '#22c55e', strokeWidth: 2, strokeDasharray: '4 2' },
                  } as any,
                ]);
                setResidualSourceId(null);
                setIsAddingResidual(false);
              }
              return;
            }

            setSelectedNodeId(node.id);

            const data = node.data as {
              label?: string;
              details?: LayerDetails;
            };
            const isAlreadyPinned = pinnedNodeId === node.id;

            if (isAlreadyPinned) {
              // Toggle off: unpin, but keep current hover-based overlay state.
              setPinnedNodeId(null);
              return;
            }

            // Pin a new node
            setPinnedNodeId(node.id);
            highlightPathTo(node.id);
            setActiveLayerLabel(data?.label ?? node.id);
            setActiveLayerDetails(data?.details ?? null);
            if (data && data.details && data.details.params) {
              const chips: ParamChip[] = Object.keys(data.details.params).map((paramName) => ({
                id: `${node.id}-${paramName}`,
                nodeId: node.id,
                layerLabel: data.label ?? node.id,
                paramName,
              }));
              setActiveNodeParams(chips);
            } else {
              setActiveNodeParams(null);
            }
          }}
          onNodeMouseEnter={(_, node) => {
            highlightPathTo(node.id);
            const data = node.data as {
              label?: string;
              details?: LayerDetails;
            };
            setActiveLayerLabel(data?.label ?? node.id);
            setActiveLayerDetails(data?.details ?? null);
            if (data && data.details && data.details.params) {
              const chips: ParamChip[] = Object.keys(data.details.params).map((paramName) => ({
                id: `${node.id}-${paramName}`,
                nodeId: node.id,
                layerLabel: data.label ?? node.id,
                paramName,
              }));
              setActiveNodeParams(chips);
            } else {
              setActiveNodeParams(null);
            }
          }}
          onNodeMouseLeave={(_, node) => {
            if (pinnedNodeId) {
              // Restore overlay state for pinned node when cursor leaves any node.
              const pinned = nodes.find((n) => n.id === pinnedNodeId);
              const data = pinned?.data as { label?: string; details?: LayerDetails } | undefined;

              if (pinned) {
                highlightPathTo(pinnedNodeId);
              } else {
                highlightPathTo(null);
              }

              setActiveLayerLabel(data?.label ?? null);
              setActiveLayerDetails(data?.details ?? null);

              if (data && data.details && data.details.params) {
                const chips: ParamChip[] = Object.keys(data.details.params).map((paramName) => ({
                  id: `${pinnedNodeId}-${paramName}`,
                  nodeId: pinnedNodeId,
                  layerLabel: data.label ?? pinnedNodeId,
                  paramName,
                }));
                setActiveNodeParams(chips);
              } else {
                setActiveNodeParams(null);
              }
            } else {
              // No pin: clear overlay when cursor leaves nodes.
              highlightPathTo(null);
              setActiveNodeParams(null);
              setActiveLayerLabel(null);
              setActiveLayerDetails(null);
            }
          }}
          fitView
          minZoom={0.1}
        >
          <Background color="#334155" variant={BackgroundVariant.Dots} gap={24} size={1} />
          <Controls className="bg-slate-800 border-slate-700" />
        </ReactFlow>
      </div>
    </div>
  );
}
