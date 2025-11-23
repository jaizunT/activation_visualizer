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
  const [editingParam, setEditingParam] = useState<ParamChip | null>(null);
  const [editingParamValue, setEditingParamValue] = useState<number | null>(null);
  const [editingParamValues, setEditingParamValues] = useState<number[] | null>(null);
  const [editingParamShape, setEditingParamShape] = useState<number[] | null>(null);
  const [hoveredInputIndex, setHoveredInputIndex] = useState<number | null>(null);
  const [pinnedInputIndex, setPinnedInputIndex] = useState<number | null>(null);
  const [hoveredParamIndex, setHoveredParamIndex] = useState<number | null>(null);
  const [pinnedParamIndex, setPinnedParamIndex] = useState<number | null>(null);

  const activeInputIndex = pinnedInputIndex ?? hoveredInputIndex;
  const activeParamIndex = pinnedParamIndex ?? hoveredParamIndex;

  const runSimulation = useCallback(() => {
    setLoading(true);
    try {
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
        paramOverrides,
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
            const overrideVals = paramOverrides[key];
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
  }, [architecture, layers, hiddenDim, activation, inputDim, attnHeads, initMode, initValue, inputValue, inputVector, setNodes, setEdges, paramOverrides]);

  useEffect(() => {
    runSimulation();
  }, [architecture, layers, hiddenDim, activation, inputDim, attnHeads, initMode, initValue, inputValue, inputVector]);

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
  const effectiveLayerLabel = pinnedNodeId
    ? (() => {
        const node = nodes.find((n) => n.id === pinnedNodeId);
        const data = node?.data as { label?: string } | undefined;
        return data?.label ?? activeLayerLabel;
      })()
    : activeLayerLabel;

  const overlayBackSignal = useMemo(
    () => (effectiveLayerLabel ? buildBackSignalForLayer(effectiveLayerLabel) : null),
    [effectiveLayerLabel],
  );

  const overlayActivations = useMemo(() => {
    if (!activeLayerDetails) return null;

    const cap = 16;

    // For MLP, prefer true forward-pass samples from the engine
    if (architecture === 'mlp') {
      const inSample = activeLayerDetails.input_sample;
      const outSample = activeLayerDetails.output_sample;

      if ((inSample && inSample.length) || (outSample && outSample.length)) {
        const inVec =
          inSample && inSample.length
            ? inSample.slice(0, cap)
            : outSample
            ? outSample.slice(0, cap)
            : [];
        const outVec =
          outSample && outSample.length
            ? outSample.slice(0, cap)
            : inSample
            ? inSample.slice(0, cap)
            : [];

        return { inVec, outVec };
      }
    }

    // Fallback: synthetic vectors based on shapes and forward_mean
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
  }, [activeLayerDetails, effectiveLayerLabel, architecture, inputVector]);

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
        if (label.startsWith('Final h_T')) return true;
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

    // Prefer real samples for MLP if available
    let inVec: number[];
    if (architecture === 'mlp' && inDetails.output_sample && inDetails.output_sample.length) {
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
    if (architecture === 'mlp' && outDetails.output_sample && outDetails.output_sample.length) {
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
    };
  }, [nodes, architecture, inputVector]);

  const handleInputEntryChange = useCallback(
    (index: number, value: number) => {
      const clamped = Math.max(-1, Math.min(1, value));
      setInputVector((prev) => {
        const dim = inputDim || prev.length || 1;
        const next = Array.from({ length: dim }, (_, i) =>
          i === index ? clamped : prev[i] ?? 0,
        );
        const mean =
          next.length > 0 ? next.reduce((a, b) => a + b, 0) / next.length : 0;
        setInputValue(mean);
        return next;
      });
    },
    [inputDim],
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
            <h1 className="font-bold text-xl tracking-tight">Backprop Visualizer</h1>
          </div>
          <button
            type="button"
            onClick={() => setIsBlockPanelOpen((open) => !open)}
            className="bg-slate-800 hover:bg-slate-700 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-600"
          >
            {isBlockPanelOpen ? 'Close Blocks' : 'Blocks'}
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
            <span className="text-xs text-slate-400">Dim:</span>
            <input
              type="number"
              value={hiddenDim}
              onChange={(e) => setHiddenDim(Number(e.target.value))}
              className="w-12 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
            />
          </div>
          <div className="flex items-center gap-2 px-3">
            <span className="text-xs text-slate-400">Input Dim:</span>
            <input
              type="number"
              value={inputDim}
              min={1}
              max={16}
              onChange={(e) => {
                const raw = Number(e.target.value) || 1;
                const dim = Math.max(1, Math.min(16, raw));
                setInputDim(dim);
                setInputVector((prev) => {
                  const next = Array.from({ length: dim }, (_, i) => prev[i] ?? 0.5);
                  const mean =
                    next.length > 0
                      ? next.reduce((a, b) => a + b, 0) / next.length
                      : 0;
                  setInputValue(mean);
                  return next;
                });
              }}
              className="w-12 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
            />
          </div>
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
            onClick={runSimulation}
            disabled={loading}
            className="bg-blue-600 hover:bg-blue-500 text-white px-4 py-1.5 rounded-md text-sm font-medium flex items-center gap-2 transition-colors disabled:opacity-50"
          >
            <Play size={14} />
            {loading ? 'Computing...' : 'Simulate'}
          </button>
          <button
            type="button"
            onClick={() => {
              setInitMode('random');
              setParamOverrides({});
              setEditingParam(null);
              setEditingParamValue(null);
              runSimulation();
            }}
            className="bg-slate-700 hover:bg-slate-600 text-white px-3 py-1.5 rounded-md text-xs font-medium border border-slate-500"
          >
            Randomize params
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
                if (!pinnedNodeId) {
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
                // Toggle behavior: if this param is already being edited, close the editor
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

                  const baseVals =
                    info.value_sample && info.value_sample.length
                      ? info.value_sample
                      : [0];

                  if (baseVals.length === total) {
                    flatVals = [...baseVals];
                  } else {
                    const seed = baseVals[0] ?? 0;
                    flatVals = Array.from({ length: total }, () => seed);
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
            {renderVector(ioVectors.inVec)}
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

      {(activeChainRules.length > 0 || overlayBackSignal || overlayActivations) && (
        <div className="pointer-events-none absolute top-40 left-4 z-30">
          <div className="pointer-events-auto max-w-[60vw] max-h-[60vh] rounded-lg border border-slate-700 bg-slate-900/95 px-4 py-2 text-xs shadow-xl flex flex-col gap-1 justify-start overflow-y-auto">
            <span className="uppercase tracking-wider text-slate-500 text-[10px] flex-shrink-0 text-left">
              Chain rule
            </span>
            {overlayBackSignal && (
              <div className="mt-1 flex items-center gap-3 justify-start text-slate-100">
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
            {overlayActivations && (
              <div className="mt-1 space-y-1 text-[10px] text-slate-300 font-mono">
                {!(effectiveLayerLabel && effectiveLayerLabel.startsWith('Input')) && (
                  <div className="flex items-center gap-2">
                    <span className="text-slate-500">x (in):</span>
                    {renderVector(overlayActivations.inVec)}
                  </div>
                )}
                <div className="flex items-center gap-2">
                  <span className="text-slate-500">
                    {effectiveLayerLabel && effectiveLayerLabel.startsWith('Input') ? 'x:' : 'y (out):'}
                  </span>
                  {renderVector(overlayActivations.outVec)}
                </div>
                {effectiveLayerLabel && effectiveLayerLabel.startsWith('Input') && architecture === 'mlp' && (
                  <div className="mt-2 space-y-1">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-500">Input vector (d = {inputDim}):</span>
                      <span className="text-slate-400 font-mono text-[10px]">
                        mean = {inputValue.toFixed(3)}
                      </span>
                    </div>
                    <div className="max-h-32 overflow-y-auto pr-1">
                      <div className="flex flex-wrap gap-[4px]">
                        {Array.from({ length: inputDim }).map((_, i) => {
                          const v = inputVector[i] ?? 0;
                          const isActive = activeInputIndex === i;
                          return (
                            <button
                              key={i}
                              type="button"
                              className={`w-4 h-4 rounded-sm border ${
                                isActive
                                  ? 'ring-2 ring-blue-400 ring-offset-1 ring-offset-slate-900 border-transparent'
                                  : 'border-slate-600'
                              }`}
                              style={{ backgroundColor: valueToColor(v) }}
                              onMouseEnter={() => setHoveredInputIndex(i)}
                              onMouseLeave={() =>
                                setHoveredInputIndex((prev) => (prev === i ? null : prev))
                              }
                              onClick={() =>
                                setPinnedInputIndex((prev) => (prev === i ? null : i))
                              }
                            />
                          );
                        })}
                      </div>
                    </div>
                    {activeInputIndex !== null && activeInputIndex < inputDim && (
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

            // If this node is already pinned, keep the overlay as-is
            if (isAlreadyPinned) {
              // Optionally refresh details in case params changed
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
            if (pinnedNodeId && pinnedNodeId === node.id) {
              return;
            }
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
            if (pinnedNodeId && pinnedNodeId === node.id) {
              return;
            }
            highlightPathTo(null);
            setActiveNodeParams(null);
            setActiveLayerLabel(null);
            setActiveLayerDetails(null);
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
