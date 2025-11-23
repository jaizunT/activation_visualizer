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
  const [isBlockPanelOpen, setIsBlockPanelOpen] = useState(false);
  const [selectedNodeId, setSelectedNodeId] = useState<string | null>(null);
  const [activeParam, setActiveParam] = useState<ParamChip | null>(null);
  const [activeNodeParams, setActiveNodeParams] = useState<ParamChip[] | null>(null);
  const [activeLayerLabel, setActiveLayerLabel] = useState<string | null>(null);
  const [activeLayerDetails, setActiveLayerDetails] = useState<LayerDetails | null>(null);
  const [isAddingResidual, setIsAddingResidual] = useState(false);
  const [residualSourceId, setResidualSourceId] = useState<string | null>(null);

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
      });

      setNodes((prevNodes) => {
        // First run or topology changed: compute fresh layout
        if (prevNodes.length === 0 || prevNodes.length !== rawNodes.length) {
          const layouted = getLayoutedElements(rawNodes, rawEdges);
          setEdges(layouted.edges);
          return layouted.nodes;
        }

        // Same topology: preserve positions, update data
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
        return merged;
      });
    } catch (err) {
      console.error(err);
    } finally {
      setLoading(false);
    }
  }, [architecture, layers, hiddenDim, activation, inputDim, setNodes, setEdges]);

  useEffect(() => {
    runSimulation();
  }, [runSimulation]);

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
  const overlayBackSignal = useMemo(
    () => (activeLayerLabel ? buildBackSignalForLayer(activeLayerLabel) : null),
    [activeLayerLabel],
  );

  const overlayActivations = useMemo(() => {
    if (!activeLayerDetails) return null;

    const getDim = (shape: number[] | string | undefined): number => {
      if (!shape || typeof shape === 'string') return 4;
      if (shape.length >= 2) return shape[shape.length - 1] ?? shape[0] ?? 4;
      if (shape.length === 1) return shape[0] ?? 4;
      return 4;
    };

    const lenIn = getDim(activeLayerDetails.in_shape as any);
    const lenOut = getDim(activeLayerDetails.out_shape as any);
    const cap = 8;

    const makeVec = (len: number, center: number): number[] => {
      const L = Math.min(len || 4, cap);
      const res: number[] = [];
      for (let i = 0; i < L; i++) {
        const t = L === 1 ? 0 : (i / (L - 1)) * 2 - 1; // in [-1,1]
        res.push(center + t * 0.2);
      }
      return res;
    };

    const inVec = makeVec(lenIn, 0);
    const outVec = makeVec(lenOut, activeLayerDetails.forward_mean ?? 0);

    return { inVec, outVec };
  }, [activeLayerDetails]);

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
              max={32}
              onChange={(e) => setInputDim(Number(e.target.value) || 1)}
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
          {architecture === 'mlp' && (
            <>
              <div className="flex items-center gap-2 px-3">
                <span className="text-xs text-slate-400">Init:</span>
                <select
                  value={initMode}
                  onChange={(e) => setInitMode(e.target.value as InitMode)}
                  className="bg-slate-700 border border-slate-600 rounded px-2 py-1 text-xs"
                >
                  <option value="random">Random</option>
                  <option value="constant">Constant</option>
                </select>
              </div>
              <div className="flex items-center gap-2 px-3">
                <span className="text-xs text-slate-400">Init value:</span>
                <input
                  type="number"
                  value={initValue}
                  onChange={(e) => setInitValue(Number(e.target.value) || 0)}
                  className="w-16 bg-slate-700 border border-slate-600 rounded px-1 text-sm"
                  step="0.1"
                />
              </div>
            </>
          )}
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
              }}
              className="px-2 py-0.5 rounded-full border border-slate-700 bg-slate-800/70 text-slate-200 hover:border-orange-400 hover:text-orange-300 whitespace-nowrap"
            >
              {p.paramName}
              <span className="text-slate-500">{` (${p.layerLabel})`}</span>
            </button>
          ))}
        </div>
      )}

      {(activeChainRules.length > 0 || overlayBackSignal || overlayActivations) && (
        <div className="pointer-events-none absolute top-24 left-4 z-30">
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
                {!(activeLayerLabel && activeLayerLabel.startsWith('Input')) && (
                  <div className="flex gap-2">
                    <span className="text-slate-500">x (in):</span>
                    <span>[ {overlayActivations.inVec.map((v) => v.toFixed(2)).join(', ')} ]</span>
                  </div>
                )}
                <div className="flex gap-2">
                  <span className="text-slate-500">
                    {activeLayerLabel && activeLayerLabel.startsWith('Input') ? 'x:' : 'y (out):'}
                  </span>
                  <span>[ {overlayActivations.outVec.map((v) => v.toFixed(2)).join(', ')} ]</span>
                </div>
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
          onNodeMouseLeave={() => {
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
