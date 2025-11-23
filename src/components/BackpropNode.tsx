import { useEffect, useState } from 'react';
import type { NodeProps } from 'reactflow';
import { Handle, Position } from 'reactflow';
import Latex from 'react-latex-next';
import 'katex/dist/katex.min.css';

import type { LayerDetails } from '../engine';

function valueToColor(v: number): string {
  const clamp = (x: number) => Math.max(-1, Math.min(1, x));
  const x = clamp(v);

  const neutral = { r: 15, g: 23, b: 42 };
  const pos = { r: 34, g: 197, b: 94 };
  const neg = { r: 248, g: 113, b: 113 };

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

const EQ_MAP: Record<string, string> = {
  // MLP
  Input: '\\mathbf{x} \\in \\mathbb{R}^{d_{in}}',
  Linear: 'y = Wx + b',
  ReLU: 'y = \\max(0, x)',
  Tanh: 'y = \\tanh(x)',
  Sigmoid: 'y = \\sigma(x)',
  MSELoss: 'L = (y - y_{true})^2',

  // CNN
  Conv: 'y = W * x + b',
  GlobalAvgPool: 'y_c = \\tfrac{1}{HW} \\sum_{h,w} x_{c,h,w}',
  Flatten: 'y = \\mathrm{vec}(x)',

  // RNN
  RNN: 'h_t = \\sigma(W_h h_{t-1} + W_x x_t + b) \\ y_t = h_t',

  LayerNorm: '\\hat{x} = \\frac{x - \\mu}{\\sigma + \\epsilon},\\quad y = \\gamma \\hat{x} + \\beta',
  BatchNorm:
    '\\hat{x} = \\frac{x - \\mu_{\\text{batch}}}{\\sigma_{\\text{batch}} + \\epsilon},\\quad y = \\gamma \\hat{x} + \\beta',

  // Transformer
  Token: 'x \\in \\mathbb{R}^{T \\times d}',
  Positional: 'x + P',
  'Self-Attn': '\\mathrm{Attn}(Q,K,V) = \\mathrm{softmax}(QK^T/\\sqrt{d_k})V',
  FFN: 'y = W_2 \\sigma(W_1 x + b_1) + b_2',
};

export type BackpropNodeData = {
  label: string;
  details: LayerDetails;
};

function getEqKey(label: string): string {
  if (label.startsWith('Linear')) return 'Linear';
  if (label.startsWith('ReLU')) return 'ReLU';
  if (label.startsWith('Tanh')) return 'Tanh';
  if (label.startsWith('Sigmoid')) return 'Sigmoid';
  if (label.startsWith('Conv')) return 'Conv';
  if (label.startsWith('GlobalAvgPool')) return 'GlobalAvgPool';
  if (label.startsWith('Flatten')) return 'Flatten';
  if (label.startsWith('RNN')) return 'RNN';
  if (label.startsWith('LayerNorm')) return 'LayerNorm';
  if (label.startsWith('BatchNorm')) return 'BatchNorm';
  if (label.startsWith('Self-Attn')) return 'Self-Attn';
  if (label.startsWith('FFN')) return 'FFN';
  if (label.startsWith('Token')) return 'Token';
  if (label.startsWith('Positional')) return 'Positional';
  if (label.startsWith('Encoder Output')) return 'Token';
  return label;
}

export default function BackpropNode({ data, selected }: NodeProps<BackpropNodeData>) {
  const { label, details } = data;

  const isActivation = ['ReLU', 'Tanh', 'Sigmoid'].some((p) => label.startsWith(p));
  const isLoss = label === 'Loss' || label === 'MSELoss';

  let borderColor = 'border-blue-500';
  let bgColor = 'bg-slate-900';
  let headerColor = 'text-blue-400';

  if (isActivation) {
    borderColor = 'border-emerald-500';
    headerColor = 'text-emerald-400';
  }
  if (isLoss) {
    borderColor = 'border-rose-500';
    headerColor = 'text-rose-400';
  }

  const hasParams = details.params && Object.keys(details.params).length > 0;

  const eqKey = getEqKey(label);
  const eq = EQ_MAP[eqKey] || 'f(x)';
  const isLongEq = eqKey === 'Self-Attn' || eqKey === 'FFN';

  const isSelfAttn = label.startsWith('Self-Attn');
  const isRNN = label.startsWith('RNN');
  const isConv = label.startsWith('Conv');
  const isResidualBlock = label.startsWith('Self-Attn') || label.startsWith('FFN');

  const isVectorInput =
    label.startsWith('Input') && Array.isArray(details.out_shape) && details.out_shape.length === 2;
  const vecDim = isVectorInput ? details.out_shape[1] ?? details.out_shape[0] : undefined;

  const isImageInput =
    label.startsWith('Input') && Array.isArray(details.out_shape) && details.out_shape.length === 3;

  const seqLen =
    Array.isArray(details.in_shape) && details.in_shape.length >= 2 ? details.in_shape[0] : undefined;
  const featureDim =
    Array.isArray(details.out_shape) && details.out_shape.length >= 2 ? details.out_shape[1] : undefined;
  const maxStepsViz = 8;
  const visibleSteps = seqLen ? Math.min(seqLen, maxStepsViz) : 4;

  const rnnInputDim =
    isRNN && Array.isArray(details.in_shape) && details.in_shape.length >= 2
      ? details.in_shape[1]
      : undefined;
  const rnnHiddenDim =
    isRNN && Array.isArray(details.out_shape) && details.out_shape.length >= 2
      ? details.out_shape[1]
      : undefined;

  const [convStep, setConvStep] = useState(0);
  const [attnHead, setAttnHead] = useState(0);

  useEffect(() => {
    if (!label.startsWith('Conv')) return;

    let Hin = 4;
    let Win = 4;
    if (Array.isArray(details.in_shape) && details.in_shape.length >= 3) {
      const shape = details.in_shape as number[];
      Hin = Math.min(8, Math.max(3, shape[shape.length - 2] ?? 4));
      Win = Math.min(8, Math.max(3, shape[shape.length - 1] ?? 4));
    }

    const maxR0 = Math.max(1, Hin - 2);
    const maxC0 = Math.max(1, Win - 2);
    const totalPositions = Math.max(1, maxR0 * maxC0);

    const id = window.setInterval(() => {
      setConvStep((s) => (s + 1) % totalPositions);
    }, 900);
    return () => window.clearInterval(id);
  }, [label, details.in_shape]);

  useEffect(() => {
    setAttnHead(0);
  }, [details.attention_heads]);

  const gradValues = hasParams ? Object.values(details.params).map((p) => p.grad_mean) : [];
  const maxGrad = gradValues.length ? Math.max(...gradValues) : 0;
  let saturationHint: 'none' | 'low' | 'high' = 'none';
  if (gradValues.length) {
    if (maxGrad < 1e-3) {
      saturationHint = 'low';
    } else if (maxGrad > 0.5) {
      saturationHint = 'high';
    }
  }

  let activationPoints: string | null = null;
  if (isActivation) {
    const N = 32;
    const xMin = -3;
    const xMax = 3;
    const svgXMin = 10;
    const svgXMax = 90;
    const svgYMin = 10;
    const svgYMax = 50;
    const coords: string[] = [];
    for (let i = 0; i < N; i++) {
      const t = i / (N - 1);
      const xVal = xMin + t * (xMax - xMin);
      let yVal: number;
      if (eqKey === 'ReLU') {
        yVal = Math.max(0, xVal);
      } else if (eqKey === 'Tanh') {
        yVal = Math.tanh(xVal);
      } else if (eqKey === 'Sigmoid') {
        yVal = 1 / (1 + Math.exp(-xVal));
      } else {
        continue;
      }

      let yNorm: number;
      if (eqKey === 'Tanh') {
        yNorm = (yVal + 1) / 2;
      } else if (eqKey === 'Sigmoid') {
        yNorm = yVal;
      } else {
        const reluMax = 3;
        const clamped = Math.max(0, Math.min(reluMax, yVal));
        yNorm = clamped / reluMax;
      }

      const xSvg = svgXMin + t * (svgXMax - svgXMin);
      const ySvg = svgYMax - yNorm * (svgYMax - svgYMin);
      coords.push(`${xSvg},${ySvg}`);
    }
    activationPoints = coords.join(' ');
  }

  const activeAttentionPattern =
    attnHead > 0 &&
    details.attention_pattern_heads &&
    details.attention_pattern_heads[attnHead - 1]
      ? details.attention_pattern_heads[attnHead - 1]
      : details.attention_pattern;

  let backSignalLatex = '\\frac{\\partial L}{\\partial x}';
  if (isRNN) {
    backSignalLatex = '\\frac{\\partial L}{\\partial h_{t-1}}';
  } else if (isSelfAttn) {
    backSignalLatex = '\\left(\\frac{\\partial L}{\\partial Q_t},\\; \\frac{\\partial L}{\\partial K_t},\\; \\frac{\\partial L}{\\partial V_t}\\right)';
  } else if (isConv) {
    backSignalLatex = '\\frac{\\partial L}{\\partial x_{c,h,w}}';
  }

  const ringClass = selected
    ? 'ring-2 ring-orange-400'
    : saturationHint === 'low'
    ? 'ring-2 ring-sky-500/40'
    : saturationHint === 'high'
    ? 'ring-2 ring-rose-500/60'
    : '';

  return (
    <div
      className={`relative w-[260px] rounded-xl border-2 ${borderColor} ${bgColor} p-3 shadow-2xl backdrop-blur-md ${ringClass}`}
    >
      <Handle type="target" position={Position.Left} className="!bg-white" />

      {/* Header */}
      <div className="mb-2 flex justify-between items-center border-b border-slate-700 pb-1">
        <div className="flex items-center gap-1">
          <span className={`font-bold text-sm ${headerColor}`}>{label}</span>
          {isResidualBlock && (
            <span className="rounded-full border border-slate-600 px-1.5 py-0.5 text-[9px] font-mono text-slate-300">
              residual
            </span>
          )}
        </div>
        <span className="text-[10px] font-mono text-slate-400">
          {JSON.stringify(details.in_shape)} {' → '} {JSON.stringify(details.out_shape)}
        </span>
      </div>

      {/* Parameter list */}
      {hasParams && (
        <div className="mb-2 flex flex-wrap gap-1 text-[9px] font-mono text-slate-400">
          {Object.keys(details.params).map((name) => (
            <span
              key={name}
              className="px-1 py-0.5 rounded border border-slate-700 bg-slate-800/60"
            >
              {name}
            </span>
          ))}
        </div>
      )}

      {isImageInput && (
        <div className="mb-2">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-sky-400 uppercase tracking-wider">Input map</span>
            <span className="text-[9px] text-slate-400 font-mono">
              {Array.isArray(details.out_shape) &&
                `${details.out_shape[0]}×${details.out_shape[1]}×${details.out_shape[2]}`}
            </span>
          </div>
          <div className="flex justify-center">
            {(() => {
              const shape = Array.isArray(details.out_shape)
                ? (details.out_shape as number[])
                : [1, 4, 4];
              const H = Math.max(1, Math.min(8, shape[1] ?? 4));
              const W = Math.max(1, Math.min(8, shape[2] ?? 4));
              const vals = (details.output_sample as number[]) || [];

              const cells: JSX.Element[] = [];
              for (let r = 0; r < H; r++) {
                for (let c = 0; c < W; c++) {
                  const idx = r * W + c;
                  const v = vals.length ? vals[idx % vals.length] : 0;
                  cells.push(
                    <div
                      key={idx}
                      className="w-3 h-3 rounded-sm border border-slate-700"
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
            })()}
          </div>
        </div>
      )}

      {/* Math */}
      <div className="mb-3 text-center text-slate-300">
        <div
          className={`${
            isLongEq ? 'text-[10px] leading-snug' : 'text-sm'
          } inline-block max-w-full overflow-x-auto whitespace-nowrap px-1`}
        >
          <Latex>{`$$ ${eq} $$`}</Latex>
        </div>
      </div>

      {isActivation && (
        <div className="mb-3">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Activation graph</span>
            <span className="text-[9px] text-slate-400 font-mono">f(x)</span>
          </div>
          <svg viewBox="0 0 100 60" className="w-full h-16 text-slate-400">
            <line x1="10" y1="50" x2="90" y2="50" stroke="#64748b" strokeWidth="1" />
            <line x1="50" y1="55" x2="50" y2="10" stroke="#64748b" strokeWidth="1" />
            {activationPoints && (
              <polyline
                points={activationPoints}
                fill="none"
                stroke="#22c55e"
                strokeWidth="2"
              />
            )}
          </svg>
        </div>
      )}

      {isConv && (
        <div className="mb-2">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-sky-400 uppercase tracking-wider">Conv maps</span>
            {Array.isArray(details.in_shape) && details.in_shape.length === 3 && (
              <span className="text-[9px] text-slate-400 font-mono">
                in: {details.in_shape[1]}×{details.in_shape[2]} · out:
                {Array.isArray(details.out_shape) && details.out_shape.length === 3
                  ? ` ${details.out_shape[1]}×${details.out_shape[2]}`
                  : ' ?×?'}
              </span>
            )}
          </div>
          <div className="flex items-center gap-4">
            {(() => {
              let Hin = 4;
              let Win = 4;
              if (Array.isArray(details.in_shape) && details.in_shape.length === 3) {
                Hin = Math.min(8, Math.max(2, details.in_shape[1] ?? 4));
                Win = Math.min(8, Math.max(2, details.in_shape[2] ?? 4));
              }

              let Hout = 4;
              let Wout = 4;
              if (Array.isArray(details.out_shape) && details.out_shape.length === 3) {
                Hout = Math.min(8, Math.max(2, details.out_shape[1] ?? 4));
                Wout = Math.min(8, Math.max(2, details.out_shape[2] ?? 4));
              }

              const inputCells: JSX.Element[] = [];
              const outputCells: JSX.Element[] = [];

              // Compute top-left of the 3x3 kernel from convStep so it sweeps the whole map.
              const maxR0 = Math.max(1, Hin - 2);
              const maxC0 = Math.max(1, Win - 2);
              const r0 = Math.floor(convStep % (maxR0 * maxC0) / maxC0);
              const c0 = convStep % maxC0;

              for (let r = 0; r < Hin; r++) {
                for (let c = 0; c < Win; c++) {
                  const inKernel = r >= r0 && r <= r0 + 2 && c >= c0 && c <= c0 + 2;
                  inputCells.push(
                    <div
                      key={`in-${r}-${c}`}
                      className={`w-3 h-3 rounded-sm border ${
                        inKernel
                          ? 'border-amber-400 bg-amber-500/70'
                          : 'border-slate-700 bg-slate-800'
                      }`}
                    />,
                  );
                }
              }

              for (let r = 0; r < Hout; r++) {
                for (let c = 0; c < Wout; c++) {
                  const isActive = r === r0 && c === c0;
                  outputCells.push(
                    <div
                      key={`out-${r}-${c}`}
                      className={`w-3 h-3 rounded-sm border ${
                        isActive
                          ? 'border-emerald-400 bg-emerald-500/80'
                          : 'border-slate-700 bg-slate-800'
                      }`}
                    />,
                  );
                }
              }

              return (
                <>
                  <div>
                    <span className="text-[9px] text-slate-400 font-mono">input</span>
                    <div
                      className="mt-1 grid gap-[2px]"
                      style={{ gridTemplateColumns: `repeat(${Win}, minmax(0, 1fr))` }}
                    >
                      {inputCells}
                    </div>
                  </div>
                  <div className="flex flex-col items-center gap-1">
                    <span className="text-[9px] text-slate-400 font-mono">kernel</span>
                    <div className="grid grid-cols-3 gap-[2px]">
                      {Array.from({ length: 9 }).map((_, i) => (
                        <div
                          key={i}
                          className="w-3 h-3 rounded-sm border border-amber-400 bg-amber-500/80"
                        />
                      ))}
                    </div>
                  </div>
                  <div>
                    <span className="text-[9px] text-slate-400 font-mono">output</span>
                    <div
                      className="mt-1 grid gap-[2px]"
                      style={{ gridTemplateColumns: `repeat(${Wout}, minmax(0, 1fr))` }}
                    >
                      {outputCells}
                    </div>
                  </div>
                </>
              );
            })()}
          </div>
          <div className="mt-1 text-[9px] text-slate-500 font-mono">kernel slides across spatial grid</div>
        </div>
      )}

      {isSelfAttn && (
        <div className="mb-2">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-sky-400 uppercase tracking-wider">Q / K / V</span>
            <span className="text-[9px] text-slate-400 font-mono">
              {seqLen && `T=${seqLen}`}
              {featureDim && `, d_model=${featureDim}`}
              {typeof details.attention_heads === 'number' && `, h=${details.attention_heads}`}
              {featureDim && typeof details.attention_heads === 'number' && details.attention_heads > 0 &&
                (() => {
                  const dk = Math.floor(featureDim / details.attention_heads!);
                  return `, d_k≈${dk}`;
                })()}
            </span>
          </div>
          {details.attention_heads && details.attention_heads > 1 && (
            <div className="mb-1 flex items-center gap-1 text-[9px] text-slate-400 font-mono">
              <span>head:</span>
              <button
                type="button"
                onClick={() => setAttnHead(0)}
                className={`px-1.5 py-0.5 rounded-full border text-[9px] ${
                  attnHead === 0
                    ? 'bg-sky-600/60 border-sky-400 text-white'
                    : 'bg-slate-800 border-slate-600 text-slate-300'
                }`}
              >
                all
              </button>
              {Array.from({ length: details.attention_heads }).map((_, idx) => (
                <button
                  key={idx}
                  type="button"
                  onClick={() => setAttnHead(idx + 1)}
                  className={`px-1.5 py-0.5 rounded-full border text-[9px] ${
                    attnHead === idx + 1
                      ? 'bg-sky-600/60 border-sky-400 text-white'
                      : 'bg-slate-800 border-slate-600 text-slate-300'
                  }`}
                >
                  h{idx + 1}
                </button>
              ))}
            </div>
          )}
          <div className="flex gap-1 text-[10px] font-mono">
            <div className="flex-1 rounded-sm bg-sky-900/70 border border-sky-500/60 text-center py-1">Q</div>
            <div className="flex-1 rounded-sm bg-violet-900/70 border border-violet-500/60 text-center py-1">K</div>
            <div className="flex-1 rounded-sm bg-emerald-900/70 border border-emerald-500/60 text-center py-1">V</div>
          </div>
          <div className="mt-2">
            <div className="text-[9px] text-slate-400 mb-1 font-mono">attention pattern</div>
            <div className="grid grid-cols-4 gap-[1px]">
              {activeAttentionPattern
                ? activeAttentionPattern.slice(0, 4).flatMap((row, i) =>
                    row.slice(0, 4).map((v, j) => (
                      <div
                        key={`${i}-${j}`}
                        className={`w-3 h-3 ${
                          v > 0.6
                            ? 'bg-amber-500/90'
                            : v > 0.3
                            ? 'bg-amber-400/70'
                            : 'bg-slate-700'
                        }`}
                      />
                    )),
                  )
                : Array.from({ length: 16 }).map((_, i) => (
                    <div
                      key={i}
                      className={`w-3 h-3 ${i % 5 === 0 ? 'bg-amber-500/80' : 'bg-slate-700'}`}
                    />
                  ))}
            </div>
          </div>
        </div>
      )}

      {isRNN && (
        <div className="mb-2">
          <div className="flex justify-between items-center mb-1">
            <span className="text-[10px] font-bold text-amber-400 uppercase tracking-wider">Hidden state over time</span>
            {seqLen && (
              <span className="text-[9px] text-slate-400 font-mono">T={seqLen}</span>
            )}
          </div>
          <div className="space-y-1">
            <div className="flex items-center gap-1">
              <span className="text-[9px] text-slate-400 font-mono w-6">h_t</span>
              <div className="flex gap-0.5">
                {Array.from({ length: visibleSteps }).map((_, i) => (
                  <div
                    key={i}
                    className={`w-3 h-3 rounded-sm border ${
                      i === visibleSteps - 1
                        ? 'bg-amber-500/80 border-amber-300'
                        : 'bg-slate-800 border-slate-700'
                    }`}
                  />
                ))}
                {seqLen && seqLen > maxStepsViz && (
                  <span className="text-[9px] text-slate-500 font-mono ml-1">…</span>
                )}
              </div>
            </div>
          </div>

          <div className="mt-2 rounded border border-slate-700 bg-slate-900/80 p-2">
            <div className="mb-1 flex justify-between items-center">
              <span className="text-[9px] text-slate-400 font-mono">RNN step</span>
              {seqLen && (
                <span className="text-[9px] text-slate-500 font-mono">t = 1 … {seqLen}</span>
              )}
            </div>
            <div className="flex items-center justify-between text-[9px] font-mono text-slate-300">
              <div className="flex flex-col items-center gap-1">
                <span>x_t</span>
                <div className="flex gap-[1px]">
                  {Array.from({ length: Math.min(rnnInputDim ?? 4, 5) }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 h-3 rounded-sm border border-slate-600 bg-slate-800"
                    />
                  ))}
                </div>
                <span className="mt-1">
                  <Latex>{'$h_{t-1}$'}</Latex>
                </span>
                <div className="flex gap-[1px]">
                  {Array.from({ length: Math.min(rnnHiddenDim ?? 4, 5) }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 h-3 rounded-sm border border-slate-600 bg-slate-800"
                    />
                  ))}
                </div>
              </div>

              <div className="flex-1 mx-2 h-px bg-slate-600" />

              <div className="flex flex-col items-center gap-1 flex-1">
                <span>linear</span>
                <span className="text-[8px] text-slate-200 inline-block max-w-[120px] overflow-x-auto whitespace-nowrap px-1">
                  <Latex>{'$W_x x_t + W_h h_{t-1} + b$'}</Latex>
                </span>
              </div>

              <div className="flex-1 mx-2 h-px bg-slate-600" />

              <div className="flex flex-col items-center gap-1">
                <span>σ</span>
                <div className="flex gap-[1px]">
                  {Array.from({ length: Math.min(rnnHiddenDim ?? 4, 5) }).map((_, i) => (
                    <div
                      key={i}
                      className="w-2 h-3 rounded-sm border border-amber-300 bg-amber-500/80"
                    />
                  ))}
                </div>
                <span>
                  <Latex>{'$h_t$'}</Latex>
                </span>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Vector preview for MLP input */}
      {isVectorInput && vecDim && (
        <div className="mb-2 flex justify-center items-center gap-1">
          {Array.from({ length: Math.min(vecDim, 8) }).map((_, i) => (
            <div
              key={i}
              className="w-3 h-3 rounded-sm border border-slate-600 bg-slate-800"
            />
          ))}
          {vecDim > 8 && (
            <span className="text-[10px] text-slate-500 font-mono ml-1">…</span>
          )}
        </div>
      )}

      {/* Gradients */}
      {hasParams && (
        <div className="space-y-1 rounded bg-black/40 p-2">
          <div className="flex items-center justify-between">
            <p className="text-[10px] font-bold text-rose-400 uppercase tracking-wider">Gradients</p>
            {saturationHint !== 'none' && (
              <span
                className={`text-[9px] font-mono ${
                  saturationHint === 'low' ? 'text-sky-400' : 'text-rose-400'
                }`}
              >
                {saturationHint === 'low' ? 'vanishing?' : 'exploding?'}
              </span>
            )}
          </div>
          {Object.entries(details.params).map(([key, val]) => {
            const latexKey = key;
            return (
              <div key={key} className="flex justify-between items-center text-[10px] font-mono text-slate-300">
                <span>
                  <Latex>{`$\\frac{\\partial L}{\\partial ${latexKey}}$`}</Latex>
                </span>
                <span className="text-rose-300">{val.grad_mean.toFixed(4)}</span>
              </div>
            );
          })}
        </div>
      )}

      <div className="mt-2 flex items-center justify-between text-[9px] text-slate-400 font-mono">
        <span>backprop to prev:</span>
        <span className="flex items-center gap-1 text-slate-200">
          <span>←</span>
          <Latex>{`$${backSignalLatex}$`}</Latex>
        </span>
      </div>

      {/* Forward mean (always) */}
      <div className="mt-1 pt-2 border-t border-slate-800 flex items-center justify-between text-[10px] text-slate-400 font-mono">
        <span>forward mean:</span>
        <span className="text-emerald-300 font-bold">{details.forward_mean.toFixed(4)}</span>
      </div>

      <Handle type="source" position={Position.Right} className="!bg-white" />
    </div>
  );
}
