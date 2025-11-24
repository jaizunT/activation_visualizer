import { useEffect, useState, FormEvent } from 'react';
import type { AiProvider, AiMessage } from '../App';

type ModelOption = {
  id: string;
  label: string;
};

type Props = {
  isOpen: boolean;
  onToggle: () => void;
  provider: AiProvider | '';
  onProviderChange: (p: AiProvider | '') => void;
  apiKey: string;
  onApiKeyChange: (v: string) => void;
  model: string;
  onModelChange: (v: string) => void;
  models: ModelOption[];
  modelsLoading: boolean;
  modelsError: string | null;
  onRefreshModels: () => void;
  messages: AiMessage[];
  onAsk: (question: string) => Promise<void> | void;
  loading: boolean;
};

export default function AiAssistantPanel({
  isOpen,
  onToggle,
  provider,
  onProviderChange,
  apiKey,
  onApiKeyChange,
  model,
  onModelChange,
  models,
  modelsLoading,
  modelsError,
  onRefreshModels,
  messages,
  onAsk,
  loading,
}: Props) {
  const [draft, setDraft] = useState('');

  useEffect(() => {
    if (!provider) return;
    if (!model && models.length) {
      onModelChange(models[0].id);
    }
  }, [provider, model, models, onModelChange]);

  const handleProviderChange = (value: string) => {
    const p = value as AiProvider | '';
    onProviderChange(p);
  };

  const handleSubmit = async (e: FormEvent) => {
    e.preventDefault();
    const q = draft.trim();
    if (!q || !provider || !apiKey || loading) return;
    await onAsk(q);
    setDraft('');
  };

  return (
    <div
      className={`absolute top-16 right-0 bottom-0 w-80 border-l border-slate-800 bg-slate-900/95 z-30 flex flex-col transition-transform duration-200 ${
        isOpen ? 'translate-x-0' : 'translate-x-full'
      }`}
    >
      <div className="px-4 py-3 border-b border-slate-800 flex items-center justify-between">
        <span className="text-xs font-semibold text-slate-100">AI Assistant</span>
        <button
          type="button"
          onClick={onToggle}
          className="text-[11px] text-slate-400 hover:text-slate-100"
        >
          {isOpen ? 'Hide' : 'Show'}
        </button>
      </div>

      <div className="px-4 py-3 border-b border-slate-800 space-y-2 text-[11px]">
        <div className="space-y-1">
          <label className="block text-slate-300">Provider</label>
          <select
            value={provider}
            onChange={(e) => handleProviderChange(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-100"
          >
            <option value="">Choose provider</option>
            <option value="openai">OpenAI</option>
            <option value="anthropic">Anthropic</option>
            <option value="google">Google</option>
          </select>
        </div>

        <div className="space-y-1">
          <div className="flex items-center justify-between">
            <label className="block text-slate-300">Model</label>
            <button
              type="button"
              onClick={() => {
                if (!provider || !apiKey || modelsLoading) return;
                onRefreshModels();
              }}
              disabled={!provider || !apiKey || modelsLoading}
              className="text-[10px] text-slate-400 hover:text-slate-100 disabled:opacity-40"
            >
              {modelsLoading ? 'Refreshing…' : 'Refresh'}
            </button>
          </div>
          <select
            value={model}
            onChange={(e) => onModelChange(e.target.value)}
            disabled={!provider || !models.length}
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-100 disabled:opacity-50"
          >
            {!provider && <option value="">Choose provider first</option>}
            {provider && !models.length && <option value="">No models</option>}
            {provider &&
              models.map((opt) => (
                <option key={opt.id} value={opt.id}>
                  {opt.label}
                </option>
              ))}
          </select>
          {modelsError && (
            <p className="text-[9px] text-rose-400">{modelsError}</p>
          )}
        </div>

        <div className="space-y-1">
          <label className="block text-slate-300">API key</label>
          <input
            type="password"
            value={apiKey}
            onChange={(e) => onApiKeyChange(e.target.value)}
            className="w-full bg-slate-800 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-100"
            placeholder="Paste key here"
          />
          <p className="text-[9px] text-slate-500">
            Keys are kept in this browser session and sent directly to the selected provider.
          </p>
        </div>
      </div>

      <div className="flex-1 overflow-y-auto px-4 py-3 space-y-2 text-[11px]">
        {messages.length === 0 && (
          <p className="text-slate-500 text-[11px]">
            Ask about activations, gradients, or the math behind the current architecture or layer.
          </p>
        )}
        {messages.map((m, idx) => (
          <div
            key={idx}
            className={`rounded-md px-2 py-1.5 whitespace-pre-wrap ${
              m.role === 'user'
                ? 'bg-slate-800 text-slate-100 self-end'
                : 'bg-slate-900 border border-slate-700 text-slate-100'
            }`}
          >
            <span className="block text-[10px] font-semibold mb-0.5 text-slate-400">
              {m.role === 'user' ? 'You' : 'Assistant'}
            </span>
            {m.content}
          </div>
        ))}
      </div>

      <form
        onSubmit={handleSubmit}
        className="border-t border-slate-800 px-3 py-2 flex flex-col gap-2"
      >
        <textarea
          rows={2}
          value={draft}
          onChange={(e) => setDraft(e.target.value)}
          placeholder="Ask a theoretical question about the model or a specific layer..."
          className="w-full resize-none bg-slate-900 border border-slate-700 rounded px-2 py-1 text-[11px] text-slate-100 placeholder:text-slate-500"
        />
        <button
          type="submit"
          disabled={!provider || !apiKey || !draft.trim() || loading}
          className="self-end px-3 py-1.5 rounded-md text-[11px] font-medium bg-blue-600 text-white disabled:opacity-40"
        >
          {loading ? 'Thinking…' : 'Ask'}
        </button>
      </form>
    </div>
  );
}
