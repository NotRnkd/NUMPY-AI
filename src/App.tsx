import React, { useState, useEffect, useRef } from 'react';
import {
  Cpu,
  MessageSquare,
  Zap,
  Layers,
  Activity,
  Play,
  Square,
  RefreshCw,
  Send,
  Download,
  Settings,
  HardDrive,
  Info,
  CheckCircle2,
  Sliders,
  ChevronRight,
  Terminal,
} from 'lucide-react';

interface ModelStatus {
  status: string;
  device: string;
  model_params: number;
  vocab_size: number;
  context_length: number;
  embedding_dim: number;
  num_heads: number;
  num_layers: number;
  norm_type: string;
  activation: string;
}

interface HardwareInfo {
  os: string;
  cpu_name: string;
  cpu_cores: number;
  ram_gb: number;
  has_cuda: boolean;
  cuda_device_name?: string;
  cuda_vram_gb?: number;
  has_npu: boolean;
  npu_details: string;
  has_avx2: boolean;
  has_avx512: boolean;
  recommended_backend: string;
  active_device: string;
}

interface ChatMessage {
  role: 'system' | 'user' | 'assistant';
  content: string;
  latency_ms?: number;
}

interface TrainMetrics {
  is_training: boolean;
  step: number;
  total_steps: number;
  loss: number;
  val_loss: number;
  lr: number;
  history: Array<{ step: number; loss: number; val_loss: number }>;
}

export default function App() {
  const [activeTab, setActiveTab] = useState<'chat' | 'generate' | 'architecture' | 'train' | 'hardware'>('chat');
  const [status, setStatus] = useState<ModelStatus | null>(null);
  const [hardware, setHardware] = useState<HardwareInfo | null>(null);

  // Chat State
  const [messages, setMessages] = useState<ChatMessage[]>([
    {
      role: 'assistant',
      content: 'Hello! I am NumPy-GPT, an offline neural language model running natively in pure NumPy and CuPy. How can I help you today?',
    },
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [systemPrompt, setSystemPrompt] = useState('You are NumPy-GPT, a helpful, honest, and offline AI assistant.');
  const [temperature, setTemperature] = useState(0.7);
  const [topK, setTopK] = useState(40);
  const [topP, setTopP] = useState(0.9);
  const [repetitionPenalty, setRepetitionPenalty] = useState(1.15);
  const [isChatLoading, setIsChatLoading] = useState(false);
  const chatBottomRef = useRef<HTMLDivElement>(null);

  // Generation Playground State
  const [genPrompt, setGenPrompt] = useState('Artificial intelligence and neural networks');
  const [genOutput, setGenOutput] = useState('');
  const [genUseCache, setGenUseCache] = useState(true);
  const [genLatency, setGenLatency] = useState<number | null>(null);
  const [genSpeed, setGenSpeed] = useState<number | null>(null);
  const [isGenLoading, setIsGenLoading] = useState(false);

  // Training State
  const [trainSteps, setTrainSteps] = useState(30);
  const [trainLr, setTrainLr] = useState(0.0005);
  const [trainMetrics, setTrainMetrics] = useState<TrainMetrics>({
    is_training: false,
    step: 0,
    total_steps: 0,
    loss: 0,
    val_loss: 0,
    lr: 0,
    history: [],
  });

  // Benchmark & Export State
  const [benchResult, setBenchResult] = useState<any>(null);
  const [isBenchmarking, setIsBenchmarking] = useState(false);
  const [exportResult, setExportResult] = useState<any>(null);
  const [isExporting, setIsExporting] = useState(false);

  // Load status and hardware on mount
  useEffect(() => {
    fetchStatus();
    fetchHardware();
  }, []);

  useEffect(() => {
    chatBottomRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, isChatLoading]);

  // Poll training status
  useEffect(() => {
    let timer: any;
    if (activeTab === 'train' || trainMetrics.is_training) {
      timer = setInterval(async () => {
        try {
          const res = await fetch('/api/train/status');
          if (res.ok) {
            const data = await res.json();
            setTrainMetrics(data);
          }
        } catch {
          // ignore transient poll failure
        }
      }, 1000);
    }
    return () => clearInterval(timer);
  }, [activeTab, trainMetrics.is_training]);

  const fetchStatus = async () => {
    try {
      const res = await fetch('/api/status');
      if (res.ok) {
        const data = await res.json();
        setStatus(data);
      }
    } catch (e) {
      console.error('Failed to fetch status', e);
    }
  };

  const fetchHardware = async () => {
    try {
      const res = await fetch('/api/hardware');
      if (res.ok) {
        const data = await res.json();
        setHardware(data);
      }
    } catch (e) {
      console.error('Failed to fetch hardware', e);
    }
  };

  const handleSendMessage = async () => {
    if (!inputMessage.trim() || isChatLoading) return;
    const userMsg: ChatMessage = { role: 'user', content: inputMessage.trim() };
    const nextMessages = [...messages, userMsg];
    setMessages(nextMessages);
    setInputMessage('');
    setIsChatLoading(true);

    try {
      const res = await fetch('/api/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          messages: nextMessages.map((m) => ({ role: m.role, content: m.content })),
          system_prompt: systemPrompt,
          temperature,
          top_k: topK,
          top_p: topP,
          repetition_penalty: repetitionPenalty,
          max_new_tokens: 120,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setMessages([
          ...nextMessages,
          {
            role: 'assistant',
            content: data.reply || '...',
            latency_ms: data.latency_ms,
          },
        ]);
      } else {
        setMessages([
          ...nextMessages,
          { role: 'assistant', content: 'Error communicating with local NumPy model.' },
        ]);
      }
    } catch (err) {
      setMessages([
        ...nextMessages,
        { role: 'assistant', content: 'Connection error to NumPy backend.' },
      ]);
    } finally {
      setIsChatLoading(false);
    }
  };

  const handleGenerate = async () => {
    if (!genPrompt.trim() || isGenLoading) return;
    setIsGenLoading(true);
    setGenOutput('');

    try {
      const res = await fetch('/api/generate', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt: genPrompt,
          max_new_tokens: 80,
          temperature,
          top_k: topK,
          top_p: topP,
          repetition_penalty: repetitionPenalty,
          use_cache: genUseCache,
        }),
      });

      if (res.ok) {
        const data = await res.json();
        setGenOutput(data.text);
        setGenLatency(data.latency_ms);
        setGenSpeed(data.speed_tok_s);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenLoading(false);
    }
  };

  const handleStartTrain = async () => {
    try {
      await fetch('/api/train/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          steps: trainSteps,
          lr: trainLr,
          batch_size: 2,
        }),
      });
      setTrainMetrics((prev) => ({ ...prev, is_training: true }));
    } catch (e) {
      console.error(e);
    }
  };

  const handleStopTrain = async () => {
    try {
      await fetch('/api/train/stop', { method: 'POST' });
    } catch (e) {
      console.error(e);
    }
  };

  const handleRunBenchmark = async () => {
    setIsBenchmarking(true);
    try {
      const res = await fetch('/api/benchmark', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setBenchResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsBenchmarking(false);
    }
  };

  const handleExportOnnx = async () => {
    setIsExporting(true);
    try {
      const res = await fetch('/api/export', { method: 'POST' });
      if (res.ok) {
        const data = await res.json();
        setExportResult(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsExporting(false);
    }
  };

  return (
    <div id="app-root" className="min-h-screen bg-neutral-950 text-neutral-100 flex flex-col font-sans">
      {/* Top Header */}
      <header id="main-header" className="border-b border-neutral-800 bg-neutral-900/60 backdrop-blur px-6 py-3.5 flex items-center justify-between sticky top-0 z-50">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 rounded-xl bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400 font-bold">
            <Cpu className="w-5 h-5" />
          </div>
          <div>
            <div className="flex items-center gap-2">
              <h1 className="text-base font-semibold tracking-tight text-neutral-100">NumPy-GPT</h1>
              <span className="text-xs px-2 py-0.5 rounded-full bg-blue-500/10 text-blue-400 border border-blue-500/20 font-mono">
                v2.0 Architecture
              </span>
            </div>
            <p className="text-xs text-neutral-400">
              Offline-first Decoder-only Transformer in pure NumPy & CuPy
            </p>
          </div>
        </div>

        {/* Global status pills */}
        <div className="flex items-center gap-3">
          <div className="flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neutral-800/80 border border-neutral-700/60 text-xs font-mono">
            <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse"></span>
            <span className="text-neutral-400">Backend:</span>
            <span className="text-neutral-200 uppercase font-semibold">{status?.device || 'CPU'}</span>
          </div>
          <div className="hidden sm:flex items-center gap-2 px-3 py-1.5 rounded-lg bg-neutral-800/80 border border-neutral-700/60 text-xs font-mono">
            <span className="text-neutral-400">Params:</span>
            <span className="text-neutral-200">
              {status?.model_params ? `${(status.model_params / 1000).toFixed(1)}k` : 'Loaded'}
            </span>
          </div>
        </div>
      </header>

      {/* Main Navigation Tabs */}
      <nav id="nav-tabs" className="border-b border-neutral-800 bg-neutral-900/30 px-6 flex gap-2">
        <button
          id="tab-chat"
          onClick={() => setActiveTab('chat')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'chat'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <MessageSquare className="w-4 h-4" />
          Chat Assistant
        </button>

        <button
          id="tab-generate"
          onClick={() => setActiveTab('generate')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'generate'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <Zap className="w-4 h-4" />
          Text Generation & KV-Cache
        </button>

        <button
          id="tab-architecture"
          onClick={() => setActiveTab('architecture')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'architecture'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <Layers className="w-4 h-4" />
          Architecture & Weights
        </button>

        <button
          id="tab-train"
          onClick={() => setActiveTab('train')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'train'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <Activity className="w-4 h-4" />
          Trainer & Loss Monitor
        </button>

        <button
          id="tab-hardware"
          onClick={() => setActiveTab('hardware')}
          className={`flex items-center gap-2 px-4 py-3 text-sm font-medium border-b-2 transition-colors ${
            activeTab === 'hardware'
              ? 'border-blue-500 text-blue-400'
              : 'border-transparent text-neutral-400 hover:text-neutral-200'
          }`}
        >
          <HardDrive className="w-4 h-4" />
          Hardware & ONNX
        </button>
      </nav>

      {/* Main Content Area */}
      <main id="tab-content" className="flex-1 p-6 max-w-7xl w-full mx-auto">
        {/* ================= CHAT TAB ================= */}
        {activeTab === 'chat' && (
          <div className="grid grid-cols-1 lg:grid-cols-4 gap-6 h-[calc(100vh-12rem)] min-h-[500px]">
            {/* Left Chat Window */}
            <div className="lg:col-span-3 flex flex-col bg-neutral-900/50 border border-neutral-800 rounded-2xl overflow-hidden shadow-xl">
              {/* Message History */}
              <div id="chat-messages" className="flex-1 overflow-y-auto p-5 space-y-4">
                {messages.map((m, idx) => (
                  <div
                    key={idx}
                    className={`flex flex-col ${
                      m.role === 'user' ? 'items-end' : 'items-start'
                    }`}
                  >
                    <div className="text-[11px] font-mono text-neutral-500 mb-1 px-1">
                      {m.role === 'user' ? 'YOU' : 'NUMPY-GPT'}
                    </div>
                    <div
                      className={`max-w-[85%] rounded-2xl px-4 py-3 text-sm leading-relaxed whitespace-pre-wrap ${
                        m.role === 'user'
                          ? 'bg-blue-600 text-white rounded-br-none shadow-md'
                          : 'bg-neutral-800/90 text-neutral-100 border border-neutral-700/60 rounded-bl-none shadow-md'
                      }`}
                    >
                      {m.content}
                    </div>
                    {m.latency_ms && (
                      <span className="text-[10px] text-neutral-500 mt-1 font-mono">
                        Generated in {m.latency_ms.toFixed(0)} ms
                      </span>
                    )}
                  </div>
                ))}
                {isChatLoading && (
                  <div className="flex items-center gap-2 text-neutral-400 text-sm font-mono p-3 bg-neutral-800/50 rounded-xl w-fit border border-neutral-700/40">
                    <span className="w-2 h-2 rounded-full bg-blue-400 animate-ping"></span>
                    Computing forward attention passes...
                  </div>
                )}
                <div ref={chatBottomRef} />
              </div>

              {/* Chat Input Bar */}
              <div className="p-4 border-t border-neutral-800 bg-neutral-900/80">
                <form
                  onSubmit={(e) => {
                    e.preventDefault();
                    handleSendMessage();
                  }}
                  className="flex gap-2"
                >
                  <input
                    id="chat-input"
                    type="text"
                    value={inputMessage}
                    onChange={(e) => setInputMessage(e.target.value)}
                    placeholder="Ask NumPy-GPT anything (e.g., 'Explain RoPE attention')..."
                    disabled={isChatLoading}
                    className="flex-1 bg-neutral-950 border border-neutral-800 focus:border-blue-500 rounded-xl px-4 py-2.5 text-sm text-neutral-100 placeholder-neutral-500 focus:outline-none transition-colors"
                  />
                  <button
                    id="chat-send-button"
                    type="submit"
                    disabled={isChatLoading || !inputMessage.trim()}
                    className="px-5 py-2.5 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-xl text-sm font-medium flex items-center gap-2 transition-colors cursor-pointer"
                  >
                    <Send className="w-4 h-4" />
                    <span>Send</span>
                  </button>
                </form>
              </div>
            </div>

            {/* Right Hyperparameter Sidebar */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-5 flex flex-col gap-5 overflow-y-auto">
              <div className="flex items-center gap-2 text-sm font-semibold text-neutral-200">
                <Sliders className="w-4 h-4 text-blue-400" />
                <span>Sampling Controls</span>
              </div>

              {/* Temperature */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <span className="text-neutral-400">Temperature</span>
                  <span className="text-neutral-200">{temperature.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.5"
                  step="0.05"
                  value={temperature}
                  onChange={(e) => setTemperature(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Top-K */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <span className="text-neutral-400">Top-K</span>
                  <span className="text-neutral-200">{topK}</span>
                </div>
                <input
                  type="range"
                  min="5"
                  max="100"
                  step="5"
                  value={topK}
                  onChange={(e) => setTopK(parseInt(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Top-P */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <span className="text-neutral-400">Top-P (Nucleus)</span>
                  <span className="text-neutral-200">{topP.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="0.1"
                  max="1.0"
                  step="0.05"
                  value={topP}
                  onChange={(e) => setTopP(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* Repetition Penalty */}
              <div>
                <div className="flex justify-between text-xs mb-1.5 font-mono">
                  <span className="text-neutral-400">Repetition Penalty</span>
                  <span className="text-neutral-200">{repetitionPenalty.toFixed(2)}</span>
                </div>
                <input
                  type="range"
                  min="1.0"
                  max="1.5"
                  step="0.05"
                  value={repetitionPenalty}
                  onChange={(e) => setRepetitionPenalty(parseFloat(e.target.value))}
                  className="w-full accent-blue-500"
                />
              </div>

              {/* System Prompt */}
              <div>
                <label className="text-xs font-mono text-neutral-400 block mb-1.5">System Prompt</label>
                <textarea
                  rows={3}
                  value={systemPrompt}
                  onChange={(e) => setSystemPrompt(e.target.value)}
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-2.5 text-xs text-neutral-300 focus:outline-none focus:border-blue-500"
                />
              </div>

              {/* Reset History Button */}
              <button
                id="chat-reset-button"
                onClick={() =>
                  setMessages([
                    {
                      role: 'assistant',
                      content: 'Conversation history reset. How can I help you today?',
                    },
                  ])
                }
                className="mt-auto px-4 py-2 border border-neutral-700/80 hover:bg-neutral-800 text-neutral-300 rounded-xl text-xs font-medium flex items-center justify-center gap-2 transition-colors cursor-pointer"
              >
                <RefreshCw className="w-3.5 h-3.5" />
                Reset Conversation
              </button>
            </div>
          </div>
        )}

        {/* ================= GENERATION & KV-CACHE TAB ================= */}
        {activeTab === 'generate' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* Input Side */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col gap-4">
              <h2 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
                <Zap className="w-5 h-5 text-blue-400" />
                Prompt & Decoding Engine
              </h2>

              <div>
                <label className="text-xs font-mono text-neutral-400 block mb-2">Prompt String</label>
                <textarea
                  id="generate-prompt-input"
                  rows={5}
                  value={genPrompt}
                  onChange={(e) => setGenPrompt(e.target.value)}
                  placeholder="Enter seed text..."
                  className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-3.5 text-sm text-neutral-100 focus:outline-none focus:border-blue-500 font-mono"
                />
              </div>

              {/* KV-Cache Acceleration Toggle */}
              <div className="p-4 rounded-xl bg-neutral-950/80 border border-neutral-800 flex items-center justify-between">
                <div>
                  <div className="text-sm font-medium text-neutral-200">Linear-Time KV-Cache</div>
                  <div className="text-xs text-neutral-500">
                    Stores past key & value projections to eliminate O(T²) redundant passes
                  </div>
                </div>
                <button
                  id="toggle-kv-cache"
                  onClick={() => setGenUseCache(!genUseCache)}
                  className={`px-3 py-1.5 rounded-lg text-xs font-semibold font-mono transition-colors ${
                    genUseCache
                      ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                      : 'bg-neutral-800 text-neutral-400 border border-neutral-700'
                  }`}
                >
                  {genUseCache ? 'ENABLED (Fast)' : 'DISABLED (Full O(T²))'}
                </button>
              </div>

              <button
                id="generate-button"
                onClick={handleGenerate}
                disabled={isGenLoading}
                className="w-full py-3 bg-blue-600 hover:bg-blue-500 disabled:opacity-50 text-white rounded-xl text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-md"
              >
                {isGenLoading ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                <span>{isGenLoading ? 'Generating...' : 'Run Autoregressive Generation'}</span>
              </button>
            </div>

            {/* Output Side */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col gap-4">
              <div className="flex items-center justify-between">
                <h2 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
                  <Terminal className="w-5 h-5 text-emerald-400" />
                  Model Output Stream
                </h2>
                {genLatency && (
                  <div className="flex gap-2">
                    <span className="text-xs font-mono px-2.5 py-1 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                      {genLatency.toFixed(0)} ms
                    </span>
                    {genSpeed && (
                      <span className="text-xs font-mono px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                        {genSpeed.toFixed(1)} tok/s
                      </span>
                    )}
                  </div>
                )}
              </div>

              <div
                id="generate-output-box"
                className="flex-1 min-h-[220px] bg-neutral-950 border border-neutral-800 rounded-xl p-4 text-sm font-mono text-neutral-200 overflow-y-auto whitespace-pre-wrap leading-relaxed"
              >
                {genOutput || (
                  <span className="text-neutral-600">Generated tokens will appear here...</span>
                )}
              </div>
            </div>
          </div>
        )}

        {/* ================= ARCHITECTURE TAB ================= */}
        {activeTab === 'architecture' && (
          <div className="space-y-6">
            {/* Quick Metrics Bar */}
            <div className="grid grid-cols-2 sm:grid-cols-4 gap-4">
              <div className="p-4 rounded-xl bg-neutral-900/50 border border-neutral-800">
                <div className="text-xs font-mono text-neutral-500 uppercase">Parameters</div>
                <div className="text-2xl font-bold text-neutral-100 mt-1 font-mono">
                  {status?.model_params ? status.model_params.toLocaleString() : '126,080'}
                </div>
                <div className="text-[11px] text-blue-400 mt-1">Weight-tied input & output</div>
              </div>

              <div className="p-4 rounded-xl bg-neutral-900/50 border border-neutral-800">
                <div className="text-xs font-mono text-neutral-500 uppercase">Context Length</div>
                <div className="text-2xl font-bold text-neutral-100 mt-1 font-mono">
                  {status?.context_length || 256}
                </div>
                <div className="text-[11px] text-emerald-400 mt-1">RoPE continuous rotation</div>
              </div>

              <div className="p-4 rounded-xl bg-neutral-900/50 border border-neutral-800">
                <div className="text-xs font-mono text-neutral-500 uppercase">Attention Heads</div>
                <div className="text-2xl font-bold text-neutral-100 mt-1 font-mono">
                  {status?.num_heads || 4}
                </div>
                <div className="text-[11px] text-neutral-400 mt-1">
                  dim={status?.embedding_dim || 128} ({((status?.embedding_dim || 128) / (status?.num_heads || 4)).toFixed(0)} per head)
                </div>
              </div>

              <div className="p-4 rounded-xl bg-neutral-900/50 border border-neutral-800">
                <div className="text-xs font-mono text-neutral-500 uppercase">Layers / Blocks</div>
                <div className="text-2xl font-bold text-neutral-100 mt-1 font-mono">
                  {status?.num_layers || 4}
                </div>
                <div className="text-[11px] text-neutral-400 mt-1">RMSNorm + SwiGLU</div>
              </div>
            </div>

            {/* Architecture Pipeline Visualizer */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6">
              <h3 className="text-base font-semibold text-neutral-100 mb-6 flex items-center gap-2">
                <Layers className="w-5 h-5 text-blue-400" />
                NumPy-GPT Transformer Pipeline
              </h3>

              <div className="flex flex-col gap-4 font-mono text-xs">
                {/* 1. Embeddings */}
                <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-lg bg-blue-500/10 border border-blue-500/20 text-blue-400 flex items-center justify-center font-bold">1</span>
                    <div>
                      <div className="text-sm font-semibold text-neutral-200">Token Embedding Matrix (WTE)</div>
                      <div className="text-neutral-400">Shape: (vocab_size={status?.vocab_size || 512}, embedding_dim={status?.embedding_dim || 128})</div>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-neutral-800 text-neutral-300">Tied with LM_Head</span>
                </div>

                {/* 2. RoPE Attention */}
                <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-lg bg-indigo-500/10 border border-indigo-500/20 text-indigo-400 flex items-center justify-center font-bold">2</span>
                    <div>
                      <div className="text-sm font-semibold text-neutral-200">Rotary Position Embeddings (RoPE)</div>
                      <div className="text-neutral-400">Formula: R_Θ,m d = Diag(e^i m θ_1, ..., e^i m θ_d/2) applied to Q & K</div>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-indigo-500/10 text-indigo-400 border border-indigo-500/20">Zero parameters</span>
                </div>

                {/* 3. RMSNorm */}
                <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-lg bg-emerald-500/10 border border-emerald-500/20 text-emerald-400 flex items-center justify-center font-bold">3</span>
                    <div>
                      <div className="text-sm font-semibold text-neutral-200">Root Mean Square Normalization (RMSNorm)</div>
                      <div className="text-neutral-400">RMS(x) = sqrt(1/d * sum(x_i^2) + eps); x_norm = (x / RMS(x)) * gamma</div>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">1D Gain Only</span>
                </div>

                {/* 4. SwiGLU */}
                <div className="p-4 rounded-xl bg-neutral-950 border border-neutral-800 flex items-center justify-between">
                  <div className="flex items-center gap-3">
                    <span className="w-7 h-7 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-400 flex items-center justify-center font-bold">4</span>
                    <div>
                      <div className="text-sm font-semibold text-neutral-200">Swish Gated Linear Unit (SwiGLU)</div>
                      <div className="text-neutral-400">SwiGLU(x) = (SiLU(x * W_gate) ⊙ (x * W_up)) * W_down</div>
                    </div>
                  </div>
                  <span className="px-2.5 py-1 rounded bg-amber-500/10 text-amber-400 border border-amber-500/20">3 Projections</span>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================= TRAIN TAB ================= */}
        {activeTab === 'train' && (
          <div className="space-y-6">
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* Training Controls */}
              <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col gap-4">
                <h3 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
                  <Activity className="w-5 h-5 text-blue-400" />
                  Training Engine Controls
                </h3>

                <div>
                  <label className="text-xs font-mono text-neutral-400 block mb-1">Steps to Run</label>
                  <input
                    id="train-steps-input"
                    type="number"
                    min="5"
                    max="1000"
                    value={trainSteps}
                    onChange={(e) => setTrainSteps(parseInt(e.target.value) || 20)}
                    disabled={trainMetrics.is_training}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-2.5 text-sm text-neutral-200 font-mono"
                  />
                </div>

                <div>
                  <label className="text-xs font-mono text-neutral-400 block mb-1">Learning Rate (AdamW)</label>
                  <input
                    id="train-lr-input"
                    type="number"
                    step="0.0001"
                    value={trainLr}
                    onChange={(e) => setTrainLr(parseFloat(e.target.value) || 0.0005)}
                    disabled={trainMetrics.is_training}
                    className="w-full bg-neutral-950 border border-neutral-800 rounded-xl p-2.5 text-sm text-neutral-200 font-mono"
                  />
                </div>

                <div className="text-xs text-neutral-400 bg-neutral-950 p-3 rounded-xl border border-neutral-800">
                  <p className="font-semibold text-neutral-300 mb-1">Features Active:</p>
                  <ul className="list-disc pl-4 space-y-1">
                    <li>Decoupled AdamW (excludes 1D norm gains)</li>
                    <li>Cosine decay with linear warmup</li>
                    <li>Global L2 gradient clipping (1.0)</li>
                  </ul>
                </div>

                {trainMetrics.is_training ? (
                  <button
                    id="stop-train-button"
                    onClick={handleStopTrain}
                    className="w-full py-3 bg-rose-600 hover:bg-rose-500 text-white rounded-xl text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors"
                  >
                    <Square className="w-4 h-4" />
                    <span>Stop & Save Checkpoint</span>
                  </button>
                ) : (
                  <button
                    id="start-train-button"
                    onClick={handleStartTrain}
                    className="w-full py-3 bg-blue-600 hover:bg-blue-500 text-white rounded-xl text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-md"
                  >
                    <Play className="w-4 h-4" />
                    <span>Start Training Run</span>
                  </button>
                )}
              </div>

              {/* Real-time Status Card */}
              <div className="lg:col-span-2 bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col justify-between">
                <div>
                  <div className="flex items-center justify-between mb-4">
                    <h3 className="text-base font-semibold text-neutral-100">Live Training Telemetry</h3>
                    <span
                      className={`px-3 py-1 rounded-full text-xs font-mono font-medium ${
                        trainMetrics.is_training
                          ? 'bg-emerald-500/20 text-emerald-400 border border-emerald-500/30'
                          : 'bg-neutral-800 text-neutral-400'
                      }`}
                    >
                      {trainMetrics.is_training ? 'TRAINING RUNNING' : 'IDLE'}
                    </span>
                  </div>

                  <div className="grid grid-cols-3 gap-4 mb-6">
                    <div className="p-4 bg-neutral-950 rounded-xl border border-neutral-800">
                      <div className="text-xs font-mono text-neutral-500">STEP</div>
                      <div className="text-xl font-bold font-mono text-neutral-100 mt-1">
                        {trainMetrics.step} / {trainMetrics.total_steps || trainSteps}
                      </div>
                    </div>

                    <div className="p-4 bg-neutral-950 rounded-xl border border-neutral-800">
                      <div className="text-xs font-mono text-neutral-500">TRAIN LOSS</div>
                      <div className="text-xl font-bold font-mono text-blue-400 mt-1">
                        {trainMetrics.loss ? trainMetrics.loss.toFixed(4) : '—'}
                      </div>
                    </div>

                    <div className="p-4 bg-neutral-950 rounded-xl border border-neutral-800">
                      <div className="text-xs font-mono text-neutral-500">VAL LOSS</div>
                      <div className="text-xl font-bold font-mono text-emerald-400 mt-1">
                        {trainMetrics.val_loss ? trainMetrics.val_loss.toFixed(4) : '—'}
                      </div>
                    </div>
                  </div>

                  {/* Progress Bar */}
                  <div className="w-full bg-neutral-950 rounded-full h-2.5 overflow-hidden border border-neutral-800 mb-6">
                    <div
                      className="bg-blue-500 h-2.5 rounded-full transition-all duration-300"
                      style={{
                        width: `${
                          trainMetrics.total_steps
                            ? Math.min(100, (trainMetrics.step / trainMetrics.total_steps) * 100)
                            : 0
                        }%`,
                      }}
                    ></div>
                  </div>
                </div>

                {/* History list */}
                <div className="border-t border-neutral-800 pt-4">
                  <div className="text-xs font-mono text-neutral-500 mb-2">RECENT LOSS CHECKPOINTS</div>
                  <div className="flex gap-2 overflow-x-auto pb-1 text-xs font-mono">
                    {trainMetrics.history.slice(-5).map((h, i) => (
                      <div key={i} className="px-3 py-1.5 rounded-lg bg-neutral-950 border border-neutral-800 whitespace-nowrap">
                        Step {h.step}: Loss {h.loss} (val: {h.val_loss})
                      </div>
                    ))}
                    {trainMetrics.history.length === 0 && (
                      <span className="text-neutral-600">No training iterations yet.</span>
                    )}
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ================= HARDWARE & ONNX TAB ================= */}
        {activeTab === 'hardware' && (
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            {/* System Hardware Specs */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col gap-4">
              <h3 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
                <HardDrive className="w-5 h-5 text-blue-400" />
                Local Hardware Diagnostics
              </h3>

              <div className="space-y-3 font-mono text-xs">
                <div className="flex justify-between p-3 bg-neutral-950 rounded-xl border border-neutral-800">
                  <span className="text-neutral-400">Operating System</span>
                  <span className="text-neutral-200">{hardware?.os || 'Linux'}</span>
                </div>
                <div className="flex justify-between p-3 bg-neutral-950 rounded-xl border border-neutral-800">
                  <span className="text-neutral-400">CPU Architecture</span>
                  <span className="text-neutral-200">{hardware?.cpu_name || 'x86_64'} ({hardware?.cpu_cores || 2} cores)</span>
                </div>
                <div className="flex justify-between p-3 bg-neutral-950 rounded-xl border border-neutral-800">
                  <span className="text-neutral-400">System Memory (RAM)</span>
                  <span className="text-neutral-200">{hardware?.ram_gb?.toFixed(1) || '4.0'} GB</span>
                </div>
                <div className="flex justify-between p-3 bg-neutral-950 rounded-xl border border-neutral-800">
                  <span className="text-neutral-400">NVIDIA CUDA GPU</span>
                  <span className={hardware?.has_cuda ? 'text-emerald-400 font-bold' : 'text-neutral-500'}>
                    {hardware?.has_cuda ? hardware.cuda_device_name : 'Not detected / CPU mode'}
                  </span>
                </div>
                <div className="flex justify-between p-3 bg-neutral-950 rounded-xl border border-neutral-800">
                  <span className="text-neutral-400">NPU / DirectML Status</span>
                  <span className="text-neutral-300">{hardware?.npu_details || 'ONNX Runtime CPU / DirectML ready'}</span>
                </div>
              </div>

              {/* Hardware Benchmark Button */}
              <div className="mt-4 pt-4 border-t border-neutral-800">
                <button
                  id="run-benchmark-button"
                  onClick={handleRunBenchmark}
                  disabled={isBenchmarking}
                  className="w-full py-2.5 bg-neutral-800 hover:bg-neutral-700 text-neutral-200 rounded-xl text-xs font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${isBenchmarking ? 'animate-spin' : ''}`} />
                  <span>{isBenchmarking ? 'Benchmarking Throughput...' : 'Run Hardware Benchmark'}</span>
                </button>

                {benchResult && (
                  <div className="mt-3 p-3 bg-neutral-950 rounded-xl border border-neutral-800 text-xs font-mono space-y-1">
                    <div className="text-emerald-400">With KV Cache: {benchResult.kv_cached_speed_tok_s} tok/s</div>
                    <div className="text-neutral-400">Without Cache: {benchResult.uncached_speed_tok_s} tok/s</div>
                    <div className="text-blue-400 font-bold">Speedup: {benchResult.kv_cache_speedup}x faster</div>
                  </div>
                )}
              </div>
            </div>

            {/* ONNX Export Card */}
            <div className="bg-neutral-900/50 border border-neutral-800 rounded-2xl p-6 flex flex-col gap-4">
              <h3 className="text-base font-semibold text-neutral-100 flex items-center gap-2">
                <Download className="w-5 h-5 text-emerald-400" />
                ONNX Model Exporter
              </h3>

              <p className="text-xs text-neutral-400 leading-relaxed">
                Export your active NumPy model directly to the standard Open Neural Network Exchange (.onnx) format. This allows local hardware acceleration via Windows DirectML (NPU/AMD/Intel/NVIDIA) or OpenVINO.
              </p>

              <button
                id="export-onnx-button"
                onClick={handleExportOnnx}
                disabled={isExporting}
                className="w-full py-3 bg-emerald-600 hover:bg-emerald-500 disabled:opacity-50 text-white rounded-xl text-sm font-medium flex items-center justify-center gap-2 cursor-pointer transition-colors shadow-md"
              >
                {isExporting ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Download className="w-4 h-4" />}
                <span>{isExporting ? 'Exporting & Verifying...' : 'Export to model.onnx'}</span>
              </button>

              {exportResult && (
                <div className="p-4 rounded-xl bg-emerald-500/10 border border-emerald-500/30 text-xs font-mono space-y-1.5">
                  <div className="flex items-center gap-1.5 text-emerald-400 font-bold">
                    <CheckCircle2 className="w-4 h-4" />
                    <span>ONNX Export Verified</span>
                  </div>
                  <div className="text-neutral-300">File: {exportResult.filename} ({exportResult.size_kb} KB)</div>
                  <div className="text-neutral-400">ONNX Runtime Output: {JSON.stringify(exportResult.output_shape)}</div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}
