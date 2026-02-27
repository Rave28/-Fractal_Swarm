import { useEffect, useState } from "react";
import MemoryLog from "./components/MemoryLog";
import ResearchTerminal from "./components/ResearchTerminal";
import NightcrawlerStatus from "./components/NightcrawlerStatus";
import SecurityAuditFeed from "./components/SecurityAuditFeed";

interface TelemetryData {
  cpu_usage: number;
  memory_percent: number;
  active_nodes: number;
  vector_cache_size: number;
  vibe_status: string;
  uptime: number;
}

function App() {
  const [telemetry, setTelemetry] = useState<TelemetryData | null>(null);

  useEffect(() => {
    const fetchTelemetry = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/telemetry");
        const data: TelemetryData = await res.json();
        setTelemetry(data);
      } catch (err) {
        console.error("Oracle disconnected", err);
      }
    };

    fetchTelemetry();
    const interval = setInterval(fetchTelemetry, 2000);
    return () => clearInterval(interval);
  }, []);

  const formatUptime = (seconds: number) => {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = Math.floor(seconds % 60);
    return `${h}h ${m}m ${s}s`;
  };

  return (
    <div className="p-6 md:p-8 max-w-7xl mx-auto">
      <header className="mb-10 text-center md:text-left">
        <h1 className="text-4xl font-extrabold tracking-tight text-transparent bg-clip-text bg-gradient-to-r from-blue-400 to-emerald-400">
          Sovereign Intelligence Dashboard
        </h1>
        <p className="text-slate-400 mt-2 text-lg">
          Fractal Swarm V3 &middot;{" "}
          <span className="font-mono text-sm">
            {telemetry
              ? `Uptime: ${formatUptime(telemetry.uptime)}`
              : "Connecting..."}
          </span>
        </p>
      </header>

      {/* === Telemetry Cards === */}
      <div className="grid grid-cols-2 lg:grid-cols-4 gap-4 md:gap-6">
        <div className="glass p-5 rounded-2xl">
          <h2 className="text-xs uppercase tracking-wider text-slate-400 mb-1.5">
            CPU Usage
          </h2>
          <div className="text-3xl font-bold font-mono">
            {telemetry ? `${telemetry.cpu_usage}%` : "---"}
          </div>
        </div>

        <div className="glass p-5 rounded-2xl">
          <h2 className="text-xs uppercase tracking-wider text-slate-400 mb-1.5">
            RAM Load
          </h2>
          <div className="text-3xl font-bold font-mono">
            {telemetry ? `${telemetry.memory_percent}%` : "---"}
          </div>
        </div>

        <div className="glass p-5 rounded-2xl">
          <h2 className="text-xs uppercase tracking-wider text-slate-400 mb-1.5">
            Vector Cache
          </h2>
          <div className="text-3xl font-bold font-mono text-emerald-400">
            {telemetry ? telemetry.vector_cache_size.toLocaleString() : "---"}
          </div>
        </div>

        <div className="glass p-5 rounded-2xl">
          <h2 className="text-xs uppercase tracking-wider text-slate-400 mb-1.5">
            Vibe Status
          </h2>
          <div className="text-base font-bold text-blue-400 truncate">
            {telemetry ? telemetry.vibe_status : "SYNCING..."}
          </div>
        </div>
      </div>

      {/* === Two-Column Layout: Memory + Research === */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mt-2">
        <MemoryLog />
        <ResearchTerminal />
      </div>

      {/* === Security Audit Feed (full-width) === */}
      <SecurityAuditFeed />

      {/* === Nightcrawler === */}
      <NightcrawlerStatus />
    </div>
  );
}

export default App;
