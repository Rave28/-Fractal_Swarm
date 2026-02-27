import { useEffect, useState } from "react";

interface NightcrawlerData {
  status: string;
  last_run: string | null;
  targets_scanned: number;
  vectors_ingested: number;
}

export default function NightcrawlerStatus() {
  const [data, setData] = useState<NightcrawlerData | null>(null);

  useEffect(() => {
    const fetchStatus = async () => {
      try {
        const res = await fetch(
          "http://127.0.0.1:8000/api/nightcrawler/status",
        );
        const json: NightcrawlerData = await res.json();
        setData(json);
      } catch {
        setData({
          status: "offline",
          last_run: null,
          targets_scanned: 0,
          vectors_ingested: 0,
        });
      }
    };
    fetchStatus();
    const interval = setInterval(fetchStatus, 10000);
    return () => clearInterval(interval);
  }, []);

  const statusColor: Record<string, string> = {
    active: "bg-emerald-400",
    dormant: "bg-yellow-400",
    offline: "bg-red-400",
    error: "bg-red-400",
  };

  const dotColor = statusColor[data?.status ?? "offline"] ?? "bg-slate-400";

  return (
    <div className="glass rounded-2xl p-6 mt-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-sm uppercase tracking-wider text-slate-400">
          Nightcrawler Daemon
        </h2>
        <div className="flex items-center gap-2">
          <span
            className={`inline-block w-2.5 h-2.5 rounded-full ${dotColor} animate-pulse`}
          />
          <span className="text-xs font-mono uppercase text-slate-300">
            {data?.status ?? "..."}
          </span>
        </div>
      </div>

      <div className="grid grid-cols-3 gap-4">
        <div>
          <p className="text-xs text-slate-500 mb-1">Last Run</p>
          <p className="text-sm font-mono text-slate-300">
            {data?.last_run
              ? new Date(data.last_run).toLocaleTimeString()
              : "Never"}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Targets Scanned</p>
          <p className="text-lg font-bold font-mono text-blue-400">
            {data?.targets_scanned ?? 0}
          </p>
        </div>
        <div>
          <p className="text-xs text-slate-500 mb-1">Vectors Ingested</p>
          <p className="text-lg font-bold font-mono text-emerald-400">
            {data?.vectors_ingested ?? 0}
          </p>
        </div>
      </div>
    </div>
  );
}
