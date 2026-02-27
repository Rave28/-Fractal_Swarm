import { useEffect, useState } from "react";

interface MemoryEntry {
  id: string;
  collection: string;
  preview: string;
  metadata: Record<string, string>;
}

interface MemoryResponse {
  entries: MemoryEntry[];
  total: number;
  error?: string;
}

export default function MemoryLog() {
  const [data, setData] = useState<MemoryResponse | null>(null);

  useEffect(() => {
    const fetchMemory = async () => {
      try {
        const res = await fetch("http://127.0.0.1:8000/api/memory?limit=15");
        const json: MemoryResponse = await res.json();
        setData(json);
      } catch {
        setData({ entries: [], total: 0, error: "Backend offline" });
      }
    };
    fetchMemory();
    const interval = setInterval(fetchMemory, 5000);
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="glass rounded-2xl p-6 mt-6">
      <h2 className="text-sm uppercase tracking-wider text-slate-400 mb-4">
        MyBrain Memory Log
      </h2>
      {data?.error && (
        <p className="text-red-400 text-sm font-mono">{data.error}</p>
      )}
      <div className="space-y-3 max-h-[350px] overflow-y-auto pr-2 custom-scrollbar">
        {data?.entries.length === 0 && !data?.error && (
          <p className="text-slate-500 font-mono text-sm">
            No vectors ingested yet.
          </p>
        )}
        {data?.entries.map((entry) => (
          <div
            key={entry.id}
            className="rounded-xl p-4 border border-white/10 hover:border-blue-400/40 transition-colors"
            style={{ background: "rgba(255,255,255,0.03)" }}
          >
            <div className="flex items-center justify-between mb-1">
              <span className="text-xs font-mono text-emerald-400 truncate max-w-[60%]">
                {entry.id}
              </span>
              <span className="text-xs text-slate-500 bg-white/5 px-2 py-0.5 rounded-full">
                {entry.collection}
              </span>
            </div>
            <p className="text-sm text-slate-300 leading-relaxed">
              {entry.preview}
            </p>
          </div>
        ))}
      </div>
    </div>
  );
}
