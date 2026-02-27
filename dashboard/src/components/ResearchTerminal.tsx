import { useState } from "react";

interface SearchResult {
  id: string;
  collection: string;
  content: string;
  distance: number;
}

interface SearchResponse {
  results: SearchResult[];
  query: string;
  error?: string;
}

export default function ResearchTerminal() {
  const [query, setQuery] = useState("");
  const [data, setData] = useState<SearchResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const executeSearch = async () => {
    if (query.trim().length < 2) return;
    setLoading(true);
    try {
      const res = await fetch(
        `http://127.0.0.1:8000/api/research?q=${encodeURIComponent(query)}&limit=5`,
      );
      const json: SearchResponse = await res.json();
      setData(json);
    } catch {
      setData({ results: [], query, error: "Backend offline" });
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter") executeSearch();
  };

  const getRelevanceBadge = (distance: number) => {
    if (distance < 0.5) return { label: "HIGH", color: "text-emerald-400" };
    if (distance < 1.0) return { label: "MED", color: "text-yellow-400" };
    return { label: "LOW", color: "text-red-400" };
  };

  return (
    <div className="glass rounded-2xl p-6 mt-6">
      <h2 className="text-sm uppercase tracking-wider text-slate-400 mb-4">
        Research Terminal
      </h2>
      <div className="flex gap-3 mb-4">
        <input
          type="text"
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Query the Oracle..."
          className="flex-1 px-4 py-2.5 rounded-xl border border-white/10 bg-white/5 text-white placeholder-slate-500 font-mono text-sm outline-none focus:border-blue-400/60 transition-colors"
        />
        <button
          onClick={executeSearch}
          disabled={loading || query.trim().length < 2}
          className="px-5 py-2.5 rounded-xl font-bold text-sm transition-all disabled:opacity-40 disabled:cursor-not-allowed"
          style={{
            background: "linear-gradient(135deg, #3b82f6, #10b981)",
          }}
        >
          {loading ? "..." : "SEARCH"}
        </button>
      </div>

      {data?.error && (
        <p className="text-red-400 text-sm font-mono">{data.error}</p>
      )}

      <div className="space-y-3 max-h-[300px] overflow-y-auto pr-2 custom-scrollbar">
        {data?.results.length === 0 && data && !data.error && (
          <p className="text-slate-500 font-mono text-sm">
            No matching vectors found.
          </p>
        )}
        {data?.results.map((result, i) => {
          const badge = getRelevanceBadge(result.distance);
          return (
            <div
              key={`${result.id}-${i}`}
              className="rounded-xl p-4 border border-white/10 hover:border-emerald-400/40 transition-colors"
              style={{ background: "rgba(255,255,255,0.03)" }}
            >
              <div className="flex items-center justify-between mb-2">
                <span className="text-xs font-mono text-blue-400 truncate max-w-[50%]">
                  {result.id}
                </span>
                <div className="flex items-center gap-2">
                  <span className={`text-xs font-bold ${badge.color}`}>
                    {badge.label}
                  </span>
                  <span className="text-xs text-slate-500 bg-white/5 px-2 py-0.5 rounded-full">
                    {result.distance.toFixed(3)}
                  </span>
                </div>
              </div>
              <p className="text-sm text-slate-300 leading-relaxed">
                {result.content}
              </p>
            </div>
          );
        })}
      </div>
    </div>
  );
}
