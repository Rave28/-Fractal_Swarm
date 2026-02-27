import { useEffect, useState, useRef } from "react";

interface AuditEvent {
  event: "SCAN_PASS" | "SCAN_BLOCKED" | "OVERRIDE_AUTHORIZED";
  filename: string;
  threat_class: string | null;
  manifest_id: string | null;
  timestamp: number;
  guard_version: string;
}

interface AuditResponse {
  events: AuditEvent[];
  total_events: number;
  guard_version: string;
  audit_capacity: number;
  error?: string;
}

const EVENT_CONFIG = {
  SCAN_PASS: {
    label: "PASS",
    dot: "bg-emerald-400",
    text: "text-emerald-400",
    border: "border-emerald-500/20",
    bg: "rgba(16,185,129,0.04)",
    pulse: false,
  },
  SCAN_BLOCKED: {
    label: "BLOCK",
    dot: "bg-red-500",
    text: "text-red-400",
    border: "border-red-500/30",
    bg: "rgba(239,68,68,0.06)",
    pulse: true,
  },
  OVERRIDE_AUTHORIZED: {
    label: "OVERRIDE",
    dot: "bg-amber-400",
    text: "text-amber-400",
    border: "border-amber-500/30",
    bg: "rgba(245,158,11,0.06)",
    pulse: false,
  },
} as const;

function formatTs(ts: number): string {
  return new Date(ts * 1000).toLocaleTimeString("en-US", {
    hour12: false,
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

function shortFile(filename: string): string {
  const parts = filename.replace(/\\/g, "/").split("/");
  return parts[parts.length - 1] || filename;
}

export default function SecurityAuditFeed() {
  const [data, setData] = useState<AuditResponse | null>(null);
  const [filter, setFilter] = useState<string>("ALL");
  const [newIds, setNewIds] = useState<Set<string>>(new Set());
  const prevTimestamps = useRef<Set<number>>(new Set());
  const listRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const fetchAudit = async () => {
      try {
        const res = await fetch(
          "http://127.0.0.1:8000/api/audit/stream?limit=50",
        );
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const json: AuditResponse = await res.json();
        setData(json);

        // Track newly arrived events for flash animation
        const fresh = new Set<string>();
        json.events.forEach((ev) => {
          const key = `${ev.filename}:${ev.timestamp}`;
          if (!prevTimestamps.current.has(ev.timestamp)) {
            fresh.add(key);
          }
        });
        if (fresh.size > 0) {
          setNewIds(fresh);
          prevTimestamps.current = new Set(json.events.map((e) => e.timestamp));
          setTimeout(() => setNewIds(new Set()), 1200);
        }
      } catch {
        setData(
          (prev) =>
            prev ?? {
              events: [],
              total_events: 0,
              guard_version: "offline",
              audit_capacity: 1000,
              error: "Backend offline",
            },
        );
      }
    };

    fetchAudit();
    const interval = setInterval(fetchAudit, 1500);
    return () => clearInterval(interval);
  }, []);

  const filtered = (data?.events ?? []).filter(
    (ev) => filter === "ALL" || ev.event === filter,
  );

  const counts = {
    PASS: data?.events.filter((e) => e.event === "SCAN_PASS").length ?? 0,
    BLOCK: data?.events.filter((e) => e.event === "SCAN_BLOCKED").length ?? 0,
    OVERRIDE:
      data?.events.filter((e) => e.event === "OVERRIDE_AUTHORIZED").length ?? 0,
  };

  const filterBtns = [
    {
      key: "ALL",
      label: "All",
      count: data?.events.length ?? 0,
      color: "text-slate-300",
    },
    {
      key: "SCAN_BLOCKED",
      label: "Blocked",
      count: counts.BLOCK,
      color: "text-red-400",
    },
    {
      key: "OVERRIDE_AUTHORIZED",
      label: "Override",
      count: counts.OVERRIDE,
      color: "text-amber-400",
    },
    {
      key: "SCAN_PASS",
      label: "Pass",
      count: counts.PASS,
      color: "text-emerald-400",
    },
  ];

  return (
    <div className="glass rounded-2xl p-6 mt-6 audit-feed-container">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center gap-3">
          <div className="relative">
            <div className="w-2.5 h-2.5 rounded-full bg-red-500" />
            {counts.BLOCK > 0 && (
              <div className="absolute inset-0 w-2.5 h-2.5 rounded-full bg-red-500 animate-ping opacity-60" />
            )}
          </div>
          <h2 className="text-sm uppercase tracking-wider text-slate-400">
            AST Guard Audit Feed
            <span className="ml-2 text-xs font-mono text-slate-600">
              {data?.guard_version ?? "—"}
            </span>
          </h2>
        </div>
        <div className="flex items-center gap-2">
          <span className="text-xs text-slate-600 font-mono">
            {data?.total_events ?? 0} / {data?.audit_capacity ?? 1000}
          </span>
          {/* Capacity bar */}
          <div className="w-16 h-1 bg-white/5 rounded-full overflow-hidden">
            <div
              className="capacity-bar-fill"
              style={{
                width: `${Math.min(100, ((data?.total_events ?? 0) / (data?.audit_capacity ?? 1000)) * 100)}%`,
              }}
            />
          </div>
        </div>
      </div>

      {/* Summary stats row */}
      <div className="grid grid-cols-3 gap-3 mb-4">
        {[
          {
            label: "Blocked",
            val: counts.BLOCK,
            type: "blocked",
            glow: counts.BLOCK > 0,
          },
          {
            label: "Overrides",
            val: counts.OVERRIDE,
            type: "override",
            glow: false,
          },
          { label: "Clean", val: counts.PASS, type: "pass", glow: false },
        ].map(({ label, val, type, glow }) => (
          <div
            key={label}
            className={`stat-card stat-${type} ${glow && val > 0 ? "glow" : ""}`}
          >
            <div className="text-2xl font-mono font-bold stat-value">{val}</div>
            <div className="text-xs text-slate-500 mt-0.5">{label}</div>
          </div>
        ))}
      </div>

      {/* Filter tabs */}
      <div className="flex gap-2 mb-3 flex-wrap">
        {filterBtns.map(({ key, label, count, color }) => (
          <button
            key={key}
            onClick={() => setFilter(key)}
            className={`px-3 py-1 rounded-lg text-xs font-mono border transition-all ${
              filter === key
                ? "border-blue-400/40 bg-blue-500/10 text-blue-300"
                : "border-white/10 bg-white/5 text-slate-500 hover:text-slate-300 hover:border-white/20"
            }`}
          >
            {label}
            <span className={`ml-1.5 ${color} opacity-80`}>{count}</span>
          </button>
        ))}
      </div>

      {/* Event list */}
      <div
        ref={listRef}
        className="space-y-1.5 max-h-[280px] overflow-y-auto pr-1 custom-scrollbar"
      >
        {data?.error && (
          <p className="text-red-400 text-sm font-mono">{data.error}</p>
        )}
        {filtered.length === 0 && !data?.error && (
          <div className="flex flex-col items-center justify-center h-[200px] gap-3">
            <div className="w-8 h-8 rounded-full border border-slate-700 flex items-center justify-center">
              <div className="w-2 h-2 rounded-full bg-slate-600 animate-pulse" />
            </div>
            <p className="text-slate-600 font-mono text-xs">
              Awaiting nanobot activity…
            </p>
          </div>
        )}
        {filtered.map((ev) => {
          const cfg = EVENT_CONFIG[ev.event] ?? EVENT_CONFIG.SCAN_PASS;
          const key = `${ev.filename}:${ev.timestamp}`;
          const isNew = newIds.has(key);
          return (
            <div
              key={key}
              className={`rounded-xl px-4 py-2.5 border flex items-start gap-3 transition-all duration-300 audit-event-row status-${ev.event === "SCAN_BLOCKED" ? "blocked" : ev.event === "OVERRIDE_AUTHORIZED" ? "override" : "pass"} ${
                isNew
                  ? "scale-[1.01] opacity-100 border-white/20 shadow-lg"
                  : "opacity-90 border-white/5"
              }`}
            >
              {/* Status dot */}
              <div className="mt-0.5 flex-shrink-0">
                <div
                  className={`w-2 h-2 rounded-full ${cfg.dot} ${cfg.pulse && isNew ? "animate-pulse" : ""}`}
                />
              </div>

              {/* Content */}
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2 flex-wrap">
                  <span className={`text-xs font-bold font-mono ${cfg.text}`}>
                    {cfg.label}
                  </span>
                  <span className="text-xs text-slate-300 font-mono truncate max-w-[180px]">
                    {shortFile(ev.filename)}
                  </span>
                  {ev.threat_class && (
                    <span className="text-xs text-slate-500 bg-white/5 px-1.5 py-px rounded font-mono">
                      {ev.threat_class}
                    </span>
                  )}
                </div>
                {ev.manifest_id && (
                  <div className="text-xs text-slate-600 font-mono mt-0.5 truncate">
                    manifest: {ev.manifest_id}
                  </div>
                )}
              </div>

              {/* Timestamp */}
              <div className="text-xs text-slate-600 font-mono flex-shrink-0 mt-0.5">
                {formatTs(ev.timestamp)}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
