import { cn } from "@/lib/utils";

export interface DotSeries {
  key: string;
  label: string;
  color: string;
  hollow?: boolean;
}

export interface DotRow {
  key: string;
  label: string;
  values: Record<string, number | null>;
}

/**
 * One row per category, one dot per series on a shared horizontal axis -
 * a dot plot for comparing a few close scores. The axis may start above zero;
 * the caller states that in the caption.
 */
export function DotRows({
  rows,
  series,
  domain,
  step,
  format,
  connect = true,
}: {
  rows: readonly DotRow[];
  series: readonly DotSeries[];
  domain: [number, number];
  step: number;
  format: (v: number) => string;
  connect?: boolean;
}) {
  const [lo, hi] = domain;
  const x = (v: number) => `${((v - lo) / (hi - lo)) * 100}%`;
  const ticks = Array.from({ length: Math.round((hi - lo) / step) + 1 }, (_, i) => lo + i * step);

  return (
    <div aria-hidden>
      {rows.map((r) => {
        const vals = series.map((s) => r.values[s.key]).filter((v): v is number => v !== null && v !== undefined);
        const min = Math.min(...vals);
        const max = Math.max(...vals);
        return (
          <div key={r.key} className="grid grid-cols-[minmax(84px,150px)_1fr] items-center gap-3">
            <span className="truncate text-sm">{r.label}</span>
            <div className="relative h-8">
              {ticks.map((t) => (
                <span key={t} className="absolute top-0 bottom-0 w-px bg-hairline" style={{ left: x(t) }} />
              ))}
              {connect && vals.length > 1 ? (
                <span
                  className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-hairline-strong"
                  style={{ left: x(min), width: `calc(${x(max)} - ${x(min)})` }}
                />
              ) : null}
              {series.map((s) => {
                const v = r.values[s.key];
                if (v === null || v === undefined) return null;
                return (
                  <span
                    key={s.key}
                    title={`${s.label}: ${format(v)}`}
                    className={cn(
                      "absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-plate",
                      s.hollow && "border-2 bg-plate",
                    )}
                    style={{
                      left: x(Math.min(hi, Math.max(lo, v))),
                      ...(s.hollow ? { borderColor: s.color } : { background: s.color }),
                    }}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
      <div className="grid grid-cols-[minmax(84px,150px)_1fr] gap-3">
        <span />
        <div className="relative h-5">
          {ticks.map((t, i) => (
            <span
              key={t}
              className={cn(
                "num absolute top-1 text-xs text-muted-ink",
                i === ticks.length - 1 ? "-translate-x-full" : i === 0 ? "" : "-translate-x-1/2",
                i % 2 === 1 && "hidden sm:inline",
              )}
              style={{ left: x(t) }}
            >
              {format(t)}
            </span>
          ))}
        </div>
      </div>
    </div>
  );
}

/** Lower bound for a score axis: the nearest `step` below the smallest value. */
export const floorTo = (values: readonly number[], step: number) =>
  Math.max(0, Math.floor(Math.min(...values) / step) * step);
