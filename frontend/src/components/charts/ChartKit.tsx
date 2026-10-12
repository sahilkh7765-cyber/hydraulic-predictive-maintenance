/**
 * Shared chart chrome: one recessive axis/grid style, one tooltip, and a
 * figure wrapper that always offers a table view of the same data.
 */
import { Table2, ChartLine } from "lucide-react";
import { useState, type ReactNode } from "react";
import { DataTable, type Column } from "../DataTable";
import { cn } from "@/lib/utils";

export const AXIS = {
  stroke: "var(--hairline-strong)",
  tick: { fill: "var(--muted-ink)", fontSize: 12, fontFamily: "var(--font-sans)" },
  tickLine: false,
} as const;

export const GRID = { stroke: "var(--hairline)", strokeDasharray: undefined, vertical: false } as const;

export const axisLabel = (value: string, axis: "x" | "y") =>
  axis === "x"
    ? { value, position: "insideBottom" as const, offset: -4, fill: "var(--muted-ink)", fontSize: 12, fontFamily: "var(--font-plate)" }
    : {
        value,
        angle: -90,
        position: "insideLeft" as const,
        style: { textAnchor: "middle" as const },
        fill: "var(--muted-ink)",
        fontSize: 12,
        fontFamily: "var(--font-plate)",
      };

export interface TooltipRow {
  label: ReactNode;
  value: ReactNode;
  color?: string;
}

export function TooltipBox({ title, rows }: { title?: ReactNode; rows: readonly TooltipRow[] }) {
  return (
    <div className="pointer-events-none min-w-36 rounded-sm border border-hairline-strong bg-raised px-3 py-2 text-xs shadow-lg">
      {title !== undefined ? <div className="mb-1 font-medium text-ink">{title}</div> : null}
      <ul className="space-y-0.5">
        {rows.map((r, i) => (
          <li key={i} className="flex items-center justify-between gap-4">
            <span className="flex items-center gap-1.5 text-ink-2">
              {r.color ? <span aria-hidden className="size-2 rounded-full" style={{ background: r.color }} /> : null}
              {r.label}
            </span>
            <span className="num text-ink">{r.value}</span>
          </li>
        ))}
      </ul>
    </div>
  );
}

export interface LegendItem {
  label: string;
  color: string;
  shape?: "dot" | "line" | "ring" | "square";
}

export function Legend({ items, className }: { items: readonly LegendItem[]; className?: string }) {
  return (
    <ul className={cn("flex flex-wrap gap-x-4 gap-y-1 text-sm text-ink-2", className)}>
      {items.map((it) => (
        <li key={it.label} className="inline-flex items-center gap-1.5">
          <LegendSwatch color={it.color} shape={it.shape ?? "dot"} />
          {it.label}
        </li>
      ))}
    </ul>
  );
}

function LegendSwatch({ color, shape }: { color: string; shape: NonNullable<LegendItem["shape"]> }) {
  if (shape === "line") return <span aria-hidden className="h-0.5 w-4 rounded" style={{ background: color }} />;
  if (shape === "ring") return <span aria-hidden className="size-3 rounded-full border-2" style={{ borderColor: color }} />;
  if (shape === "square") return <span aria-hidden className="size-3 rounded-[2px]" style={{ background: color }} />;
  return <span aria-hidden className="size-2.5 rounded-full" style={{ background: color }} />;
}

/**
 * A chart with a caption and a "Table" toggle that swaps in the same data as
 * an accessible table (the chart's WCAG-clean twin).
 */
export function ChartFigure<T>({
  caption,
  legend,
  table,
  height,
  children,
  className,
}: {
  caption: ReactNode;
  legend?: ReactNode;
  table: { caption: string; columns: readonly Column<T>[]; rows: readonly T[]; rowKey: (row: T) => string };
  height: number;
  children: ReactNode;
  className?: string;
}) {
  const [asTable, setAsTable] = useState(false);
  return (
    <figure className={cn("min-w-0", className)}>
      <div className="mb-2 flex flex-wrap items-center justify-between gap-2">
        <div className="min-w-0">{legend}</div>
        <button
          type="button"
          aria-pressed={asTable}
          onClick={() => setAsTable((v) => !v)}
          className="inline-flex items-center gap-1.5 rounded-sm border border-hairline-strong px-2 py-1 text-xs text-ink-2 hover:text-ink"
        >
          {asTable ? <ChartLine aria-hidden className="size-3.5" /> : <Table2 aria-hidden className="size-3.5" />}
          {asTable ? "Chart" : "Table"}
        </button>
      </div>
      {asTable ? (
        <DataTable caption={table.caption} columns={table.columns} rows={table.rows} rowKey={table.rowKey} maxHeight={height + 40} />
      ) : (
        <div style={{ height }} className="w-full min-w-0">
          {children}
        </div>
      )}
      <figcaption className="mt-1.5 text-xs text-muted-ink">{caption}</figcaption>
    </figure>
  );
}
