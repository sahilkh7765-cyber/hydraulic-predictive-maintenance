import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface BarListRow {
  key: string;
  label: ReactNode;
  value: number;
  display: string;
  color?: string;
  title?: string;
}

/**
 * Ranked horizontal bars in plain HTML: labels wrap instead of truncating,
 * which matters for long feature names on a phone. Values sit in a muted
 * right-hand column, so the list doubles as its own table.
 */
export function BarList({
  rows,
  max,
  className,
  labelClassName,
  ariaLabel,
}: {
  rows: readonly BarListRow[];
  max?: number;
  className?: string;
  labelClassName?: string;
  ariaLabel: string;
}) {
  const top = max ?? Math.max(...rows.map((r) => r.value), 0);
  return (
    <ol
      aria-label={ariaLabel}
      className={cn("space-y-1.5 sm:grid sm:grid-cols-[minmax(110px,auto)_minmax(80px,1fr)_auto] sm:gap-x-3 sm:gap-y-1.5 sm:space-y-0", className)}
    >
      {rows.map((r) => (
        <li
          key={r.key}
          title={r.title}
          className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm sm:col-span-3 sm:grid-cols-subgrid"
        >
          <span className={cn("min-w-0 break-words", labelClassName)}>{r.label}</span>
          <span aria-hidden className="order-last col-span-2 h-2 rounded-full bg-raised sm:order-none sm:col-span-1">
            <span
              className="block h-full rounded-full"
              style={{ width: `${top > 0 ? Math.max(0, (r.value / top) * 100) : 0}%`, background: r.color ?? "var(--series-1)" }}
            />
          </span>
          <span className="num text-right text-ink-2">{r.display}</span>
        </li>
      ))}
    </ol>
  );
}
