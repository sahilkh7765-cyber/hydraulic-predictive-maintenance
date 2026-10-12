import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

export interface Column<T> {
  key: string;
  header: ReactNode;
  cell: (row: T) => ReactNode;
  align?: "left" | "right";
  className?: string;
}

/** Semantic table that scrolls inside its own box on narrow screens. */
export function DataTable<T>({
  caption,
  columns,
  rows,
  rowKey,
  rowClassName,
  captionHidden = true,
  maxHeight,
}: {
  caption: string;
  columns: readonly Column<T>[];
  rows: readonly T[];
  rowKey: (row: T) => string;
  rowClassName?: (row: T) => string | undefined;
  captionHidden?: boolean;
  maxHeight?: number;
}) {
  return (
    <div
      className="overflow-auto rounded-sm border border-hairline"
      style={maxHeight ? { maxHeight } : undefined}
      tabIndex={0}
      role="region"
      aria-label={caption}
    >
      <table className="w-full border-collapse text-sm">
        <caption className={cn("px-3 py-2 text-left text-xs text-muted-ink", captionHidden && "sr-only")}>{caption}</caption>
        <thead className="sticky top-0 bg-raised">
          <tr>
            {columns.map((c) => (
              <th
                key={c.key}
                scope="col"
                className={cn(
                  "nameplate px-3 py-2 font-medium whitespace-nowrap",
                  c.align === "right" ? "text-right" : "text-left",
                  c.className,
                )}
              >
                {c.header}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row) => (
            <tr key={rowKey(row)} className={cn("border-t border-hairline", rowClassName?.(row))}>
              {columns.map((c) => (
                <td
                  key={c.key}
                  className={cn(
                    "px-3 py-2 align-top",
                    c.align === "right" ? "num text-right whitespace-nowrap" : "text-left",
                    c.className,
                  )}
                >
                  {c.cell(row)}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
