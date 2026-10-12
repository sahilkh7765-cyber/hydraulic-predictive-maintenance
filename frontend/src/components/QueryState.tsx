import type { UseQueryResult } from "@tanstack/react-query";
import { AlertOctagon, Inbox, LoaderCircle, RotateCw } from "lucide-react";
import type { ReactNode } from "react";
import { ApiError, errorMessage } from "@/lib/api";
import { cn } from "@/lib/utils";
import { Button } from "./ui/button";

export function Loading({ label = "Loading", className }: { label?: string; className?: string }) {
  return (
    <div role="status" className={cn("flex items-center gap-2 py-6 text-sm text-muted-ink", className)}>
      <LoaderCircle aria-hidden className="size-4 animate-spin motion-reduce:animate-none" />
      {label}…
    </div>
  );
}

export function ErrorNote({ error, onRetry, className }: { error: unknown; onRetry?: () => void; className?: string }) {
  const code = error instanceof ApiError ? error.code : null;
  return (
    <div
      role="alert"
      className={cn("flex flex-wrap items-start gap-3 rounded-md border border-crit/40 bg-crit/5 px-4 py-3 text-sm", className)}
    >
      <AlertOctagon aria-hidden className="mt-0.5 size-4 shrink-0 text-crit" />
      <div className="min-w-0 flex-1">
        <p className="text-ink">{errorMessage(error)}</p>
        {code ? <p className="mt-0.5 font-mono text-xs text-muted-ink">{code}</p> : null}
      </div>
      {onRetry ? (
        <Button size="sm" variant="outline" onClick={onRetry}>
          <RotateCw aria-hidden />
          Retry
        </Button>
      ) : null}
    </div>
  );
}

export function Empty({ children, className }: { children: ReactNode; className?: string }) {
  return (
    <div className={cn("flex items-start gap-3 rounded-md border border-dashed border-hairline-strong px-4 py-5 text-sm text-ink-2", className)}>
      <Inbox aria-hidden className="mt-0.5 size-4 shrink-0 text-muted-ink" />
      <div>{children}</div>
    </div>
  );
}

/**
 * Renders loading / error / empty / success for one query. While a refetch
 * for new parameters is in flight the previous result stays visible but dimmed,
 * so the layout never jumps.
 */
export function QueryState<T>({
  query,
  children,
  isEmpty,
  empty,
  loadingLabel,
}: {
  query: UseQueryResult<T>;
  children: (data: T) => ReactNode;
  isEmpty?: (data: T) => boolean;
  empty?: ReactNode;
  loadingLabel?: string;
}) {
  if (query.isPending) return <Loading label={loadingLabel} />;
  if (query.isError && query.data === undefined) {
    return <ErrorNote error={query.error} onRetry={() => void query.refetch()} />;
  }
  const data = query.data as T;
  if (isEmpty?.(data)) return <Empty>{empty ?? "Nothing to show."}</Empty>;
  return (
    <div
      aria-busy={query.isFetching}
      className={cn("transition-opacity", query.isPlaceholderData && query.isFetching && "opacity-50")}
    >
      {query.isError ? <ErrorNote className="mb-3" error={query.error} onRetry={() => void query.refetch()} /> : null}
      {children(data)}
    </div>
  );
}
