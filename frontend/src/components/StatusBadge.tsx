import { CircleCheck, CircleHelp, OctagonX, TriangleAlert, type LucideIcon } from "lucide-react";
import { SEVERITY_COLOR } from "@/lib/format";
import type { Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

export const SEVERITY_ICON: Record<Severity, LucideIcon> = {
  Normal: CircleCheck,
  Degraded: TriangleAlert,
  Critical: OctagonX,
  Unknown: CircleHelp,
};

/** Status is never color alone: every use carries a distinct icon and the word. */
export function StatusBadge({
  severity,
  size = "sm",
  className,
}: {
  severity: Severity;
  size?: "sm" | "lg";
  className?: string;
}) {
  const Icon = SEVERITY_ICON[severity];
  return (
    <span
      className={cn(
        "inline-flex items-center gap-1.5 font-medium",
        size === "lg" ? "text-xl" : "text-sm",
        className,
      )}
      style={{ color: SEVERITY_COLOR[severity] }}
    >
      <Icon aria-hidden className={size === "lg" ? "size-6" : "size-4"} strokeWidth={2.25} />
      {severity}
    </span>
  );
}
