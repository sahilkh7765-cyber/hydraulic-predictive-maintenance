import type { ReactNode } from "react";
import { cn } from "@/lib/utils";

/** A plate on the panel: a nameplate heading, optional controls, then content. */
export function Panel({
  title,
  description,
  aside,
  children,
  className,
  id,
}: {
  title: string;
  description?: ReactNode;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  id?: string;
}) {
  const headingId = id ? `${id}-title` : undefined;
  return (
    <section
      id={id}
      aria-labelledby={headingId}
      className={cn("min-w-0 rounded-md border border-hairline bg-plate p-4 sm:p-5", className)}
    >
      <div className="mb-3 flex flex-wrap items-start justify-between gap-x-4 gap-y-2">
        <div className="min-w-0">
          <h2 id={headingId} className="nameplate text-[15px] text-ink">
            {title}
          </h2>
          {description ? <p className="prose-measure mt-1 text-sm text-muted-ink">{description}</p> : null}
        </div>
        {aside ? <div className="flex flex-wrap items-center gap-2">{aside}</div> : null}
      </div>
      {children}
    </section>
  );
}
