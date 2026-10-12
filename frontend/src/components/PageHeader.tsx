import type { ReactNode } from "react";

/** Every page opens with its title, the one-sentence takeaway, then the readouts that prove it. */
export function PageHeader({
  title,
  takeaway,
  children,
}: {
  title: string;
  takeaway: ReactNode;
  children?: ReactNode;
}) {
  return (
    <header className="mb-6">
      <h1 className="text-2xl font-semibold tracking-tight text-ink sm:text-3xl">{title}</h1>
      <p className="prose-measure mt-2 text-lg text-ink-2">{takeaway}</p>
      {children ? <div className="mt-5">{children}</div> : null}
    </header>
  );
}

/** A horizontal strip of instrument readouts, divided by hairlines rather than boxed into cards. */
export function ReadoutStrip({ children }: { children: ReactNode }) {
  return (
    <dl className="grid grid-cols-2 overflow-hidden rounded-md border border-hairline bg-plate sm:grid-cols-[repeat(auto-fit,minmax(170px,1fr))] [&>div]:border-hairline [&>div]:border-b [&>div]:border-r">
      {children}
    </dl>
  );
}

export function Readout({
  label,
  value,
  hint,
  tone,
}: {
  label: string;
  value: ReactNode;
  hint?: ReactNode;
  tone?: string;
}) {
  return (
    <div className="-mb-px -mr-px min-w-0 px-4 py-3">
      <dt className="nameplate">{label}</dt>
      <dd className="mt-1 text-xl font-semibold text-ink" style={tone ? { color: tone } : undefined}>
        {value}
      </dd>
      {hint ? <dd className="mt-0.5 text-xs text-muted-ink">{hint}</dd> : null}
    </div>
  );
}
