import { CircleCheck, Circle, LoaderCircle, PlugZap, ServerCrash } from "lucide-react";
import type { ReactNode } from "react";
import { useHealth } from "@/lib/queries";
import { Loading } from "../QueryState";

/**
 * Holds the pages back until the API has loaded its models. While warming,
 * it shows the API's own progress list instead of letting requests fail.
 */
export function WarmupGate({ children }: { children: ReactNode }) {
  const health = useHealth();

  if (health.isPending) return <Loading label="Connecting to the API" className="px-1" />;

  if (health.isError && !health.data) {
    return (
      <GateCard icon={<PlugZap aria-hidden className="size-6 text-warn" />} title="The API isn't reachable">
        <p>
          Start it from the project root, in the virtual environment:
        </p>
        <pre className="mt-3 overflow-x-auto rounded-sm border border-hairline bg-ground px-3 py-2 font-mono text-sm text-ink">
          uvicorn api.main:app --port 8000
        </pre>
        <p className="mt-3 text-sm text-muted-ink">This page reconnects on its own every few seconds.</p>
      </GateCard>
    );
  }

  const data = health.data;
  if (!data) return null;

  if (data.status === "warming") {
    const done = data.steps.filter((s) => s.done).length;
    return (
      <GateCard
        icon={<LoaderCircle aria-hidden className="size-6 animate-spin text-signal motion-reduce:animate-none" />}
        title="Loading models"
      >
        <p>
          The API is loading the four saved pipelines and scoring the held-out test cycles. This takes a few seconds
          and happens once per server start.
        </p>
        <ol className="mt-4 space-y-2" aria-live="polite">
          {data.steps.map((step) => (
            <li key={step.name} className="flex items-center gap-2 text-sm">
              {step.done ? (
                <CircleCheck aria-hidden className="size-4 text-ok" />
              ) : (
                <Circle aria-hidden className="size-4 text-muted-ink" />
              )}
              <span className={step.done ? "text-ink" : "text-muted-ink"}>{step.name}</span>
              <span className="sr-only">{step.done ? "(done)" : "(pending)"}</span>
            </li>
          ))}
        </ol>
        <p className="num mt-4 text-xs text-muted-ink">
          {done} of {data.steps.length} steps done
        </p>
      </GateCard>
    );
  }

  if (data.status === "unavailable") {
    return (
      <GateCard icon={<ServerCrash aria-hidden className="size-6 text-crit" />} title="The models aren't available">
        <p>{data.message ?? "The API could not load the trained models."}</p>
        <p className="mt-3 text-sm text-muted-ink">Train and evaluate once, then restart the API:</p>
        <pre className="mt-2 overflow-x-auto rounded-sm border border-hairline bg-ground px-3 py-2 font-mono text-sm text-ink">
          python src/train.py{"\n"}python src/evaluate.py
        </pre>
      </GateCard>
    );
  }

  return <>{children}</>;
}

function GateCard({ icon, title, children }: { icon: ReactNode; title: string; children: ReactNode }) {
  return (
    <div className="mx-auto mt-6 max-w-xl rounded-md border border-hairline bg-plate p-6" role="status">
      <div className="flex items-center gap-3">
        {icon}
        <h1 className="text-xl font-semibold">{title}</h1>
      </div>
      <div className="mt-3 text-ink-2">{children}</div>
    </div>
  );
}
