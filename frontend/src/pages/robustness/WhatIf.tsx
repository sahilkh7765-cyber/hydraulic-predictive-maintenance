import { ArrowRight, Check, RotateCcw, Shuffle } from "lucide-react";
import { useEffect, useState, type FormEvent } from "react";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { Button } from "@/components/ui/button";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { Slider } from "@/components/ui/slider";
import { SEVERITY_COLOR, fmt, valueWithUnit } from "@/lib/format";
import { useWhatIf, useWhatIfOptions } from "@/lib/queries";
import type { ClassProbability, ComponentKey, WhatIfOptions } from "@/lib/types";

export function WhatIf({ component, unit, nCycles }: { component: ComponentKey; unit: string; nCycles: number }) {
  const [cycle, setCycle] = useState<number | undefined>(undefined);
  const options = useWhatIfOptions(component, cycle);
  return (
    <Panel
      title="What-if probe"
      description="Change one input of one recorded cycle and watch the model's probabilities. Real features move together, so large single-feature moves can describe cycles that never occur physically: treat this as a sensitivity check, not a simulation."
    >
      <QueryState query={options} loadingLabel="Loading the probe">
        {(o) => <Probe key={`${component}-${o.cycle}`} o={o} unit={unit} nCycles={nCycles} onCycle={setCycle} />}
      </QueryState>
    </Panel>
  );
}

function Probe({
  o,
  unit,
  nCycles,
  onCycle,
}: {
  o: WhatIfOptions;
  unit: string;
  nCycles: number;
  onCycle: (c: number) => void;
}) {
  const [feature, setFeature] = useState(o.features[0]?.name ?? "");
  const f = o.features.find((x) => x.name === feature) ?? o.features[0];
  const lo = f ? Math.min(f.p01, f.current) : 0;
  const hi = f ? Math.max(f.p99, f.current) : 1;
  const [value, setValue] = useState(f?.current ?? 0);
  const [committed, setCommitted] = useState(f?.current ?? 0);
  const [draftCycle, setDraftCycle] = useState(String(o.cycle));

  useEffect(() => {
    if (f) {
      setValue(f.current);
      setCommitted(f.current);
    }
  }, [f]);

  const result = useWhatIf(o.component, f ? { cycle: o.cycle, feature: f.name, value: committed } : null);
  const name = (v: number) => valueWithUnit(v, unit);
  const n = Number(draftCycle);
  const cycleValid = draftCycle.trim() !== "" && Number.isInteger(n) && n >= 0 && n < nCycles;
  const submitCycle = (e: FormEvent) => {
    e.preventDefault();
    if (cycleValid) onCycle(n);
  };

  if (!f) return <p className="text-sm text-ink-2">No probe-able features for this model.</p>;

  return (
    <div className="space-y-5">
      <div className="grid gap-4 md:grid-cols-[auto_minmax(0,1fr)]">
        <form onSubmit={submitCycle} className="flex flex-wrap items-end gap-2" noValidate>
          <div>
            <label htmlFor="whatif-cycle" className="nameplate mb-1.5 block">
              Cycle (0 to {nCycles - 1})
            </label>
            <input
              id="whatif-cycle"
              type="number"
              inputMode="numeric"
              value={draftCycle}
              min={0}
              max={nCycles - 1}
              onChange={(e) => setDraftCycle(e.target.value)}
              aria-invalid={!cycleValid}
              className="num h-9 w-28 rounded-sm border border-hairline-strong bg-ground px-3 aria-[invalid=true]:border-crit"
            />
          </div>
          <Button type="submit" variant="outline" disabled={!cycleValid || n === o.cycle}>
            Load cycle
          </Button>
          <Button
            type="button"
            variant="ghost"
            size="icon"
            aria-label="Load a random cycle"
            onClick={() => {
              const r = Math.floor(Math.random() * nCycles);
              setDraftCycle(String(r));
              onCycle(r);
            }}
          >
            <Shuffle aria-hidden />
          </Button>
        </form>
        <div className="min-w-0">
          <label htmlFor="whatif-feature" className="nameplate mb-1.5 block">
            Feature to vary (the model's ten most important inputs)
          </label>
          <Select value={f.name} onValueChange={setFeature}>
            <SelectTrigger id="whatif-feature" className="w-full max-w-md bg-ground font-mono text-xs">
              <SelectValue />
            </SelectTrigger>
            <SelectContent>
              {o.features.map((x) => (
                <SelectItem key={x.name} value={x.name} className="font-mono text-xs">
                  {x.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
        </div>
      </div>

      <div className="max-w-2xl">
        <div className="mb-2 flex flex-wrap items-baseline justify-between gap-2">
          <label htmlFor="whatif-value" className="nameplate">
            Value of <span className="font-mono text-ink">{f.name}</span>
          </label>
          <span className="num text-sm">
            {fmt.sig(f.current)} <ArrowRight aria-hidden className="inline size-3.5 text-muted-ink" />{" "}
            <strong className="text-ink">{fmt.sig(value)}</strong>
          </span>
        </div>
        <Slider
          id="whatif-value"
          min={lo}
          max={hi}
          step={(hi - lo) / 200 || 1}
          value={[value]}
          onValueChange={(v) => setValue(v[0] ?? f.current)}
          onValueCommit={(v) => setCommitted(v[0] ?? f.current)}
          aria-label={`Value of ${f.name}`}
        />
        <div className="num mt-1 flex justify-between text-xs text-muted-ink">
          <span>{fmt.sig(lo)}</span>
          <span>training 1st to 99th percentile</span>
          <span>{fmt.sig(hi)}</span>
        </div>
        <Button
          type="button"
          variant="ghost"
          size="sm"
          className="mt-2"
          disabled={value === f.current}
          onClick={() => {
            setValue(f.current);
            setCommitted(f.current);
          }}
        >
          <RotateCcw aria-hidden />
          Reset to the recorded value
        </Button>
      </div>

      <QueryState query={result} loadingLabel="Re-scoring">
        {(r) => (
          <div>
            <p className="mb-3 text-sm" aria-live="polite">
              {r.changed ? (
                <span className="inline-flex items-center gap-1.5 font-medium text-warn">
                  <ArrowRight aria-hidden className="size-4" />
                  Decision changed: {name(r.original_prediction)} to {name(r.modified_prediction)}
                </span>
              ) : (
                <span className="inline-flex items-center gap-1.5 font-medium text-ink">
                  <Check aria-hidden className="size-4 text-ok" />
                  Decision unchanged: {name(r.modified_prediction)}
                </span>
              )}
              <span className="text-muted-ink"> (recorded condition {name(r.true_value)})</span>
            </p>
            <ProbabilityPairs original={r.original} modified={r.modified} name={name} />
          </div>
        )}
      </QueryState>
    </div>
  );
}

function ProbabilityPairs({
  original,
  modified,
  name,
}: {
  original: readonly ClassProbability[];
  modified: readonly ClassProbability[];
  name: (v: number) => string;
}) {
  return (
    <table className="w-full max-w-2xl text-sm">
      <caption className="sr-only">Class probabilities before and after the change</caption>
      <thead>
        <tr className="text-left">
          <th scope="col" className="nameplate pb-1 font-medium">
            Condition
          </th>
          <th scope="col" className="nameplate w-2/5 pb-1 font-medium">
            Recorded value / what-if
          </th>
          <th scope="col" className="nameplate pb-1 text-right font-medium">
            Change
          </th>
        </tr>
      </thead>
      <tbody>
        {original.map((o, i) => {
          const m = modified[i];
          if (!m) return null;
          return (
            <tr key={o.value} className="border-t border-hairline">
              <td className="py-1.5 pr-3">
                <span className="num">{name(o.value)}</span> <span className="text-muted-ink">{o.label}</span>
              </td>
              <td className="py-1.5">
                <div className="space-y-1" aria-hidden>
                  <div className="h-1.5 rounded-full bg-raised">
                    <div className="h-full rounded-full bg-muted-ink" style={{ width: `${o.p * 100}%` }} />
                  </div>
                  <div className="h-1.5 rounded-full bg-raised">
                    <div className="h-full rounded-full" style={{ width: `${m.p * 100}%`, background: SEVERITY_COLOR[m.severity] }} />
                  </div>
                </div>
              </td>
              <td className="num py-1.5 pl-3 text-right whitespace-nowrap">
                {fmt.pct(o.p)} → {fmt.pct(m.p)}
              </td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}
