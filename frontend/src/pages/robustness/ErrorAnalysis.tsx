import { useState } from "react";
import { Link } from "react-router";
import { BarList } from "@/components/charts/BarList";
import { ChartFigure, Legend } from "@/components/charts/ChartKit";
import { DotRows } from "@/components/charts/DotRows";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from "@/components/ui/select";
import { SERIES_HEX, SEVERITY_COLOR, fmt, valueWithUnit } from "@/lib/format";
import { useErrorCases, useErrorProfile } from "@/lib/queries";
import type { ComponentKey, ErrorCasesResponse } from "@/lib/types";

export function ErrorAnalysis({ component, unit }: { component: ComponentKey; unit: string }) {
  const cases = useErrorCases(component);
  return (
    <Panel
      title="Why the hard cycles are hard"
      description="For the model's ten most important features: where a cycle sits against the average training cycle of its recorded class and of the class it was confused with."
    >
      <QueryState query={cases} loadingLabel="Finding misclassified cycles">
        {(c) => <ErrorBody key={component} cases={c} unit={unit} />}
      </QueryState>
    </Panel>
  );
}

function ErrorBody({ cases, unit }: { cases: ErrorCasesResponse; unit: string }) {
  const [cycle, setCycle] = useState<number | null>(cases.cases[0]?.cycle ?? null);
  const profile = useErrorProfile(cases.component, cycle);
  const name = (v: number) => valueWithUnit(v, unit);

  if (cases.cases.length === 0) return <p className="text-sm text-ink-2">No test cycles to analyse.</p>;

  return (
    <div className="space-y-4">
      <p className="text-sm text-ink-2">
        {cases.mode === "errors" ? (
          <>
            <strong className="num text-ink">{cases.n_errors}</strong> of {cases.n_test} held-out test cycles are
            misclassified. Pick one to see what misled the model.
          </>
        ) : (
          <>
            No held-out test cycle is misclassified for this component, so these are the ten <strong>least confident correct</strong>{" "}
            predictions: the cycles closest to a decision boundary.
          </>
        )}
      </p>
      <div className="max-w-xl">
        <label htmlFor="error-cycle" className="nameplate mb-1.5 block">
          Cycle
        </label>
        <Select value={cycle === null ? undefined : String(cycle)} onValueChange={(v) => setCycle(Number(v))}>
          <SelectTrigger id="error-cycle" className="w-full bg-ground">
            <SelectValue placeholder="Choose a cycle" />
          </SelectTrigger>
          <SelectContent>
            {cases.cases.map((c) => (
              <SelectItem key={c.cycle} value={String(c.cycle)}>
                Cycle {c.cycle}: recorded {name(c.true_value)}, predicted {name(c.predicted_value)} ({fmt.pct(c.confidence, 0)})
              </SelectItem>
            ))}
          </SelectContent>
        </Select>
      </div>

      <QueryState query={profile} loadingLabel="Building the feature profile">
        {(p) => {
          const differs = p.predicted_value !== p.true_value;
          const series = [
            { key: "cycle", label: `Cycle ${p.cycle}`, color: "var(--ink)", hollow: true },
            { key: "true", label: `Recorded class mean (${name(p.true_value)})`, color: SERIES_HEX[2] },
            ...(differs ? [{ key: "pred", label: `Predicted class mean (${name(p.predicted_value)})`, color: SERIES_HEX[1] }] : []),
          ];
          const extent = Math.max(1, ...p.features.flatMap((f) => [f.z_cycle, f.z_true_mean, f.z_pred_mean ?? 0].map(Math.abs)));
          const m = Math.ceil(extent);
          const rows = p.features.map((f) => ({
            key: f.feature,
            label: f.feature,
            values: { cycle: f.z_cycle, true: f.z_true_mean, pred: f.z_pred_mean } as Record<string, number | null>,
          }));
          return (
            <div className="grid gap-6 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
              <ChartFigure
                height={p.features.length * 32 + 24}
                legend={<Legend items={series.map((s) => ({ label: s.label, color: s.color, shape: s.hollow ? "ring" : "dot" }))} />}
                caption="z-score against the training set (0 = average training cycle). A cycle that sits nearer the wrong class on important features explains the error."
                table={{
                  caption: "Feature z-scores",
                  rowKey: (r) => r.feature,
                  rows: p.features,
                  columns: [
                    { key: "f", header: "Feature", cell: (r) => <span className="font-mono text-xs">{r.feature}</span> },
                    { key: "c", header: "This cycle", align: "right", cell: (r) => fmt.fixed(r.z_cycle, 2) },
                    { key: "t", header: "Recorded class", align: "right", cell: (r) => fmt.fixed(r.z_true_mean, 2) },
                    { key: "p", header: "Predicted class", align: "right", cell: (r) => (r.z_pred_mean === null ? "same" : fmt.fixed(r.z_pred_mean, 2)) },
                  ],
                }}
              >
                <DotRows rows={rows} series={series} domain={[-m, m]} step={m > 4 ? 2 : 1} format={(v) => (v > 0 ? `+${v}` : String(v))} />
              </ChartFigure>

              <div className="space-y-4">
                <div>
                  <h3 className="mb-2 font-medium">Class probabilities for cycle {p.cycle}</h3>
                  <BarList
                    ariaLabel="Class probabilities"
                    max={1}
                    rows={p.probabilities.map((cp) => ({
                      key: String(cp.value),
                      label: `${name(cp.value)} ${cp.label}`,
                      value: cp.p,
                      display: fmt.pct(cp.p),
                      color: SEVERITY_COLOR[cp.severity],
                    }))}
                  />
                </div>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <dt className="text-muted-ink">Recorded</dt>
                  <dd>{name(p.true_value)}</dd>
                  <dt className="text-muted-ink">Predicted</dt>
                  <dd>{name(p.predicted_value)}</dd>
                  <dt className="text-muted-ink">Margin over 2nd</dt>
                  <dd className="num">{fmt.fixed(p.margin)}</dd>
                  <dt className="text-muted-ink">Stable flag</dt>
                  <dd>{p.stable_flag === 1 ? "1, static conditions may not have been reached" : p.stable_flag === 0 ? "0, stable" : "n/a"}</dd>
                </dl>
                <p className="text-sm">
                  <Link to={`/predict?cycle=${p.cycle}`}>Score cycle {p.cycle} from its raw files</Link>
                </p>
              </div>
            </div>
          );
        }}
      </QueryState>
    </div>
  );
}
