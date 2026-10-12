import { BadgeCheck, CircleAlert } from "lucide-react";
import { Link } from "react-router";
import {
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, ChartFigure, GRID, Legend, TooltipBox, axisLabel } from "@/components/charts/ChartKit";
import { DataTable } from "@/components/DataTable";
import { Readout, ReadoutStrip } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { SERIES_HEX, fmt, rampColor, valueWithUnit } from "@/lib/format";
import { useEvaluation } from "@/lib/queries";
import type { ComponentKey, EvaluationResponse } from "@/lib/types";

export function Evaluation({ component, unit }: { component: ComponentKey; unit: string }) {
  const evaluation = useEvaluation(component);
  return (
    <QueryState query={evaluation} loadingLabel="Scoring the held-out test set">
      {(ev) => <EvaluationBody ev={ev} unit={unit} />}
    </QueryState>
  );
}

function EvaluationBody({ ev, unit }: { ev: EvaluationResponse; unit: string }) {
  const name = (v: number) => valueWithUnit(v, unit);
  const colorOf = (v: number) => SERIES_HEX[ev.classes.findIndex((c) => c.value === v)] ?? SERIES_HEX[0];
  const critical = ev.report.find((r) => ev.classes.find((c) => c.value === r.value)?.severity === "Critical");

  return (
    <div className="space-y-5">
      <ReadoutStrip>
        <Readout label="Accuracy" value={fmt.fixed(ev.accuracy, 4)} hint={`${ev.n_test - ev.n_errors} of ${ev.n_test} test cycles`} />
        <Readout label="Macro-F1" value={fmt.fixed(ev.f1_macro, 4)} hint="every class weighted equally" />
        <Readout label="ROC-AUC (one-vs-rest)" value={ev.roc_auc === null ? "n/a" : fmt.fixed(ev.roc_auc, 4)} hint="ranking quality" />
        <Readout label="Misclassified" value={ev.n_errors} hint="held-out test cycles" />
        <Readout label="Calibration error" value={fmt.fixed(ev.ece, 3)} hint="ECE; 0 means confidence equals accuracy" />
      </ReadoutStrip>

      <p className="flex items-center gap-2 text-sm">
        {ev.parity.matches ? (
          <>
            <BadgeCheck aria-hidden className="size-4 text-ok" />
            <span className="text-ink-2">
              Scored live by the saved pipeline; accuracy and macro-F1 match <code>outputs/metrics/test_results.csv</code>.
            </span>
          </>
        ) : (
          <>
            <CircleAlert aria-hidden className="size-4 text-warn" />
            <span className="text-ink-2">
              Live scores differ from <code>test_results.csv</code> (saved accuracy{" "}
              {ev.parity.saved_accuracy === null ? "n/a" : fmt.fixed(ev.parity.saved_accuracy, 4)}). Check library versions.
            </span>
          </>
        )}
      </p>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Confusion matrix" description="Rows are the recorded condition, columns the prediction. Shade is the share of each row.">
          <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Confusion matrix">
            <table className="w-full border-separate border-spacing-0.5 text-sm">
              <caption className="sr-only">Confusion matrix, counts of held-out test cycles</caption>
              <thead>
                <tr>
                  <th scope="col" className="nameplate pb-1 text-left font-medium">
                    Recorded ↓ / predicted →
                  </th>
                  {ev.classes.map((c) => (
                    <th key={c.value} scope="col" className="nameplate pb-1 text-center font-medium" title={c.label}>
                      {name(c.value)}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {ev.confusion.map((row, i) => {
                  const total = row.reduce((s, x) => s + x, 0);
                  const cls = ev.classes[i];
                  return (
                    <tr key={cls?.value ?? i}>
                      <th scope="row" className="pr-2 text-left font-normal whitespace-nowrap text-ink-2">
                        {cls ? name(cls.value) : i} <span className="text-xs text-muted-ink">{cls?.label}</span>
                      </th>
                      {row.map((count, j) => {
                        const share = total ? count / total : 0;
                        const { bg, fg } = rampColor(share);
                        return (
                          <td
                            key={j}
                            className="num h-11 min-w-14 rounded-sm text-center font-medium"
                            style={{ background: count === 0 ? "var(--ground)" : bg, color: count === 0 ? "var(--muted-ink)" : fg }}
                            title={`${count} cycles (${fmt.pct(share)} of row)`}
                          >
                            {count}
                          </td>
                        );
                      })}
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </Panel>

        <Panel title="Per-class report" description="Precision, recall and F1 for every documented condition on the held-out test set.">
          <DataTable
            caption="Per-class report"
            rowKey={(r) => String(r.value)}
            rows={ev.report}
            columns={[
              { key: "c", header: "Condition", cell: (r) => <span><span className="num">{name(r.value)}</span> <span className="text-muted-ink">{r.label}</span></span> },
              { key: "p", header: "Precision", align: "right", cell: (r) => fmt.fixed(r.precision, 4) },
              { key: "r", header: "Recall", align: "right", cell: (r) => fmt.fixed(r.recall, 4) },
              { key: "f", header: "F1", align: "right", cell: (r) => fmt.fixed(r.f1, 4) },
              { key: "s", header: "Support", align: "right", cell: (r) => r.support },
            ]}
          />
          {critical ? (
            <p className="mt-3 text-sm text-ink-2">
              Recall on the critical class ({name(critical.value)}, {critical.label}) is{" "}
              <strong className="num text-ink">{fmt.fixed(critical.recall, 3)}</strong>: the share of genuinely critical cycles
              the model catches. Missing a critical state is the costly error.
            </p>
          ) : null}
          {ev.saved_report_text ? (
            <details className="mt-3">
              <summary className="cursor-pointer text-sm text-signal">Saved classification report (outputs/reports)</summary>
              <pre className="mt-2 overflow-x-auto rounded-sm border border-hairline bg-ground p-3 font-mono text-xs text-ink-2">
                {ev.saved_report_text}
              </pre>
            </details>
          ) : null}
        </Panel>
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <CurvePanel
          kind="roc"
          ev={ev}
          name={name}
          colorOf={colorOf}
        />
        <CurvePanel kind="pr" ev={ev} name={name} colorOf={colorOf} />
      </div>

      <div className="grid gap-5 lg:grid-cols-2">
        <Panel title="Calibration" description="Does 80% confidence mean right 80% of the time? Points on the diagonal are perfectly calibrated.">
          <ChartFigure
            height={260}
            caption="Bins of width 0.1 on the top-class probability; only non-empty bins are drawn."
            table={{
              caption: "Calibration bins",
              rowKey: (r) => String(r.mean_confidence),
              rows: ev.calibration,
              columns: [
                { key: "c", header: "Mean confidence", align: "right", cell: (r) => fmt.fixed(r.mean_confidence, 3) },
                { key: "a", header: "Observed accuracy", align: "right", cell: (r) => fmt.fixed(r.accuracy, 3) },
                { key: "n", header: "Cycles", align: "right", cell: (r) => r.count },
              ],
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={ev.calibration} margin={{ top: 8, right: 12, bottom: 22, left: 4 }}>
                <CartesianGrid {...GRID} vertical />
                <XAxis {...AXIS} type="number" dataKey="mean_confidence" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} label={axisLabel("mean predicted confidence", "x")} />
                <YAxis {...AXIS} type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={40} />
                <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-ink)" strokeDasharray="4 4" ifOverflow="extendDomain" />
                <Tooltip
                  content={({ active, payload }) => {
                    const r = payload?.[0]?.payload as EvaluationResponse["calibration"][number] | undefined;
                    if (!active || !r) return null;
                    return (
                      <TooltipBox
                        title={`Confidence ${fmt.fixed(r.mean_confidence, 3)}`}
                        rows={[
                          { label: "observed accuracy", value: fmt.fixed(r.accuracy, 3), color: "var(--series-1)" },
                          { label: "cycles", value: r.count },
                        ]}
                      />
                    );
                  }}
                />
                <Line dataKey="accuracy" stroke="var(--series-1)" strokeWidth={2} dot={{ r: 4, fill: "var(--series-1)", stroke: "var(--plate)", strokeWidth: 2 }} isAnimationActive={false} />
              </LineChart>
            </ResponsiveContainer>
          </ChartFigure>
        </Panel>

        <Panel title="Confidence of correct and wrong predictions" description="Wrong predictions should sit at low confidence.">
          <ChartFigure
            height={260}
            legend={<Legend items={[{ label: "correct", color: "var(--ok)", shape: "square" }, { label: "wrong", color: "var(--crit)", shape: "square" }]} />}
            caption="Vertical axis uses a square-root scale so the few low-confidence cycles stay visible next to the many confident ones."
            table={{
              caption: "Confidence histogram",
              rowKey: (r) => String(r.lo),
              rows: ev.confidence_hist.filter((b) => b.correct + b.wrong > 0),
              columns: [
                { key: "b", header: "Confidence bin", cell: (r) => `${r.lo.toFixed(2)} to ${r.hi.toFixed(2)}` },
                { key: "c", header: "Correct", align: "right", cell: (r) => r.correct },
                { key: "w", header: "Wrong", align: "right", cell: (r) => r.wrong },
              ],
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={ev.confidence_hist} margin={{ top: 8, right: 12, bottom: 22, left: 4 }} barCategoryGap={2}>
                <CartesianGrid {...GRID} />
                <XAxis {...AXIS} dataKey="lo" tickFormatter={(v: number) => v.toFixed(1)} interval={3} label={axisLabel("top-class probability", "x")} />
                <YAxis {...AXIS} scale="sqrt" domain={[0, "auto"]} width={40} allowDecimals={false} />
                <Tooltip
                  cursor={{ fill: "var(--raised)" }}
                  content={({ active, payload }) => {
                    const r = payload?.[0]?.payload as EvaluationResponse["confidence_hist"][number] | undefined;
                    if (!active || !r) return null;
                    return (
                      <TooltipBox
                        title={`${r.lo.toFixed(2)} to ${r.hi.toFixed(2)}`}
                        rows={[
                          { label: "correct", value: r.correct, color: "var(--ok)" },
                          { label: "wrong", value: r.wrong, color: "var(--crit)" },
                        ]}
                      />
                    );
                  }}
                />
                <Bar dataKey="correct" stackId="a" fill="var(--ok)" isAnimationActive={false} />
                <Bar dataKey="wrong" stackId="a" fill="var(--crit)" radius={[3, 3, 0, 0]} isAnimationActive={false} />
              </BarChart>
            </ResponsiveContainer>
          </ChartFigure>
        </Panel>
      </div>

      <Panel title="Misclassified test cycles" description="Open a cycle on the Predict page to see its full explanation.">
        {ev.misclassified.length === 0 ? (
          <p className="text-sm text-ink-2">
            No held-out test cycle is misclassified for this component. The realistic, blocked estimate further down is still
            below a perfect score.
          </p>
        ) : (
          <DataTable
            caption="Misclassified test cycles"
            rowKey={(r) => String(r.cycle)}
            rows={ev.misclassified}
            columns={[
              { key: "c", header: "Cycle", cell: (r) => <Link className="num" to={`/predict?cycle=${r.cycle}`}>{r.cycle}</Link> },
              { key: "t", header: "Recorded", cell: (r) => name(r.true_value) },
              { key: "p", header: "Predicted", cell: (r) => name(r.predicted_value) },
              { key: "cf", header: "Confidence", align: "right", cell: (r) => fmt.fixed(r.confidence, 3) },
              { key: "pt", header: "P(recorded class)", align: "right", cell: (r) => fmt.fixed(r.p_true, 3) },
              { key: "s", header: "Stable flag", align: "right", cell: (r) => r.stable_flag ?? "n/a" },
            ]}
          />
        )}
      </Panel>
    </div>
  );
}

function CurvePanel({
  kind,
  ev,
  name,
  colorOf,
}: {
  kind: "roc" | "pr";
  ev: EvaluationResponse;
  name: (v: number) => string;
  colorOf: (v: number) => string;
}) {
  const curves = kind === "roc" ? ev.roc : ev.pr;
  const scoreName = kind === "roc" ? "AUC" : "Average precision";
  return (
    <Panel
      title={kind === "roc" ? "ROC curves, one class against the rest" : "Precision-recall curves, one class against the rest"}
      description={kind === "roc" ? "Closer to the top-left corner is better." : "Closer to the top-right corner is better."}
    >
      <ChartFigure
        height={260}
        legend={<Legend items={curves.map((c) => ({ label: `${name(c.value)} (${scoreName} ${c.score.toFixed(3)})`, color: colorOf(c.value), shape: "line" }))} />}
        caption={kind === "roc" ? "Dashed diagonal: a model that guesses." : "Recall on the horizontal axis, precision on the vertical."}
        table={{
          caption: `${scoreName} per class`,
          rowKey: (c) => String(c.value),
          rows: curves,
          columns: [
            { key: "c", header: "Class", cell: (c) => `${name(c.value)} ${c.label}` },
            { key: "s", header: scoreName, align: "right", cell: (c) => fmt.fixed(c.score, 4) },
            { key: "n", header: "Curve points", align: "right", cell: (c) => c.points.length },
          ],
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart margin={{ top: 8, right: 12, bottom: 22, left: 4 }}>
            <CartesianGrid {...GRID} vertical />
            <XAxis
              {...AXIS}
              type="number"
              dataKey="x"
              domain={[-0.02, 1.02]}
              ticks={[0, 0.25, 0.5, 0.75, 1]}
              label={axisLabel(kind === "roc" ? "false positive rate" : "recall", "x")}
              allowDuplicatedCategory={false}
            />
            <YAxis {...AXIS} type="number" domain={[0, 1.02]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={40} />
            {kind === "roc" ? (
              <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="var(--muted-ink)" strokeDasharray="4 4" />
            ) : null}
            <Tooltip
              content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const first = payload[0]?.payload as { x: number } | undefined;
                return (
                  <TooltipBox
                    title={`${kind === "roc" ? "FPR" : "recall"} ${first ? first.x.toFixed(3) : ""}`}
                    rows={payload.map((p) => ({
                      label: String(p.name),
                      value: typeof p.value === "number" ? p.value.toFixed(3) : String(p.value),
                      color: p.color,
                    }))}
                  />
                );
              }}
            />
            {curves.map((c) => (
              <Line
                key={c.value}
                data={c.points}
                dataKey="y"
                name={name(c.value)}
                stroke={colorOf(c.value)}
                strokeWidth={2}
                dot={false}
                type="linear"
                isAnimationActive={false}
              />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </ChartFigure>
    </Panel>
  );
}
