import { useQueries } from "@tanstack/react-query";
import { useState } from "react";
import {
  Area,
  CartesianGrid,
  ComposedChart,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import { AXIS, ChartFigure, GRID, Legend, TooltipBox, axisLabel } from "@/components/charts/ChartKit";
import { Readout, ReadoutStrip } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { ErrorNote, Loading, QueryState } from "@/components/QueryState";
import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { api } from "@/lib/api";
import { COMPONENTS, COMPONENT_SHORT, SERIES_HEX, fmt } from "@/lib/format";
import { useSelective } from "@/lib/queries";
import type { ComponentKey, NoiseResponse, SelectiveResponse } from "@/lib/types";

const componentColor = (c: ComponentKey) => SERIES_HEX[COMPONENTS.indexOf(c)] ?? SERIES_HEX[0];

export function NoiseTest({ component }: { component: ComponentKey }) {
  const [all, setAll] = useState(false);
  const keys = all ? COMPONENTS : [component];
  const results = useQueries({
    queries: keys.map((c) => ({ queryKey: ["noise", c], queryFn: () => api.noise(c) })),
  });
  const pending = results.some((r) => r.isPending);
  const failed = results.find((r) => r.isError);
  const data = results.map((r) => r.data).filter((d): d is NoiseResponse => d !== undefined);
  const focus = data.find((d) => d.component === component);

  return (
    <Panel
      title="Noise stress test"
      description="Gaussian noise is added to every model input, scaled to that feature's spread on the training set, and the held-out test set is re-scored (mean of three draws, fixed seed)."
      aside={
        <div className="flex items-center gap-2">
          <Switch id="noise-all" checked={all} onCheckedChange={setAll} />
          <Label htmlFor="noise-all" className="text-sm text-ink-2">
            Compare all four components
          </Label>
        </div>
      }
    >
      {failed && !data.length ? (
        <ErrorNote error={failed.error} onRetry={() => void failed.refetch()} />
      ) : pending && !data.length ? (
        <Loading label="Re-scoring the test set under noise (first run takes a few seconds)" />
      ) : (
        <>
          {focus ? (
            <div className="mb-4">
              <ReadoutStrip>
                {[0, 0.1, 0.3, 1].map((lvl) => {
                  const pt = focus.points.find((p) => p.level === lvl);
                  return (
                    <Readout
                      key={lvl}
                      label={lvl === 0 ? "No noise" : `Noise ${lvl} × spread`}
                      value={pt ? fmt.fixed(pt.f1_macro) : "n/a"}
                      hint={lvl === 0 ? "test macro-F1" : lvl <= 0.1 ? "sensor jitter" : lvl <= 0.3 ? "moderate" : "severe"}
                    />
                  );
                })}
              </ReadoutStrip>
            </div>
          ) : null}
          <ChartFigure
            height={300}
            legend={
              all ? (
                <Legend items={data.map((d) => ({ label: COMPONENT_SHORT[d.component], color: componentColor(d.component), shape: "line" }))} />
              ) : (
                <Legend items={[{ label: `${COMPONENT_SHORT[component]}: mean, with min to max band`, color: componentColor(component), shape: "line" }]} />
              )
            }
            caption={`Vertical axis starts at 0. ${pending ? "Still computing some components…" : ""}`}
            table={{
              caption: "Macro-F1 under noise",
              rowKey: (r) => String(r.level),
              rows: (data[0]?.points ?? []).map((p, i) => ({
                level: p.level,
                values: Object.fromEntries(data.map((d) => [d.component, d.points[i]])),
              })),
              columns: [
                { key: "l", header: "Noise level", align: "right", cell: (r) => `${r.level} ×` },
                ...data.map((d) => ({
                  key: d.component,
                  header: COMPONENT_SHORT[d.component],
                  align: "right" as const,
                  cell: (r: { values: Record<string, NoiseResponse["points"][number] | undefined> }) => {
                    const p = r.values[d.component];
                    return p ? `${fmt.fixed(p.f1_macro)} (${fmt.fixed(p.min)} to ${fmt.fixed(p.max)})` : "n/a";
                  },
                })),
              ],
            }}
          >
            <ResponsiveContainer width="100%" height="100%">
              <ComposedChart margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
                <CartesianGrid {...GRID} />
                <XAxis
                  {...AXIS}
                  type="number"
                  dataKey="level"
                  domain={[0, 1]}
                  ticks={[0, 0.1, 0.2, 0.3, 0.5, 0.75, 1]}
                  allowDuplicatedCategory={false}
                  label={axisLabel("noise level (× each feature's training spread)", "x")}
                />
                <YAxis {...AXIS} type="number" domain={[0, 1]} ticks={[0, 0.25, 0.5, 0.75, 1]} width={40} />
                <Tooltip
                  content={({ active, payload }) => {
                    if (!active || !payload?.length) return null;
                    const lvl = (payload[0]?.payload as { level: number } | undefined)?.level;
                    return (
                      <TooltipBox
                        title={`Noise ${lvl ?? ""} × spread`}
                        rows={payload
                          .filter((p) => p.dataKey === "f1_macro")
                          .map((p) => ({ label: String(p.name), value: typeof p.value === "number" ? p.value.toFixed(3) : "", color: p.color }))}
                      />
                    );
                  }}
                />
                {data.map((d) =>
                  all ? null : (
                    <Area
                      key={`${d.component}-band`}
                      data={d.points.map((p) => ({ level: p.level, band: [p.min, p.max] }))}
                      dataKey="band"
                      stroke="none"
                      fill={componentColor(d.component)}
                      fillOpacity={0.18}
                      isAnimationActive={false}
                      activeDot={false}
                      tooltipType="none"
                    />
                  ),
                )}
                {data.map((d) => (
                  <Line
                    key={d.component}
                    data={d.points}
                    dataKey="f1_macro"
                    name={COMPONENT_SHORT[d.component]}
                    stroke={componentColor(d.component)}
                    strokeWidth={2}
                    dot={{ r: 3.5, fill: componentColor(d.component), stroke: "var(--plate)", strokeWidth: 2 }}
                    isAnimationActive={false}
                  />
                ))}
              </ComposedChart>
            </ResponsiveContainer>
          </ChartFigure>
        </>
      )}
    </Panel>
  );
}

export function Selective({ component }: { component: ComponentKey }) {
  const selective = useSelective(component);
  return (
    <Panel
      title="Selective prediction"
      description="If the model may answer 'not sure' below a confidence threshold, how many cycles does it answer, and how accurate is it on those?"
    >
      <QueryState query={selective} loadingLabel="Loading test-set confidences">
        {(s) => <SelectiveBody key={component} s={s} />}
      </QueryState>
    </Panel>
  );
}

function SelectiveBody({ s }: { s: SelectiveResponse }) {
  const [threshold, setThreshold] = useState(0.6);
  const point = s.grid.reduce((best, g) => (Math.abs(g.threshold - threshold) < Math.abs(best.threshold - threshold) ? g : best), s.grid[0]!);
  const accValues = s.grid.map((g) => g.accuracy).filter((a): a is number => a !== null);
  const lo = Math.max(0, Math.floor(Math.min(...accValues, ...s.grid.map((g) => g.coverage)) * 10) / 10);

  return (
    <div className="space-y-4">
      <div className="max-w-md">
        <label htmlFor="sel-threshold" className="nameplate mb-2 block">
          Confidence threshold <span className="num text-ink">{threshold.toFixed(2)}</span>
        </label>
        <Slider
          id="sel-threshold"
          min={0.3}
          max={1}
          step={0.01}
          value={[threshold]}
          onValueChange={(v) => setThreshold(v[0] ?? 0.6)}
          aria-label="Confidence threshold"
        />
      </div>
      <ReadoutStrip>
        <Readout label="Answered" value={fmt.pct(point.coverage)} hint={`${point.answered} of ${s.n_test} cycles`} />
        <Readout label="Deferred to an engineer" value={s.n_test - point.answered} hint="below the threshold" />
        <Readout
          label="Accuracy when answering"
          value={point.accuracy === null ? "n/a" : fmt.fixed(point.accuracy, 4)}
          hint="on answered cycles only"
        />
        <Readout label="Errors still answered" value={`${point.errors_answered} of ${s.n_errors}`} hint="test-set errors" />
      </ReadoutStrip>
      <ChartFigure
        height={260}
        legend={
          <Legend
            items={[
              { label: "Coverage", color: SERIES_HEX[0], shape: "line" },
              { label: "Accuracy on answered cycles", color: SERIES_HEX[2], shape: "line" },
            ]}
          />
        }
        caption={`Vertical axis starts at ${lo.toFixed(1)}. The marker follows the threshold slider.`}
        table={{
          caption: "Coverage and accuracy per threshold",
          rowKey: (g) => String(g.threshold),
          rows: s.grid.filter((_, i) => i % 5 === 0 || i === s.grid.length - 1),
          columns: [
            { key: "t", header: "Threshold", align: "right", cell: (g) => g.threshold.toFixed(2) },
            { key: "c", header: "Coverage", align: "right", cell: (g) => fmt.pct(g.coverage) },
            { key: "a", header: "Accuracy", align: "right", cell: (g) => (g.accuracy === null ? "n/a" : fmt.fixed(g.accuracy, 4)) },
            { key: "e", header: "Errors answered", align: "right", cell: (g) => g.errors_answered },
          ],
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={s.grid} margin={{ top: 8, right: 16, bottom: 22, left: 4 }}>
            <CartesianGrid {...GRID} />
            <XAxis {...AXIS} type="number" dataKey="threshold" domain={[0.3, 1]} ticks={[0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]} label={axisLabel("confidence threshold", "x")} />
            <YAxis
              {...AXIS}
              type="number"
              domain={[lo, 1]}
              ticks={Array.from({ length: Math.round((1 - lo) / (1 - lo > 0.5 ? 0.2 : 0.1)) + 1 }, (_, i) => Number((lo + i * (1 - lo > 0.5 ? 0.2 : 0.1)).toFixed(2)))}
              width={40}
              tickFormatter={(v: number) => v.toFixed(1)}
            />
            <ReferenceLine x={point.threshold} stroke="var(--ink-2)" />
            <Tooltip
              content={({ active, payload }) => {
                const g = payload?.[0]?.payload as SelectiveResponse["grid"][number] | undefined;
                if (!active || !g) return null;
                return (
                  <TooltipBox
                    title={`Threshold ${g.threshold.toFixed(2)}`}
                    rows={[
                      { label: "coverage", value: fmt.pct(g.coverage), color: SERIES_HEX[0] },
                      { label: "accuracy", value: g.accuracy === null ? "n/a" : fmt.fixed(g.accuracy, 4), color: SERIES_HEX[2] },
                    ]}
                  />
                );
              }}
            />
            <Line dataKey="coverage" stroke={SERIES_HEX[0]} strokeWidth={2} dot={false} isAnimationActive={false} />
            <Line dataKey="accuracy" stroke={SERIES_HEX[2]} strokeWidth={2} dot={false} connectNulls={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </ChartFigure>
    </div>
  );
}
