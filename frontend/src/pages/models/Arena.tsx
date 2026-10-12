import { Trophy } from "lucide-react";
import { useMemo, useState } from "react";
import { CartesianGrid, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, ChartFigure, GRID, Legend, TooltipBox, axisLabel } from "@/components/charts/ChartKit";
import { DataTable } from "@/components/DataTable";
import { Panel } from "@/components/Panel";
import { SegmentedControl } from "@/components/SegmentedControl";
import { Slider } from "@/components/ui/slider";
import {
  COMPONENTS,
  COMPONENT_SHORT,
  FEATURE_SET_NAMES,
  FEATURE_SET_ORDER,
  MODEL_NAMES,
  MODEL_ORDER,
  MODEL_SHORT,
  fmt,
  modelColor,
  rampColor,
} from "@/lib/format";
import type { ComponentKey, CvRow, MetricsResponse } from "@/lib/types";

const SHAPES = { baseline: "circle", engineered: "triangle", selected: "square" } as const;

export function Arena({ metrics }: { metrics: MetricsResponse }) {
  const [floor, setFloor] = useState(0.9);
  const winners = useMemo(
    () => new Map(metrics.test.map((t) => [t.component, `${t.model}|${t.feature_set}`])),
    [metrics.test],
  );

  return (
    <>
      <Panel
        title="All configurations, cross-validated"
        description={`Macro-F1 from ${metrics.cv.length} cross-validated runs on the training set: ${MODEL_ORDER.length} models × ${FEATURE_SET_ORDER.length} feature sets × ${COMPONENTS.length} components. The outlined cell is the model each component ships with.`}
        aside={
          <div className="w-56">
            <label htmlFor="arena-floor" className="nameplate mb-2 block">
              Colour scale starts at <span className="num text-ink">{floor.toFixed(2)}</span>
            </label>
            <Slider
              id="arena-floor"
              min={0.5}
              max={0.95}
              step={0.05}
              value={[floor]}
              onValueChange={(v) => setFloor(v[0] ?? 0.5)}
              aria-label="Colour scale starts at"
            />
          </div>
        }
      >
        <div className="grid gap-5 sm:grid-cols-2 xl:grid-cols-4">
          {COMPONENTS.map((c) => (
            <Heatmap key={c} component={c} rows={metrics.cv.filter((r) => r.component === c)} floor={floor} winner={winners.get(c)} />
          ))}
        </div>
        <p className="mt-3 text-xs text-muted-ink">
          Cell shade encodes macro-F1 from {floor.toFixed(2)} or below (darkest) to 1.00 (brightest); every cell prints its value. Lower the start to see the weak baselines in context.
        </p>
      </Panel>

      <div className="mt-5 grid gap-5 lg:grid-cols-2">
        <CostChart rows={metrics.cv} />
        <Stability rows={metrics.cv} />
      </div>

      <Panel className="mt-5" title="How each winner was chosen" description="Highest mean cross-validated macro-F1 on the training set only; ties go to the simpler feature set and model. The test set is used once, after selection.">
        <DataTable
          caption="Top three configurations per component"
          rowKey={(r) => `${r.component}-${r.model}-${r.feature_set}`}
          rows={COMPONENTS.flatMap((c) =>
            metrics.cv
              .filter((r) => r.component === c)
              .sort((a, b) => b.f1_macro - a.f1_macro)
              .slice(0, 3),
          )}
          rowClassName={(r) => (winners.get(r.component) === `${r.model}|${r.feature_set}` ? "bg-raised/60" : undefined)}
          columns={[
            { key: "c", header: "Component", cell: (r) => COMPONENT_SHORT[r.component] },
            {
              key: "m",
              header: "Model",
              cell: (r) => (
                <span className="inline-flex items-center gap-1.5">
                  {MODEL_NAMES[r.model] ?? r.model}
                  {winners.get(r.component) === `${r.model}|${r.feature_set}` ? (
                    <span className="inline-flex items-center gap-1 text-xs text-signal">
                      <Trophy aria-hidden className="size-3.5" /> shipped
                    </span>
                  ) : null}
                </span>
              ),
            },
            { key: "fs", header: "Feature set", cell: (r) => FEATURE_SET_NAMES[r.feature_set] ?? r.feature_set },
            { key: "f1", header: "CV macro-F1", align: "right", cell: (r) => fmt.fixed(r.f1_macro, 4) },
            { key: "sd", header: "± fold std", align: "right", cell: (r) => fmt.fixed(r.f1_macro_std, 4) },
            { key: "t", header: "Fit time per fold", align: "right", cell: (r) => `${fmt.fixed(r.training_time, 2)} s` },
          ]}
        />
        <p className="mt-2 text-xs text-muted-ink">
          Where the top three sit within about one fold standard deviation of each other, their order is not statistically
          meaningful.
        </p>
      </Panel>
    </>
  );
}

function Heatmap({
  component,
  rows,
  floor,
  winner,
}: {
  component: ComponentKey;
  rows: readonly CvRow[];
  floor: number;
  winner: string | undefined;
}) {
  const cell = (m: string, fs: string) => rows.find((r) => r.model === m && r.feature_set === fs);
  return (
    <div className="min-w-0">
      <table className="w-full table-fixed border-separate border-spacing-0.5 text-xs">
        <caption className="mb-1 text-left font-medium text-ink">{COMPONENT_SHORT[component]}</caption>
        <thead>
          <tr>
            <th scope="col" className="w-[30%]">
              <span className="sr-only">Feature set</span>
            </th>
            {MODEL_ORDER.map((m) => (
              <th key={m} scope="col" className="nameplate pb-1 text-center font-medium">
                <abbr title={MODEL_NAMES[m]} className="no-underline">
                  {MODEL_SHORT[m]}
                </abbr>
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {FEATURE_SET_ORDER.map((fs) => (
            <tr key={fs}>
              <th scope="row" className="pr-1 text-left font-normal text-ink-2">
                {FEATURE_SET_NAMES[fs]}
              </th>
              {MODEL_ORDER.map((m) => {
                const r = cell(m, fs);
                if (!r) return <td key={m} className="bg-raised text-center text-muted-ink">n/a</td>;
                const { bg, fg } = rampColor((r.f1_macro - floor) / (1 - floor));
                const isWinner = winner === `${m}|${fs}`;
                return (
                  <td
                    key={m}
                    title={`${MODEL_NAMES[m]}, ${FEATURE_SET_NAMES[fs]}: ${fmt.fixed(r.f1_macro, 4)} ± ${fmt.fixed(r.f1_macro_std, 4)}`}
                    className="num h-10 rounded-sm text-center"
                    style={{ background: bg, color: fg, outline: isWinner ? "2px solid var(--ink)" : undefined, outlineOffset: 1 }}
                  >
                    {r.f1_macro.toFixed(3)}
                    {isWinner ? <span className="sr-only"> (shipped model)</span> : null}
                  </td>
                );
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function CostChart({ rows }: { rows: readonly CvRow[] }) {
  const lo = Math.min(...rows.map((r) => r.f1_macro));
  const yMin = Math.max(0, Math.floor(lo * 10) / 10);
  return (
    <Panel title="Accuracy against training cost" description="Each point is one configuration: mean fit time per fold (log scale) against CV macro-F1.">
      <ChartFigure
        height={300}
        legend={
          <div className="space-y-1">
            <Legend items={MODEL_ORDER.map((m) => ({ label: MODEL_SHORT[m] ?? m, color: modelColor(m) }))} />
            <p className="text-xs text-muted-ink">Circle baseline, triangle engineered, square selected.</p>
          </div>
        }
        caption={`Vertical axis starts at ${yMin.toFixed(1)}.`}
        table={{
          caption: "Fit time and macro-F1 per configuration",
          rowKey: (r) => `${r.component}-${r.model}-${r.feature_set}`,
          rows,
          columns: [
            { key: "c", header: "Component", cell: (r) => COMPONENT_SHORT[r.component] },
            { key: "m", header: "Model", cell: (r) => MODEL_SHORT[r.model] ?? r.model },
            { key: "fs", header: "Feature set", cell: (r) => FEATURE_SET_NAMES[r.feature_set] ?? r.feature_set },
            { key: "t", header: "Fit time [s]", align: "right", cell: (r) => fmt.fixed(r.training_time, 3) },
            { key: "f1", header: "CV macro-F1", align: "right", cell: (r) => fmt.fixed(r.f1_macro, 4) },
          ],
        }}
      >
        <ResponsiveContainer width="100%" height="100%">
          <ScatterChart margin={{ top: 8, right: 12, bottom: 22, left: 4 }}>
            <CartesianGrid {...GRID} vertical />
            <XAxis
              {...AXIS}
              type="number"
              dataKey="training_time"
              scale="log"
              domain={["auto", "auto"]}
              tickFormatter={(v: number) => (v < 0.01 ? v.toFixed(3) : v < 0.1 ? v.toFixed(2) : v < 1 ? v.toFixed(1) : v.toFixed(0))}
              label={axisLabel("fit time per fold [s, log]", "x")}
            />
            <YAxis {...AXIS} type="number" dataKey="f1_macro" domain={[yMin, 1]} width={44} tickFormatter={(v: number) => v.toFixed(2)} />
            <Tooltip
              cursor={false}
              content={({ active, payload }) => {
                const r = payload?.[0]?.payload as CvRow | undefined;
                if (!active || !r) return null;
                return (
                  <TooltipBox
                    title={`${COMPONENT_SHORT[r.component]}: ${MODEL_NAMES[r.model]}, ${FEATURE_SET_NAMES[r.feature_set]?.toLowerCase()}`}
                    rows={[
                      { label: "CV macro-F1", value: fmt.fixed(r.f1_macro, 4), color: modelColor(r.model) },
                      { label: "fit time", value: `${fmt.fixed(r.training_time, 3)} s` },
                    ]}
                  />
                );
              }}
            />
            {MODEL_ORDER.flatMap((m) =>
              FEATURE_SET_ORDER.map((fs) => (
                <Scatter
                  key={`${m}-${fs}`}
                  name={`${MODEL_SHORT[m]} ${fs}`}
                  data={rows.filter((r) => r.model === m && r.feature_set === fs)}
                  fill={modelColor(m)}
                  shape={SHAPES[fs]}
                  stroke="var(--plate)"
                  strokeWidth={2}
                  isAnimationActive={false}
                />
              )),
            )}
          </ScatterChart>
        </ResponsiveContainer>
      </ChartFigure>
    </Panel>
  );
}

function Stability({ rows }: { rows: readonly CvRow[] }) {
  const [component, setComponent] = useState<ComponentKey>("accumulator");
  const data = rows
    .filter((r) => r.component === component)
    .sort((a, b) => FEATURE_SET_ORDER.indexOf(a.feature_set as never) - FEATURE_SET_ORDER.indexOf(b.feature_set as never) || MODEL_ORDER.indexOf(a.model as never) - MODEL_ORDER.indexOf(b.model as never));
  const lo = Math.max(0, Math.floor(Math.min(...data.map((r) => r.f1_macro - r.f1_macro_std)) * 20) / 20);
  const x = (v: number) => `${((Math.min(1, Math.max(lo, v)) - lo) / (1 - lo)) * 100}%`;

  return (
    <Panel
      title="Fold-to-fold stability"
      description="Mean CV macro-F1 with one fold standard deviation either side."
      aside={
        <SegmentedControl
          label="Component"
          hideLabel
          options={COMPONENTS.map((c) => ({ value: c, label: COMPONENT_SHORT[c] }))}
          value={component}
          onChange={setComponent}
        />
      }
    >
      <ul className="space-y-1" aria-label={`Stability for ${COMPONENT_SHORT[component]}`}>
        {data.map((r) => (
          <li key={`${r.model}-${r.feature_set}`} className="grid grid-cols-[minmax(110px,150px)_1fr_auto] items-center gap-3 text-sm">
            <span className="truncate">
              {MODEL_SHORT[r.model]} <span className="text-muted-ink">{FEATURE_SET_NAMES[r.feature_set]?.toLowerCase()}</span>
            </span>
            <span aria-hidden className="relative h-5">
              <span className="absolute inset-y-0 left-0 w-px bg-hairline" />
              <span
                className="absolute top-1/2 h-0.5 -translate-y-1/2 rounded"
                style={{ left: x(r.f1_macro - r.f1_macro_std), width: `calc(${x(r.f1_macro + r.f1_macro_std)} - ${x(r.f1_macro - r.f1_macro_std)})`, background: modelColor(r.model) }}
              />
              <span
                className="absolute top-1/2 size-3 -translate-x-1/2 -translate-y-1/2 rounded-full ring-2 ring-plate"
                style={{ left: x(r.f1_macro), background: modelColor(r.model) }}
              />
            </span>
            <span className="num text-right text-xs text-ink-2">
              {r.f1_macro.toFixed(3)} ± {r.f1_macro_std.toFixed(3)}
            </span>
          </li>
        ))}
      </ul>
      <p className="mt-2 text-xs text-muted-ink">Axis runs from {lo.toFixed(2)} to 1.00. Colour follows the model family.</p>
    </Panel>
  );
}
