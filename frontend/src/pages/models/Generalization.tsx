import { useState } from "react";
import { ChartFigure, Legend } from "@/components/charts/ChartKit";
import { DotRows, floorTo } from "@/components/charts/DotRows";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { Slider } from "@/components/ui/slider";
import { COMPONENT_SHORT, SERIES_HEX, fmt, valueWithUnit } from "@/lib/format";
import { useGeneralization, useTimeline } from "@/lib/queries";
import type { ComponentKey, TimelineResponse } from "@/lib/types";

const ESTIMATES = [
  { key: "block", label: "Blocked CV (realistic)", color: SERIES_HEX[0], hollow: true },
  { key: "cv", label: "Random CV", color: SERIES_HEX[1] },
  { key: "test", label: "Test, random split", color: SERIES_HEX[2] },
] as const;

export function Generalization({ component, unit, nCycles }: { component: ComponentKey; unit: string; nCycles: number }) {
  const gen = useGeneralization();
  return (
    <Panel
      title="Random splits flatter; blocked validation does not"
      description="Neighbouring cycles were recorded seconds apart under the same settings. A random split puts near-duplicates on both sides of the boundary; blocked CV holds out whole runs of consecutive cycles."
    >
      <QueryState query={gen} loadingLabel="Loading validation results">
        {(g) => {
          const rows = g.rows.map((r) => ({
            key: r.component,
            label: COMPONENT_SHORT[r.component],
            values: { block: r.block_cv_f1_macro, cv: r.cv_f1_macro, test: r.test_f1_macro } as Record<string, number | null>,
          }));
          const lo = floorTo(g.rows.flatMap((r) => [r.block_cv_f1_macro, r.cv_f1_macro, r.test_f1_macro]), 0.02);
          return (
            <ChartFigure
              height={4 * 32 + 24}
              legend={<Legend items={ESTIMATES.map((e) => ({ label: e.label, color: e.color, shape: "hollow" in e ? "ring" : "dot" }))} />}
              caption={`Macro-F1; axis starts at ${lo.toFixed(2)}. Blocks of ${g.block_size} consecutive cycles.`}
              table={{
                caption: "Validation estimates per component",
                rowKey: (r) => r.component,
                rows: g.rows,
                columns: [
                  { key: "c", header: "Component", cell: (r) => r.display_name },
                  { key: "b", header: "Blocked CV", align: "right", cell: (r) => fmt.fixed(r.block_cv_f1_macro, 4) },
                  { key: "r", header: "Random CV", align: "right", cell: (r) => fmt.fixed(r.cv_f1_macro, 4) },
                  { key: "t", header: "Test", align: "right", cell: (r) => fmt.fixed(r.test_f1_macro, 4) },
                  { key: "o", header: "Optimism", align: "right", cell: (r) => fmt.signed(r.optimism) },
                ],
              }}
            >
              <DotRows rows={rows} series={ESTIMATES} domain={[lo, 1]} step={0.02} format={(v) => v.toFixed(2)} />
            </ChartFigure>
          );
        }}
      </QueryState>
      <SplitTimeline component={component} unit={unit} nCycles={nCycles} />
    </Panel>
  );
}

function SplitTimeline({ component, unit, nCycles }: { component: ComponentKey; unit: string; nCycles: number }) {
  const [draft, setDraft] = useState<[number, number]>([0, 300]);
  const [range, setRange] = useState<[number, number]>([0, 300]);
  const timeline = useTimeline(component, range[0], range[1]);

  return (
    <div className="mt-6 border-t border-hairline pt-5">
      <h3 className="font-medium">Why random splits leak, {COMPONENT_SHORT[component].toLowerCase()}</h3>
      <p className="prose-measure mt-1 text-sm text-ink-2">
        Each tick is one cycle in recording order. Test cycles (bottom row) sit between training cycles recorded moments
        before and after, under the same condition.
      </p>
      <div className="mt-3 max-w-md">
        <label htmlFor="timeline-window" className="nameplate mb-2 block">
          Cycle window <span className="num text-ink">{draft[0]} to {draft[1]}</span>
        </label>
        <Slider
          id="timeline-window"
          min={0}
          max={nCycles - 1}
          step={1}
          minStepsBetweenThumbs={20}
          value={draft}
          onValueChange={(v) => setDraft([v[0] ?? 0, v[1] ?? 300])}
          onValueCommit={(v) => setRange([v[0] ?? 0, v[1] ?? 300])}
          aria-label="Cycle window"
        />
      </div>
      <div className="mt-4">
        <QueryState query={timeline} loadingLabel="Loading the split">
          {(t) => <Ticks t={t} unit={unit} />}
        </QueryState>
      </div>
    </div>
  );
}

function Ticks({ t, unit }: { t: TimelineResponse; unit: string }) {
  const span = Math.max(1, t.end - t.start);
  const colorOf = (v: number) => SERIES_HEX[t.classes.findIndex((c) => c.value === v)] ?? SERIES_HEX[0];
  const nTest = t.points.filter((p) => p.split === "test").length;
  return (
    <figure>
      <Legend
        className="mb-2"
        items={t.classes.map((c) => ({ label: `${valueWithUnit(c.value, unit)} ${c.label}`, color: colorOf(c.value), shape: "square" }))}
      />
      <div aria-hidden className="space-y-1">
        {(["train", "test"] as const).map((split) => (
          <div key={split} className="grid grid-cols-[3.5rem_1fr] items-center gap-2">
            <span className="text-xs text-muted-ink">{split}</span>
            <div className="relative h-7 rounded-sm bg-ground">
              {t.points
                .filter((p) => p.split === split)
                .map((p) => (
                  <span
                    key={p.cycle}
                    className="absolute top-1 bottom-1 w-[2px]"
                    style={{ left: `${((p.cycle - t.start) / span) * 100}%`, background: colorOf(p.value) }}
                    title={`cycle ${p.cycle}: ${valueWithUnit(p.value, unit)}, ${split}`}
                  />
                ))}
            </div>
          </div>
        ))}
        <div className="grid grid-cols-[3.5rem_1fr] gap-2">
          <span />
          <div className="num flex justify-between text-xs text-muted-ink">
            <span>{t.start}</span>
            <span>cycle index (recording order)</span>
            <span>{t.end}</span>
          </div>
        </div>
      </div>
      <figcaption className="mt-1 text-xs text-muted-ink">
        {nTest} of {t.points.length} cycles in this window are held-out test cycles.
      </figcaption>
    </figure>
  );
}
