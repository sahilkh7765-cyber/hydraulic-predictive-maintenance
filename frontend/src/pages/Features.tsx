import { ArrowRight } from "lucide-react";
import { useMemo, useState } from "react";
import { BarList } from "@/components/charts/BarList";
import { ChartFigure, Legend } from "@/components/charts/ChartKit";
import { DotRows, floorTo, type DotRow } from "@/components/charts/DotRows";
import { PageHeader } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { SegmentedControl } from "@/components/SegmentedControl";
import {
  BLOCK_NAMES,
  COMPONENTS,
  COMPONENT_SHORT,
  FEATURE_SET_NAMES,
  FEATURE_SET_ORDER,
  MODEL_NAMES,
  MODEL_ORDER,
  SERIES_HEX,
  fmt,
} from "@/lib/format";
import { useFeatures, useImportance, useMetrics } from "@/lib/queries";
import type { ComponentKey, FeaturesResponse, MetricsResponse } from "@/lib/types";

/** What each block computes - a description of src/feature_engineering.py, not a measurement. */
const BLOCK_DESCRIPTIONS: Record<string, string> = {
  statistical:
    "Distribution shape within the cycle: mean, median, spread, extremes, RMS, skewness, kurtosis, quartiles, coefficient of variation.",
  time_domain:
    "How the signal evolves: peak-to-peak, energy, first-difference statistics, linear slope, zero crossings, peak count.",
  rolling: "Local behaviour in a one-second window sized from each sensor's own rate. Only for channels at 10 Hz and above.",
  frequency:
    "A small FFT summary: dominant frequency, spectral energy and centroid, log-spaced band energy ratios. Only for channels at 10 Hz and above.",
  domain:
    "Hydraulic relationships between channels: pressure differentials and ratios, flow imbalance, hydraulic power, efficiency, temperature spreads, vibration per unit power.",
};

export default function FeaturesPage() {
  const features = useFeatures();
  const metrics = useMetrics();
  return (
    <QueryState query={features} loadingLabel="Loading feature engineering results">
      {(f) => (
        <QueryState query={metrics} loadingLabel="Loading experiment results">
          {(m) => <FeaturesBody features={f} metrics={m} />}
        </QueryState>
      )}
    </QueryState>
  );
}

function FeaturesBody({ features, metrics }: { features: FeaturesResponse; metrics: MetricsResponse }) {
  const engineered = features.blocks.reduce((s, b) => s + b.count, 0);
  const perModel = features.selection.map((s) => s.n_selected);
  const gains = metrics.experiment_effect
    .map((r) => ({
      component: r.component,
      gain: Math.max(r.gain_engineered_vs_baseline, r.gain_selected_vs_baseline),
    }))
    .sort((a, b) => b.gain - a.gain);
  const [first, second] = gains;
  const flat = gains.filter((g) => g.gain < 0.001).map((g) => COMPONENT_SHORT[g.component].toLowerCase());

  return (
    <>
      <PageHeader
        title="Feature engineering"
        takeaway={
          first && second ? (
            <>
              Each cycle's {fmt.int(features.raw_dimensions)} raw values become {fmt.int(engineered)} physical features.
              They lift cross-validated macro-F1 most for the {COMPONENT_SHORT[first.component].toLowerCase()} (
              {fmt.signed(first.gain)}) and {COMPONENT_SHORT[second.component].toLowerCase()} ({fmt.signed(second.gain)})
              {flat.length ? `; for the ${flat.join(" and ")}, per-sensor means are already enough` : ""}.
            </>
          ) : (
            "How raw sensor samples become model inputs."
          )
        }
      >
        <ol className="grid gap-px overflow-hidden rounded-md border border-hairline bg-hairline sm:grid-cols-4" aria-label="Feature reduction steps">
          {[
            { n: fmt.int(features.raw_dimensions), what: "raw values per cycle" },
            { n: fmt.int(engineered), what: `engineered features in ${features.blocks.length} blocks` },
            { n: fmt.int(features.cleaning.n_output), what: "left after cleaning" },
            { n: `${Math.min(...perModel)} to ${fmt.int(Math.max(...perModel))}`, what: "reach each classifier" },
          ].map((step, i, all) => (
            <li key={step.what} className="relative flex items-center gap-3 bg-plate px-4 py-3">
              <div className="min-w-0">
                <div className="text-xl font-semibold">{step.n}</div>
                <div className="text-xs text-muted-ink">{step.what}</div>
              </div>
              {i < all.length - 1 ? (
                <ArrowRight aria-hidden className="ml-auto hidden size-4 shrink-0 text-muted-ink sm:block" />
              ) : null}
            </li>
          ))}
        </ol>
      </PageHeader>

      <ExperimentPanel metrics={metrics} />

      <Panel
        className="mt-5"
        title="The five feature blocks"
        description="Feature counts before cleaning, what each block computes, and what cleaning removed."
      >
        <div className="grid gap-8 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.3fr)]">
          <div className="min-w-0">
          <BarList
            ariaLabel="Features per block"
            rows={features.blocks.map((b) => ({
              key: b.key,
              label: BLOCK_NAMES[b.key] ?? b.key,
              value: b.count,
              display: fmt.int(b.count),
            }))}
          />
          <CleaningList features={features} />
          </div>
          <dl className="space-y-3">
            {features.blocks.map((b) => (
              <div key={b.key}>
                <dt className="font-medium">{BLOCK_NAMES[b.key] ?? b.key}</dt>
                <dd className="prose-measure text-sm text-ink-2">{BLOCK_DESCRIPTIONS[b.key]}</dd>
                <dd className="mt-1 flex flex-wrap gap-1">
                  {b.examples.slice(0, 5).map((e) => (
                    <code key={e} className="rounded-sm bg-raised px-1.5 py-0.5 text-xs text-ink-2">
                      {e}
                    </code>
                  ))}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      </Panel>

      <SelectionPanel features={features} />
    </>
  );
}

function CleaningList({ features }: { features: FeaturesResponse }) {
  return (
    <div className="mt-6 border-t border-hairline pt-4">
      <h3 className="font-medium">Removed during cleaning</h3>
      <p className="mb-3 text-sm text-muted-ink">
        {fmt.int(features.cleaning.n_input)} features in, {fmt.int(features.cleaning.n_output)} out. None were removed for
        missing values; the dataset has none.
      </p>
          {(
            [
              ["Constant", features.cleaning.dropped_constant, "carry no information"],
              ["Near-zero variance", features.cleaning.dropped_near_zero_variance, "almost constant across cycles"],
              ["Exact duplicates", features.cleaning.dropped_duplicate, "identical to another column"],
            ] as const
          ).map(([name, list, why]) => (
            <details key={name} className="group border-t border-hairline py-3 first:border-t-0 first:pt-0">
              <summary className="flex cursor-pointer list-none items-baseline justify-between gap-3 [&::-webkit-details-marker]:hidden">
                <span>
                  <span className="font-medium">{name}</span> <span className="text-sm text-muted-ink">{why}</span>
                </span>
                <span className="num shrink-0 text-ink-2">
                  {list.length}
                  <span className="ml-2 text-xs text-signal group-open:hidden">show</span>
                  <span className="ml-2 hidden text-xs text-signal group-open:inline">hide</span>
                </span>
              </summary>
              <div className="mt-2 flex flex-wrap gap-1">
                {list.map((f) => (
                  <code key={f} className="rounded-sm bg-raised px-1.5 py-0.5 text-xs text-ink-2">
                    {f}
                  </code>
                ))}
              </div>
            </details>
          ))}
    </div>
  );
}

// --------------------------------------------------------------------------
// Baseline vs engineered vs selected
// --------------------------------------------------------------------------
type Basis = "best" | (typeof MODEL_ORDER)[number];

function ExperimentPanel({ metrics }: { metrics: MetricsResponse }) {
  const [basis, setBasis] = useState<Basis>("best");
  const sizes = metrics.feature_set_sizes;

  const rows: DotRow[] = useMemo(
    () =>
      COMPONENTS.map((c) => {
        const values: Record<string, number | null> = {};
        for (const fs of FEATURE_SET_ORDER) {
          if (basis === "best") {
            const r = metrics.experiment_effect.find((e) => e.component === c);
            values[fs] = r ? r[fs] : null;
          } else {
            const r = metrics.cv.find((x) => x.component === c && x.model === basis && x.feature_set === fs);
            values[fs] = r ? r.f1_macro : null;
          }
        }
        return { key: c, label: COMPONENT_SHORT[c], values };
      }),
    [basis, metrics],
  );

  const all = rows.flatMap((r) => Object.values(r.values)).filter((v): v is number => v !== null);
  const span = 1 - Math.min(...all);
  const step = span > 0.3 ? 0.1 : span > 0.1 ? 0.05 : 0.01;
  const lo = floorTo(all, step);
  const series = FEATURE_SET_ORDER.map((fs, i) => ({
    key: fs,
    label: `${FEATURE_SET_NAMES[fs]} (${sizes[fs] ?? "?"} features)`,
    color: SERIES_HEX[i] ?? SERIES_HEX[0],
  }));

  return (
    <Panel
      className="mt-5"
      title="What the engineered features buy"
      description="Cross-validated macro-F1 on the training set for the three feature sets. A linear model cannot build its own features, so logistic regression shows the effect of the representation most clearly."
      aside={
        <SegmentedControl
          label="Compare using"
          options={[
            { value: "best" as Basis, label: "Best model" },
            ...MODEL_ORDER.map((m) => ({ value: m as Basis, label: MODEL_NAMES[m] ?? m })),
          ]}
          value={basis}
          onChange={setBasis}
        />
      }
    >
      <ChartFigure
        height={4 * 32 + 24}
        legend={<Legend items={series} />}
        caption={`Axis starts at ${lo.toFixed(2)}. "Best model" is the highest-scoring model for each feature set (feature_experiment_effect.csv); the others come from model_results.csv.`}
        table={{
          caption: "Cross-validated macro-F1 per feature set",
          rowKey: (r) => r.key,
          rows,
          columns: [
            { key: "c", header: "Component", cell: (r) => r.label },
            ...FEATURE_SET_ORDER.map((fs) => ({
              key: fs,
              header: FEATURE_SET_NAMES[fs],
              align: "right" as const,
              cell: (r: DotRow) => (r.values[fs] == null ? "n/a" : fmt.fixed(r.values[fs], 4)),
            })),
            {
              key: "gain",
              header: "Best gain vs baseline",
              align: "right" as const,
              cell: (r: DotRow) => {
                const b = r.values["baseline"];
                const best = Math.max(r.values["engineered"] ?? -1, r.values["selected"] ?? -1);
                return b == null || best < 0 ? "n/a" : fmt.signed(best - b, 4);
              },
            },
          ],
        }}
      >
        <DotRows rows={rows} series={series} domain={[lo, 1]} step={step} format={(v) => v.toFixed(2)} />
      </ChartFigure>
    </Panel>
  );
}

// --------------------------------------------------------------------------
// Selected features and importance
// --------------------------------------------------------------------------
function SelectionPanel({ features }: { features: FeaturesResponse }) {
  const [component, setComponent] = useState<ComponentKey>("valve");
  const [kind, setKind] = useState<"impurity" | "permutation">("impurity");
  const importance = useImportance(component, 15);
  const sel = features.selection.find((s) => s.component === component);

  const grouped = useMemo(() => {
    const map = new Map<string, string[]>();
    for (const f of sel?.features ?? []) {
      const k = f.block ?? "other";
      map.set(k, [...(map.get(k) ?? []), f.name]);
    }
    return [...map.entries()].sort((a, b) => b[1].length - a[1].length);
  }, [sel]);

  return (
    <Panel
      className="mt-5"
      title="Features each model actually uses"
      description="Global importance from the saved model, and the features that survive selection, grouped by block."
      aside={
        <SegmentedControl
          label="Component"
          options={COMPONENTS.map((c) => ({ value: c, label: COMPONENT_SHORT[c] }))}
          value={component}
          onChange={setComponent}
        />
      }
    >
      <div className="grid gap-6 lg:grid-cols-2">
        <div className="min-w-0">
          <div className="mb-3 flex flex-wrap items-end justify-between gap-2">
            <h3 className="font-medium">Top 15 features by importance</h3>
            <SegmentedControl
              label="Importance type"
              hideLabel
              options={[
                { value: "impurity" as const, label: "Impurity" },
                { value: "permutation" as const, label: "Permutation" },
              ]}
              value={kind}
              onChange={setKind}
            />
          </div>
          <QueryState
            query={importance}
            isEmpty={(d) => d[kind].length === 0}
            empty="No importance values were saved for this model."
          >
            {(d) => (
              <>
                <BarList
                  ariaLabel={`${kind} importance`}
                  labelClassName="font-mono text-xs"
                  rows={d[kind].map((it) => ({
                    key: it.feature,
                    label: it.feature,
                    value: Math.max(0, it.importance),
                    display: fmt.sig(it.importance, 3),
                    title: it.block ? BLOCK_NAMES[it.block] : undefined,
                  }))}
                />
                <p className="mt-3 text-xs text-muted-ink">
                  {kind === "impurity"
                    ? "Impurity importance: how much each feature's splits reduce impurity inside the trees."
                    : "Permutation importance: the drop in test macro-F1 when the feature is shuffled. Correlated features share credit, so it spreads out on the 553-input accumulator model."}
                </p>
              </>
            )}
          </QueryState>
        </div>
        <div className="min-w-0">
          {sel ? (
            <>
              <h3 className="font-medium">
                {fmt.int(sel.features.length)} inputs reach the{" "}
                {MODEL_NAMES[sel.model]?.toLowerCase() ?? sel.model}
              </h3>
              <p className="mt-0.5 text-sm text-muted-ink">
                {FEATURE_SET_NAMES[sel.feature_set]} feature set, CV macro-F1 {fmt.fixed(sel.cv_f1_macro, 4)}
              </p>
              <div className="mt-3 space-y-3">
                {grouped.map(([block, names]) => (
                  <details key={block} open={names.length <= 20} className="group">
                    <summary className="flex cursor-pointer list-none items-baseline justify-between gap-3 [&::-webkit-details-marker]:hidden">
                      <span className="text-sm font-medium">{BLOCK_NAMES[block] ?? block}</span>
                      <span className="num text-sm text-ink-2">
                        {names.length}
                        {names.length > 20 ? (
                          <>
                            <span className="ml-2 text-xs text-signal group-open:hidden">show</span>
                            <span className="ml-2 hidden text-xs text-signal group-open:inline">hide</span>
                          </>
                        ) : null}
                      </span>
                    </summary>
                    <div className="mt-1.5 flex flex-wrap gap-1">
                      {names.map((n) => (
                        <code key={n} className="rounded-sm bg-raised px-1.5 py-0.5 text-xs text-ink-2">
                          {n}
                        </code>
                      ))}
                    </div>
                  </details>
                ))}
              </div>
            </>
          ) : null}
        </div>
      </div>
    </Panel>
  );
}
