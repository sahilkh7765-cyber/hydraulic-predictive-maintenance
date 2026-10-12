import { Link } from "react-router";
import { DataTable, type Column } from "@/components/DataTable";
import { PageHeader, Readout, ReadoutStrip } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { RigSchematic } from "@/components/RigSchematic";
import { StatusBadge } from "@/components/StatusBadge";
import { FEATURE_SET_NAMES, MODEL_NAMES, fmt, valueWithUnit } from "@/lib/format";
import { useOverview, useSensors } from "@/lib/queries";
import type { Headline, Overview } from "@/lib/types";
import { cn } from "@/lib/utils";

export function OverviewPage() {
  const overview = useOverview();
  return (
    <QueryState query={overview} loadingLabel="Loading the overview">
      {(data) => <OverviewBody data={data} />}
    </QueryState>
  );
}

function OverviewBody({ data }: { data: Overview }) {
  const { dataset, feature_counts: fc, headline, run } = data;
  const tests = headline.map((h) => h.test_f1_macro);
  const blocks = headline.map((h) => h.block_cv_f1_macro);
  const range = (xs: number[]) => {
    const lo = Math.min(...xs);
    const hi = Math.max(...xs);
    return lo === hi ? fmt.fixed(lo) : `${fmt.fixed(lo)} to ${fmt.fixed(hi)}`;
  };

  return (
    <>
      <PageHeader
        title="Hydraulic rig condition monitor"
        takeaway={
          <>
            Four classifiers read one {dataset.cycle_duration_s}-second operating cycle and report the condition of the
            cooler, valve, pump and accumulator. On held-out test cycles they score {range(tests)} macro-F1; the
            realistic estimate, with whole blocks of {run.block_size} consecutive cycles held out, is {range(blocks)}.
          </>
        }
      >
        <ReadoutStrip>
          <Readout label="Operating cycles" value={fmt.int(dataset.n_cycles)} hint={`${dataset.cycle_duration_s} s each`} />
          <Readout
            label="Sensors"
            value={dataset.n_sensors}
            hint={`${dataset.n_sensors - dataset.n_virtual_sensors} physical, ${dataset.n_virtual_sensors} computed`}
          />
          <Readout
            label="Raw values per cycle"
            value={fmt.int(dataset.raw_values_per_cycle)}
            hint={`at ${dataset.sampling_rates_hz.join(", ")} Hz`}
          />
          <Readout
            label="Engineered features"
            value={fmt.int(fc.engineered)}
            hint={`${Math.round(dataset.raw_values_per_cycle / fc.engineered)}× fewer than raw values`}
          />
        </ReadoutStrip>
      </PageHeader>

      <div className="grid gap-5 xl:grid-cols-[minmax(0,1.65fr)_minmax(0,1fr)]">
        <Panel
          title="The rig"
          description="Four components are deliberately degraded to documented levels while the rig repeats a constant load cycle."
        >
          <RigSchematic />
          <p className="mt-2 text-xs text-muted-ink">
            Simplified drawing for orientation; the rig's exact plumbing and sensor positions are in the dataset's
            documentation.
          </p>
          <SensorSummary />
        </Panel>

        <Panel title="What each model reports" description="Condition levels exactly as documented in the dataset.">
          <ul className="space-y-4">
            {data.components.map((c) => (
              <li key={c.key}>
                <h3 className="font-medium text-ink">{c.display_name}</h3>
                <ul className="mt-1.5 space-y-1">
                  {c.classes.map((cls) => (
                    <li key={cls.value} className="flex flex-wrap items-baseline justify-between gap-x-3 text-sm">
                      <span className="text-ink-2">
                        <span className="num text-ink">{valueWithUnit(cls.value, c.unit)}</span> {cls.label}
                      </span>
                      <StatusBadge severity={cls.severity} />
                    </li>
                  ))}
                </ul>
              </li>
            ))}
          </ul>
        </Panel>
      </div>

      <Panel
        className="mt-5"
        title="Headline scores per component"
        description={
          <>
            The test score comes from a random 80/20 split and is an upper bound: neighbouring cycles are near-duplicates.
            The blocked score holds out whole runs of {run.block_size} cycles and is the one to quote.
          </>
        }
      >
        <ScoreDumbbell rows={headline} />
        <div className="mt-4">
          <DataTable caption="Headline scores per component" columns={HEADLINE_COLUMNS} rows={headline} rowKey={(r) => r.component} />
        </div>
        <p className="mt-3 text-sm text-ink-2">
          Details are on <Link to="/models">Models</Link>; how the models fail is on{" "}
          <Link to="/robustness">Robustness</Link>.
        </p>
      </Panel>

      <div className="mt-5 grid gap-5 md:grid-cols-2">
        <Panel title="How the scores were produced">
          <dl className="grid grid-cols-[auto_1fr] gap-x-6 gap-y-2 text-sm">
            <dt className="text-muted-ink">Configurations compared</dt>
            <dd className="num">
              {run.n_configurations} ({run.models.length} models × {run.feature_sets.length} feature sets ×{" "}
              {data.components.length} targets)
            </dd>
            <dt className="text-muted-ink">Train / test cycles</dt>
            <dd className="num">
              {fmt.int(run.n_train_cycles)} / {fmt.int(run.n_test_cycles)} ({fmt.pct(run.test_size, 0)} held out)
            </dd>
            <dt className="text-muted-ink">Cross-validation</dt>
            <dd className="num">
              {run.cv_folds}-fold stratified, plus blocked by {run.block_size} cycles
            </dd>
            <dt className="text-muted-ink">Random state</dt>
            <dd className="num">{run.random_state}</dd>
            <dt className="text-muted-ink">Boosting backend</dt>
            <dd>{run.xgboost_available ? "XGBoost" : "HistGradientBoosting"}</dd>
          </dl>
        </Panel>
        <Panel title="What is deliberately not claimed">
          <p className="prose-measure text-sm text-ink-2">
            The dataset labels a condition level for every cycle. It contains no maintenance timestamps and no failure
            times, so remaining useful life cannot be computed and is not reported anywhere in this app. Every number
            shown comes from the saved models, the saved evaluation outputs, or a live prediction.
          </p>
        </Panel>
      </div>
    </>
  );
}

/** What the rig measures, grouped by quantity, straight from the sensor specification. */
function SensorSummary() {
  const sensors = useSensors();
  return (
    <div className="mt-5 border-t border-hairline pt-4">
      <h3 className="nameplate mb-2">What the rig measures</h3>
      <QueryState query={sensors} loadingLabel="Loading sensors">
        {(data) => (
          <ul className="grid gap-x-6 gap-y-1.5 text-sm sm:grid-cols-2">
            {data.groups.map((g) => {
              const members = data.sensors.filter((s) => g.sensors.includes(s.name));
              const first = members[0];
              if (!first) return null;
              const rates = [...new Set(members.map((s) => s.sampling_rate_hz))].join(" / ");
              const units = [...new Set(members.map((s) => s.unit))].join(", ");
              return (
                <li key={g.key} className="flex items-baseline justify-between gap-3">
                  <span>
                    <span className="text-ink">{g.label}</span>{" "}
                    <span className="font-mono text-xs text-muted-ink">{g.sensors.join(" ")}</span>
                  </span>
                  <span className="num shrink-0 text-xs text-muted-ink">
                    {units}, {rates} Hz
                  </span>
                </li>
              );
            })}
          </ul>
        )}
      </QueryState>
    </div>
  );
}

const HEADLINE_COLUMNS: readonly Column<Headline>[] = [
  { key: "component", header: "Component", cell: (r) => <span className="font-medium">{r.display_name}</span> },
  {
    key: "model",
    header: "Model",
    cell: (r) => (
      <span className="whitespace-nowrap">
        {MODEL_NAMES[r.model] ?? r.model}
        <span className="text-muted-ink">
          , {FEATURE_SET_NAMES[r.feature_set]?.toLowerCase() ?? r.feature_set} set, {r.n_features} inputs
        </span>
      </span>
    ),
  },
  { key: "test", header: "Test macro-F1", align: "right", cell: (r) => fmt.fixed(r.test_f1_macro) },
  { key: "block", header: "Blocked CV macro-F1", align: "right", cell: (r) => fmt.fixed(r.block_cv_f1_macro) },
  { key: "gap", header: "Random-split optimism", align: "right", cell: (r) => fmt.signed(r.optimism) },
  {
    key: "auc",
    header: "Test ROC-AUC",
    align: "right",
    cell: (r) => (r.test_roc_auc === null ? "n/a" : fmt.fixed(r.test_roc_auc, 4)),
  },
];

/** Blocked CV (hollow) to test score (filled) per component on one shared axis. */
function ScoreDumbbell({ rows }: { rows: readonly Headline[] }) {
  const lowest = Math.min(...rows.flatMap((r) => [r.block_cv_f1_macro, r.test_f1_macro]));
  const floor = Math.max(0, Math.floor(lowest * 20) / 20);
  const x = (v: number) => `${((v - floor) / (1 - floor)) * 100}%`;
  const ticks = Array.from({ length: Math.round((1 - floor) / 0.025) + 1 }, (_, i) => floor + i * 0.025);

  return (
    <figure>
      <div className="mb-3 flex flex-wrap gap-x-5 gap-y-1 text-sm text-ink-2" aria-hidden>
        <span className="inline-flex items-center gap-2">
          <span className="size-3 rounded-full border-2 border-[var(--series-1)] bg-plate" />
          Blocked CV (realistic)
        </span>
        <span className="inline-flex items-center gap-2">
          <span className="size-3 rounded-full bg-[var(--series-1)]" />
          Test, random split (upper bound)
        </span>
      </div>
      <div className="space-y-1" aria-hidden>
        {rows.map((r) => (
          <div key={r.component} className="grid grid-cols-[minmax(92px,150px)_1fr] items-center gap-3">
            <span className="truncate text-sm">{r.display_name}</span>
            <div className="relative h-8">
              {ticks.map((t) => (
                <span key={t} className="absolute top-0 bottom-0 w-px bg-hairline" style={{ left: x(t) }} />
              ))}
              <span
                className="absolute top-1/2 h-0.5 -translate-y-1/2 bg-[var(--series-1)]"
                style={{ left: x(r.block_cv_f1_macro), width: `calc(${x(r.test_f1_macro)} - ${x(r.block_cv_f1_macro)})` }}
              />
              <span
                className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full border-2 border-[var(--series-1)] bg-plate"
                style={{ left: x(r.block_cv_f1_macro) }}
                title={`Blocked CV ${fmt.fixed(r.block_cv_f1_macro)}`}
              />
              <span
                className="absolute top-1/2 size-3.5 -translate-x-1/2 -translate-y-1/2 rounded-full bg-[var(--series-1)] ring-2 ring-plate"
                style={{ left: x(r.test_f1_macro) }}
                title={`Test ${fmt.fixed(r.test_f1_macro)}`}
              />
            </div>
          </div>
        ))}
        <div className="grid grid-cols-[minmax(92px,150px)_1fr] gap-3">
          <span />
          <div className="relative h-5">
            {ticks.map((t, i) => (
              <span
                key={t}
                className={cn(
                  "num absolute top-1 text-xs text-muted-ink",
                  i === ticks.length - 1 ? "-translate-x-full" : i === 0 ? "" : "-translate-x-1/2",
                  i % 2 === 1 && "hidden sm:inline",
                )}
                style={{ left: x(t) }}
              >
                {t.toFixed(3)}
              </span>
            ))}
          </div>
        </div>
      </div>
      <figcaption className="mt-1 text-xs text-muted-ink">
        Macro-F1; the axis starts at {floor.toFixed(2)} to make the gaps visible. Values are in the table below.
      </figcaption>
    </figure>
  );
}
