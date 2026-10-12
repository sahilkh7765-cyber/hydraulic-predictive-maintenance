import { ChevronLeft, ChevronRight, Dices, Download } from "lucide-react";
import { useMemo, useState, type FormEvent } from "react";
import { useSearchParams } from "react-router";
import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { AXIS, GRID, TooltipBox, axisLabel } from "@/components/charts/ChartKit";
import { DataTable, type Column } from "@/components/DataTable";
import { PageHeader, Readout, ReadoutStrip } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { ErrorNote, Loading, QueryState } from "@/components/QueryState";
import { SegmentedControl } from "@/components/SegmentedControl";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { api } from "@/lib/api";
import { COMPONENTS, COMPONENT_SHORT, SEVERITY_COLOR, fmt, valueWithUnit } from "@/lib/format";
import { useCycle, useLabels, useSensors } from "@/lib/queries";
import type { CycleResponse, LabelsResponse, Sensor, SensorsResponse, Signal } from "@/lib/types";

export default function DatasetPage() {
  const labels = useLabels();
  const sensors = useSensors();
  return (
    <QueryState query={labels} loadingLabel="Loading the dataset summary">
      {(l) => (
        <QueryState query={sensors} loadingLabel="Loading sensors">
          {(s) => <DatasetBody labels={l} sensors={s} />}
        </QueryState>
      )}
    </QueryState>
  );
}

function DatasetBody({ labels, sensors }: { labels: LabelsResponse; sensors: SensorsResponse }) {
  const n = labels.n_cycles;
  const rawPerCycle = sensors.sensors.reduce((sum, s) => sum + s.n_samples, 0);
  const worst = [...labels.components].sort((a, b) => b.imbalance_ratio - a.imbalance_ratio)[0];
  const stable = labels.stable_flag["stable"] ?? 0;

  return (
    <>
      <PageHeader
        title="Dataset"
        takeaway={
          <>
            {fmt.int(n)} recorded cycles, each {fmt.int(rawPerCycle)} raw values from {sensors.sensors.length} sensors
            sampled at three different rates.
            {worst ? (
              <>
                {" "}
                Class balance is uneven: the {worst.display_name.toLowerCase()} has an imbalance ratio of{" "}
                {fmt.fixed(worst.imbalance_ratio, 2)}, which is why macro-F1 is the headline metric.
              </>
            ) : null}
          </>
        }
      >
        <ReadoutStrip>
          <Readout label="Cycles" value={fmt.int(n)} hint="rows in every sensor file" />
          <Readout label="Sensors" value={sensors.sensors.length} hint={`${sensors.groups.length} groups`} />
          <Readout label="Raw values per cycle" value={fmt.int(rawPerCycle)} hint="before feature engineering" />
          <Readout label="Stable cycles" value={fmt.pct(stable / n)} hint={`${fmt.int(stable)} of ${fmt.int(n)} reached steady state`} />
        </ReadoutStrip>
      </PageHeader>

      <CycleExplorer nCycles={n} sensors={sensors} units={Object.fromEntries(labels.components.map((c) => [c.component, c.unit]))} />

      <Panel
        className="mt-5"
        title="Condition labels across all cycles"
        description="How many cycles carry each documented condition. Macro-F1 weights every class equally, so a rare class counts as much as a common one."
      >
        <div className="grid gap-x-8 gap-y-6 md:grid-cols-2">
          {labels.components.map((c) => {
            const max = Math.max(...c.counts.map((x) => x.count));
            return (
              <div key={c.component}>
                <div className="mb-2 flex items-baseline justify-between gap-3">
                  <h3 className="font-medium">{c.display_name}</h3>
                  <span className="num text-xs text-muted-ink">imbalance {fmt.fixed(c.imbalance_ratio, 2)}</span>
                </div>
                <ul className="space-y-1.5">
                  {c.counts.map((x) => (
                    <li key={x.value} className="grid grid-cols-[minmax(0,1fr)_auto] items-center gap-x-3 gap-y-1 text-sm sm:grid-cols-[minmax(0,1fr)_minmax(80px,1.4fr)_auto]">
                      <span className="min-w-0">
                        <span className="num">{valueWithUnit(x.value, c.unit)}</span>{" "}
                        <span className="text-muted-ink">{x.label}</span>
                      </span>
                      <span className="order-last col-span-2 h-2 rounded-full bg-raised sm:order-none sm:col-span-1" aria-hidden>
                        <span
                          className="block h-full rounded-full"
                          style={{ width: `${(x.count / max) * 100}%`, background: SEVERITY_COLOR[x.severity] }}
                        />
                      </span>
                      <span className="num w-28 text-right text-ink-2">
                        {fmt.int(x.count)} <span className="text-muted-ink">({fmt.pct(x.share)})</span>
                      </span>
                    </li>
                  ))}
                </ul>
              </div>
            );
          })}
        </div>
        <p className="mt-4 text-xs text-muted-ink">
          Bar colour follows the documented severity: green normal, amber degraded, red critical. The stable flag
          ({fmt.int(labels.stable_flag["not_stable"] ?? 0)} cycles may not have reached static conditions) describes the
          rig, not a component, so it is metadata rather than a prediction target.
        </p>
      </Panel>

      <Panel className="mt-5" title="Sensors and sampling rates" description="Computed channels (CE, CP, SE) are derived by the rig rather than measured.">
        <DataTable caption="Sensors" columns={SENSOR_COLUMNS} rows={sensors.sensors} rowKey={(s) => s.name} />
      </Panel>
    </>
  );
}

const SENSOR_COLUMNS: readonly Column<Sensor>[] = [
  { key: "name", header: "Sensor", cell: (s) => <span className="font-mono">{s.name}</span> },
  { key: "q", header: "Quantity", cell: (s) => s.quantity },
  { key: "unit", header: "Unit", cell: (s) => s.unit },
  { key: "rate", header: "Rate", align: "right", cell: (s) => `${s.sampling_rate_hz} Hz` },
  { key: "n", header: "Samples per cycle", align: "right", cell: (s) => fmt.int(s.n_samples) },
  { key: "v", header: "Kind", cell: (s) => (s.virtual ? "computed" : "physical") },
];

// --------------------------------------------------------------------------
// Cycle explorer
// --------------------------------------------------------------------------
function CycleExplorer({
  nCycles,
  sensors,
  units,
}: {
  nCycles: number;
  sensors: SensorsResponse;
  units: Record<string, string>;
}) {
  const [params, setParams] = useSearchParams();
  const parsed = Number.parseInt(params.get("cycle") ?? "0", 10);
  const cycleId = Number.isFinite(parsed) && parsed >= 0 && parsed < nCycles ? parsed : 0;
  const [draft, setDraft] = useState(String(cycleId));
  const [group, setGroup] = useState(sensors.groups[0]?.key ?? "pressure");
  const cycle = useCycle(cycleId);

  const go = (id: number) => {
    const clamped = Math.min(Math.max(0, id), nCycles - 1);
    setDraft(String(clamped));
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set("cycle", String(clamped));
        return next;
      },
      { replace: true },
    );
  };

  const draftNumber = Number(draft);
  const draftValid = draft.trim() !== "" && Number.isInteger(draftNumber) && draftNumber >= 0 && draftNumber < nCycles;
  const submit = (e: FormEvent) => {
    e.preventDefault();
    if (draftValid) go(draftNumber);
  };

  return (
    <Panel
      title="One cycle, sensor by sensor"
      description="Pick a cycle to see its raw signals and the conditions recorded for it."
      aside={
        <a
          className="inline-flex items-center gap-1.5 text-sm"
          href={api.cycleCsvUrl(cycleId)}
          download={`cycle_${cycleId}.csv`}
        >
          <Download aria-hidden className="size-4" />
          Download cycle {cycleId} as CSV
        </a>
      }
    >
      <form onSubmit={submit} className="flex flex-wrap items-end gap-2" noValidate>
        <div>
          <label htmlFor="cycle-input" className="nameplate mb-1.5 block">
            Cycle number (0 to {nCycles - 1})
          </label>
          <input
            id="cycle-input"
            type="number"
            inputMode="numeric"
            min={0}
            max={nCycles - 1}
            value={draft}
            onChange={(e) => setDraft(e.target.value)}
            aria-invalid={!draftValid}
            aria-describedby={!draftValid ? "cycle-input-error" : undefined}
            className="num h-9 w-32 rounded-sm border border-hairline-strong bg-ground px-3 text-ink aria-[invalid=true]:border-crit"
          />
        </div>
        <Button type="submit" disabled={!draftValid || cycle.isFetching}>
          Show cycle
        </Button>
        <Button type="button" variant="outline" size="icon" aria-label="Previous cycle" disabled={cycleId <= 0} onClick={() => go(cycleId - 1)}>
          <ChevronLeft aria-hidden />
        </Button>
        <Button
          type="button"
          variant="outline"
          size="icon"
          aria-label="Next cycle"
          disabled={cycleId >= nCycles - 1}
          onClick={() => go(cycleId + 1)}
        >
          <ChevronRight aria-hidden />
        </Button>
        <Button type="button" variant="outline" onClick={() => go(Math.floor(Math.random() * nCycles))}>
          <Dices aria-hidden />
          Random cycle
        </Button>
      </form>
      {!draftValid ? (
        <p id="cycle-input-error" className="mt-1.5 text-sm text-crit">
          Enter a whole number from 0 to {nCycles - 1}.
        </p>
      ) : null}

      <div className="mt-5">
        {cycle.isPending ? (
          <Loading label={`Reading cycle ${cycleId} from the raw files`} />
        ) : cycle.isError && !cycle.data ? (
          <ErrorNote error={cycle.error} onRetry={() => void cycle.refetch()} />
        ) : cycle.data ? (
          <div className={cycle.isFetching ? "opacity-50 transition-opacity" : undefined} aria-busy={cycle.isFetching}>
            <CycleFacts cycle={cycle.data} units={units} />
            <div className="mt-5 flex flex-wrap items-end justify-between gap-3">
              <SegmentedControl
                label="Sensor group"
                options={sensors.groups.map((g) => ({ value: g.key, label: g.label }))}
                value={group}
                onChange={setGroup}
              />
            </div>
            <SignalGrid signals={cycle.data.signals.filter((s) => s.group === group)} />
          </div>
        ) : null}
      </div>
    </Panel>
  );
}

function CycleFacts({ cycle, units }: { cycle: CycleResponse; units: Record<string, string> }) {
  return (
    <div>
      <h3 className="nameplate mb-2">Recorded condition of cycle {cycle.cycle_id}</h3>
      <ul className="grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
        {COMPONENTS.map((c) => {
          const l = cycle.labels[c];
          if (!l) return null;
          return (
            <li key={c} className="rounded-sm border border-hairline bg-ground px-3 py-2">
              <div className="flex items-center justify-between gap-2">
                <span className="text-sm text-ink-2">{COMPONENT_SHORT[c]}</span>
                <StatusBadge severity={l.severity} />
              </div>
              <div className="mt-0.5 text-sm">
                <span className="num font-medium">{valueWithUnit(l.value, units[c] ?? "")}</span>{" "}
                <span className="text-muted-ink">{l.label}</span>
              </div>
              <div className="mt-0.5 text-xs text-muted-ink">
                {cycle.split[c] === "test" ? "held-out test cycle" : "training cycle"} for this model
              </div>
            </li>
          );
        })}
      </ul>
      <p className="mt-2 text-xs text-muted-ink">
        Stable flag {cycle.stable_flag ?? "n/a"}:{" "}
        {cycle.stable_flag === 0 ? "conditions were stable." : "static conditions might not have been reached."}
      </p>
    </div>
  );
}

function SignalGrid({ signals }: { signals: readonly Signal[] }) {
  const [asTable, setAsTable] = useState(false);
  return (
    <div className="mt-4">
      <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
        <p className="text-xs text-muted-ink">
          Each sensor has its own vertical scale. 100 Hz channels are drawn as a min/max envelope (600 points) so short
          spikes stay visible.
        </p>
        <button
          type="button"
          aria-pressed={asTable}
          onClick={() => setAsTable((v) => !v)}
          className="rounded-sm border border-hairline-strong px-2 py-1 text-xs text-ink-2 hover:text-ink"
        >
          {asTable ? "Show charts" : "Show as table"}
        </button>
      </div>
      {asTable ? (
        <DataTable
          caption="Signal summary for this cycle"
          rowKey={(s) => s.sensor}
          rows={signals}
          columns={[
            { key: "s", header: "Sensor", cell: (s) => <span className="font-mono">{s.sensor}</span> },
            { key: "q", header: "Quantity", cell: (s) => `${s.quantity} [${s.unit}]` },
            { key: "n", header: "Samples", align: "right", cell: (s) => fmt.int(s.n_samples) },
            { key: "mean", header: "Mean", align: "right", cell: (s) => fmt.sig(s.mean) },
            { key: "min", header: "Min", align: "right", cell: (s) => fmt.sig(s.min) },
            { key: "max", header: "Max", align: "right", cell: (s) => fmt.sig(s.max) },
          ]}
        />
      ) : (
        <div className="grid gap-4 sm:grid-cols-2 xl:grid-cols-3">
          {signals.map((s) => (
            <SignalChart key={s.sensor} signal={s} />
          ))}
        </div>
      )}
    </div>
  );
}

function SignalChart({ signal }: { signal: Signal }) {
  const data = useMemo(() => signal.t.map((t, i) => ({ t, v: signal.v[i] ?? null })), [signal]);
  return (
    <figure className="min-w-0 rounded-sm border border-hairline bg-ground p-3">
      <figcaption className="mb-1 flex items-baseline justify-between gap-2">
        <span>
          <span className="font-mono text-sm text-ink">{signal.sensor}</span>{" "}
          <span className="text-xs text-muted-ink">
            {signal.quantity}, {signal.sampling_rate_hz} Hz
          </span>
        </span>
        <span className="num text-xs text-muted-ink">mean {fmt.sig(signal.mean)} {signal.unit}</span>
      </figcaption>
      <div className="h-40">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 8, bottom: 14, left: 0 }}>
            <CartesianGrid {...GRID} />
            <XAxis
              {...AXIS}
              dataKey="t"
              type="number"
              domain={[0, 60]}
              ticks={[0, 15, 30, 45, 60]}
              label={axisLabel("time in cycle [s]", "x")}
            />
            <YAxis {...AXIS} width={52} domain={["auto", "auto"]} tickCount={4} interval={0} tickFormatter={(v: number) => fmt.sig(v, 3)} />
            <Tooltip
              cursor={{ stroke: "var(--hairline-strong)" }}
              content={({ active, payload }) => {
                const p = payload?.[0];
                if (!active || !p) return null;
                const point = p.payload as { t: number; v: number };
                return (
                  <TooltipBox
                    title={`${point.t.toFixed(2)} s`}
                    rows={[{ label: signal.sensor, value: `${fmt.sig(point.v)} ${signal.unit}`, color: "var(--series-1)" }]}
                  />
                );
              }}
            />
            <Line type="linear" dataKey="v" stroke="var(--series-1)" strokeWidth={1.5} dot={false} isAnimationActive={false} />
          </LineChart>
        </ResponsiveContainer>
      </div>
      <p className="sr-only">
        {signal.sensor} ranges from {fmt.sig(signal.min)} to {fmt.sig(signal.max)} {signal.unit} over the cycle.
      </p>
    </figure>
  );
}
