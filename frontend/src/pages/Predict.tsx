import { useMutation } from "@tanstack/react-query";
import { Check, Dices, FileUp, LoaderCircle, ScanSearch, X } from "lucide-react";
import { useEffect, useRef, useState, type DragEvent, type FormEvent } from "react";
import { Link, useSearchParams } from "react-router";
import { BarList } from "@/components/charts/BarList";
import { DataTable } from "@/components/DataTable";
import { PageHeader } from "@/components/PageHeader";
import { Panel } from "@/components/Panel";
import { Empty, ErrorNote } from "@/components/QueryState";
import { RigSchematic } from "@/components/RigSchematic";
import { SegmentedControl } from "@/components/SegmentedControl";
import { StatusBadge } from "@/components/StatusBadge";
import { Button } from "@/components/ui/button";
import { ApiError, api } from "@/lib/api";
import { BLOCK_NAMES, COMPONENT_SHORT, MODEL_NAMES, SEVERITY_COLOR, fmt, valueWithUnit } from "@/lib/format";
import { useOverview, useSensors } from "@/lib/queries";
import type { ComponentKey, ComponentPrediction, PredictionResponse, Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

type Request = { kind: "cycle"; id: number } | { kind: "upload"; file: File };
type Source = "cycle" | "upload";

const MAX_UPLOAD_BYTES = 25 * 1024 * 1024;

export default function PredictPage() {
  const overview = useOverview();
  const nCycles = overview.data?.dataset.n_cycles ?? null;
  const [params, setParams] = useSearchParams();
  const [source, setSource] = useState<Source>("cycle");
  const [draft, setDraft] = useState(params.get("cycle") ?? "0");
  const [file, setFile] = useState<File | null>(null);
  const [fileError, setFileError] = useState<string | null>(null);

  const predict = useMutation({
    mutationFn: (req: Request) => (req.kind === "cycle" ? api.predictCycle(req.id) : api.predictUpload(req.file)),
  });

  const runCycle = (id: number) => {
    setDraft(String(id));
    setParams(
      (prev) => {
        const next = new URLSearchParams(prev);
        next.set("cycle", String(id));
        return next;
      },
      { replace: true },
    );
    predict.mutate({ kind: "cycle", id });
  };

  // A link such as /predict?cycle=499 scores that cycle on arrival.
  const autoRan = useRef(false);
  useEffect(() => {
    const initial = params.get("cycle");
    if (autoRan.current || initial === null || nCycles === null) return;
    autoRan.current = true;
    const id = Number(initial);
    if (Number.isInteger(id) && id >= 0 && id < nCycles) predict.mutate({ kind: "cycle", id });
  }, [params, nCycles, predict]);

  const draftNumber = Number(draft);
  const draftValid =
    nCycles !== null && draft.trim() !== "" && Number.isInteger(draftNumber) && draftNumber >= 0 && draftNumber < nCycles;

  const submitCycle = (e: FormEvent) => {
    e.preventDefault();
    if (draftValid) runCycle(draftNumber);
  };

  const chooseFile = (f: File | null) => {
    setFileError(null);
    setFile(null);
    if (!f) return;
    if (!f.name.toLowerCase().endsWith(".csv")) return setFileError(`"${f.name}" is not a .csv file.`);
    if (f.size === 0) return setFileError(`"${f.name}" is empty.`);
    if (f.size > MAX_UPLOAD_BYTES) return setFileError(`"${f.name}" is larger than 25 MB; one cycle is about 1 MB.`);
    setFile(f);
  };

  const submitUpload = (e: FormEvent) => {
    e.preventDefault();
    if (file) predict.mutate({ kind: "upload", file });
  };

  const pending = predict.isPending;
  const pendingLabel =
    predict.variables?.kind === "upload"
      ? `Reading ${predict.variables.file.name} and extracting features`
      : `Reading cycle ${predict.variables?.id ?? ""} from the raw files and extracting features`;

  return (
    <>
      <PageHeader
        title="Predict"
        takeaway="Score one 60-second cycle with the four saved pipelines. Pick a recorded cycle, roll a random one, or upload a raw-sensor CSV."
      />

      <Panel title="Choose a cycle">
        <SegmentedControl
          label="Input"
          options={[
            { value: "cycle" as Source, label: "Recorded cycle" },
            { value: "upload" as Source, label: "Upload a CSV" },
          ]}
          value={source}
          onChange={setSource}
          disabled={pending}
        />

        {source === "cycle" ? (
          <form onSubmit={submitCycle} noValidate className="mt-4">
            <div className="flex flex-wrap items-end gap-2">
              <div>
                <label htmlFor="predict-cycle" className="nameplate mb-1.5 block">
                  Cycle number{nCycles !== null ? ` (0 to ${nCycles - 1})` : ""}
                </label>
                <input
                  id="predict-cycle"
                  type="number"
                  inputMode="numeric"
                  min={0}
                  max={nCycles !== null ? nCycles - 1 : undefined}
                  value={draft}
                  disabled={pending || nCycles === null}
                  onChange={(e) => setDraft(e.target.value)}
                  aria-invalid={!draftValid}
                  aria-describedby="predict-cycle-error"
                  className="num h-9 w-32 rounded-sm border border-hairline-strong bg-ground px-3 text-ink disabled:opacity-50 aria-[invalid=true]:border-crit"
                />
              </div>
              <Button type="submit" disabled={!draftValid || pending}>
                {pending && predict.variables?.kind === "cycle" ? (
                  <LoaderCircle aria-hidden className="animate-spin motion-reduce:animate-none" />
                ) : (
                  <ScanSearch aria-hidden />
                )}
                Predict this cycle
              </Button>
              <Button
                type="button"
                variant="outline"
                disabled={pending || nCycles === null}
                onClick={() => nCycles !== null && runCycle(Math.floor(Math.random() * nCycles))}
              >
                <Dices aria-hidden />
                Random cycle
              </Button>
            </div>
            <p id="predict-cycle-error" className={cn("mt-1.5 text-sm text-crit", draftValid && "sr-only")}>
              {draftValid ? "" : nCycles !== null ? `Enter a whole number from 0 to ${nCycles - 1}.` : "Loading the cycle count."}
            </p>
          </form>
        ) : (
          <UploadForm
            file={file}
            fileError={fileError}
            onChoose={chooseFile}
            onSubmit={submitUpload}
            pending={pending}
            exampleCycle={draftValid ? draftNumber : 0}
          />
        )}
      </Panel>

      <div className="mt-5" aria-live="polite">
        {pending ? (
          <div role="status" className="flex items-center gap-3 rounded-md border border-hairline bg-plate px-5 py-6 text-ink-2">
            <LoaderCircle aria-hidden className="size-5 animate-spin text-signal motion-reduce:animate-none" />
            {pendingLabel}…
          </div>
        ) : predict.isError ? (
          <PredictError error={predict.error} />
        ) : predict.data ? (
          <Result result={predict.data} />
        ) : (
          <Empty>
            No cycle scored yet. Choose <strong>Predict this cycle</strong>, roll a <strong>random cycle</strong>, or upload
            a CSV to see the condition of all four components.
          </Empty>
        )}
      </div>
    </>
  );
}

function PredictError({ error }: { error: Error }) {
  const isCsv = error instanceof ApiError && error.code === "invalid_csv";
  return (
    <div>
      <ErrorNote error={error} className="whitespace-pre-line" />
      {isCsv ? (
        <p className="mt-2 text-sm text-ink-2">
          Check the column names and sample counts in the format notes above, or start from a downloaded example file.
        </p>
      ) : null}
    </div>
  );
}

// --------------------------------------------------------------------------
// Upload
// --------------------------------------------------------------------------
function UploadForm({
  file,
  fileError,
  onChoose,
  onSubmit,
  pending,
  exampleCycle,
}: {
  file: File | null;
  fileError: string | null;
  onChoose: (f: File | null) => void;
  onSubmit: (e: FormEvent) => void;
  pending: boolean;
  exampleCycle: number;
}) {
  const sensors = useSensors();
  const [dragging, setDragging] = useState(false);
  const onDrop = (e: DragEvent) => {
    e.preventDefault();
    setDragging(false);
    onChoose(e.dataTransfer.files[0] ?? null);
  };

  return (
    <form onSubmit={onSubmit} className="mt-4" noValidate>
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <div>
          <label
            htmlFor="predict-file"
            onDragOver={(e) => {
              e.preventDefault();
              setDragging(true);
            }}
            onDragLeave={() => setDragging(false)}
            onDrop={onDrop}
            className={cn(
              "flex cursor-pointer flex-col items-center justify-center gap-2 rounded-md border border-dashed border-hairline-strong bg-ground px-4 py-8 text-center",
              "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-2 has-[:focus-visible]:outline-signal",
              dragging && "border-signal bg-signal/5",
            )}
          >
            <FileUp aria-hidden className="size-6 text-muted-ink" />
            <span className="font-medium text-ink">{file ? file.name : "Choose a raw-sensor CSV"}</span>
            <span className="text-sm text-muted-ink">
              {file ? `${fmt.int(Math.round(file.size / 1024))} KB, ready to score` : "or drop it here. One cycle per file."}
            </span>
            <input
              id="predict-file"
              type="file"
              accept=".csv,text/csv"
              className="sr-only"
              disabled={pending}
              aria-describedby="predict-file-help predict-file-error"
              onChange={(e) => onChoose(e.target.files?.[0] ?? null)}
            />
          </label>
          {fileError ? (
            <p id="predict-file-error" className="mt-1.5 text-sm text-crit">
              {fileError}
            </p>
          ) : null}
          <div className="mt-3 flex flex-wrap items-center gap-3">
            <Button type="submit" disabled={!file || pending}>
              {pending ? <LoaderCircle aria-hidden className="animate-spin motion-reduce:animate-none" /> : <ScanSearch aria-hidden />}
              Predict from file
            </Button>
            <a href={api.cycleCsvUrl(exampleCycle)} download={`cycle_${exampleCycle}.csv`} className="text-sm">
              Download an example file (cycle {exampleCycle})
            </a>
          </div>
        </div>
        <div id="predict-file-help" className="text-sm text-ink-2">
          <p>
            One column per sensor, named as in the dataset. Each column holds that sensor's samples for one 60-second
            cycle, top to bottom; shorter columns are padded with empty cells.
          </p>
          {sensors.data ? (
            <div className="mt-3">
              <DataTable
                caption="Samples required per sensor"
                rowKey={(r) => String(r.rate)}
                rows={[...new Set(sensors.data.sensors.map((s) => s.sampling_rate_hz))].map((rate) => ({
                  rate,
                  names: sensors.data.sensors.filter((s) => s.sampling_rate_hz === rate).map((s) => s.name),
                  n: sensors.data.sensors.find((s) => s.sampling_rate_hz === rate)?.n_samples ?? 0,
                }))}
                columns={[
                  { key: "names", header: "Columns", cell: (r) => <span className="font-mono text-xs">{r.names.join(", ")}</span> },
                  { key: "rate", header: "Rate", align: "right", cell: (r) => `${r.rate} Hz` },
                  { key: "n", header: "Rows", align: "right", cell: (r) => fmt.int(r.n) },
                ]}
              />
            </div>
          ) : null}
        </div>
      </div>
    </form>
  );
}

// --------------------------------------------------------------------------
// Result
// --------------------------------------------------------------------------
function Result({ result }: { result: PredictionResponse }) {
  const statuses = Object.fromEntries(result.components.map((c) => [c.component, c.severity])) as Record<
    ComponentKey,
    Severity
  >;
  const fromDataset = result.source.kind === "dataset";
  const matches = result.components.filter((c) => c.correct).length;
  const label = fromDataset ? `Cycle ${result.source.cycle_id}` : (result.source.filename ?? "Uploaded cycle");
  const [focus, setFocus] = useState<ComponentKey>(
    result.components.reduce((worst, c) => (sevRank(c.severity) > sevRank(worst.severity) ? c : worst)).component,
  );

  return (
    <div className="space-y-5">
      <section
        aria-labelledby="result-title"
        className="grid gap-5 rounded-md border bg-plate p-4 sm:p-5 xl:grid-cols-[minmax(0,340px)_minmax(0,1fr)]"
        style={{ borderColor: `color-mix(in srgb, ${SEVERITY_COLOR[result.system_status]} 45%, var(--hairline))` }}
      >
        <div>
          <h2 id="result-title" className="nameplate">
            {label}: overall system status
          </h2>
          <div className="mt-2">
            <StatusBadge severity={result.system_status} size="lg" />
          </div>
          <p className="mt-1 text-sm text-ink-2">
            Driven by the {result.driving_component.toLowerCase()}. A system is only as healthy as its weakest component.
          </p>
          <dl className="mt-4 grid grid-cols-2 gap-3 text-sm">
            <div>
              <dt className="text-muted-ink">Mean confidence</dt>
              <dd className="text-lg font-semibold">{result.mean_confidence === null ? "n/a" : fmt.pct(result.mean_confidence)}</dd>
            </div>
            <div>
              <dt className="text-muted-ink">Features extracted</dt>
              <dd className="text-lg font-semibold">{fmt.int(result.n_engineered_features)}</dd>
            </div>
            {fromDataset ? (
              <div className="col-span-2">
                <dt className="text-muted-ink">Agreement with the recorded condition</dt>
                <dd className="text-lg font-semibold">
                  {matches} of {result.components.length} components
                </dd>
              </div>
            ) : null}
          </dl>
          {fromDataset ? (
            <p className="mt-3 text-xs text-muted-ink">
              Most recorded cycles were used in training, so agreement here is not an accuracy estimate. Each card says
              whether this cycle was held out for that model.
            </p>
          ) : (
            <p className="mt-3 text-xs text-muted-ink">An uploaded cycle has no recorded condition to compare against.</p>
          )}
        </div>
        <RigSchematic statuses={statuses} animate />
      </section>

      <ul className="lamp-sequence grid gap-4 lg:grid-cols-2" aria-label="Component predictions">
        {result.components.map((c) => (
          <li key={c.component}>
            <ComponentCard p={c} />
          </li>
        ))}
      </ul>

      <Panel
        title="Why the models decided this"
        description="The features each saved model relies on most, with the value measured in this cycle, and the closest training cycles in the model's own input space."
        aside={
          <SegmentedControl
            label="Component"
            options={result.components.map((c) => ({ value: c.component, label: COMPONENT_SHORT[c.component] }))}
            value={focus}
            onChange={setFocus}
          />
        }
      >
        <Explanation p={result.components.find((c) => c.component === focus) ?? result.components[0]!} fromDataset={fromDataset} />
      </Panel>
    </div>
  );
}

const sevRank = (s: Severity) => ({ Normal: 0, Unknown: 1, Degraded: 2, Critical: 3 })[s];

function ComponentCard({ p }: { p: ComponentPrediction }) {
  return (
    <article
      className="h-full rounded-md border border-hairline bg-plate p-4"
      style={{ boxShadow: `inset 3px 0 0 ${SEVERITY_COLOR[p.severity]}` }}
      aria-labelledby={`card-${p.component}`}
    >
      <header className="flex flex-wrap items-start justify-between gap-2">
        <div>
          <h3 id={`card-${p.component}`} className="font-medium">
            {p.display_name}
          </h3>
          <p className="text-xs text-muted-ink">
            {MODEL_NAMES[p.model] ?? p.model}, {p.n_features} inputs
          </p>
        </div>
        <StatusBadge severity={p.severity} />
      </header>

      <p className="mt-3 text-lg">
        <span className="num font-semibold">{valueWithUnit(p.predicted_value, p.unit)}</span>{" "}
        <span className="text-ink-2">{p.predicted_label}</span>
      </p>

      <dl className="mt-2 grid grid-cols-3 gap-2 text-sm">
        <div>
          <dt className="text-xs text-muted-ink">Confidence</dt>
          <dd className="num font-medium">{p.confidence === null ? "n/a" : fmt.pct(p.confidence)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted-ink">Margin over 2nd</dt>
          <dd className="num font-medium">{p.margin === null ? "n/a" : fmt.fixed(p.margin)}</dd>
        </div>
        <div>
          <dt className="text-xs text-muted-ink">Entropy (0 to 1)</dt>
          <dd className="num font-medium">{p.entropy === null ? "n/a" : fmt.fixed(p.entropy)}</dd>
        </div>
      </dl>

      {p.probabilities.length ? (
        <div className="mt-3">
          <h4 className="sr-only">Class probabilities</h4>
          <BarList
            ariaLabel={`${p.display_name} class probabilities`}
            max={1}
            rows={p.probabilities.map((cp) => ({
              key: String(cp.value),
              label: (
                <span className={cn(cp.value === p.predicted_value ? "font-medium text-ink" : "text-ink-2")}>
                  {valueWithUnit(cp.value, p.unit)} {cp.label}
                  {cp.value === p.predicted_value ? <span className="sr-only"> (predicted)</span> : null}
                </span>
              ),
              value: cp.p,
              display: fmt.pct(cp.p),
              color: SEVERITY_COLOR[cp.severity],
            }))}
          />
        </div>
      ) : (
        <p className="mt-3 text-sm text-muted-ink">This model does not expose probabilities.</p>
      )}

      <div className="mt-3 space-y-1 border-t border-hairline pt-3 text-sm">
        {p.true_value !== null ? (
          <p className="flex flex-wrap items-center gap-x-2">
            {p.correct ? (
              <span className="inline-flex items-center gap-1 font-medium text-ok">
                <Check aria-hidden className="size-4" /> Matches
              </span>
            ) : (
              <span className="inline-flex items-center gap-1 font-medium text-crit">
                <X aria-hidden className="size-4" /> Differs
              </span>
            )}
            <span className="text-ink-2">
              recorded {valueWithUnit(p.true_value, p.unit)} {p.true_label}
            </span>
            <span className="text-muted-ink">({p.in_test_set ? "held-out test cycle" : "training cycle"})</span>
          </p>
        ) : null}
        {p.reliability && p.reliability.n > 0 && p.reliability.accuracy !== null ? (
          <p className="text-muted-ink">
            On held-out cycles scored with {fmt.pct(p.reliability.lo, 0)} to {fmt.pct(p.reliability.hi, 0)} confidence,
            this model was right {fmt.pct(p.reliability.accuracy, 0)} of the time (n = {p.reliability.n}).
          </p>
        ) : (
          <p className="text-muted-ink">No held-out cycle was scored at this confidence, so there is no track record for it.</p>
        )}
      </div>
    </article>
  );
}

function Explanation({ p, fromDataset }: { p: ComponentPrediction; fromDataset: boolean }) {
  const agree = p.neighbours.filter((n) => n.agrees).length;
  const hasImportance = p.explanation.some((e) => e.importance !== null);
  return (
    <div className="grid gap-6 lg:grid-cols-[minmax(0,1.3fr)_minmax(0,1fr)]">
      <div className="min-w-0">
        <h3 className="mb-2 font-medium">Top {p.explanation.length} features for the {p.display_name.toLowerCase()} model</h3>
        {hasImportance ? (
          <BarList
            ariaLabel="Feature importance with this cycle's values"
            labelClassName="font-mono text-xs"
            rows={p.explanation.map((e) => ({
              key: e.feature,
              label: (
                <span>
                  {e.feature}
                  <span className="block font-sans text-xs text-muted-ink">
                    {e.block ? BLOCK_NAMES[e.block] : "feature"}, this cycle: {e.value === null ? "n/a" : fmt.sig(e.value)}
                  </span>
                </span>
              ),
              value: e.importance ?? 0,
              display: e.importance === null ? "n/a" : fmt.sig(e.importance, 3),
            }))}
          />
        ) : (
          <p className="text-sm text-muted-ink">This model type does not expose feature importances.</p>
        )}
        <p className="mt-2 text-xs text-muted-ink">
          Bars show the model's global importance; the value is what this cycle measured. Compare them with the class
          profiles on the <Link to={`/robustness?c=${p.component}`}>Robustness</Link> page.
        </p>
      </div>
      <div className="min-w-0">
        <h3 className="mb-1 font-medium">Nearest training cycles</h3>
        <p className="mb-2 text-sm text-ink-2">
          <span className="num font-medium text-ink">
            {agree} of {p.neighbours.length}
          </span>{" "}
          share the predicted condition.
        </p>
        <DataTable
          caption="Nearest training cycles"
          rowKey={(n) => String(n.cycle)}
          rows={p.neighbours}
          columns={[
            {
              key: "c",
              header: "Cycle",
              cell: (n) => (
                <Link to={`/dataset?cycle=${n.cycle}`} className="num">
                  {n.cycle}
                </Link>
              ),
            },
            { key: "d", header: "Distance", align: "right", cell: (n) => fmt.fixed(n.distance) },
            { key: "t", header: "Recorded", cell: (n) => valueWithUnit(n.true_value, p.unit) },
            {
              key: "a",
              header: "Agrees",
              cell: (n) =>
                n.agrees ? (
                  <span className="inline-flex items-center gap-1 text-ok">
                    <Check aria-hidden className="size-3.5" /> yes
                  </span>
                ) : (
                  <span className="inline-flex items-center gap-1 text-crit">
                    <X aria-hidden className="size-3.5" /> no
                  </span>
                ),
            },
          ]}
        />
        <p className="mt-2 text-xs text-muted-ink">
          Distance is measured after the pipeline's own imputation, scaling and selection.
          {fromDataset ? " Consecutive cycles are near-duplicates, so a recorded cycle's neighbours are usually its recording neighbours." : ""}
        </p>
      </div>
    </div>
  );
}
