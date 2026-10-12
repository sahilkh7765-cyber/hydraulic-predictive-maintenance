import { BarList } from "@/components/charts/BarList";
import { DataTable } from "@/components/DataTable";
import { Panel } from "@/components/Panel";
import { QueryState } from "@/components/QueryState";
import { FEATURE_SET_NAMES, MODEL_NAMES, fmt, valueWithUnit } from "@/lib/format";
import { usePipeline } from "@/lib/queries";
import type { ComponentKey } from "@/lib/types";

export function Pipeline({ component, unit }: { component: ComponentKey; unit: string }) {
  const pipeline = usePipeline(component);
  return (
    <Panel
      title="Inside the saved pipeline"
      description={`What models/${component}/model.joblib contains, read from the fitted object. Every step is re-fitted inside each cross-validation fold, so no validation cycle leaks into training.`}
    >
      <QueryState query={pipeline} loadingLabel="Reading the pipeline">
        {(p) => (
          <div className="space-y-5">
            <ol className="flex flex-wrap items-stretch gap-2" aria-label="Pipeline steps in order">
              {p.steps.map((s, i) => (
                <li key={s.name} className="flex min-w-0 grow basis-36 items-start gap-2 rounded-sm border border-hairline bg-ground px-3 py-2">
                  <span className="num mt-0.5 text-xs text-muted-ink">{i + 1}</span>
                  <span className="min-w-0">
                    <span className="block font-medium">{s.name.replaceAll("_", " ")}</span>
                    <span className="block font-mono text-xs break-all text-muted-ink">{s.type}</span>
                    {s.n_in !== null ? (
                      <span className="num block text-xs text-ink-2">
                        {s.n_out !== null && s.n_out !== s.n_in ? `${s.n_in} → ${s.n_out} features` : `${s.n_in} features`}
                      </span>
                    ) : null}
                  </span>
                </li>
              ))}
            </ol>

            <div className="grid gap-6 lg:grid-cols-3">
              <div>
                <h3 className="mb-2 font-medium">
                  {MODEL_NAMES[p.model] ?? p.model}, {FEATURE_SET_NAMES[p.feature_set]?.toLowerCase()} set
                </h3>
                <dl className="grid grid-cols-[auto_1fr] gap-x-4 gap-y-1 text-sm">
                  <dt className="text-muted-ink">Estimator</dt>
                  <dd className="font-mono text-xs leading-5">{p.estimator}</dd>
                  {p.params.map((x) => (
                    <div key={x.key} className="contents">
                      <dt className="text-muted-ink">{x.key.replaceAll("_", " ")}</dt>
                      <dd className="num font-mono text-xs leading-5 break-all">{x.value}</dd>
                    </div>
                  ))}
                  <dt className="text-muted-ink">CV macro-F1</dt>
                  <dd className="num">
                    {fmt.fixed(p.cv_f1_macro, 4)} ± {fmt.fixed(p.cv_f1_macro_std, 4)}
                  </dd>
                  <dt className="text-muted-ink">Blocked CV macro-F1</dt>
                  <dd className="num">{fmt.fixed(p.block_cv_f1_macro, 4)}</dd>
                </dl>
              </div>

              <div>
                <h3 className="mb-2 font-medium">Feature funnel inside the pipeline</h3>
                <BarList
                  ariaLabel="Features remaining after each selection step"
                  rows={p.funnel.map((f) => ({ key: f.stage, label: f.stage, value: f.features, display: fmt.int(f.features) }))}
                />
              </div>

              <div>
                <h3 className="mb-2 font-medium">Class weights on the training set</h3>
                <DataTable
                  caption="Class weights"
                  rowKey={(r) => String(r.value)}
                  rows={p.class_weights}
                  columns={[
                    { key: "c", header: "Condition", cell: (r) => valueWithUnit(r.value, unit) },
                    { key: "n", header: "Cycles", align: "right", cell: (r) => fmt.int(r.count) },
                    { key: "w", header: "Weight", align: "right", cell: (r) => fmt.fixed(r.weight, 2) },
                  ]}
                />
                <p className="mt-2 text-xs text-muted-ink">
                  Imbalance ratio {fmt.fixed(p.imbalance_ratio, 2)}.{" "}
                  {p.balanced ? "Inverse-frequency weights, w = N / (K × N_c)." : "Classes are near-uniform, so no weighting is applied."}
                </p>
              </div>
            </div>
            <p className="text-xs text-muted-ink">Selection rule: {p.selection_rule}.</p>
          </div>
        )}
      </QueryState>
    </Panel>
  );
}
