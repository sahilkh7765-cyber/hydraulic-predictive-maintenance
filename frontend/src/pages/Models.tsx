import { ComponentPicker } from "@/components/ComponentPicker";
import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { useComponentParam } from "@/hooks/useComponentParam";
import { COMPONENT_SHORT, MODEL_NAMES, fmt } from "@/lib/format";
import { useMetrics, useOverview } from "@/lib/queries";
import type { MetricsResponse, Overview } from "@/lib/types";
import { Arena } from "./models/Arena";
import { Evaluation } from "./models/Evaluation";
import { Generalization } from "./models/Generalization";
import { Pipeline } from "./models/Pipeline";

export default function ModelsPage() {
  const metrics = useMetrics();
  const overview = useOverview();
  return (
    <QueryState query={metrics} loadingLabel="Loading model results">
      {(m) => (
        <QueryState query={overview} loadingLabel="Loading the overview">
          {(o) => <ModelsBody metrics={m} overview={o} />}
        </QueryState>
      )}
    </QueryState>
  );
}

function ModelsBody({ metrics, overview }: { metrics: MetricsResponse; overview: Overview }) {
  const [component] = useComponentParam();
  const unit = overview.components.find((c) => c.key === component)?.unit ?? "";
  const families = [...new Set(metrics.test.map((t) => MODEL_NAMES[t.model]?.toLowerCase() ?? t.model))];
  const hardest = [...metrics.test].sort((a, b) => a.block_cv_f1_macro - b.block_cv_f1_macro)[0];

  return (
    <>
      <PageHeader
        title="Models and evaluation"
        takeaway={
          hardest ? (
            <>
              Of {metrics.cv.length} cross-validated configurations, {families.join(" and ")} win every component. The{" "}
              {COMPONENT_SHORT[hardest.component].toLowerCase()} is the hardest: {fmt.fixed(hardest.f1_macro)} test macro-F1,
              falling to {fmt.fixed(hardest.block_cv_f1_macro)} under blocked validation.
            </>
          ) : (
            "How the four classifiers were chosen and how they perform on held-out cycles."
          )
        }
      />

      <Arena metrics={metrics} />

      <div className="sticky top-[65px] z-20 -mx-4 mt-8 mb-5 border-y border-hairline bg-ground/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6 lg:top-0 lg:-mx-10 lg:px-10">
        <div className="flex flex-wrap items-end justify-between gap-3">
          <div>
            <h2 className="text-xl font-semibold">{COMPONENT_SHORT[component]} model on held-out cycles</h2>
            <p className="text-sm text-muted-ink">Everything below is scoped to this component.</p>
          </div>
          <ComponentPicker />
        </div>
      </div>

      <div className="space-y-5">
        <Evaluation component={component} unit={unit} />
        <Pipeline component={component} unit={unit} />
        <Generalization component={component} unit={unit} nCycles={overview.dataset.n_cycles} />
      </div>
    </>
  );
}
