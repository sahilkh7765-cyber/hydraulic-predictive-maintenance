import { ComponentPicker } from "@/components/ComponentPicker";
import { PageHeader } from "@/components/PageHeader";
import { QueryState } from "@/components/QueryState";
import { useComponentParam } from "@/hooks/useComponentParam";
import { COMPONENT_SHORT, fmt } from "@/lib/format";
import { useNoise, useOverview, useSelective } from "@/lib/queries";
import type { Overview } from "@/lib/types";
import { ErrorAnalysis } from "./robustness/ErrorAnalysis";
import { NoiseTest, Selective } from "./robustness/Stress";
import { WhatIf } from "./robustness/WhatIf";

export default function RobustnessPage() {
  const overview = useOverview();
  return (
    <QueryState query={overview} loadingLabel="Loading">
      {(o) => <RobustnessBody overview={o} />}
    </QueryState>
  );
}

function RobustnessBody({ overview }: { overview: Overview }) {
  const [component] = useComponentParam();
  const unit = overview.components.find((c) => c.key === component)?.unit ?? "";
  const noise = useNoise(component);
  const selective = useSelective(component);
  const at = (lvl: number) => noise.data?.points.find((p) => p.level === lvl)?.f1_macro;
  const sel60 = selective.data?.grid.find((g) => Math.abs(g.threshold - 0.6) < 1e-9);
  const name = COMPONENT_SHORT[component].toLowerCase();

  const clean = at(0);
  const moderate = at(0.3);
  return (
    <>
      <PageHeader
        title="Robustness and errors"
        takeaway={
          clean !== undefined && moderate !== undefined ? (
            <>
              The {name} model scores {fmt.fixed(clean)} macro-F1 on clean test cycles and {fmt.fixed(moderate)} with noise
              at 0.3 × each feature's spread.
              {sel60 && selective.data
                ? ` Deferring predictions below 60% confidence would hand ${selective.data.n_test - sel60.answered} of ${selective.data.n_test} cycles to an engineer${selective.data.n_errors === 0 ? "; this model has no test errors to catch." : ` and leave ${sel60.errors_answered} of ${selective.data.n_errors} errors answered.`}`
                : ""}
            </>
          ) : (
            `How the ${name} model fails, how sensitive it is to single inputs, and how it holds up under sensor noise.`
          )
        }
      />

      <div className="sticky top-[65px] z-20 -mx-4 mb-5 border-y border-hairline bg-ground/95 px-4 py-3 backdrop-blur sm:-mx-6 sm:px-6 lg:top-0 lg:-mx-10 lg:px-10">
        <ComponentPicker />
      </div>

      <div className="space-y-5">
        <ErrorAnalysis component={component} unit={unit} />
        <WhatIf key={component} component={component} unit={unit} nCycles={overview.dataset.n_cycles} />
        <NoiseTest component={component} />
        <Selective component={component} />
      </div>
    </>
  );
}
