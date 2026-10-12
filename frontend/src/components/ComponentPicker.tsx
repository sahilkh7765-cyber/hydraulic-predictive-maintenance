import { useComponentParam } from "@/hooks/useComponentParam";
import { COMPONENTS, COMPONENT_SHORT } from "@/lib/format";
import type { ComponentKey } from "@/lib/types";
import { SegmentedControl } from "./SegmentedControl";

const OPTIONS = COMPONENTS.map((c) => ({ value: c, label: COMPONENT_SHORT[c] }));

/** The page-level filter row: one control scopes every panel below it. */
export function ComponentPicker({ fallback }: { fallback?: ComponentKey }) {
  const [component, setComponent] = useComponentParam(fallback);
  return <SegmentedControl label="Component" options={OPTIONS} value={component} onChange={setComponent} />;
}
