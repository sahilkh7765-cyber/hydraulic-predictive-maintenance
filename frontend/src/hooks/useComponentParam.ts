import { useCallback } from "react";
import { useSearchParams } from "react-router";
import { COMPONENTS } from "@/lib/format";
import type { ComponentKey } from "@/lib/types";

const isComponent = (v: string | null): v is ComponentKey => v !== null && (COMPONENTS as readonly string[]).includes(v);

/** The focus component lives in the URL (?c=valve) so a view can be shared and survives reloads. */
export function useComponentParam(fallback: ComponentKey = "valve"): [ComponentKey, (c: ComponentKey) => void] {
  const [params, setParams] = useSearchParams();
  const raw = params.get("c");
  const component = isComponent(raw) ? raw : fallback;
  const set = useCallback(
    (c: ComponentKey) =>
      setParams(
        (prev) => {
          const next = new URLSearchParams(prev);
          next.set("c", c);
          return next;
        },
        { replace: true },
      ),
    [setParams],
  );
  return [component, set];
}
