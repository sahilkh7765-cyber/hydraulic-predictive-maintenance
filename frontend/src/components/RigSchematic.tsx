import { SEVERITY_COLOR } from "@/lib/format";
import type { ComponentKey, Severity } from "@/lib/types";
import { cn } from "@/lib/utils";

/**
 * Simplified ISO 1219-style drawing of the test rig: a working circuit
 * (motor, main pump, accumulator, switching valve, load) and a cooling and
 * filtration circuit (motor, pump, filter, cooler), joined through the tank.
 * It is an orientation drawing, not the rig's exact plumbing.
 *
 * With `statuses`, the four monitored components carry a status lamp
 * (colour + glyph + word), which makes the drawing the Predict page's display.
 */
const LINE = "var(--ink-2)";
const DIM = "var(--muted-ink)";

interface Props {
  statuses?: Partial<Record<ComponentKey, Severity>>;
  labels?: Partial<Record<ComponentKey, string>>;
  className?: string;
  animate?: boolean;
}

export function RigSchematic({ statuses, labels, className, animate }: Props) {
  const lit = statuses !== undefined;
  const describe = lit
    ? Object.entries(statuses)
        .map(([k, s]) => `${k}: ${s}`)
        .join(", ")
    : "no prediction shown";
  return (
    <div className={cn("min-w-0", className)}>
      <p className="mb-2 text-xs text-muted-ink sm:hidden">Swipe sideways to see the whole circuit.</p>
      <div className="overflow-x-auto" tabIndex={0} role="region" aria-label="Rig circuit drawing, scrollable">
      <svg
        viewBox="0 0 760 335"
        role="img"
        aria-labelledby="rig-title rig-desc"
        className="block h-auto w-full min-w-[600px]"
      >
        <title id="rig-title">Simplified circuit of the hydraulic test rig</title>
        <desc id="rig-desc">
          Working circuit: motor, main pump, accumulator, switching valve and load. Cooling and filtration circuit:
          motor, pump, filter and cooler. Both return to the tank. Status: {describe}.
        </desc>

        <g fontFamily="var(--font-plate)" fontSize="13" fill={DIM}>
          <text x="24" y="30">Working circuit</text>
          <text x="300" y="232">Cooling and filtration circuit</text>
        </g>

        {/* ---------------- working circuit ---------------- */}
        <g fill="none" stroke={LINE} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round">
          {/* pressure line */}
          <path d="M145 168 V90 H430" />
          <path d="M530 90 H612" />
          {/* motor + shaft */}
          <circle cx="80" cy="190" r="20" stroke={DIM} />
          <path d="M100 187 H123 M100 193 H123" stroke={DIM} />
          {/* suction + tank */}
          <path d="M145 212 V262" />
          <path d="M127 262 v16 h36 v-16" stroke={DIM} />
          {/* accumulator branch */}
          <path d="M300 90 V128" />
          {/* valve return to tank */}
          <path d="M505 110 V150" />
          <path d="M487 150 v16 h36 v-16" stroke={DIM} />
          {/* load: double-acting cylinder */}
          <rect x="612" y="72" width="92" height="36" stroke={DIM} />
          <path d="M646 72 V108 M646 90 H742" stroke={DIM} />
        </g>
        <text x="80" y="195" textAnchor="middle" fontSize="14" fontWeight="600" fill={DIM} fontFamily="var(--font-sans)">
          M
        </text>
        <text x="658" y="128" textAnchor="middle" fontSize="12" fill={DIM} fontFamily="var(--font-plate)">
          load
        </text>

        <Monitored
          k="pump"
          lit={lit}
          status={statuses?.pump}
          label={labels?.pump ?? "Main pump"}
          labelX={176}
          labelY={186}
          animate={animate}
          order={2}
        >
          <circle cx="145" cy="190" r="22" />
          <path d="M145 168 l-8 13 h16 z" fill="currentColor" />
        </Monitored>

        <Monitored
          k="accumulator"
          lit={lit}
          status={statuses?.accumulator}
          label={labels?.accumulator ?? "Accumulator"}
          labelX={330}
          labelY={156}
          animate={animate}
          order={3}
        >
          <rect x="282" y="128" width="36" height="74" rx="18" />
          <path d="M282 166 H318" />
        </Monitored>

        <Monitored
          k="valve"
          lit={lit}
          status={statuses?.valve}
          label={labels?.valve ?? "Switching valve"}
          labelX={430}
          labelY={34}
          animate={animate}
          order={1}
        >
          <rect x="430" y="70" width="50" height="40" />
          <rect x="480" y="70" width="50" height="40" />
          <path d="M440 102 L470 78 M470 78 l-9 1 M470 78 l-2 9" />
          <path d="M492 80 V100 M486 100 H498 M518 80 V100 M512 80 H524" />
        </Monitored>

        {/* ---------------- cooling and filtration circuit ---------------- */}
        <g fill="none" stroke={LINE} strokeWidth="1.75" strokeLinejoin="round" strokeLinecap="round">
          <circle cx="330" cy="275" r="18" stroke={DIM} />
          <path d="M348 272 H372 M348 278 H372" stroke={DIM} />
          <circle cx="392" cy="275" r="20" stroke={DIM} />
          <path d="M412 275 l-12 -7 v14 z" fill={DIM} stroke={DIM} />
          <path d="M392 295 V306" />
          <path d="M374 306 v14 h36 v-14" stroke={DIM} />
          <path d="M412 275 H482 M518 275 H590 M630 275 H720 V306" />
          <path d="M702 306 v14 h36 v-14" stroke={DIM} />
          {/* filter: diamond with a dashed element */}
          <path d="M500 257 L518 275 L500 293 L482 275 Z" stroke={DIM} />
          <path d="M500 261 V289" stroke={DIM} strokeDasharray="3 3" />
        </g>
        <text x="330" y="280" textAnchor="middle" fontSize="13" fontWeight="600" fill={DIM} fontFamily="var(--font-sans)">
          M
        </text>
        <text x="500" y="251" textAnchor="middle" fontSize="12" fill={DIM} fontFamily="var(--font-plate)">
          filter
        </text>

        <Monitored
          k="cooler"
          lit={lit}
          status={statuses?.cooler}
          label={labels?.cooler ?? "Cooler"}
          labelX={640}
          labelY={244}
          animate={animate}
          order={0}
        >
          <path d="M610 255 L630 275 L610 295 L590 275 Z" />
          <path d="M610 259 V291" />
          <path d="M610 259 l-4 7 h8 z M610 291 l-4 -7 h8 z" fill="currentColor" />
        </Monitored>
      </svg>
      </div>
    </div>
  );
}

function Monitored({
  k,
  lit,
  status,
  label,
  labelX,
  labelY,
  children,
  animate,
  order,
}: {
  k: ComponentKey;
  lit: boolean;
  status: Severity | undefined;
  label: string;
  labelX: number;
  labelY: number;
  children: React.ReactNode;
  animate?: boolean;
  order: number;
}) {
  const colour = status ? SEVERITY_COLOR[status] : "var(--ink)";
  return (
    <g data-component={k}>
      <g
        fill="none"
        stroke={colour}
        color={colour}
        strokeWidth="2.25"
        strokeLinejoin="round"
        strokeLinecap="round"
      >
        {children}
      </g>
      <text x={labelX} y={labelY} fontSize="14" fontWeight="600" fill="var(--ink)" fontFamily="var(--font-sans)">
        {label}
      </text>
      {lit && status ? (
        <g
          transform={`translate(${labelX} ${labelY + 10})`}
          style={animate ? { animation: "lamp-on 420ms ease-out both", animationDelay: `${order * 120}ms` } : undefined}
        >
          <Glyph severity={status} />
          <text x="22" y="13" fontSize="13" fontWeight="600" fill={colour} fontFamily="var(--font-sans)">
            {status}
          </text>
        </g>
      ) : null}
    </g>
  );
}

/** Same shapes as the lucide status icons: circle-check, triangle, octagon. */
function Glyph({ severity }: { severity: Severity }) {
  const c = SEVERITY_COLOR[severity];
  const common = { fill: "none", stroke: c, strokeWidth: 2, strokeLinecap: "round" as const, strokeLinejoin: "round" as const };
  if (severity === "Normal")
    return (
      <g {...common}>
        <circle cx="8" cy="8" r="7" />
        <path d="M5 8.2 7.2 10.4 11 6" />
      </g>
    );
  if (severity === "Degraded")
    return (
      <g {...common}>
        <path d="M8 1.5 15 14.5 H1 Z" />
        <path d="M8 6 V9.5 M8 12 V12.1" />
      </g>
    );
  if (severity === "Critical")
    return (
      <g {...common}>
        <path d="M5 1 H11 L15 5 V11 L11 15 H5 L1 11 V5 Z" />
        <path d="M5.5 5.5 10.5 10.5 M10.5 5.5 5.5 10.5" />
      </g>
    );
  return (
    <g {...common}>
      <circle cx="8" cy="8" r="7" />
    </g>
  );
}
