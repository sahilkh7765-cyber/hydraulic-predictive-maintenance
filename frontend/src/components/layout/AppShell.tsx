import {
  ChartColumn,
  Gauge,
  Layers,
  Menu,
  ScanSearch,
  ShieldCheck,
  Waves,
  X,
  type LucideIcon,
} from "lucide-react";
import { Suspense, useEffect, useRef, useState } from "react";
import { NavLink, Outlet, useLocation } from "react-router";
import { useHealth } from "@/lib/queries";
import { cn } from "@/lib/utils";
import { Loading } from "../QueryState";
import { WarmupGate } from "./WarmupGate";

interface NavItem {
  to: string;
  label: string;
  hint: string;
  icon: LucideIcon;
}

export const NAV: readonly NavItem[] = [
  { to: "/", label: "Overview", hint: "Rig and headline scores", icon: Gauge },
  { to: "/dataset", label: "Dataset", hint: "Sensor signals per cycle", icon: Waves },
  { to: "/features", label: "Features", hint: "Raw samples to features", icon: Layers },
  { to: "/predict", label: "Predict", hint: "Score one cycle", icon: ScanSearch },
  { to: "/models", label: "Models", hint: "Comparison, evaluation", icon: ChartColumn },
  { to: "/robustness", label: "Robustness", hint: "Errors, noise, what-if", icon: ShieldCheck },
];

export function AppShell() {
  const [menuOpen, setMenuOpen] = useState(false);
  const location = useLocation();

  // On route changes (not the first load, which would swallow the skip link)
  // move focus to the new page's content so screen readers announce it.
  const previousPath = useRef(location.pathname);
  useEffect(() => {
    setMenuOpen(false);
    if (previousPath.current === location.pathname) return;
    previousPath.current = location.pathname;
    document.getElementById("main")?.focus({ preventScroll: true });
    window.scrollTo(0, 0);
  }, [location.pathname]);

  return (
    <div className="min-h-dvh lg:grid lg:grid-cols-[232px_minmax(0,1fr)]">
      <a
        href="#main"
        className="sr-only z-50 rounded-sm bg-signal px-3 py-2 text-[var(--signal-ink)] focus:not-sr-only focus:fixed focus:top-2 focus:left-2"
      >
        Skip to content
      </a>

      {/* phone / tablet top bar */}
      <header className="sticky top-0 z-30 flex items-center justify-between border-b border-hairline bg-ground/95 px-4 py-3 backdrop-blur lg:hidden">
        <Brand />
        <button
          type="button"
          className="inline-flex size-10 items-center justify-center rounded-sm border border-hairline-strong text-ink"
          aria-expanded={menuOpen}
          aria-controls="site-nav"
          onClick={() => setMenuOpen((o) => !o)}
        >
          {menuOpen ? <X aria-hidden className="size-5" /> : <Menu aria-hidden className="size-5" />}
          <span className="sr-only">{menuOpen ? "Close navigation" : "Open navigation"}</span>
        </button>
      </header>

      <aside
        id="site-nav"
        className={cn(
          "border-hairline bg-ground lg:sticky lg:top-0 lg:flex lg:h-dvh lg:flex-col lg:border-r",
          menuOpen ? "block border-b" : "hidden lg:flex",
        )}
      >
        <div className="hidden px-5 pt-6 pb-5 lg:block">
          <Brand />
        </div>
        <nav aria-label="Main" className="px-3 py-2 lg:flex-1">
          <ul className="space-y-0.5">
            {NAV.map((item) => (
              <li key={item.to}>
                <NavLink
                  to={item.to}
                  end={item.to === "/"}
                  onClick={() => setMenuOpen(false)}
                  className={({ isActive }) =>
                    cn(
                      "group flex items-center gap-3 rounded-sm px-3 py-2 text-ink-2 no-underline hover:bg-plate hover:text-ink",
                      isActive && "bg-plate text-ink shadow-[inset_2px_0_0_var(--signal)]",
                    )
                  }
                >
                  <item.icon aria-hidden className="size-[18px] shrink-0" strokeWidth={1.75} />
                  <span className="min-w-0">
                    <span className="block font-medium">{item.label}</span>
                    <span className="block truncate text-xs text-muted-ink">{item.hint}</span>
                  </span>
                </NavLink>
              </li>
            ))}
          </ul>
        </nav>
        <div className="border-t border-hairline px-5 py-4">
          <ApiLamp />
        </div>
      </aside>

      <main id="main" tabIndex={-1} className="min-w-0 px-4 pt-6 pb-16 outline-none sm:px-6 lg:px-10 lg:pt-10">
        <div className="mx-auto max-w-[1180px]">
          <WarmupGate>
            <Suspense fallback={<Loading label="Loading page" />}>
              <Outlet />
            </Suspense>
          </WarmupGate>
        </div>
      </main>
    </div>
  );
}

function Brand() {
  return (
    <NavLink to="/" className="flex items-center gap-2.5 text-ink no-underline" aria-label="Rig monitor, overview">
      <svg aria-hidden viewBox="0 0 32 32" className="size-8 shrink-0">
        <circle cx="16" cy="16" r="13" fill="none" stroke="var(--hairline-strong)" strokeWidth="2" />
        <path d="M7.5 21.5A10 10 0 0 1 16 6a10 10 0 0 1 8.5 15.5" fill="none" stroke="var(--ink-2)" strokeWidth="1.5" />
        <path d="M16 16 22 10" stroke="var(--signal)" strokeWidth="2.25" strokeLinecap="round" />
        <circle cx="16" cy="16" r="2.25" fill="var(--ink)" />
      </svg>
      <span className="leading-tight">
        <span className="block font-semibold">Rig monitor</span>
        <span className="block text-xs text-muted-ink">UCI hydraulic test rig</span>
      </span>
    </NavLink>
  );
}

function ApiLamp() {
  const health = useHealth();
  let tone = "var(--muted-ink)";
  let text = "Connectingâ€¦";
  if (health.isError && !health.data) {
    tone = "var(--crit)";
    text = "API offline";
  } else if (health.data?.status === "warming") {
    tone = "var(--warn)";
    text = "Loading models";
  } else if (health.data?.status === "unavailable") {
    tone = "var(--crit)";
    text = "Models missing";
  } else if (health.data?.status === "ready") {
    tone = "var(--ok)";
    text = "API ready";
  }
  return (
    <div className="flex items-center gap-2 text-sm" role="status" aria-live="polite">
      <span aria-hidden className="size-2.5 rounded-full" style={{ background: tone, boxShadow: `0 0 0 3px color-mix(in srgb, ${tone} 25%, transparent)` }} />
      <span className="text-ink-2">{text}</span>
      {health.data?.versions["scikit-learn"] ? (
        <span className="num ml-auto text-xs text-muted-ink">sklearn {health.data.versions["scikit-learn"]}</span>
      ) : null}
    </div>
  );
}
