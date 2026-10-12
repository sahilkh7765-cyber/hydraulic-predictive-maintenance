import { Link } from "react-router";

export function NotFound() {
  return (
    <div className="max-w-xl">
      <h1 className="text-2xl font-semibold">No page at this address</h1>
      <p className="mt-2 text-ink-2">
        The link may be out of date. <Link to="/">Go to the overview</Link>.
      </p>
    </div>
  );
}
