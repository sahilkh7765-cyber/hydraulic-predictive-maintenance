import { useId } from "react";
import { cn } from "@/lib/utils";

export interface SegmentOption<T extends string> {
  value: T;
  label: string;
}

/**
 * A row of mutually exclusive options built from native radio inputs, so
 * arrow keys, Tab and screen readers work without extra code.
 */
export function SegmentedControl<T extends string>({
  label,
  options,
  value,
  onChange,
  disabled,
  hideLabel,
}: {
  label: string;
  options: readonly SegmentOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  hideLabel?: boolean;
}) {
  const name = useId();
  return (
    <fieldset disabled={disabled} className="min-w-0">
      <legend className={cn("nameplate mb-1.5", hideLabel && "sr-only")}>{label}</legend>
      <div className="inline-flex max-w-full flex-wrap gap-1 rounded-md border border-hairline-strong bg-ground p-1">
        {options.map((option) => {
          const checked = option.value === value;
          return (
            <label
              key={option.value}
              className={cn(
                "relative cursor-pointer rounded-sm px-3 py-1.5 text-sm font-medium text-ink-2 select-none",
                "has-[:focus-visible]:outline-2 has-[:focus-visible]:outline-offset-1 has-[:focus-visible]:outline-signal",
                "hover:text-ink has-[:disabled]:cursor-not-allowed has-[:disabled]:opacity-50",
                checked && "bg-raised text-ink shadow-[inset_0_-2px_0_var(--signal)]",
              )}
            >
              <input
                type="radio"
                className="sr-only"
                name={name}
                value={option.value}
                checked={checked}
                onChange={() => onChange(option.value)}
              />
              {option.label}
            </label>
          );
        })}
      </div>
    </fieldset>
  );
}
