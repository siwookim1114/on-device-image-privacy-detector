import { useId, useRef } from 'react';

export interface SegmentOption {
  value: string;
  label: string;
  description: string;
  default?: boolean;
}

interface SegmentedControlProps {
  options: SegmentOption[];
  value: string;
  onChange: (value: string) => void;
  accentColor?: 'blue' | 'amber' | 'purple' | 'teal';
  label?: string;
}

const ACCENT_CLASSES: Record<
  NonNullable<SegmentedControlProps['accentColor']>,
  {
    border: string;
    bg: string;
    text: string;
    ring: string;
    descText: string;
  }
> = {
  blue: {
    border: 'border-blue-500',
    bg: 'bg-blue-500/10',
    text: 'text-blue-400',
    ring: 'focus-visible:ring-blue-500',
    descText: 'text-blue-300/80',
  },
  amber: {
    border: 'border-amber-500',
    bg: 'bg-amber-500/10',
    text: 'text-amber-400',
    ring: 'focus-visible:ring-amber-500',
    descText: 'text-amber-300/80',
  },
  purple: {
    border: 'border-purple-500',
    bg: 'bg-purple-500/10',
    text: 'text-purple-400',
    ring: 'focus-visible:ring-purple-500',
    descText: 'text-purple-300/80',
  },
  teal: {
    border: 'border-teal-500',
    bg: 'bg-teal-500/10',
    text: 'text-teal-400',
    ring: 'focus-visible:ring-teal-500',
    descText: 'text-teal-300/80',
  },
};

export function SegmentedControl({
  options,
  value,
  onChange,
  accentColor = 'blue',
  label,
}: SegmentedControlProps) {
  const groupId = useId();
  const accent = ACCENT_CLASSES[accentColor];
  const selectedOption = options.find((o) => o.value === value);
  const buttonRefs = useRef<(HTMLButtonElement | null)[]>([]);

  const handleKeyDown = (e: React.KeyboardEvent<HTMLButtonElement>, index: number) => {
    let nextIndex: number | undefined;
    if (e.key === 'ArrowRight' || e.key === 'ArrowDown') {
      nextIndex = (index + 1) % options.length;
    } else if (e.key === 'ArrowLeft' || e.key === 'ArrowUp') {
      nextIndex = (index - 1 + options.length) % options.length;
    } else if (e.key === 'Home') {
      nextIndex = 0;
    } else if (e.key === 'End') {
      nextIndex = options.length - 1;
    }

    if (nextIndex !== undefined) {
      e.preventDefault();
      const nextOption = options[nextIndex];
      if (nextOption) {
        onChange(nextOption.value);
        buttonRefs.current[nextIndex]?.focus();
      }
    }
  };

  return (
    <div className="space-y-2">
      {label && (
        <p id={`${groupId}-label`} className="text-xs font-medium text-gray-400 uppercase tracking-wide">
          {label}
        </p>
      )}

      <div
        role="radiogroup"
        aria-label={label}
        aria-labelledby={label ? `${groupId}-label` : undefined}
        className="flex gap-2"
      >
        {options.map((option, index) => {
          const isSelected = option.value === value;
          return (
            <button
              key={option.value}
              ref={(el) => { buttonRefs.current[index] = el; }}
              type="button"
              role="radio"
              aria-checked={isSelected}
              tabIndex={isSelected ? 0 : -1}
              onClick={() => onChange(option.value)}
              onKeyDown={(e) => handleKeyDown(e, index)}
              className={`
                relative flex-1 rounded-lg border-2 px-3 py-2.5 text-left text-sm
                transition-all duration-150 cursor-pointer
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-offset-2
                focus-visible:ring-offset-gray-900 ${accent.ring}
                ${isSelected
                  ? `${accent.border} ${accent.bg}`
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600 hover:bg-gray-800'
                }
              `}
            >
              <span
                className={`block font-semibold capitalize ${
                  isSelected ? accent.text : 'text-gray-300'
                }`}
              >
                {option.label}
              </span>
            </button>
          );
        })}
      </div>

      {/* Description area — updates based on selection */}
      <div
        aria-live="polite"
        aria-atomic="true"
        className={`min-h-[2rem] rounded-md px-3 py-2 text-xs transition-all duration-200 ${
          selectedOption
            ? `${accent.bg} ${accent.descText}`
            : 'bg-gray-800/50 text-gray-500'
        }`}
      >
        {selectedOption?.description ?? ''}
      </div>
    </div>
  );
}
