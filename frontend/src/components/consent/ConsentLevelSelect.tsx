import type { ConsentStatus } from '../../types/consent';

interface ConsentOption {
  value: ConsentStatus;
  label: string;
  description: string;
  accentColor: string;
  borderColor: string;
  checkColor: string;
  badgeColor: string;
}

const OPTIONS: ConsentOption[] = [
  {
    value: 'explicit',
    label: 'Explicit',
    description: 'Person has directly given permission',
    accentColor: 'bg-green-500/10',
    borderColor: 'border-green-500',
    checkColor: 'text-green-400',
    badgeColor: 'bg-green-500/20 text-green-300',
  },
  {
    value: 'assumed',
    label: 'Assumed',
    description: 'Implied consent from relationship',
    accentColor: 'bg-yellow-500/10',
    borderColor: 'border-yellow-500',
    checkColor: 'text-yellow-400',
    badgeColor: 'bg-yellow-500/20 text-yellow-300',
  },
  {
    value: 'none',
    label: 'None',
    description: 'No consent recorded',
    accentColor: 'bg-red-500/10',
    borderColor: 'border-red-500',
    checkColor: 'text-red-400',
    badgeColor: 'bg-red-500/20 text-red-300',
  },
];

interface ConsentLevelSelectProps {
  value: ConsentStatus;
  onChange: (status: ConsentStatus) => void;
}

function CheckIcon({ className }: { className: string }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className={className}
      aria-hidden="true"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

export function ConsentLevelSelect({ value, onChange }: ConsentLevelSelectProps) {
  return (
    <fieldset>
      <legend className="sr-only">Consent level</legend>
      <div className="grid grid-cols-3 gap-3">
        {OPTIONS.map((option) => {
          const isSelected = value === option.value;
          return (
            <button
              key={option.value}
              type="button"
              role="radio"
              aria-checked={isSelected}
              onClick={() => onChange(option.value)}
              className={`
                relative flex flex-col gap-1.5 rounded-lg border-2 p-3 text-left
                transition-all duration-150 cursor-pointer focus-visible:outline-none
                focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2
                focus-visible:ring-offset-gray-900
                ${isSelected
                  ? `${option.borderColor} ${option.accentColor}`
                  : 'border-gray-700 bg-gray-800/50 hover:border-gray-600 hover:bg-gray-800'
                }
              `}
            >
              <div className="flex items-center justify-between">
                <span
                  className={`text-xs font-semibold uppercase tracking-wide ${
                    isSelected ? option.checkColor : 'text-gray-300'
                  }`}
                >
                  {option.label}
                </span>
                {isSelected && (
                  <CheckIcon className={`w-4 h-4 ${option.checkColor}`} />
                )}
              </div>
              <p className="text-xs text-gray-400 leading-snug">{option.description}</p>
            </button>
          );
        })}
      </div>
    </fieldset>
  );
}
