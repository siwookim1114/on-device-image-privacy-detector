import type { RiskLevel } from '../../types/risk';
import { SEVERITY_COLORS } from '../../lib/colors';

interface SeverityOverrideProps {
  currentSeverity: RiskLevel;
  detectionId: string;
  locked: boolean;
  onOverride: (id: string, severity: RiskLevel) => void;
}

const SEVERITY_OPTIONS: { value: RiskLevel; label: string }[] = [
  { value: 'critical', label: 'Critical' },
  { value: 'high', label: 'High' },
  { value: 'medium', label: 'Medium' },
  { value: 'low', label: 'Low' },
];

function LockIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-3.5 h-3.5 shrink-0"
      aria-hidden="true"
    >
      <rect x="3" y="11" width="18" height="11" rx="2" ry="2" />
      <path d="M7 11V7a5 5 0 0 1 10 0v4" />
    </svg>
  );
}

export function SeverityOverride({
  currentSeverity,
  detectionId,
  locked,
  onOverride,
}: SeverityOverrideProps) {
  if (locked) {
    return (
      <div
        className="flex items-center gap-1.5 text-xs text-gray-500 cursor-not-allowed"
        title="Cannot change CRITICAL PII"
        aria-disabled="true"
      >
        <LockIcon />
        <span>Severity locked</span>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-2">
      <label className="text-xs text-gray-400 shrink-0" htmlFor={`severity-override-${detectionId}`}>
        Override:
      </label>
      <select
        id={`severity-override-${detectionId}`}
        value={currentSeverity}
        onChange={(e) => onOverride(detectionId, e.target.value as RiskLevel)}
        className="text-xs rounded-md bg-gray-700 border border-gray-600 text-gray-100 px-2 py-1 focus:outline-none focus:ring-1 focus:ring-blue-500 cursor-pointer"
        aria-label="Override severity"
      >
        {SEVERITY_OPTIONS.map((opt) => (
          <option
            key={opt.value}
            value={opt.value}
            style={{ color: SEVERITY_COLORS[opt.value].stroke }}
          >
            {opt.label}
          </option>
        ))}
      </select>
    </div>
  );
}
