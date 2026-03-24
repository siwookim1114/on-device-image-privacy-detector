import type { RiskLevel } from '../../types/risk';
import { SEVERITY_COLORS } from '../../lib/colors';

interface SeverityBadgeProps {
  severity: RiskLevel;
  size: 'sm' | 'md';
}

const LABELS: Record<RiskLevel, string> = {
  critical: 'CRITICAL',
  high: 'HIGH',
  medium: 'MEDIUM',
  low: 'LOW',
};

export function SeverityBadge({ severity, size }: SeverityBadgeProps) {
  const colors = SEVERITY_COLORS[severity];
  const isFilled = severity === 'critical' || severity === 'high';

  const sizeClasses =
    size === 'sm'
      ? 'text-xs px-2 py-0.5'
      : 'text-sm px-3 py-1';

  const colorClasses = isFilled
    ? `${colors.bg} ${colors.text} border border-transparent`
    : `bg-transparent ${colors.text} border`;

  // For the outline variant we need the border color to match the text/stroke color.
  // We use inline style for the border color since Tailwind doesn't have dynamic stroke
  // colors in this project's config.
  const borderStyle = isFilled
    ? undefined
    : { borderColor: SEVERITY_COLORS[severity].stroke };

  return (
    <span
      className={`inline-flex items-center font-semibold rounded-full leading-none ${sizeClasses} ${colorClasses}`}
      style={borderStyle}
    >
      {LABELS[severity]}
    </span>
  );
}
