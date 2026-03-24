import type { RiskAssessment, RiskLevel } from '../../types/risk';

import { SEVERITY_COLORS } from '../../lib/colors';

interface DetectionStatsProps {
  assessments: RiskAssessment[];
}

const SEVERITY_ORDER: RiskLevel[] = ['critical', 'high', 'medium', 'low'];

const SEVERITY_LABELS: Record<RiskLevel, string> = {
  critical: 'CRITICAL',
  high: 'HIGH',
  medium: 'MEDIUM',
  low: 'LOW',
};

export function DetectionStats({ assessments }: DetectionStatsProps) {
  const counts: Record<RiskLevel, number> = { critical: 0, high: 0, medium: 0, low: 0 };

  for (const a of assessments) {
    counts[a.severity] += 1;
  }

  const nonZero = SEVERITY_ORDER.filter((s) => counts[s] > 0);

  return (
    <div className="flex items-center gap-3 px-4 py-2 bg-gray-800 border-b border-gray-700 text-sm flex-wrap">
      <span className="text-gray-300 font-medium">
        {assessments.length} {assessments.length === 1 ? 'element' : 'elements'}
      </span>

      {nonZero.length > 0 && (
        <>
          <span className="text-gray-600 select-none">:</span>
          <div className="flex items-center gap-3 flex-wrap">
            {nonZero.map((severity, idx) => (
              <span key={severity} className="flex items-center gap-1.5">
                {/* Colored dot */}
                <span
                  className="inline-block w-2 h-2 rounded-full shrink-0"
                  style={{ backgroundColor: SEVERITY_COLORS[severity].stroke }}
                />
                <span className={`font-medium ${SEVERITY_COLORS[severity].text}`}>
                  {counts[severity]}
                </span>
                <span className="text-gray-500 text-xs">{SEVERITY_LABELS[severity]}</span>
                {idx < nonZero.length - 1 && (
                  <span className="text-gray-700 ml-1 select-none">&middot;</span>
                )}
              </span>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
