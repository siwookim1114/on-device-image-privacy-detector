import type { RiskAssessment } from '../../types/risk';

import { SEVERITY_COLORS } from '../../lib/colors';
import { SeverityBadge } from './SeverityBadge';

interface ElementCardProps {
  assessment: RiskAssessment;
  selected: boolean;
  onClick: () => void;
}

function elementIcon(type: string): string {
  switch (type) {
    case 'face': return '\u{1F464}'; // person silhouette
    case 'text': return '\u{1F4DD}'; // memo
    case 'object': return '\u{1F5A5}'; // desktop computer
    default: return '\u{2022}'; // bullet
  }
}

function ConsentDot({ status }: { status: string }) {
  const colors: Record<string, string> = {
    explicit: 'bg-green-500',
    assumed: 'bg-yellow-400',
    unclear: 'bg-orange-400',
    none: 'bg-red-500',
  };
  const label: Record<string, string> = {
    explicit: 'Explicit',
    assumed: 'Assumed',
    unclear: 'Unclear',
    none: 'No consent',
  };
  return (
    <span className="flex items-center gap-1 text-xs text-gray-400">
      <span className={`inline-block w-1.5 h-1.5 rounded-full ${colors[status] ?? 'bg-gray-500'}`} />
      {label[status] ?? status}
    </span>
  );
}

export function ElementCard({ assessment, selected, onClick }: ElementCardProps) {
  const borderColor = SEVERITY_COLORS[assessment.severity].stroke;

  const baseClasses =
    'flex items-center gap-3 px-3 py-2 cursor-pointer transition-colors duration-100 min-h-[56px] select-none';

  const stateClasses = selected
    ? 'bg-gray-800 ring-1 ring-inset ring-blue-500'
    : 'hover:bg-gray-800/50';

  const truncated =
    assessment.element_description.length > 40
      ? assessment.element_description.slice(0, 37) + '...'
      : assessment.element_description;

  return (
    <div
      role="button"
      tabIndex={0}
      aria-pressed={selected}
      className={`${baseClasses} ${stateClasses}`}
      style={{ borderLeft: `4px solid ${borderColor}` }}
      onClick={onClick}
      onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') onClick(); }}
    >
      {/* Icon */}
      <span className="text-lg leading-none shrink-0" aria-hidden="true">
        {elementIcon(assessment.element_type)}
      </span>

      {/* Description + meta */}
      <div className="flex-1 min-w-0">
        <p className="text-xs text-gray-200 truncate leading-tight">{truncated}</p>
        <div className="flex items-center gap-2 mt-0.5 flex-wrap">
          {assessment.element_type === 'face' && assessment.consent_status && (
            <ConsentDot status={assessment.consent_status} />
          )}
          {assessment.element_type === 'object' && assessment.screen_state && (
            <span className="text-xs text-gray-400">
              {assessment.screen_state === 'verified_on' ? 'Screen ON' : 'Screen OFF'}
            </span>
          )}
        </div>
      </div>

      {/* Badge */}
      <div className="shrink-0">
        <SeverityBadge severity={assessment.severity} size="sm" />
      </div>
    </div>
  );
}
