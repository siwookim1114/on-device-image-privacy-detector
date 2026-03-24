import type { ReactNode } from 'react';
import type { RiskAssessment, RiskLevel } from '../../types/risk';
import { usePipelineStore } from '../../stores/pipelineStore';
import { SEVERITY_COLORS } from '../../lib/colors';
import { DetectionStats } from './DetectionStats';
import { ElementList } from './ElementList';
import { SeverityBadge } from './SeverityBadge';
import { SeverityOverride } from './SeverityOverride';

interface DetectionReviewProps {
  assessments: RiskAssessment[];
  selectedId: string | null;
  onSelectElement: (id: string | null) => void;
  onOverrideSeverity: (id: string, severity: RiskLevel) => void;
  onApprove?: () => void;
}

function DetailRow({ label, value }: { label: string; value: ReactNode }) {
  return (
    <div className="flex gap-2 py-1.5 border-b border-gray-800 last:border-0">
      <span className="text-xs text-gray-500 w-28 shrink-0">{label}</span>
      <span className="text-xs text-gray-200 break-words flex-1">{value}</span>
    </div>
  );
}

function ElementDetail({
  assessment,
  onClose,
  onOverride,
}: {
  assessment: RiskAssessment;
  onClose: () => void;
  onOverride: (id: string, severity: RiskLevel) => void;
}) {
  // CRITICAL text items are safety-locked and cannot be overridden
  const locked =
    assessment.element_type === 'text' && assessment.severity === 'critical';

  const borderColor = SEVERITY_COLORS[assessment.severity].stroke;

  return (
    <div
      className="shrink-0 bg-gray-850 border-t border-gray-700"
      style={{ borderTop: `2px solid ${borderColor}` }}
    >
      {/* Header */}
      <div className="flex items-center justify-between px-3 py-2 border-b border-gray-800">
        <span className="text-xs font-semibold text-gray-300 uppercase tracking-wide">
          Element Detail
        </span>
        <button
          onClick={onClose}
          className="text-gray-500 hover:text-gray-300 transition-colors text-lg leading-none p-0.5 rounded"
          aria-label="Close detail panel"
        >
          &times;
        </button>
      </div>

      {/* Fields */}
      <div className="px-3 py-1 max-h-64 overflow-y-auto">
        <DetailRow label="ID" value={<code className="text-gray-400 text-xs">{assessment.detection_id}</code>} />
        <DetailRow label="Type" value={assessment.element_type} />
        <DetailRow label="Description" value={assessment.element_description} />
        <DetailRow label="Risk type" value={assessment.risk_type.replace(/_/g, ' ')} />
        <DetailRow label="Severity" value={<SeverityBadge severity={assessment.severity} size="sm" />} />
        <DetailRow label="Requires protection" value={assessment.requires_protection ? 'Yes' : 'No'} />
        <DetailRow label="Reasoning" value={assessment.reasoning} />

        {assessment.element_type === 'face' && assessment.consent_status && (
          <DetailRow label="Consent" value={assessment.consent_status} />
        )}
        {assessment.classification && (
          <DetailRow label="Classification" value={assessment.classification} />
        )}
        {assessment.person_label && (
          <DetailRow label="Person" value={assessment.person_label} />
        )}
        {assessment.screen_state && (
          <DetailRow
            label="Screen state"
            value={assessment.screen_state === 'verified_on' ? 'ON' : 'OFF'}
          />
        )}

        <DetailRow
          label="Bounding box"
          value={`x:${assessment.bbox.x} y:${assessment.bbox.y} ${assessment.bbox.width}\u00d7${assessment.bbox.height}`}
        />
      </div>

      {/* Override */}
      <div className="px-3 py-2 border-t border-gray-800">
        <SeverityOverride
          currentSeverity={assessment.severity}
          detectionId={assessment.detection_id}
          locked={locked}
          onOverride={onOverride}
        />
      </div>
    </div>
  );
}

export function DetectionReview({
  assessments,
  selectedId,
  onSelectElement,
  onOverrideSeverity,
  onApprove,
}: DetectionReviewProps) {
  const selectElement = usePipelineStore((s) => s.selectElement);

  const selected = assessments.find((a) => a.detection_id === selectedId) ?? null;

  function handleSelect(id: string) {
    selectElement(id);
    onSelectElement(id);
  }

  function handleDeselect() {
    selectElement(null);
    onSelectElement(null);
  }

  return (
    <div className="flex flex-col h-full bg-gray-900 overflow-hidden">
      {/* Stats bar */}
      <DetectionStats assessments={assessments} />

      {/* Scrollable element list — fills available vertical space */}
      <div className="flex-1 min-h-0 overflow-hidden flex flex-col">
        <ElementList
          assessments={assessments}
          selectedId={selectedId}
          onSelect={handleSelect}
        />
      </div>

      {/* Detail panel for selected element */}
      {selected !== null && (
        <ElementDetail
          assessment={selected}
          onClose={handleDeselect}
          onOverride={onOverrideSeverity}
        />
      )}

      {/* Approve & Continue */}
      <div className="shrink-0 px-4 py-3 border-t border-gray-800 bg-gray-900">
        <button
          onClick={onApprove}
          className="w-full py-2 rounded-lg bg-blue-600 hover:bg-blue-500 active:bg-blue-700 text-white text-sm font-semibold transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900"
        >
          Approve &amp; Continue
        </button>
      </div>
    </div>
  );
}
