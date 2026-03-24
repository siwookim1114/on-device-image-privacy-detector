import type { ProtectionStrategy, ObfuscationMethod } from '../../types/strategy';
import type { RiskAssessment } from '../../types/risk';

import { StrategyList } from './StrategyList';
import { ApproveBar } from './ApproveBar';

interface StrategyReviewProps {
  strategies: ProtectionStrategy[];
  riskAssessments: RiskAssessment[];
  onMethodChange: (detectionId: string, method: ObfuscationMethod) => void;
  onParamsChange: (detectionId: string, params: Record<string, unknown>) => void;
  onApprove: () => void;
}

function SummaryStats({
  strategies,
  riskAssessments,
}: {
  strategies: ProtectionStrategy[];
  riskAssessments: RiskAssessment[];
}) {
  const totalElements = strategies.length;
  const willProtect = strategies.filter((s) => s.recommended_method !== null && s.recommended_method !== 'none').length;
  const requiresDecision = strategies.filter((s) => s.requires_user_decision).length;
  const overallRisk = riskAssessments.some((r) => r.severity === 'critical')
    ? 'critical'
    : riskAssessments.some((r) => r.severity === 'high')
      ? 'high'
      : riskAssessments.some((r) => r.severity === 'medium')
        ? 'medium'
        : 'low';

  const riskColors: Record<string, string> = {
    critical: 'text-red-400 bg-red-500/10 border-red-500/30',
    high: 'text-orange-400 bg-orange-500/10 border-orange-500/30',
    medium: 'text-yellow-400 bg-yellow-500/10 border-yellow-500/30',
    low: 'text-green-400 bg-green-500/10 border-green-500/30',
  };

  return (
    <div className="grid grid-cols-3 gap-3 mb-4">
      <div className="rounded-lg bg-gray-800/60 border border-gray-700/50 p-3 text-center">
        <p className="text-xl font-bold text-gray-100 tabular-nums">{totalElements}</p>
        <p className="text-xs text-gray-500 mt-0.5">Detected</p>
      </div>
      <div className="rounded-lg bg-blue-500/10 border border-blue-500/30 p-3 text-center">
        <p className="text-xl font-bold text-blue-300 tabular-nums">{willProtect}</p>
        <p className="text-xs text-blue-400/70 mt-0.5">To Protect</p>
      </div>
      <div
        className={[
          'rounded-lg border p-3 text-center',
          riskColors[overallRisk] ?? riskColors['low'],
        ].join(' ')}
      >
        <p className="text-xl font-bold tabular-nums capitalize">{overallRisk}</p>
        <p className="text-xs mt-0.5 opacity-70">Risk Level</p>
      </div>

      {requiresDecision > 0 && (
        <div className="col-span-3 rounded-lg bg-yellow-500/8 border border-yellow-500/25 px-3 py-2 flex items-center gap-2">
          <svg className="w-4 h-4 text-yellow-400 shrink-0" viewBox="0 0 16 16" fill="none" aria-hidden="true">
            <path
              d="M8 2.5l5.5 10H2.5L8 2.5z"
              stroke="currentColor"
              strokeWidth="1.5"
              strokeLinejoin="round"
            />
            <path d="M8 6.5v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
            <circle cx="8" cy="11.5" r="0.75" fill="currentColor" />
          </svg>
          <p className="text-xs text-yellow-300">
            <span className="font-semibold tabular-nums">{requiresDecision}</span>
            {' '}
            {requiresDecision === 1 ? 'element requires' : 'elements require'} your review before proceeding.
          </p>
        </div>
      )}
    </div>
  );
}

export function StrategyReview({
  strategies,
  riskAssessments,
  onMethodChange,
  onParamsChange,
  onApprove,
}: StrategyReviewProps) {
  const totalProtections = strategies.filter(
    (s) => s.recommended_method !== null && s.recommended_method !== 'none',
  ).length;
  const requiresConfirmation = strategies.filter((s) => s.requires_user_decision).length;

  return (
    <div className="flex flex-col h-full">
      {/* Scrollable body */}
      <div className="flex-1 overflow-y-auto px-4 py-4">
        <h2 className="text-sm font-semibold text-gray-200 mb-4">Strategy Review</h2>
        <SummaryStats strategies={strategies} riskAssessments={riskAssessments} />
        <StrategyList
          strategies={strategies}
          onMethodChange={onMethodChange}
          onParamsChange={onParamsChange}
        />
      </div>

      {/* Sticky footer */}
      <ApproveBar
        onApproveAll={onApprove}
        onReviewIndividually={() => undefined}
        requiresConfirmation={requiresConfirmation}
        totalProtections={totalProtections}
      />
    </div>
  );
}
