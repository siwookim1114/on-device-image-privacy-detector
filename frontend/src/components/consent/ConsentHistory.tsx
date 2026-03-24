import type { ConsentHistory as ConsentHistoryType } from '../../types/consent';

interface ConsentHistoryProps {
  history: ConsentHistoryType;
}

interface ProgressBarProps {
  value: number;
  colorClass: string;
  label: string;
}

function ProgressBar({ value, colorClass, label }: ProgressBarProps) {
  const pct = Math.min(100, Math.max(0, Math.round(value * 100)));
  return (
    <div className="space-y-1">
      <div className="flex items-center justify-between">
        <span className="text-xs text-gray-400">{label}</span>
        <span className="text-xs font-medium text-gray-300">{pct}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-gray-700 overflow-hidden" aria-hidden="true">
        <div
          className={`h-full rounded-full transition-all duration-300 ${colorClass}`}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  );
}

const DECISION_BADGE: Record<string, string> = {
  approved: 'bg-green-500/15 text-green-400',
  protected: 'bg-blue-500/15 text-blue-400',
  rejected: 'bg-red-500/15 text-red-400',
};

export function ConsentHistory({ history }: ConsentHistoryProps) {
  const {
    times_appeared,
    times_approved,
    times_protected,
    approval_rate,
    protection_rate,
    last_consent_decision,
  } = history;

  const hasApprovalRate = approval_rate !== undefined;
  const hasProtectionRate = protection_rate !== undefined;

  const decisionBadgeClass =
    last_consent_decision !== null
      ? (DECISION_BADGE[last_consent_decision] ?? 'bg-gray-700 text-gray-400')
      : null;

  return (
    <div className="space-y-3">
      {/* Progress bars */}
      <div className="space-y-2">
        {hasApprovalRate && (
          <ProgressBar value={approval_rate!} colorClass="bg-green-500" label="Approval rate" />
        )}
        {hasProtectionRate && (
          <ProgressBar value={protection_rate!} colorClass="bg-blue-500" label="Protection rate" />
        )}
      </div>

      {/* Count stats */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="rounded-md bg-gray-900/60 px-2 py-1.5">
          <p className="text-sm font-semibold text-gray-100">{times_appeared}</p>
          <p className="text-xs text-gray-500 leading-tight">appeared</p>
        </div>
        <div className="rounded-md bg-gray-900/60 px-2 py-1.5">
          <p className="text-sm font-semibold text-green-400">{times_approved}</p>
          <p className="text-xs text-gray-500 leading-tight">approved</p>
        </div>
        <div className="rounded-md bg-gray-900/60 px-2 py-1.5">
          <p className="text-sm font-semibold text-blue-400">{times_protected}</p>
          <p className="text-xs text-gray-500 leading-tight">protected</p>
        </div>
      </div>

      {/* Last decision badge */}
      {last_consent_decision !== null && decisionBadgeClass !== null && (
        <div className="flex items-center gap-1.5">
          <span className="text-xs text-gray-500">Last decision:</span>
          <span className={`inline-flex items-center rounded-full px-2 py-0.5 text-xs font-medium ${decisionBadgeClass}`}>
            {last_consent_decision}
          </span>
        </div>
      )}
    </div>
  );
}
