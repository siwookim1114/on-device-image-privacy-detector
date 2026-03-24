import type { SessionSummary } from '../../api/history';

interface SessionCardProps {
  session: SessionSummary;
  onClick: () => void;
}

function formatRelativeTime(dateStr: string): string {
  const date = new Date(dateStr);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffSecs = Math.floor(diffMs / 1000);
  const diffMins = Math.floor(diffSecs / 60);
  const diffHours = Math.floor(diffMins / 60);
  const diffDays = Math.floor(diffHours / 24);

  if (diffSecs < 60) return 'just now';
  if (diffMins < 60) return `${diffMins} minute${diffMins !== 1 ? 's' : ''} ago`;
  if (diffHours < 24) return `${diffHours} hour${diffHours !== 1 ? 's' : ''} ago`;
  if (diffDays < 7) return `${diffDays} day${diffDays !== 1 ? 's' : ''} ago`;
  return date.toLocaleDateString('en-US', { year: 'numeric', month: 'short', day: 'numeric' });
}

function StatusBadge({ status }: { status: string }) {
  const isCompleted = status === 'completed';
  const isFailed = status === 'failed';

  const colorClass = isCompleted
    ? 'bg-green-900/50 text-green-300 border border-green-700/50'
    : isFailed
    ? 'bg-red-900/50 text-red-300 border border-red-700/50'
    : 'bg-yellow-900/50 text-yellow-300 border border-yellow-700/50';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colorClass}`}>
      {status}
    </span>
  );
}

function RiskBadge({ risk }: { risk: string }) {
  const riskUpper = risk.toUpperCase();
  const colorClass =
    riskUpper === 'CRITICAL'
      ? 'bg-red-900/50 text-red-300 border border-red-700/50'
      : riskUpper === 'HIGH'
      ? 'bg-orange-900/50 text-orange-300 border border-orange-700/50'
      : riskUpper === 'MEDIUM'
      ? 'bg-yellow-900/50 text-yellow-300 border border-yellow-700/50'
      : 'bg-gray-700/50 text-gray-300 border border-gray-600/50';

  return (
    <span className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${colorClass}`}>
      {riskUpper}
    </span>
  );
}

export function SessionCard({ session, onClick }: SessionCardProps) {
  const processingTimeSec = session.total_time_ms !== undefined
    ? (session.total_time_ms / 1000).toFixed(1)
    : null;

  return (
    <div
      role="button"
      tabIndex={0}
      onClick={onClick}
      onKeyDown={(e) => {
        if (e.key === 'Enter' || e.key === ' ') {
          e.preventDefault();
          onClick();
        }
      }}
      className="bg-gray-800 rounded-lg p-4 hover:bg-gray-750 cursor-pointer transition-colors duration-150 border border-gray-700/50 hover:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500"
    >
      <div className="flex items-start justify-between gap-3 mb-3">
        <p
          className="text-sm font-medium text-gray-100 truncate flex-1 min-w-0"
          title={session.image_filename}
        >
          {session.image_filename}
        </p>
        <div className="flex items-center gap-2 flex-shrink-0">
          <StatusBadge status={session.status} />
          {session.overall_risk !== undefined && <RiskBadge risk={session.overall_risk} />}
        </div>
      </div>

      <div className="grid grid-cols-3 gap-3 mb-3">
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-100">{session.total_elements}</p>
          <p className="text-xs text-gray-500">elements</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-blue-400">{session.protections_applied}</p>
          <p className="text-xs text-gray-500">protected</p>
        </div>
        <div className="text-center">
          <p className="text-lg font-semibold text-gray-300">
            {processingTimeSec !== null ? `${processingTimeSec}s` : '—'}
          </p>
          <p className="text-xs text-gray-500">processing</p>
        </div>
      </div>

      <p className="text-xs text-gray-500">{formatRelativeTime(session.created_at)}</p>
    </div>
  );
}
