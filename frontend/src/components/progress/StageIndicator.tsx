interface StageIndicatorProps {
  stage: string;
  label: string;
  status: 'completed' | 'active' | 'pending';
  timingMs?: number;
}

function formatTiming(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  return `${(ms / 1000).toFixed(1)}s`;
}

function CheckIcon() {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={2.5}
      strokeLinecap="round"
      strokeLinejoin="round"
      className="w-3 h-3"
    >
      <polyline points="20 6 9 17 4 12" />
    </svg>
  );
}

export function StageIndicator({ label, status, timingMs }: StageIndicatorProps) {
  return (
    <div className="flex flex-col items-center gap-1.5">
      {/* Dot */}
      <div className="relative flex items-center justify-center">
        {status === 'completed' && (
          <div className="w-7 h-7 rounded-full bg-green-500 flex items-center justify-center text-white">
            <CheckIcon />
          </div>
        )}

        {status === 'active' && (
          <div className="relative w-7 h-7">
            <span className="absolute inset-0 rounded-full bg-blue-500 opacity-30 animate-ping" />
            <span className="relative flex w-7 h-7 rounded-full bg-blue-500 items-center justify-center">
              <span className="w-2.5 h-2.5 rounded-full bg-white" />
            </span>
          </div>
        )}

        {status === 'pending' && (
          <div className="w-7 h-7 rounded-full bg-gray-700 border-2 border-gray-600" />
        )}
      </div>

      {/* Label */}
      <span
        className={`text-xs font-medium text-center leading-tight max-w-[72px] ${
          status === 'completed'
            ? 'text-green-400'
            : status === 'active'
            ? 'text-blue-300'
            : 'text-gray-500'
        }`}
      >
        {label}
      </span>

      {/* Timing */}
      {timingMs !== undefined && (
        <span className="text-xs tabular-nums text-gray-500">{formatTiming(timingMs)}</span>
      )}
    </div>
  );
}
