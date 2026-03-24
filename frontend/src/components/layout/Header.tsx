import type { PipelineStage } from '../../types';

import { STAGE_LABELS } from '../../lib/colors';

interface HeaderProps {
  sessionId: string;
  currentStage: PipelineStage;
  elapsedMs: number;
}

function formatElapsed(ms: number): string {
  if (ms < 1000) return `${ms}ms`;
  const s = (ms / 1000).toFixed(1);
  return `${s}s`;
}

const STAGE_BADGE_COLORS: Record<PipelineStage, string> = {
  detection: 'bg-blue-500/20 text-blue-300 border border-blue-500/40',
  risk: 'bg-orange-500/20 text-orange-300 border border-orange-500/40',
  consent: 'bg-purple-500/20 text-purple-300 border border-purple-500/40',
  strategy: 'bg-yellow-500/20 text-yellow-300 border border-yellow-500/40',
  sam: 'bg-cyan-500/20 text-cyan-300 border border-cyan-500/40',
  execution: 'bg-red-500/20 text-red-300 border border-red-500/40',
  export: 'bg-green-500/20 text-green-300 border border-green-500/40',
  done: 'bg-green-600/20 text-green-300 border border-green-600/40',
};

export function Header({ sessionId, currentStage, elapsedMs }: HeaderProps) {
  const stageLabel: string = STAGE_LABELS[currentStage] ?? currentStage;
  const badgeColor = STAGE_BADGE_COLORS[currentStage];

  return (
    <header className="flex items-center justify-between h-14 px-4 bg-gray-900 border-b border-gray-800 shrink-0">
      {/* Left: Branding */}
      <div className="flex items-center gap-2">
        <span className="text-base font-semibold text-gray-100 tracking-tight">
          Privacy Guard
        </span>
        <span className="text-gray-700 select-none">/</span>
        <span className="text-sm text-gray-400">Review</span>
      </div>

      {/* Center: Session info */}
      <div className="flex items-center gap-3 text-sm text-gray-400">
        <div className="flex items-center gap-1.5">
          <span className="text-gray-600 text-xs uppercase tracking-wider font-medium">Session</span>
          <code className="text-gray-300 font-mono text-xs bg-gray-800 px-2 py-0.5 rounded">
            {sessionId.length > 12 ? `${sessionId.slice(0, 12)}\u2026` : sessionId}
          </code>
        </div>
        {elapsedMs > 0 && (
          <>
            <span className="text-gray-700">·</span>
            <span className="text-gray-400 text-xs tabular-nums">
              {formatElapsed(elapsedMs)}
            </span>
          </>
        )}
      </div>

      {/* Right: Stage badge */}
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 uppercase tracking-wider font-medium hidden sm:block">
          Stage
        </span>
        <span
          className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-medium ${badgeColor}`}
        >
          {/* Pulsing dot for non-done stages */}
          {currentStage !== 'done' && (
            <span className="relative flex h-1.5 w-1.5">
              <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-current opacity-50" />
              <span className="relative inline-flex rounded-full h-1.5 w-1.5 bg-current" />
            </span>
          )}
          {stageLabel}
        </span>
      </div>
    </header>
  );
}
