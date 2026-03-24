interface ApproveBarProps {
  onApproveAll: () => void;
  onReviewIndividually: () => void;
  requiresConfirmation: number;
  totalProtections: number;
}

export function ApproveBar({
  onApproveAll,
  onReviewIndividually,
  requiresConfirmation,
  totalProtections,
}: ApproveBarProps) {
  return (
    <div className="border-t border-gray-800 bg-gray-900/80 backdrop-blur-sm px-4 py-3 space-y-2.5">
      {/* Status line */}
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-300">
          <span className="font-semibold text-white tabular-nums">{totalProtections}</span>
          {' '}
          {totalProtections === 1 ? 'element' : 'elements'} will be protected
        </span>
        {requiresConfirmation > 0 && (
          <div className="flex items-center gap-1.5 text-yellow-400 text-xs font-medium">
            <svg className="w-3.5 h-3.5 shrink-0" viewBox="0 0 16 16" fill="none" aria-hidden="true">
              <path
                d="M8 2.5l5.5 10H2.5L8 2.5z"
                stroke="currentColor"
                strokeWidth="1.5"
                strokeLinejoin="round"
              />
              <path d="M8 6.5v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
              <circle cx="8" cy="11.5" r="0.75" fill="currentColor" />
            </svg>
            <span>
              <span className="tabular-nums">{requiresConfirmation}</span>
              {' '}
              {requiresConfirmation === 1 ? 'element needs' : 'elements need'} your confirmation
            </span>
          </div>
        )}
      </div>

      {/* Action buttons */}
      <div className="flex items-center gap-2">
        <button
          type="button"
          onClick={onApproveAll}
          className={[
            'flex-1 px-4 py-2 rounded-lg text-sm font-semibold transition-colors',
            'bg-blue-600 hover:bg-blue-500 text-white',
            'focus:outline-none focus:ring-2 focus:ring-blue-500/50',
          ].join(' ')}
        >
          Approve All
        </button>
        <button
          type="button"
          onClick={onReviewIndividually}
          className={[
            'flex-1 px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700',
            'focus:outline-none focus:ring-2 focus:ring-gray-500/50',
          ].join(' ')}
        >
          Review Individually
        </button>
      </div>
    </div>
  );
}
