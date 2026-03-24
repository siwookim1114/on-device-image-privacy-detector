interface ActionBarProps {
  onAccept: () => void;
  onAdjust: () => void;
  onReprocess: () => void;
  patchCount: number;
}

export function ActionBar({ onAccept, onAdjust, onReprocess, patchCount }: ActionBarProps) {
  return (
    <div className="sticky bottom-0 bg-gray-900 border-t border-gray-800 p-4">
      {patchCount > 0 && (
        <p className="text-xs text-blue-400 mb-3 text-center">
          <span className="tabular-nums font-semibold">{patchCount}</span>
          {' '}
          {patchCount === 1 ? 'patch' : 'patches'} applied during verification
        </p>
      )}

      <div className="flex items-center justify-between gap-3">
        <button
          type="button"
          onClick={onReprocess}
          className={[
            'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
            'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700',
            'focus:outline-none focus:ring-2 focus:ring-gray-500/50',
          ].join(' ')}
        >
          Re-process
        </button>

        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onAdjust}
            className={[
              'px-4 py-2 rounded-lg text-sm font-medium transition-colors',
              'bg-gray-800 hover:bg-gray-700 text-gray-300 border border-gray-700',
              'focus:outline-none focus:ring-2 focus:ring-gray-500/50',
            ].join(' ')}
          >
            Adjust Protection
          </button>

          <button
            type="button"
            onClick={onAccept}
            className={[
              'px-5 py-2 rounded-lg text-sm font-semibold transition-colors',
              'bg-green-600 hover:bg-green-500 text-white',
              'focus:outline-none focus:ring-2 focus:ring-green-500/50',
            ].join(' ')}
          >
            Accept &amp; Export
          </button>
        </div>
      </div>
    </div>
  );
}
