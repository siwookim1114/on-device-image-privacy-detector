import { useState } from 'react';

import { BeforeAfterSlider } from './BeforeAfterSlider';

type ComparisonMode = 'slider' | 'sidebyside' | 'toggle';

interface ComparisonViewProps {
  originalUrl: string;
  protectedUrl: string;
}

const TABS: { id: ComparisonMode; label: string }[] = [
  { id: 'slider', label: 'Slider' },
  { id: 'sidebyside', label: 'Side-by-side' },
  { id: 'toggle', label: 'Toggle' },
];

export function ComparisonView({ originalUrl, protectedUrl }: ComparisonViewProps) {
  const [mode, setMode] = useState<ComparisonMode>('slider');
  const [showOriginal, setShowOriginal] = useState(true);

  return (
    <div className="flex flex-col flex-1 overflow-hidden min-h-0">
      {/* Tab bar */}
      <div className="flex items-center gap-1 px-3 py-2 border-b border-gray-800 shrink-0">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            type="button"
            onClick={() => setMode(tab.id)}
            className={[
              'px-3 py-1.5 rounded-md text-xs font-medium transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500/50',
              mode === tab.id
                ? 'bg-blue-600 text-white'
                : 'text-gray-400 hover:text-gray-200 hover:bg-gray-800',
            ].join(' ')}
            aria-pressed={mode === tab.id}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Content area */}
      <div className="flex-1 overflow-hidden min-h-0">
        {mode === 'slider' && (
          <div className="w-full h-full">
            <BeforeAfterSlider originalUrl={originalUrl} protectedUrl={protectedUrl} />
          </div>
        )}

        {mode === 'sidebyside' && (
          <div className="flex h-full gap-2 p-2">
            <div className="flex-1 relative overflow-hidden rounded-lg bg-gray-950">
              <img
                src={originalUrl}
                alt="Original"
                className="w-full h-full object-contain"
              />
              <span className="absolute top-2 left-2 text-xs font-semibold text-white
                               bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded pointer-events-none">
                Original
              </span>
            </div>
            <div className="flex-1 relative overflow-hidden rounded-lg bg-gray-950">
              <img
                src={protectedUrl}
                alt="Protected"
                className="w-full h-full object-contain"
              />
              <span className="absolute top-2 left-2 text-xs font-semibold text-white
                               bg-black/50 backdrop-blur-sm px-2 py-0.5 rounded pointer-events-none">
                Protected
              </span>
            </div>
          </div>
        )}

        {mode === 'toggle' && (
          <div
            className="relative w-full h-full cursor-pointer"
            onClick={() => setShowOriginal((v) => !v)}
            role="button"
            tabIndex={0}
            aria-label={`Currently showing ${showOriginal ? 'original' : 'protected'} — click to toggle`}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') setShowOriginal((v) => !v);
            }}
          >
            <img
              src={showOriginal ? originalUrl : protectedUrl}
              alt={showOriginal ? 'Original' : 'Protected'}
              className="w-full h-full object-contain"
            />
            <div className="absolute bottom-3 left-1/2 -translate-x-1/2 flex flex-col items-center gap-1 pointer-events-none">
              <span className="text-xs font-semibold text-white bg-black/60 backdrop-blur-sm px-3 py-1 rounded-full">
                {showOriginal ? 'Original' : 'Protected'}
              </span>
              <span className="text-xs text-gray-300 bg-black/40 backdrop-blur-sm px-2 py-0.5 rounded">
                Click to toggle
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
