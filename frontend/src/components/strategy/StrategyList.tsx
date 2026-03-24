import { useState } from 'react';
import type { ProtectionStrategy, ObfuscationMethod } from '../../types/strategy';
import { StrategyCard } from './StrategyCard';

interface StrategyListProps {
  strategies: ProtectionStrategy[];
  onMethodChange: (detectionId: string, method: ObfuscationMethod) => void;
  onParamsChange: (detectionId: string, params: Record<string, unknown>) => void;
}

interface GroupHeaderProps {
  label: string;
  count: number;
  isOpen: boolean;
  onToggle: () => void;
  variant: 'protect' | 'none';
}

function GroupHeader({ label, count, isOpen, onToggle, variant }: GroupHeaderProps) {
  const accentColor =
    variant === 'protect'
      ? 'text-blue-400 border-blue-500/30 bg-blue-500/10'
      : 'text-gray-500 border-gray-700 bg-gray-800/50';

  return (
    <button
      type="button"
      onClick={onToggle}
      className="w-full flex items-center justify-between py-2 px-1 rounded-lg hover:bg-gray-800/40 transition-colors group"
      aria-expanded={isOpen}
      aria-controls={`group-${variant}`}
    >
      <div className="flex items-center gap-2">
        <svg
          className={[
            'w-3.5 h-3.5 transition-transform duration-150',
            isOpen ? 'rotate-90' : '',
            variant === 'protect' ? 'text-blue-400' : 'text-gray-500',
          ].join(' ')}
          viewBox="0 0 16 16"
          fill="none"
          aria-hidden="true"
        >
          <path d="M6 4l4 4-4 4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
        <span
          className={[
            'text-xs font-semibold uppercase tracking-wider',
            variant === 'protect' ? 'text-blue-300' : 'text-gray-500',
          ].join(' ')}
        >
          {label}
        </span>
        <span
          className={[
            'inline-flex items-center justify-center text-xs font-semibold px-1.5 py-0.5 rounded-full border min-w-[1.25rem]',
            accentColor,
          ].join(' ')}
        >
          {count}
        </span>
      </div>
    </button>
  );
}

export function StrategyList({ strategies, onMethodChange, onParamsChange }: StrategyListProps) {
  const willProtect = strategies.filter((s) => s.recommended_method !== null && s.recommended_method !== 'none');
  const noAction = strategies.filter((s) => s.recommended_method === null || s.recommended_method === 'none');

  const [protectOpen, setProtectOpen] = useState(true);
  const [noActionOpen, setNoActionOpen] = useState(false);

  return (
    <div className="space-y-4">
      {/* Will Protect group */}
      {willProtect.length > 0 && (
        <section>
          <GroupHeader
            label="Will Protect"
            count={willProtect.length}
            isOpen={protectOpen}
            onToggle={() => setProtectOpen((v) => !v)}
            variant="protect"
          />
          {protectOpen && (
            <div id="group-protect" className="mt-2 space-y-3">
              {willProtect.map((strategy) => (
                <StrategyCard
                  key={strategy.detection_id}
                  strategy={strategy}
                  onMethodChange={onMethodChange}
                  onParamsChange={onParamsChange}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {/* No Action group */}
      {noAction.length > 0 && (
        <section>
          <GroupHeader
            label="No Action Needed"
            count={noAction.length}
            isOpen={noActionOpen}
            onToggle={() => setNoActionOpen((v) => !v)}
            variant="none"
          />
          {noActionOpen && (
            <div id="group-none" className="mt-2 space-y-3">
              {noAction.map((strategy) => (
                <StrategyCard
                  key={strategy.detection_id}
                  strategy={strategy}
                  onMethodChange={onMethodChange}
                  onParamsChange={onParamsChange}
                />
              ))}
            </div>
          )}
        </section>
      )}

      {strategies.length === 0 && (
        <p className="text-sm text-gray-500 text-center py-8">No strategies to display.</p>
      )}
    </div>
  );
}
