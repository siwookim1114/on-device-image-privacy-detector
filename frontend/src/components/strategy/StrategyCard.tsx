import type { ProtectionStrategy, ObfuscationMethod } from '../../types/strategy';
import { SeverityBadge } from '../detection/SeverityBadge';
import { MethodSelector } from './MethodSelector';
import { ParamSliders } from './ParamSliders';

interface StrategyCardProps {
  strategy: ProtectionStrategy;
  onMethodChange: (detectionId: string, method: ObfuscationMethod) => void;
  onParamsChange: (detectionId: string, params: Record<string, unknown>) => void;
}

function paramSummary(method: ObfuscationMethod | null, params: Record<string, unknown>): string {
  if (!method || method === 'none') return '';
  if (method === 'blur') {
    const k = params['kernel_size'];
    return `kernel ${typeof k === 'number' ? k : 35}`;
  }
  if (method === 'pixelate') {
    const b = params['block_size'];
    return `block ${typeof b === 'number' ? b : 16}`;
  }
  if (method === 'solid_overlay') {
    const c = params['color'];
    return `color ${typeof c === 'string' ? c : '#000000'}`;
  }
  if (method === 'avatar_replace') return 'generic avatar';
  return '';
}

export function StrategyCard({ strategy, onMethodChange, onParamsChange }: StrategyCardProps) {
  const needsDecision = strategy.requires_user_decision;
  const summary = paramSummary(strategy.recommended_method, strategy.parameters);
  const hasParams =
    strategy.recommended_method !== null &&
    strategy.recommended_method !== 'none' &&
    strategy.recommended_method !== 'avatar_replace' &&
    strategy.recommended_method !== 'inpaint';

  return (
    <article
      className={[
        'rounded-xl border p-4 space-y-3 transition-colors',
        needsDecision
          ? 'border-yellow-500/40 bg-yellow-500/5'
          : 'border-gray-800 bg-gray-900/60',
      ].join(' ')}
      aria-label={`Strategy for ${strategy.element}`}
    >
      {/* Header row */}
      <div className="flex items-start justify-between gap-2">
        <div className="min-w-0 flex-1">
          <div className="flex items-center gap-2 flex-wrap">
            <span className="text-sm font-medium text-gray-100 truncate">{strategy.element}</span>
            <SeverityBadge severity={strategy.severity} size="sm" />
            {needsDecision && (
              <span className="inline-flex items-center gap-1 text-xs px-2 py-0.5 rounded-full bg-yellow-500/15 text-yellow-400 border border-yellow-500/30">
                <svg className="w-3 h-3" viewBox="0 0 16 16" fill="none" aria-hidden="true">
                  <path
                    d="M8 2.5l5.5 10H2.5L8 2.5z"
                    stroke="currentColor"
                    strokeWidth="1.5"
                    strokeLinejoin="round"
                  />
                  <path d="M8 6.5v3" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
                  <circle cx="8" cy="11.5" r="0.75" fill="currentColor" />
                </svg>
                Review needed
              </span>
            )}
          </div>
          {summary && (
            <p className="text-xs text-gray-500 mt-0.5 font-mono">{summary}</p>
          )}
        </div>
      </div>

      {/* Method selector */}
      <div>
        <label className="block text-xs text-gray-500 mb-1.5">Method</label>
        <MethodSelector
          currentMethod={strategy.recommended_method}
          onChange={(method) => onMethodChange(strategy.detection_id, method)}
          disabled={false}
        />
      </div>

      {/* Parameter sliders — only shown when method has configurable params */}
      {hasParams && strategy.recommended_method && (
        <div>
          <label className="block text-xs text-gray-500 mb-1.5">Parameters</label>
          <ParamSliders
            method={strategy.recommended_method}
            parameters={strategy.parameters}
            onChange={(params) => onParamsChange(strategy.detection_id, params)}
          />
        </div>
      )}

      {/* Reasoning */}
      <p className="text-xs text-gray-500 leading-relaxed line-clamp-2">{strategy.reasoning}</p>
    </article>
  );
}
