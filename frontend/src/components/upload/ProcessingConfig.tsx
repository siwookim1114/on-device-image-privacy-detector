import type { ReactNode } from 'react';

import type { ProcessingMode, EthicalMode } from '../../types/pipeline';

interface ProcessingConfigProps {
  mode: ProcessingMode;
  ethicalMode: EthicalMode;
  onModeChange: (mode: ProcessingMode) => void;
  onEthicalModeChange: (mode: EthicalMode) => void;
}

interface ModeOption {
  value: ProcessingMode;
  label: string;
  badge?: string;
  description: string;
}

interface EthicalOption {
  value: EthicalMode;
  label: string;
  description: string;
}

const MODE_OPTIONS: ModeOption[] = [
  { value: 'auto', label: 'Auto', badge: 'Recommended', description: 'Fully automatic, no review needed' },
  { value: 'hybrid', label: 'Hybrid', description: 'AI processes, you review critical decisions' },
  { value: 'manual', label: 'Manual', description: 'Review every stage' },
];

const ETHICAL_OPTIONS: EthicalOption[] = [
  { value: 'strict', label: 'Strict', description: 'No deception — blur, pixelate, overlay only' },
  { value: 'balanced', label: 'Balanced', description: 'Context-preserving — includes inpainting' },
  { value: 'creative', label: 'Creative', description: 'Full power — generative replacement, requires watermark' },
];

function SectionLabel({ children }: { children: ReactNode }) {
  return <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">{children}</p>;
}

export function ProcessingConfig({ mode, ethicalMode, onModeChange, onEthicalModeChange }: ProcessingConfigProps) {
  return (
    <div className="space-y-6">
      {/* Processing mode */}
      <div>
        <SectionLabel>Processing Mode</SectionLabel>
        <div className="grid grid-cols-3 gap-3">
          {MODE_OPTIONS.map((opt) => {
            const selected = mode === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onModeChange(opt.value)}
                aria-pressed={selected}
                className={`relative rounded-xl border px-4 py-3 text-left transition-colors duration-150 hover:border-blue-500/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                  selected
                    ? 'border-blue-500 bg-blue-500/8'
                    : 'border-gray-700 bg-gray-900 hover:bg-gray-800/60'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-sm font-medium ${selected ? 'text-blue-300' : 'text-gray-200'}`}
                  >
                    {opt.label}
                  </span>
                  {opt.badge && (
                    <span className="rounded-full bg-blue-500/20 px-1.5 py-0.5 text-[10px] font-semibold text-blue-400 leading-none">
                      {opt.badge}
                    </span>
                  )}
                </div>
                <p className="text-xs text-gray-500 leading-snug">{opt.description}</p>

                {selected && (
                  <span className="absolute top-2.5 right-2.5 w-2 h-2 rounded-full bg-blue-500" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      </div>

      {/* Ethical mode */}
      <div>
        <SectionLabel>Ethical Mode</SectionLabel>
        <div className="grid grid-cols-3 gap-3">
          {ETHICAL_OPTIONS.map((opt) => {
            const selected = ethicalMode === opt.value;
            return (
              <button
                key={opt.value}
                type="button"
                onClick={() => onEthicalModeChange(opt.value)}
                aria-pressed={selected}
                className={`relative rounded-xl border px-4 py-3 text-left transition-colors duration-150 hover:border-blue-500/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                  selected
                    ? 'border-blue-500 bg-blue-500/8'
                    : 'border-gray-700 bg-gray-900 hover:bg-gray-800/60'
                }`}
              >
                <div className="flex items-center gap-2 mb-1">
                  <span
                    className={`text-sm font-medium ${selected ? 'text-blue-300' : 'text-gray-200'}`}
                  >
                    {opt.label}
                  </span>
                </div>
                <p className="text-xs text-gray-500 leading-snug">{opt.description}</p>

                {selected && (
                  <span className="absolute top-2.5 right-2.5 w-2 h-2 rounded-full bg-blue-500" aria-hidden="true" />
                )}
              </button>
            );
          })}
        </div>
      </div>
    </div>
  );
}
