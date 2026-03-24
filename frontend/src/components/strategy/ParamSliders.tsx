import type { ObfuscationMethod } from '../../types/strategy';

interface ParamSlidersProps {
  method: ObfuscationMethod;
  parameters: Record<string, unknown>;
  onChange: (params: Record<string, unknown>) => void;
}

const SOLID_OVERLAY_PRESETS = [
  { label: 'Black', value: '#000000' },
  { label: 'Dark gray', value: '#1f2937' },
  { label: 'Gray', value: '#6b7280' },
  { label: 'White', value: '#ffffff' },
];

function SliderRow({
  label,
  value,
  min,
  max,
  step,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  onChange: (v: number) => void;
}) {
  return (
    <div className="flex items-center gap-3">
      <label className="text-xs text-gray-400 w-24 shrink-0">{label}</label>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(e) => onChange(Number(e.target.value))}
        className={[
          'flex-1 h-1.5 rounded-full appearance-none cursor-pointer',
          'bg-gray-700 accent-blue-500',
        ].join(' ')}
        aria-label={label}
      />
      <span className="text-xs text-gray-300 tabular-nums w-8 text-right shrink-0">{value}</span>
    </div>
  );
}

export function ParamSliders({ method, parameters, onChange }: ParamSlidersProps) {
  if (method === 'blur') {
    const kernelSize = typeof parameters['kernel_size'] === 'number' ? parameters['kernel_size'] : 35;
    return (
      <div className="space-y-2">
        <SliderRow
          label="Kernel size"
          value={kernelSize}
          min={15}
          max={99}
          step={2}
          onChange={(v) => onChange({ ...parameters, kernel_size: v })}
        />
      </div>
    );
  }

  if (method === 'pixelate') {
    const blockSize = typeof parameters['block_size'] === 'number' ? parameters['block_size'] : 16;
    return (
      <div className="space-y-2">
        <SliderRow
          label="Block size"
          value={blockSize}
          min={4}
          max={32}
          step={1}
          onChange={(v) => onChange({ ...parameters, block_size: v })}
        />
      </div>
    );
  }

  if (method === 'solid_overlay') {
    const currentColor = typeof parameters['color'] === 'string' ? parameters['color'] : '#000000';
    return (
      <div className="space-y-2">
        <span className="text-xs text-gray-400">Color</span>
        <div className="flex flex-wrap gap-2 mt-1">
          {SOLID_OVERLAY_PRESETS.map((preset) => (
            <button
              key={preset.value}
              type="button"
              onClick={() => onChange({ ...parameters, color: preset.value })}
              aria-label={preset.label}
              aria-pressed={currentColor === preset.value}
              className={[
                'flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border transition-colors',
                currentColor === preset.value
                  ? 'border-blue-500 text-blue-300 bg-blue-500/10'
                  : 'border-gray-700 text-gray-400 hover:border-gray-600 bg-gray-800',
              ].join(' ')}
            >
              <span
                className="w-3 h-3 rounded-sm border border-gray-600 shrink-0"
                style={{ backgroundColor: preset.value }}
                aria-hidden="true"
              />
              {preset.label}
            </button>
          ))}
          <label className="flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs border border-gray-700 text-gray-400 hover:border-gray-600 bg-gray-800 cursor-pointer transition-colors">
            <input
              type="color"
              value={currentColor}
              onChange={(e) => onChange({ ...parameters, color: e.target.value })}
              className="w-3 h-3 border-0 bg-transparent cursor-pointer p-0"
              aria-label="Custom color"
            />
            Custom
          </label>
        </div>
      </div>
    );
  }

  return null;
}
