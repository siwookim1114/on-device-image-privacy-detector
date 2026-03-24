import type { ReactNode } from 'react';
import type { ObfuscationMethod } from '../../types/strategy';

interface MethodSelectorProps {
  currentMethod: ObfuscationMethod | null;
  onChange: (method: ObfuscationMethod) => void;
  disabled: boolean;
}

interface MethodOption {
  value: ObfuscationMethod;
  label: string;
  icon: ReactNode;
  description: string;
}

const BlurIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <circle cx="8" cy="8" r="5" stroke="currentColor" strokeWidth="1.5" strokeDasharray="2 1.5" />
    <circle cx="8" cy="8" r="2" fill="currentColor" opacity="0.4" />
  </svg>
);

const PixelateIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <rect x="2" y="2" width="5" height="5" fill="currentColor" opacity="0.7" rx="0.5" />
    <rect x="9" y="2" width="5" height="5" fill="currentColor" opacity="0.4" rx="0.5" />
    <rect x="2" y="9" width="5" height="5" fill="currentColor" opacity="0.4" rx="0.5" />
    <rect x="9" y="9" width="5" height="5" fill="currentColor" opacity="0.7" rx="0.5" />
  </svg>
);

const SolidOverlayIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <rect x="2" y="2" width="12" height="12" fill="currentColor" opacity="0.8" rx="1.5" />
  </svg>
);

const AvatarReplaceIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <circle cx="8" cy="6" r="3" stroke="currentColor" strokeWidth="1.5" />
    <path d="M2 14c0-3.314 2.686-5 6-5s6 1.686 6 5" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const NoneIcon = () => (
  <svg className="w-3.5 h-3.5" viewBox="0 0 16 16" fill="none" aria-hidden="true">
    <circle cx="8" cy="8" r="5.5" stroke="currentColor" strokeWidth="1.5" />
    <path d="M4.5 11.5l7-7" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" />
  </svg>
);

const METHOD_OPTIONS: MethodOption[] = [
  { value: 'blur', label: 'Blur', icon: <BlurIcon />, description: 'Gaussian blur over region' },
  { value: 'pixelate', label: 'Pixelate', icon: <PixelateIcon />, description: 'Block pixelation effect' },
  { value: 'solid_overlay', label: 'Solid Overlay', icon: <SolidOverlayIcon />, description: 'Opaque color block' },
  { value: 'avatar_replace', label: 'Avatar Replace', icon: <AvatarReplaceIcon />, description: 'Replace with generic avatar' },
  { value: 'none', label: 'None', icon: <NoneIcon />, description: 'Remove protection' },
];

export function MethodSelector({ currentMethod, onChange, disabled }: MethodSelectorProps) {
  const current = METHOD_OPTIONS.find((o) => o.value === currentMethod) ?? null;

  return (
    <div className="relative">
      <select
        value={currentMethod ?? ''}
        onChange={(e) => {
          const val = e.target.value as ObfuscationMethod;
          if (val) onChange(val);
        }}
        disabled={disabled}
        aria-label="Select obfuscation method"
        className={[
          'w-full appearance-none rounded-lg px-3 py-2 text-sm',
          'bg-gray-800 border border-gray-700 text-gray-200',
          'focus:outline-none focus:ring-2 focus:ring-blue-500/50 focus:border-blue-500',
          'disabled:opacity-50 disabled:cursor-not-allowed',
          current?.value === 'none' ? 'text-red-400 border-red-500/40 bg-red-950/20' : '',
          'pr-8',
        ]
          .filter(Boolean)
          .join(' ')}
      >
        <option value="" disabled>
          Select method...
        </option>
        {METHOD_OPTIONS.map((option) => (
          <option
            key={option.value}
            value={option.value}
            className={option.value === 'none' ? 'text-red-400' : 'text-gray-200'}
          >
            {option.label} — {option.description}
          </option>
        ))}
      </select>

      <div className="pointer-events-none absolute inset-y-0 right-2 flex items-center">
        <svg className="w-4 h-4 text-gray-500" viewBox="0 0 16 16" fill="none" aria-hidden="true">
          <path d="M4 6l4 4 4-4" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" />
        </svg>
      </div>

      {current && (
        <div
          className={[
            'mt-1.5 flex items-center gap-1.5 text-xs px-1',
            current.value === 'none' ? 'text-red-400' : 'text-gray-400',
          ].join(' ')}
        >
          <span className={current.value === 'none' ? 'text-red-400' : 'text-gray-500'}>{current.icon}</span>
          <span>{current.description}</span>
          {current.value === 'none' && (
            <span className="ml-1 text-red-400 font-medium">(removes protection)</span>
          )}
        </div>
      )}
    </div>
  );
}
