import { useOnboardingStore } from '../../stores/onboardingStore';
import { useStudyLogger } from '../../hooks/useStudyLogger';
import { SegmentedControl } from './SegmentedControl';
import type { SegmentOption } from './SegmentedControl';
import type { OnboardingState } from '../../types/profile';

const FACE_OPTIONS: SegmentOption[] = [
  {
    value: 'paranoid',
    label: 'Paranoid',
    description: 'Blur every visible face regardless of consent or context. Maximum privacy, minimum risk.',
  },
  {
    value: 'cautious',
    label: 'Cautious',
    description: 'Protect unknown faces and bystanders. Skip faces with explicit consent on record.',
    default: true,
  },
  {
    value: 'relaxed',
    label: 'Relaxed',
    description: 'Only protect faces flagged as CRITICAL risk. Trusted contacts are always skipped.',
  },
];

const TEXT_OPTIONS: SegmentOption[] = [
  {
    value: 'maximum',
    label: 'Maximum',
    description: 'Redact all detected text including names, labels, and numeric fragments.',
  },
  {
    value: 'standard',
    label: 'Standard',
    description: 'Redact PII (SSN, credit card, passwords, addresses). Skip generic labels.',
    default: true,
  },
  {
    value: 'minimal',
    label: 'Minimal',
    description: 'Only redact CRITICAL items (SSN, credit card, passwords). Allow other text.',
  },
];

const SCREEN_OPTIONS: SegmentOption[] = [
  {
    value: 'always',
    label: 'Always',
    description: 'Blur all detected screens whether on or off. No VLM verification needed.',
  },
  {
    value: 'smart',
    label: 'Smart',
    description: 'Use VLM to verify if screen is visibly on before protecting. Best balance.',
    default: true,
  },
  {
    value: 'never',
    label: 'Never',
    description: 'Never protect screen content. Useful when screens contain non-sensitive data.',
  },
];

const OBJECT_OPTIONS: SegmentOption[] = [
  {
    value: 'include',
    label: 'Include',
    description: 'Protect privacy-relevant objects (ID cards, documents). Full object coverage.',
  },
  {
    value: 'standard',
    label: 'Standard',
    description: 'Protect high-risk objects only. Ignore generic items like cups or furniture.',
    default: true,
  },
  {
    value: 'minimal',
    label: 'Minimal',
    description: 'Skip object detection entirely. Focus only on faces and text.',
  },
];

interface PreviewProps {
  element: 'faces' | 'text' | 'screens' | 'objects';
  level: string;
  accentColor: 'blue' | 'amber' | 'purple' | 'teal';
}

const INTENSITY_MAP: Record<string, number> = {
  paranoid: 3, maximum: 3, always: 3, include: 3,
  cautious: 2, standard: 2, smart: 2,
  relaxed: 1, minimal: 1, never: 0,
};

const ACCENT_BG: Record<string, string> = {
  blue: 'bg-blue-500',
  amber: 'bg-amber-500',
  purple: 'bg-purple-500',
  teal: 'bg-teal-500',
};

function SensitivityPreview({ element, level, accentColor }: PreviewProps) {
  const intensity = INTENSITY_MAP[level] ?? 1;
  const bgClass = ACCENT_BG[accentColor] ?? 'bg-blue-500';

  const bars = element === 'faces'
    ? [{ w: 'w-10 h-10', label: 'face region' }]
    : element === 'text'
      ? [{ w: 'w-24 h-3', label: 'text line 1' }, { w: 'w-16 h-3', label: 'text line 2' }]
      : element === 'screens'
        ? [{ w: 'w-20 h-14', label: 'screen area' }]
        : [{ w: 'w-16 h-8', label: 'object region' }];

  return (
    <div
      aria-hidden="true"
      className="flex items-center justify-center gap-3 rounded-lg bg-gray-900 border border-gray-700/50 h-20 px-4 overflow-hidden"
    >
      {/* "Before" — raw element */}
      <div className="flex flex-col items-center gap-1 opacity-60">
        <div className="text-[9px] text-gray-600 uppercase tracking-wide">Before</div>
        <div className="flex flex-col items-center gap-1">
          {bars.map((b, i) => (
            <div
              key={i}
              className={`${b.w} rounded bg-gray-600/50 border border-gray-600`}
              aria-label={b.label}
            />
          ))}
        </div>
      </div>

      {/* Arrow */}
      <svg className="w-4 h-4 text-gray-600 shrink-0" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
        <path strokeLinecap="round" strokeLinejoin="round" d="M13 7l5 5m0 0l-5 5m5-5H6" />
      </svg>

      {/* "After" — protected */}
      <div className="flex flex-col items-center gap-1">
        <div className="text-[9px] text-gray-600 uppercase tracking-wide">After</div>
        <div className="flex flex-col items-center gap-1">
          {intensity === 0
            ? bars.map((b, i) => (
                <div
                  key={i}
                  className={`${b.w} rounded bg-gray-600/50 border border-gray-600`}
                />
              ))
            : bars.map((b, i) => (
                <div
                  key={i}
                  className={`${b.w} rounded ${bgClass} opacity-${intensity === 3 ? '90' : intensity === 2 ? '60' : '30'}`}
                  style={{
                    filter: intensity === 3 ? 'blur(3px) saturate(0)' : intensity === 2 ? 'blur(2px)' : 'blur(1px)',
                  }}
                />
              ))
          }
        </div>
      </div>
    </div>
  );
}

interface StepSensitivityProps {
  onContinue: () => void;
}

export function StepSensitivity({ onContinue }: StepSensitivityProps) {
  const { sensitivity, setSensitivity } = useOnboardingStore();
  const { log } = useStudyLogger();

  const handleChange = (
    key: keyof OnboardingState['sensitivity'],
    newValue: string,
  ) => {
    const oldValue = sensitivity[key];
    setSensitivity(key, newValue as OnboardingState['sensitivity'][typeof key]);
    log({
      event_type: 'onboarding_sensitivity_change',
      source: 'mouse',
      old_value: oldValue,
      new_value: newValue,
      metadata: { element_type: key },
    });
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold text-gray-100">Privacy Sensitivity</h2>
        <p className="text-sm text-gray-400">
          Set how aggressively each element type is protected. These become your default
          preferences for every image processed.
        </p>
      </div>

      <div className="space-y-8">
        {/* Faces */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-blue-500 shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-gray-200">Faces</h3>
          </div>
          <SensitivityPreview element="faces" level={sensitivity.faces} accentColor="blue" />
          <SegmentedControl
            options={FACE_OPTIONS}
            value={sensitivity.faces}
            onChange={(v) => handleChange('faces', v)}
            accentColor="blue"
            label="Face sensitivity"
          />
        </div>

        <div className="h-px bg-gray-800" aria-hidden="true" />

        {/* Text */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-amber-500 shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-gray-200">Text &amp; PII</h3>
          </div>
          <SensitivityPreview element="text" level={sensitivity.text} accentColor="amber" />
          <SegmentedControl
            options={TEXT_OPTIONS}
            value={sensitivity.text}
            onChange={(v) => handleChange('text', v)}
            accentColor="amber"
            label="Text sensitivity"
          />
        </div>

        <div className="h-px bg-gray-800" aria-hidden="true" />

        {/* Screens */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-purple-500 shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-gray-200">Screens &amp; Displays</h3>
          </div>
          <SensitivityPreview element="screens" level={sensitivity.screens} accentColor="purple" />
          <SegmentedControl
            options={SCREEN_OPTIONS}
            value={sensitivity.screens}
            onChange={(v) => handleChange('screens', v)}
            accentColor="purple"
            label="Screen sensitivity"
          />
        </div>

        <div className="h-px bg-gray-800" aria-hidden="true" />

        {/* Objects */}
        <div className="space-y-3">
          <div className="flex items-center gap-2">
            <span className="w-2 h-2 rounded-full bg-teal-500 shrink-0" aria-hidden="true" />
            <h3 className="text-sm font-semibold text-gray-200">Objects</h3>
          </div>
          <SensitivityPreview element="objects" level={sensitivity.objects} accentColor="teal" />
          <SegmentedControl
            options={OBJECT_OPTIONS}
            value={sensitivity.objects}
            onChange={(v) => handleChange('objects', v)}
            accentColor="teal"
            label="Object sensitivity"
          />
        </div>
      </div>

      {/* Continue button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onContinue}
          className="rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold px-5 py-2.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
        >
          Continue
        </button>
      </div>
    </div>
  );
}
