import { useState } from 'react';
import { useOnboardingStore } from '../../stores/onboardingStore';
import { SegmentedControl } from './SegmentedControl';
import type { SegmentOption } from './SegmentedControl';
import type { AutoAdvanceLevel } from '../../types/profile';

const METHOD_OPTIONS: Record<'face' | 'text' | 'screen' | 'object', string[]> = {
  face: ['blur', 'pixelate', 'solid_overlay'],
  text: ['solid_overlay', 'blur', 'pixelate'],
  screen: ['blur', 'solid_overlay', 'pixelate'],
  object: ['blur', 'solid_overlay', 'pixelate'],
};

const THRESHOLD_OPTIONS: SegmentOption[] = [
  { value: 'never', label: 'Never', description: 'Pipeline runs end-to-end without any manual review pauses.' },
  { value: 'low', label: 'Low', description: 'Pause when LOW risk items are flagged for manual confirmation.' },
  { value: 'medium', label: 'Medium', description: 'Pause on MEDIUM+ risk. LOW items auto-advance.', default: true },
  { value: 'high', label: 'High', description: 'Pause only on HIGH/CRITICAL risk items. Most decisions automated.' },
];

interface SummaryCardProps {
  title: string;
  onEdit: () => void;
  children: React.ReactNode;
}

function SummaryCard({ title, onEdit, children }: SummaryCardProps) {
  return (
    <div className="rounded-xl border border-gray-700 bg-gray-800/30 overflow-hidden">
      <div className="flex items-center justify-between px-4 py-3 border-b border-gray-700/60">
        <h3 className="text-sm font-semibold text-gray-200">{title}</h3>
        <button
          type="button"
          onClick={onEdit}
          className="text-xs text-blue-400 hover:text-blue-300 transition-colors focus-visible:outline-none focus-visible:ring-1 focus-visible:ring-blue-500 rounded"
        >
          Edit
        </button>
      </div>
      <div className="px-4 py-3">{children}</div>
    </div>
  );
}

interface ChevronProps {
  isOpen: boolean;
}

function ChevronIcon({ isOpen }: ChevronProps) {
  return (
    <svg
      className={`w-4 h-4 text-gray-400 transition-transform duration-200 ${isOpen ? 'rotate-180' : ''}`}
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
    </svg>
  );
}

const ACCENT_DOT: Record<string, string> = {
  faces: 'bg-blue-500',
  text: 'bg-amber-500',
  screens: 'bg-purple-500',
  objects: 'bg-teal-500',
};

interface StepReviewProps {
  onEdit: (step: number) => void;
  onFinish: () => void;
  isSaving: boolean;
}

export function StepReview({ onEdit, onFinish, isSaving }: StepReviewProps) {
  const { identity, contacts, sensitivity, advanced, setAdvanced } = useOnboardingStore();
  const [advancedOpen, setAdvancedOpen] = useState(false);

  const sensitivityEntries = [
    { key: 'faces', label: 'Faces' },
    { key: 'text', label: 'Text & PII' },
    { key: 'screens', label: 'Screens' },
    { key: 'objects', label: 'Objects' },
  ] as const;

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold text-gray-100">Review Your Setup</h2>
        <p className="text-sm text-gray-400">
          Confirm your preferences before finishing. You can change these anytime in Settings.
        </p>
      </div>

      {/* Identity summary */}
      <SummaryCard title="Identity" onEdit={() => onEdit(0)}>
        {identity.skipped ? (
          <p className="text-sm text-gray-500 italic">Skipped — no identity registered</p>
        ) : (
          <div className="flex items-center gap-3">
            {identity.photos.length > 0 ? (
              <img
                src={URL.createObjectURL(identity.photos[0]!)}
                alt=""
                className="w-10 h-10 rounded-full object-cover border border-gray-600"
              />
            ) : (
              <div className="w-10 h-10 rounded-full bg-gray-700 flex items-center justify-center text-gray-500 text-sm">
                ?
              </div>
            )}
            <div>
              <p className="text-sm font-medium text-gray-200">
                {identity.label || <span className="text-gray-500 italic">No label</span>}
              </p>
              <p className="text-xs text-gray-500">
                {identity.photos.length} photo{identity.photos.length !== 1 ? 's' : ''} &middot; self &middot; explicit consent
              </p>
            </div>
          </div>
        )}
      </SummaryCard>

      {/* Contacts summary */}
      <SummaryCard title={`Contacts (${contacts.length})`} onEdit={() => onEdit(1)}>
        {contacts.length === 0 ? (
          <p className="text-sm text-gray-500 italic">No contacts registered</p>
        ) : (
          <ul className="space-y-1.5">
            {contacts.map((c, i) => (
              <li key={i} className="flex items-center gap-2 text-sm text-gray-300">
                <span className="w-1.5 h-1.5 rounded-full bg-gray-500 shrink-0" aria-hidden="true" />
                <span className="font-medium">{c.display_name}</span>
                <span className="text-gray-500 text-xs capitalize">
                  {c.relationship} &middot; {c.consent_level}
                </span>
              </li>
            ))}
          </ul>
        )}
      </SummaryCard>

      {/* Sensitivity summary */}
      <SummaryCard title="Sensitivity Preferences" onEdit={() => onEdit(2)}>
        <ul className="space-y-2">
          {sensitivityEntries.map(({ key, label }) => (
            <li key={key} className="flex items-center justify-between">
              <div className="flex items-center gap-2">
                <span
                  className={`w-2 h-2 rounded-full ${ACCENT_DOT[key] ?? 'bg-gray-500'}`}
                  aria-hidden="true"
                />
                <span className="text-sm text-gray-300">{label}</span>
              </div>
              <span className="text-xs font-medium text-gray-400 capitalize bg-gray-700/60 rounded px-2 py-0.5">
                {sensitivity[key]}
              </span>
            </li>
          ))}
        </ul>
      </SummaryCard>

      {/* Advanced settings (collapsible) */}
      <div className="rounded-xl border border-gray-700 overflow-hidden">
        <button
          type="button"
          onClick={() => setAdvancedOpen((prev) => !prev)}
          aria-expanded={advancedOpen}
          className="w-full flex items-center justify-between px-4 py-3 bg-gray-800/30 hover:bg-gray-800/50 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-inset"
        >
          <span className="text-sm font-semibold text-gray-200">Advanced Settings</span>
          <ChevronIcon isOpen={advancedOpen} />
        </button>

        {advancedOpen && (
          <div className="px-4 py-4 space-y-5 border-t border-gray-700/60">
            {/* Method preferences */}
            {(
              [
                { key: 'preferred_face_method', label: 'Face method', type: 'face' },
                { key: 'preferred_text_method', label: 'Text method', type: 'text' },
                { key: 'preferred_screen_method', label: 'Screen method', type: 'screen' },
                { key: 'preferred_object_method', label: 'Object method', type: 'object' },
              ] as const
            ).map(({ key, label, type }) => (
              <div key={key} className="space-y-1.5">
                <label
                  htmlFor={`method-${key}`}
                  className="block text-xs font-medium text-gray-400"
                >
                  {label}
                </label>
                <select
                  id={`method-${key}`}
                  value={advanced[key]}
                  onChange={(e) => setAdvanced(key, e.target.value)}
                  className="w-full rounded-lg px-3 py-2 text-sm bg-gray-800 text-gray-100 border border-gray-700 hover:border-gray-600 transition-colors focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 appearance-none cursor-pointer"
                >
                  {METHOD_OPTIONS[type].map((m) => (
                    <option key={m} value={m}>
                      {m.replace(/_/g, ' ')}
                    </option>
                  ))}
                </select>
              </div>
            ))}

            {/* Review threshold */}
            <div className="space-y-2">
              <SegmentedControl
                options={THRESHOLD_OPTIONS}
                value={advanced.auto_advance_threshold}
                onChange={(v) => setAdvanced('auto_advance_threshold', v as AutoAdvanceLevel)}
                accentColor="blue"
                label="Review threshold"
              />
            </div>

            {/* Pause on critical */}
            <div className="flex items-center justify-between">
              <div>
                <p className="text-sm font-medium text-gray-300">Pause on CRITICAL</p>
                <p className="text-xs text-gray-500">Always pause for manual review when CRITICAL items are detected</p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={advanced.pause_on_critical}
                onClick={() => setAdvanced('pause_on_critical', !advanced.pause_on_critical)}
                className={`
                  relative inline-flex h-6 w-11 shrink-0 cursor-pointer rounded-full border-2 border-transparent
                  transition-colors duration-200 ease-in-out
                  focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900
                  ${advanced.pause_on_critical ? 'bg-blue-600' : 'bg-gray-700'}
                `}
              >
                <span
                  aria-hidden="true"
                  className={`
                    pointer-events-none inline-block h-5 w-5 rounded-full bg-white shadow-md
                    transition-transform duration-200 ease-in-out
                    ${advanced.pause_on_critical ? 'translate-x-5' : 'translate-x-0'}
                  `}
                />
              </button>
            </div>
          </div>
        )}
      </div>

      {/* Finish button */}
      <div className="flex justify-end pt-2">
        <button
          type="button"
          onClick={onFinish}
          disabled={isSaving}
          aria-disabled={isSaving}
          className="rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-semibold px-6 py-2.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
        >
          {isSaving ? (
            <span className="flex items-center gap-2">
              <svg className="w-4 h-4 animate-spin" fill="none" viewBox="0 0 24 24" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Saving...
            </span>
          ) : (
            'Finish Setup'
          )}
        </button>
      </div>
    </div>
  );
}
