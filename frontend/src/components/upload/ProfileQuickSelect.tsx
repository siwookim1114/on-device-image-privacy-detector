import type { ReactNode } from 'react';

type ProfileKey = 'strict' | 'balanced' | 'creative';

interface ProfileQuickSelectProps {
  selected: ProfileKey;
  onSelect: (profile: ProfileKey) => void;
}

interface Profile {
  key: ProfileKey;
  title: string;
  description: string;
  icon: ReactNode;
}

function ShieldIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z"
      />
    </svg>
  );
}

function ScaleIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 3v17.25m0 0c-1.472 0-2.882.265-4.185.75M12 20.25c1.472 0 2.882.265 4.185.75M18.75 4.97A48.416 48.416 0 0012 4.5c-2.291 0-4.545.16-6.75.47m13.5 0c1.01.143 2.01.317 3 .52m-3-.52l2.62 10.726c.122.499-.106 1.028-.589 1.202a5.988 5.988 0 01-2.031.352 5.988 5.988 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L18.75 4.97zm-16.5.52c.99-.203 1.99-.377 3-.52m0 0l2.62 10.726c.122.499-.106 1.028-.59 1.202A5.989 5.989 0 015.25 17.25a5.989 5.989 0 01-2.031-.352c-.483-.174-.711-.703-.59-1.202L5.25 4.971z"
      />
    </svg>
  );
}

function SparklesIcon() {
  return (
    <svg className="w-5 h-5" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={1.8} aria-hidden="true">
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M9.813 15.904L9 18.75l-.813-2.846a4.5 4.5 0 00-3.09-3.09L2.25 12l2.846-.813a4.5 4.5 0 003.09-3.09L9 5.25l.813 2.846a4.5 4.5 0 003.09 3.09L15.75 12l-2.846.813a4.5 4.5 0 00-3.09 3.09zM18.259 8.715L18 9.75l-.259-1.035a3.375 3.375 0 00-2.455-2.456L14.25 6l1.036-.259a3.375 3.375 0 002.455-2.456L18 2.25l.259 1.035a3.375 3.375 0 002.456 2.456L21.75 6l-1.035.259a3.375 3.375 0 00-2.456 2.456zM16.894 20.567L16.5 21.75l-.394-1.183a2.25 2.25 0 00-1.423-1.423L13.5 18.75l1.183-.394a2.25 2.25 0 001.423-1.423l.394-1.183.394 1.183a2.25 2.25 0 001.423 1.423l1.183.394-1.183.394a2.25 2.25 0 00-1.423 1.423z"
      />
    </svg>
  );
}

const PROFILES: Profile[] = [
  {
    key: 'strict',
    title: 'Strict',
    description: 'Maximum protection for all detected elements',
    icon: <ShieldIcon />,
  },
  {
    key: 'balanced',
    title: 'Balanced',
    description: 'Context-aware protection, respects consent',
    icon: <ScaleIcon />,
  },
  {
    key: 'creative',
    title: 'Creative',
    description: 'Artistic obfuscation with watermarks',
    icon: <SparklesIcon />,
  },
];

const ICON_COLOR: Record<ProfileKey, string> = {
  strict: 'text-red-400',
  balanced: 'text-blue-400',
  creative: 'text-purple-400',
};

export function ProfileQuickSelect({ selected, onSelect }: ProfileQuickSelectProps) {
  return (
    <div>
      <p className="text-xs font-semibold uppercase tracking-wider text-gray-500 mb-3">Privacy Profile</p>
      <div className="grid grid-cols-3 gap-3">
        {PROFILES.map((profile) => {
          const isSelected = selected === profile.key;
          return (
            <button
              key={profile.key}
              type="button"
              onClick={() => onSelect(profile.key)}
              aria-pressed={isSelected}
              className={`relative rounded-xl border px-4 py-4 text-left transition-colors duration-150 hover:border-blue-500/60 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 ${
                isSelected
                  ? 'border-blue-500 bg-blue-500/8'
                  : 'border-gray-700 bg-gray-900 hover:bg-gray-800/60'
              }`}
            >
              {/* Checkmark for selected */}
              {isSelected && (
                <span className="absolute top-2.5 right-2.5" aria-hidden="true">
                  <svg className="w-4 h-4 text-blue-400" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2.5}>
                    <path strokeLinecap="round" strokeLinejoin="round" d="M4.5 12.75l6 6 9-13.5" />
                  </svg>
                </span>
              )}

              <div className={`mb-2 ${ICON_COLOR[profile.key]}`}>{profile.icon}</div>
              <p className={`text-sm font-medium mb-1 ${isSelected ? 'text-blue-300' : 'text-gray-200'}`}>
                {profile.title}
              </p>
              <p className="text-xs text-gray-500 leading-snug">{profile.description}</p>
            </button>
          );
        })}
      </div>
    </div>
  );
}
