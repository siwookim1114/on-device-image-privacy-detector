interface ProgressDotsProps {
  currentStep: number;
  totalSteps: number;
}

const STEP_LABELS = ['Identity', 'Contacts', 'Sensitivity', 'Review'];

export function ProgressDots({ currentStep, totalSteps }: ProgressDotsProps) {
  return (
    <nav aria-label="Onboarding progress" className="flex items-center justify-center gap-0">
      {Array.from({ length: totalSteps }, (_, i) => {
        const isCompleted = i < currentStep;
        const isActive = i === currentStep;
        const label = STEP_LABELS[i] ?? `Step ${i + 1}`;

        return (
          <div key={i} className="flex items-center">
            {/* Connecting line before dot (not before first) */}
            {i > 0 && (
              <div
                className={`h-0.5 w-10 sm:w-16 transition-colors duration-300 ${
                  isCompleted ? 'bg-green-500' : 'bg-gray-700'
                }`}
                aria-hidden="true"
              />
            )}

            {/* Dot */}
            <div className="flex flex-col items-center gap-1.5">
              <div
                role="img"
                aria-label={`${label}: ${isCompleted ? 'completed' : isActive ? 'current' : 'pending'}`}
                className={`
                  relative flex items-center justify-center rounded-full transition-all duration-300
                  ${isCompleted
                    ? 'w-7 h-7 bg-green-500 shadow-lg shadow-green-500/30'
                    : isActive
                      ? 'w-7 h-7 bg-blue-600 shadow-lg shadow-blue-500/40 ring-4 ring-blue-500/20'
                      : 'w-6 h-6 bg-gray-700 border border-gray-600'
                  }
                `}
              >
                {/* Pulse ring for active */}
                {isActive && (
                  <span
                    className="absolute inset-0 rounded-full animate-ping bg-blue-500 opacity-20"
                    aria-hidden="true"
                  />
                )}

                {/* Checkmark for completed */}
                {isCompleted && (
                  <svg
                    className="w-3.5 h-3.5 text-white"
                    viewBox="0 0 24 24"
                    fill="none"
                    stroke="currentColor"
                    strokeWidth={3}
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    aria-hidden="true"
                  >
                    <polyline points="20 6 9 17 4 12" />
                  </svg>
                )}

                {/* Step number for active / pending */}
                {!isCompleted && (
                  <span
                    className={`text-xs font-bold select-none ${
                      isActive ? 'text-white' : 'text-gray-500'
                    }`}
                  >
                    {i + 1}
                  </span>
                )}
              </div>

              {/* Label below dot */}
              <span
                className={`text-xs font-medium whitespace-nowrap transition-colors duration-200 ${
                  isCompleted
                    ? 'text-green-400'
                    : isActive
                      ? 'text-blue-400'
                      : 'text-gray-600'
                }`}
              >
                {label}
              </span>
            </div>
          </div>
        );
      })}
    </nav>
  );
}
