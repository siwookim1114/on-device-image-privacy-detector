import { useEffect, useRef, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { ProgressDots } from '../components/onboarding/ProgressDots';
import { StepIdentity } from '../components/onboarding/StepIdentity';
import { StepContacts } from '../components/onboarding/StepContacts';
import { StepSensitivity } from '../components/onboarding/StepSensitivity';
import { StepReview } from '../components/onboarding/StepReview';
import { useOnboardingStore } from '../stores/onboardingStore';
import { useStudyLogger } from '../hooks/useStudyLogger';
import { createProfile } from '../api/profile';

const TOTAL_STEPS = 4;
const STEP_NAMES = ['identity', 'contacts', 'sensitivity', 'review'];

export function Onboarding() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const participantId = searchParams.get('pid');

  const { currentStep, sensitivity, advanced, identity, contacts, setStep, setCompleted, reset } =
    useOnboardingStore();
  const { log } = useStudyLogger();

  const [isSaving, setIsSaving] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const stepEntryTimeRef = useRef<number>(Date.now());

  useEffect(() => {
    stepEntryTimeRef.current = Date.now();
    log({
      event_type: 'onboarding_step_enter',
      source: 'system',
      metadata: {
        step: currentStep,
        step_name: STEP_NAMES[currentStep],
      },
    });
  }, [currentStep, log]);

  const logStepComplete = (step: number) => {
    const duration = Date.now() - stepEntryTimeRef.current;
    log({
      event_type: 'onboarding_step_complete',
      source: 'mouse',
      metadata: {
        step,
        step_name: STEP_NAMES[step],
        duration_ms: duration,
      },
    });
  };

  const goToStep = (step: number) => {
    setStep(Math.max(0, Math.min(step, TOTAL_STEPS - 1)));
  };

  const handleBack = () => {
    if (currentStep > 0) goToStep(currentStep - 1);
  };

  const handleStepAdvance = () => {
    logStepComplete(currentStep);
    if (currentStep < TOTAL_STEPS - 1) {
      goToStep(currentStep + 1);
    }
  };

  const handleFinish = async () => {
    setIsSaving(true);
    setSaveError(null);
    try {
      await createProfile({
        label: identity.label || undefined,
        sensitivity: {
          faces: sensitivity.faces,
          text: sensitivity.text,
          screens: sensitivity.screens,
          objects: sensitivity.objects,
        },
        advanced: {
          preferred_face_method: advanced.preferred_face_method,
          preferred_text_method: advanced.preferred_text_method,
          preferred_screen_method: advanced.preferred_screen_method,
          preferred_object_method: advanced.preferred_object_method,
          auto_advance_threshold: advanced.auto_advance_threshold,
          pause_on_critical: advanced.pause_on_critical,
        },
      });

      setCompleted();
      localStorage.setItem('onboarding_complete', 'true');

      log({
        event_type: 'onboarding_complete',
        source: 'mouse',
        metadata: {
          contact_count: contacts.length,
          identity_skipped: identity.skipped,
        },
      });

      reset();
      navigate('/');
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to save profile. Please try again.';
      setSaveError(message);
    } finally {
      setIsSaving(false);
    }
  };

  const renderStep = () => {
    switch (currentStep) {
      case 0:
        return <StepIdentity onSkip={handleStepAdvance} />;
      case 1:
        return <StepContacts onSkip={handleStepAdvance} />;
      case 2:
        return <StepSensitivity onContinue={handleStepAdvance} />;
      case 3:
        return (
          <StepReview
            onEdit={goToStep}
            onFinish={() => void handleFinish()}
            isSaving={isSaving}
          />
        );
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen bg-gray-950 flex flex-col">
      {/* Header */}
      <header className="shrink-0 border-b border-gray-800 px-6 py-4">
        <div className="max-w-xl mx-auto flex items-center justify-between">
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-blue-600 flex items-center justify-center" aria-hidden="true">
              <svg className="w-3.5 h-3.5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth={2}>
                <path strokeLinecap="round" strokeLinejoin="round" d="M9 12.75L11.25 15 15 9.75m-3-7.036A11.959 11.959 0 013.598 6 11.99 11.99 0 003 9.749c0 5.592 3.824 10.29 9 11.623 5.176-1.332 9-6.03 9-11.622 0-1.31-.21-2.571-.598-3.751h-.152c-3.196 0-6.1-1.248-8.25-3.285z" />
              </svg>
            </div>
            <span className="text-sm font-semibold text-gray-200">Privacy Setup</span>
          </div>
          <span className="text-xs text-gray-600">
            Step {currentStep + 1} of {TOTAL_STEPS}
          </span>
        </div>
      </header>

      {/* Progress */}
      <div className="shrink-0 pt-8 pb-6 px-6">
        <div className="max-w-xl mx-auto">
          <ProgressDots currentStep={currentStep} totalSteps={TOTAL_STEPS} />
        </div>
      </div>

      {/* Main content */}
      <main className="flex-1 px-6 pb-8">
        <div className="max-w-xl mx-auto">
          {saveError && (
            <div
              role="alert"
              className="mb-5 flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3"
            >
              <svg
                className="w-4 h-4 text-red-400 mt-0.5 shrink-0"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
                aria-hidden="true"
              >
                <path
                  strokeLinecap="round"
                  strokeLinejoin="round"
                  d="M12 9v3.75m-9.303 3.376c-.866 1.5.217 3.374 1.948 3.374h14.71c1.73 0 2.813-1.874 1.948-3.374L13.949 3.378c-.866-1.5-3.032-1.5-3.898 0L2.697 16.126zM12 15.75h.007v.008H12v-.008z"
                />
              </svg>
              <p className="text-xs text-red-400">{saveError}</p>
            </div>
          )}

          {renderStep()}
        </div>
      </main>

      {/* Back navigation (not shown on first step or last step — last step manages its own nav) */}
      {currentStep > 0 && currentStep < TOTAL_STEPS - 1 && (
        <footer className="shrink-0 border-t border-gray-800 px-6 py-4">
          <div className="max-w-xl mx-auto">
            <button
              type="button"
              onClick={handleBack}
              className="flex items-center gap-1.5 text-sm text-gray-500 hover:text-gray-300 transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 rounded"
            >
              <svg
                className="w-4 h-4"
                fill="none"
                viewBox="0 0 24 24"
                stroke="currentColor"
                strokeWidth={2}
                aria-hidden="true"
              >
                <path strokeLinecap="round" strokeLinejoin="round" d="M10 19l-7-7m0 0l7-7m-7 7h18" />
              </svg>
              Back
            </button>
          </div>
        </footer>
      )}

      {/* Study bar */}
      {participantId && (
        <div
          aria-hidden="true"
          className="shrink-0 border-t border-gray-800 bg-gray-900 px-6 py-2 text-center"
        >
          <span className="text-xs text-gray-600">
            Study session &middot; Participant: {participantId}
          </span>
        </div>
      )}
    </div>
  );
}
