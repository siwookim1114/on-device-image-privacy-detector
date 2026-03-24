import { useState } from 'react';
import { useNavigate } from 'react-router-dom';

import type { ProcessingMode, EthicalMode } from '../types/pipeline';
import { startPipeline } from '../api/pipeline';
import { usePipelineStore } from '../stores/pipelineStore';
import { DropZone } from '../components/upload/DropZone';
import { ProcessingConfig } from '../components/upload/ProcessingConfig';
import { ProfileQuickSelect } from '../components/upload/ProfileQuickSelect';

type ProfileKey = 'strict' | 'balanced' | 'creative';

const PROFILE_TO_MODE: Record<ProfileKey, { mode: ProcessingMode; ethicalMode: EthicalMode }> = {
  strict: { mode: 'auto', ethicalMode: 'strict' },
  balanced: { mode: 'auto', ethicalMode: 'balanced' },
  creative: { mode: 'hybrid', ethicalMode: 'creative' },
};

export function Dashboard() {
  const navigate = useNavigate();
  const { setSession, reset } = usePipelineStore();

  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [mode, setMode] = useState<ProcessingMode>('auto');
  const [ethicalMode, setEthicalMode] = useState<EthicalMode>('balanced');
  const [profile, setProfile] = useState<ProfileKey>('balanced');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);

  const handleProfileSelect = (p: ProfileKey) => {
    setProfile(p);
    const preset = PROFILE_TO_MODE[p];
    setMode(preset.mode);
    setEthicalMode(preset.ethicalMode);
  };

  const handleModeChange = (m: ProcessingMode) => {
    setMode(m);
  };

  const handleEthicalModeChange = (e: EthicalMode) => {
    setEthicalMode(e);
  };

  const handleSubmit = async () => {
    if (!selectedFile || isSubmitting) return;

    setIsSubmitting(true);
    setSubmitError(null);

    try {
      reset();
      const result = await startPipeline(selectedFile, mode, ethicalMode);
      const imageUrl = URL.createObjectURL(selectedFile);
      setSession(result.session_id, imageUrl);
      navigate(`/process/${result.session_id}`);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to start pipeline. Please try again.';
      setSubmitError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="min-h-full flex flex-col items-center py-12 px-4">
      <div className="w-full max-w-2xl space-y-8">
        {/* Header */}
        <div className="text-center">
          <h1 className="text-3xl font-bold text-gray-100 tracking-tight">Privacy Guard</h1>
          <p className="mt-2 text-sm text-gray-400">On-device image privacy protection</p>
        </div>

        {/* Upload zone */}
        <section aria-label="Image upload">
          <DropZone onFileSelect={setSelectedFile} />
        </section>

        {/* Config sections */}
        <section aria-label="Privacy profile">
          <ProfileQuickSelect selected={profile} onSelect={handleProfileSelect} />
        </section>

        <section aria-label="Processing configuration">
          <ProcessingConfig
            mode={mode}
            ethicalMode={ethicalMode}
            onModeChange={handleModeChange}
            onEthicalModeChange={handleEthicalModeChange}
          />
        </section>

        {/* Error banner */}
        {submitError && (
          <div
            role="alert"
            className="flex items-start gap-3 rounded-xl bg-red-500/10 border border-red-500/30 px-4 py-3"
          >
            <svg
              className="w-5 h-5 text-red-400 mt-0.5 shrink-0"
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
            <p className="text-sm text-red-400">{submitError}</p>
          </div>
        )}

        {/* Submit */}
        <button
          type="button"
          onClick={() => void handleSubmit()}
          disabled={!selectedFile || isSubmitting}
          aria-disabled={!selectedFile || isSubmitting}
          className="w-full rounded-xl bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-semibold py-3.5 text-sm transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-950"
        >
          {isSubmitting ? (
            <span className="flex items-center justify-center gap-2">
              <svg
                className="w-4 h-4 animate-spin"
                fill="none"
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path
                  className="opacity-75"
                  fill="currentColor"
                  d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z"
                />
              </svg>
              Starting pipeline...
            </span>
          ) : (
            'Start Processing'
          )}
        </button>

        {!selectedFile && (
          <p className="text-center text-xs text-gray-600">Upload an image above to enable processing</p>
        )}
      </div>
    </div>
  );
}
