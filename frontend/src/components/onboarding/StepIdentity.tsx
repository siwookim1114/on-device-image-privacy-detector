import { useRef, useState, useCallback } from 'react';
import { useOnboardingStore } from '../../stores/onboardingStore';
import { useStudyLogger } from '../../hooks/useStudyLogger';

const MAX_PHOTOS = 5;
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

function UploadIcon() {
  return (
    <svg
      className="w-8 h-8 text-gray-500"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M3 16.5v2.25A2.25 2.25 0 005.25 21h13.5A2.25 2.25 0 0021 18.75V16.5m-13.5-9L12 3m0 0l4.5 4.5M12 3v13.5"
      />
    </svg>
  );
}

function XIcon() {
  return (
    <svg
      className="w-3 h-3"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2.5}
      aria-hidden="true"
    >
      <path strokeLinecap="round" strokeLinejoin="round" d="M6 18L18 6M6 6l12 12" />
    </svg>
  );
}

interface StepIdentityProps {
  onSkip: () => void;
}

export function StepIdentity({ onSkip }: StepIdentityProps) {
  const { identity, setIdentity } = useOnboardingStore();
  const { log } = useStudyLogger();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [label, setLabel] = useState(identity.label);
  const [photos, setPhotos] = useState<File[]>(identity.photos);
  const [previews, setPreviews] = useState<string[]>(
    identity.photos.map((f) => URL.createObjectURL(f)),
  );
  const [isDragOver, setIsDragOver] = useState(false);
  const [labelError, setLabelError] = useState<string | null>(null);

  const addPhotos = useCallback(
    (files: FileList | File[]) => {
      const valid = Array.from(files).filter((f) => ACCEPTED_IMAGE_TYPES.includes(f.type));
      const combined = [...photos, ...valid].slice(0, MAX_PHOTOS);
      setPhotos(combined);
      setPreviews(combined.map((f) => URL.createObjectURL(f)));
      log({ event_type: 'onboarding_identity_save', source: 'mouse', metadata: { action: 'photo_add', count: combined.length } });
    },
    [photos, log],
  );

  const removePhoto = (index: number) => {
    const next = photos.filter((_, i) => i !== index);
    setPhotos(next);
    setPreviews(next.map((f) => URL.createObjectURL(f)));
  };

  const handleDragOver = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(true);
  };

  const handleDragLeave = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
  };

  const handleDrop = (e: React.DragEvent<HTMLDivElement>) => {
    e.preventDefault();
    setIsDragOver(false);
    addPhotos(e.dataTransfer.files);
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) addPhotos(e.target.files);
    e.target.value = '';
  };

  const handleContinue = () => {
    if (!label.trim()) {
      setLabelError('Please enter a name or label to identify yourself.');
      return;
    }
    setLabelError(null);
    setIdentity(label.trim(), photos, false);
    log({
      event_type: 'onboarding_identity_save',
      source: 'mouse',
      metadata: {
        has_label: true,
        photo_count: photos.length,
      },
    });
  };

  const handleSkip = () => {
    setIdentity('', [], true);
    log({
      event_type: 'onboarding_identity_skip',
      source: 'mouse',
    });
    onSkip();
  };

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold text-gray-100">Your Identity</h2>
        <p className="text-sm text-gray-400">
          Add your name and a few face photos so the system can recognize you and skip
          protecting your own face when appropriate.
        </p>
      </div>

      {/* Name / Label */}
      <div className="space-y-1.5">
        <label htmlFor="identity-label" className="block text-sm font-medium text-gray-300">
          Your name or label
        </label>
        <input
          id="identity-label"
          type="text"
          value={label}
          onChange={(e) => {
            setLabel(e.target.value);
            if (labelError) setLabelError(null);
          }}
          placeholder="e.g. Alice, Me, Owner"
          aria-invalid={!!labelError}
          aria-describedby={labelError ? 'identity-label-error' : undefined}
          className={`
            w-full rounded-lg px-3 py-2.5 text-sm bg-gray-800 text-gray-100 placeholder-gray-500
            border transition-colors duration-150
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
            ${labelError ? 'border-red-500' : 'border-gray-700 hover:border-gray-600'}
          `}
        />
        {labelError && (
          <p id="identity-label-error" role="alert" className="text-xs text-red-400">
            {labelError}
          </p>
        )}
      </div>

      {/* Photo upload dropzone */}
      <div className="space-y-2">
        <label className="block text-sm font-medium text-gray-300">
          Face photos
          <span className="ml-1.5 text-xs text-gray-500 font-normal">(optional, up to 5)</span>
        </label>

        {photos.length < MAX_PHOTOS && (
          <div
            role="button"
            tabIndex={0}
            aria-label="Upload face photos by clicking or dragging images here"
            onClick={() => fileInputRef.current?.click()}
            onKeyDown={(e) => {
              if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
            }}
            onDragOver={handleDragOver}
            onDragLeave={handleDragLeave}
            onDrop={handleDrop}
            className={`
              flex flex-col items-center justify-center gap-3 rounded-xl border-2 border-dashed
              min-h-[140px] px-6 py-8 cursor-pointer transition-colors duration-150
              focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
              focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900
              ${isDragOver
                ? 'border-blue-500 bg-blue-500/5'
                : 'border-gray-700 hover:border-gray-600 hover:bg-gray-800/30'
              }
            `}
          >
            <UploadIcon />
            <div className="text-center space-y-1">
              <p className="text-sm text-gray-400">
                Drop photos here or <span className="text-blue-400">click to browse</span>
              </p>
              <p className="text-xs text-gray-600">PNG, JPEG, WebP — up to {MAX_PHOTOS} photos</p>
            </div>
          </div>
        )}

        <input
          ref={fileInputRef}
          type="file"
          accept="image/png,image/jpeg,image/webp"
          multiple
          className="hidden"
          onChange={handleFileChange}
          aria-hidden="true"
        />

        {previews.length > 0 && (
          <div className="grid grid-cols-5 gap-2 mt-2">
            {previews.map((src, idx) => (
              <div key={idx} className="relative group aspect-square">
                <img
                  src={src}
                  alt={`Face photo ${idx + 1}`}
                  className="w-full h-full rounded-lg object-cover border border-gray-700"
                />
                <button
                  type="button"
                  onClick={() => removePhoto(idx)}
                  aria-label={`Remove photo ${idx + 1}`}
                  className="absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-gray-900 border border-gray-700 text-gray-400 hover:text-red-400 hover:border-red-500 transition-colors opacity-0 group-hover:opacity-100 focus-visible:opacity-100"
                >
                  <XIcon />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>

      {/* Info callout */}
      <div className="flex gap-3 rounded-lg bg-blue-500/5 border border-blue-500/20 px-4 py-3">
        <svg
          className="w-4 h-4 text-blue-400 mt-0.5 shrink-0"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path
            strokeLinecap="round"
            strokeLinejoin="round"
            d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z"
          />
        </svg>
        <p className="text-xs text-blue-300/80">
          Photos are stored locally and processed entirely on-device. Nothing is sent to the cloud.
        </p>
      </div>

      {/* Action row */}
      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={handleSkip}
          className="text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
        >
          Skip for now
        </button>
        <button
          type="button"
          onClick={handleContinue}
          className="rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold px-5 py-2.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
        >
          Continue
        </button>
      </div>
    </div>
  );
}
