import { useRef, useState, useCallback } from 'react';
import type { ConsentStatus } from '../../types/consent';
import { useConsentStore } from '../../stores/consentStore';
import { ConsentLevelSelect } from './ConsentLevelSelect';

const MAX_PHOTOS = 5;
const MIN_PHOTOS = 1;
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

type Relationship = 'self' | 'family' | 'friend' | 'colleague' | 'other';

interface ValidationErrors {
  label?: string;
  photos?: string;
}

function UploadIcon() {
  return (
    <svg
      className="w-6 h-6 text-gray-400"
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
      className="w-3.5 h-3.5"
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

interface ToastProps {
  message: string;
}

function SuccessToast({ message }: ToastProps) {
  return (
    <div
      role="status"
      aria-live="polite"
      className="flex items-center gap-2 rounded-lg bg-green-500/10 border border-green-500/30 px-4 py-3"
    >
      <svg
        className="w-4 h-4 text-green-400 shrink-0"
        fill="none"
        viewBox="0 0 24 24"
        stroke="currentColor"
        strokeWidth={2}
        aria-hidden="true"
      >
        <polyline points="20 6 9 17 4 12" />
      </svg>
      <p className="text-sm text-green-400">{message}</p>
    </div>
  );
}

export function FaceRegistration() {
  const { registerPerson } = useConsentStore();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [label, setLabel] = useState('');
  const [relationship, setRelationship] = useState<Relationship>('other');
  const [consentStatus, setConsentStatus] = useState<ConsentStatus>('none');
  const [selectedPhotos, setSelectedPhotos] = useState<File[]>([]);
  const [photoPreviews, setPhotoPreviews] = useState<string[]>([]);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [submitError, setSubmitError] = useState<string | null>(null);
  const [successMessage, setSuccessMessage] = useState<string | null>(null);
  const [validationErrors, setValidationErrors] = useState<ValidationErrors>({});
  const [isDragOver, setIsDragOver] = useState(false);

  const addPhotos = useCallback(
    (files: FileList | File[], currentPhotos: File[]) => {
      const validFiles = Array.from(files).filter((f) =>
        ACCEPTED_IMAGE_TYPES.includes(f.type),
      );
      const combined = [...currentPhotos, ...validFiles].slice(0, MAX_PHOTOS);
      setSelectedPhotos(combined);
      setPhotoPreviews(combined.map((f) => URL.createObjectURL(f)));
      setValidationErrors((prev) => ({ ...prev, photos: undefined }));
    },
    [],
  );

  const removePhoto = (index: number) => {
    const next = selectedPhotos.filter((_, i) => i !== index);
    setSelectedPhotos(next);
    setPhotoPreviews(next.map((f) => URL.createObjectURL(f)));
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
    addPhotos(e.dataTransfer.files, selectedPhotos);
  };

  const handleFileInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) {
      addPhotos(e.target.files, selectedPhotos);
    }
    e.target.value = '';
  };

  const validate = (): boolean => {
    const errors: ValidationErrors = {};
    if (!label.trim()) {
      errors.label = 'Name or label is required.';
    }
    if (selectedPhotos.length < MIN_PHOTOS) {
      errors.photos = 'At least 1 face photo is required.';
    }
    setValidationErrors(errors);
    return Object.keys(errors).length === 0;
  };

  const handleSubmit = async (e: React.FormEvent<HTMLFormElement>) => {
    e.preventDefault();
    if (!validate() || isSubmitting) return;

    setIsSubmitting(true);
    setSubmitError(null);
    setSuccessMessage(null);

    try {
      const formData = new FormData();
      formData.append('label', label.trim());
      formData.append('relationship', relationship);
      formData.append('consent_status', consentStatus);
      selectedPhotos.forEach((photo) => {
        formData.append('photos', photo);
      });

      await registerPerson(formData);

      setLabel('');
      setRelationship('other');
      setConsentStatus('none');
      setSelectedPhotos([]);
      setPhotoPreviews([]);
      setValidationErrors({});
      setSuccessMessage(`"${label.trim()}" has been registered successfully.`);

      setTimeout(() => setSuccessMessage(null), 4000);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Registration failed. Please try again.';
      setSubmitError(message);
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div className="bg-gray-900 rounded-xl p-6 max-w-lg w-full">
      <h2 className="text-base font-semibold text-gray-100 mb-5">Register New Face</h2>

      <form onSubmit={(e) => void handleSubmit(e)} noValidate className="space-y-5">
        {/* Photo upload dropzone */}
        <div className="space-y-2">
          <label className="block text-sm font-medium text-gray-300">
            Face Photos
            <span className="ml-1 text-xs text-gray-500 font-normal">(1–5 images)</span>
          </label>

          {selectedPhotos.length < MAX_PHOTOS && (
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
                flex flex-col items-center justify-center gap-2 rounded-lg border-2 border-dashed
                min-h-[100px] px-4 py-5 cursor-pointer transition-colors duration-150
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900
                ${isDragOver
                  ? 'border-blue-500 bg-blue-500/5'
                  : validationErrors.photos
                    ? 'border-red-500'
                    : 'border-gray-700 hover:border-gray-600'
                }
              `}
            >
              <UploadIcon />
              <p className="text-xs text-gray-400 text-center">
                Drop photos here or{' '}
                <span className="text-blue-400">click to browse</span>
              </p>
              <p className="text-xs text-gray-600">PNG, JPEG, WebP — up to {MAX_PHOTOS} photos</p>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            className="hidden"
            onChange={handleFileInputChange}
            aria-hidden="true"
          />

          {/* Photo thumbnails */}
          {photoPreviews.length > 0 && (
            <div className="grid grid-cols-5 gap-2">
              {photoPreviews.map((src, idx) => (
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
                    className="absolute -top-1.5 -right-1.5 w-5 h-5 flex items-center justify-center rounded-full bg-gray-900 border border-gray-700 text-gray-400 hover:text-red-400 hover:border-red-500 transition-colors opacity-0 group-hover:opacity-100"
                  >
                    <XIcon />
                  </button>
                </div>
              ))}
            </div>
          )}

          {validationErrors.photos && (
            <p role="alert" className="text-xs text-red-400">{validationErrors.photos}</p>
          )}
        </div>

        {/* Name / Label */}
        <div className="space-y-1.5">
          <label htmlFor="consent-label" className="block text-sm font-medium text-gray-300">
            Name or Label
          </label>
          <input
            id="consent-label"
            type="text"
            value={label}
            onChange={(e) => {
              setLabel(e.target.value);
              if (validationErrors.label) {
                setValidationErrors((prev) => ({ ...prev, label: undefined }));
              }
            }}
            placeholder="e.g. Alice, Dad, Colleague A"
            aria-describedby={validationErrors.label ? 'consent-label-error' : undefined}
            aria-invalid={!!validationErrors.label}
            className={`
              w-full rounded-lg px-3 py-2.5 text-sm bg-gray-800 text-gray-100 placeholder-gray-500
              border transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
              ${validationErrors.label ? 'border-red-500' : 'border-gray-700 hover:border-gray-600'}
            `}
          />
          {validationErrors.label && (
            <p id="consent-label-error" role="alert" className="text-xs text-red-400">
              {validationErrors.label}
            </p>
          )}
        </div>

        {/* Relationship */}
        <div className="space-y-1.5">
          <label htmlFor="consent-relationship" className="block text-sm font-medium text-gray-300">
            Relationship
          </label>
          <select
            id="consent-relationship"
            value={relationship}
            onChange={(e) => setRelationship(e.target.value as Relationship)}
            className="w-full rounded-lg px-3 py-2.5 text-sm bg-gray-800 text-gray-100 border border-gray-700 hover:border-gray-600 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 appearance-none cursor-pointer"
          >
            <option value="self">Self</option>
            <option value="family">Family</option>
            <option value="friend">Friend</option>
            <option value="colleague">Colleague</option>
            <option value="other">Other</option>
          </select>
        </div>

        {/* Consent Level */}
        <div className="space-y-1.5">
          <span className="block text-sm font-medium text-gray-300">Consent Level</span>
          <ConsentLevelSelect value={consentStatus} onChange={setConsentStatus} />
        </div>

        {/* Error banner */}
        {submitError && (
          <div
            role="alert"
            className="flex items-start gap-2 rounded-lg bg-red-500/10 border border-red-500/30 px-4 py-3"
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
            <p className="text-xs text-red-400">{submitError}</p>
          </div>
        )}

        {/* Success toast */}
        {successMessage && <SuccessToast message={successMessage} />}

        {/* Submit */}
        <button
          type="submit"
          disabled={isSubmitting}
          aria-disabled={isSubmitting}
          className="w-full rounded-lg bg-blue-600 hover:bg-blue-500 disabled:bg-gray-700 disabled:text-gray-500 disabled:cursor-not-allowed text-white font-semibold py-2.5 text-sm transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
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
              Registering...
            </span>
          ) : (
            'Register Person'
          )}
        </button>
      </form>
    </div>
  );
}
