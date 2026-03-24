import { useRef, useState, useCallback } from 'react';

import type { ContactEntry } from '../../types/profile';
import type { ConsentStatus } from '../../types/consent';
import { useOnboardingStore } from '../../stores/onboardingStore';
import { useStudyLogger } from '../../hooks/useStudyLogger';
import { ConsentLevelSelect } from '../consent/ConsentLevelSelect';

const MAX_PHOTOS = 5;
const ACCEPTED_IMAGE_TYPES = ['image/jpeg', 'image/png', 'image/webp'];

type Relationship = 'family' | 'friend' | 'colleague' | 'other';

const RELATIONSHIP_OPTIONS: { value: Relationship; label: string }[] = [
  { value: 'family', label: 'Family' },
  { value: 'friend', label: 'Friend' },
  { value: 'colleague', label: 'Colleague' },
  { value: 'other', label: 'Other' },
];

function XIcon({ className = 'w-3.5 h-3.5' }: { className?: string }) {
  return (
    <svg
      className={className}
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

function PersonIcon() {
  return (
    <svg
      className="w-5 h-5 text-gray-500"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.5}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M15.75 6a3.75 3.75 0 11-7.5 0 3.75 3.75 0 017.5 0zM4.501 20.118a7.5 7.5 0 0114.998 0A17.933 17.933 0 0112 21.75c-2.676 0-5.216-.584-7.499-1.632z"
      />
    </svg>
  );
}

interface ContactFormState {
  display_name: string;
  relationship: Relationship;
  consent_level: 'explicit' | 'assumed' | 'none';
  photos: File[];
  previews: string[];
}

const blankForm = (): ContactFormState => ({
  display_name: '',
  relationship: 'other',
  consent_level: 'assumed',
  photos: [],
  previews: [],
});

const CONSENT_STATUS_MAP: Record<'explicit' | 'assumed' | 'none', ConsentStatus> = {
  explicit: 'explicit',
  assumed: 'assumed',
  none: 'none',
};

interface StepContactsProps {
  onSkip: () => void;
}

export function StepContacts({ onSkip }: StepContactsProps) {
  const { contacts, addContact, removeContact } = useOnboardingStore();
  const { log } = useStudyLogger();
  const fileInputRef = useRef<HTMLInputElement>(null);

  const [form, setForm] = useState<ContactFormState>(blankForm());
  const [isDragOver, setIsDragOver] = useState(false);
  const [nameError, setNameError] = useState<string | null>(null);

  const addPhotos = useCallback(
    (files: FileList | File[]) => {
      const valid = Array.from(files).filter((f) => ACCEPTED_IMAGE_TYPES.includes(f.type));
      const combined = [...form.photos, ...valid].slice(0, MAX_PHOTOS);
      setForm((prev) => ({
        ...prev,
        photos: combined,
        previews: combined.map((f) => URL.createObjectURL(f)),
      }));
    },
    [form.photos],
  );

  const removePhoto = (index: number) => {
    const nextPhotos = form.photos.filter((_, i) => i !== index);
    setForm((prev) => ({
      ...prev,
      photos: nextPhotos,
      previews: nextPhotos.map((f) => URL.createObjectURL(f)),
    }));
  };

  const handleAdd = () => {
    if (!form.display_name.trim()) {
      setNameError('Name is required.');
      return;
    }
    setNameError(null);

    const contact: ContactEntry = {
      display_name: form.display_name.trim(),
      relationship: form.relationship,
      consent_level: form.consent_level,
      photos: form.photos,
    };
    addContact(contact);
    setForm(blankForm());
    log({
      event_type: 'onboarding_contact_add',
      source: 'mouse',
      metadata: {
        relationship: contact.relationship,
        consent_level: contact.consent_level,
        photo_count: contact.photos.length,
      },
    });
  };

  const handleRemove = (index: number) => {
    removeContact(index);
    log({
      event_type: 'onboarding_contact_remove',
      source: 'mouse',
      metadata: { index },
    });
  };

  const consentStatusValue: ConsentStatus = CONSENT_STATUS_MAP[form.consent_level];

  return (
    <div className="space-y-6">
      <div className="space-y-1">
        <h2 className="text-xl font-semibold text-gray-100">Known Contacts</h2>
        <p className="text-sm text-gray-400">
          Register people you know so the system can apply their consent preferences automatically.
        </p>
      </div>

      {/* Existing contacts list */}
      {contacts.length > 0 && (
        <div className="space-y-2">
          <p className="text-xs font-medium text-gray-500 uppercase tracking-wide">
            Added contacts ({contacts.length})
          </p>
          <ul className="space-y-2" aria-label="Registered contacts">
            {contacts.map((contact, index) => (
              <li
                key={index}
                className="flex items-center gap-3 rounded-lg bg-gray-800/60 border border-gray-700 px-4 py-3"
              >
                <div className="w-8 h-8 rounded-full bg-gray-700 flex items-center justify-center shrink-0">
                  {contact.photos.length > 0 ? (
                    <img
                      src={URL.createObjectURL(contact.photos[0]!)}
                      alt=""
                      className="w-full h-full rounded-full object-cover"
                    />
                  ) : (
                    <PersonIcon />
                  )}
                </div>
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-medium text-gray-100 truncate">{contact.display_name}</p>
                  <p className="text-xs text-gray-500 capitalize">
                    {contact.relationship} &middot; {contact.consent_level} consent
                  </p>
                </div>
                <button
                  type="button"
                  onClick={() => handleRemove(index)}
                  aria-label={`Remove ${contact.display_name}`}
                  className="w-7 h-7 flex items-center justify-center rounded-md text-gray-500 hover:text-red-400 hover:bg-red-500/10 transition-colors"
                >
                  <XIcon />
                </button>
              </li>
            ))}
          </ul>
        </div>
      )}

      {/* Add form */}
      <div className="rounded-xl border border-gray-700 bg-gray-800/30 p-5 space-y-4">
        <p className="text-sm font-medium text-gray-300">Add a contact</p>

        {/* Name */}
        <div className="space-y-1.5">
          <label htmlFor="contact-name" className="block text-xs font-medium text-gray-400">
            Name or label
          </label>
          <input
            id="contact-name"
            type="text"
            value={form.display_name}
            onChange={(e) => {
              setForm((prev) => ({ ...prev, display_name: e.target.value }));
              if (nameError) setNameError(null);
            }}
            placeholder="e.g. Bob, Mom, Colleague B"
            aria-invalid={!!nameError}
            aria-describedby={nameError ? 'contact-name-error' : undefined}
            className={`
              w-full rounded-lg px-3 py-2 text-sm bg-gray-800 text-gray-100 placeholder-gray-500
              border transition-colors duration-150
              focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900
              ${nameError ? 'border-red-500' : 'border-gray-700 hover:border-gray-600'}
            `}
          />
          {nameError && (
            <p id="contact-name-error" role="alert" className="text-xs text-red-400">
              {nameError}
            </p>
          )}
        </div>

        {/* Relationship */}
        <div className="space-y-1.5">
          <label htmlFor="contact-relationship" className="block text-xs font-medium text-gray-400">
            Relationship
          </label>
          <select
            id="contact-relationship"
            value={form.relationship}
            onChange={(e) => setForm((prev) => ({ ...prev, relationship: e.target.value as Relationship }))}
            className="w-full rounded-lg px-3 py-2 text-sm bg-gray-800 text-gray-100 border border-gray-700 hover:border-gray-600 transition-colors duration-150 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-900 appearance-none cursor-pointer"
          >
            {RELATIONSHIP_OPTIONS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Consent level */}
        <div className="space-y-1.5">
          <span className="block text-xs font-medium text-gray-400">Consent level</span>
          <ConsentLevelSelect
            value={consentStatusValue}
            onChange={(status) => {
              if (status !== 'unclear') {
                setForm((prev) => ({ ...prev, consent_level: status as 'explicit' | 'assumed' | 'none' }));
              }
            }}
          />
        </div>

        {/* Photo upload */}
        <div className="space-y-2">
          <span className="block text-xs font-medium text-gray-400">
            Face photos <span className="text-gray-600 font-normal">(optional)</span>
          </span>

          {form.photos.length < MAX_PHOTOS && (
            <div
              role="button"
              tabIndex={0}
              aria-label="Upload contact photos"
              onClick={() => fileInputRef.current?.click()}
              onKeyDown={(e) => {
                if (e.key === 'Enter' || e.key === ' ') fileInputRef.current?.click();
              }}
              onDragOver={(e) => { e.preventDefault(); setIsDragOver(true); }}
              onDragLeave={(e) => { e.preventDefault(); setIsDragOver(false); }}
              onDrop={(e) => { e.preventDefault(); setIsDragOver(false); addPhotos(e.dataTransfer.files); }}
              className={`
                flex items-center justify-center gap-2 rounded-lg border border-dashed
                h-16 cursor-pointer transition-colors duration-150
                focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500
                focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900
                ${isDragOver ? 'border-blue-500 bg-blue-500/5' : 'border-gray-700 hover:border-gray-600'}
              `}
            >
              <svg
                className="w-4 h-4 text-gray-500"
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
              <span className="text-xs text-gray-500">
                Drop or <span className="text-blue-400">click to browse</span>
              </span>
            </div>
          )}

          <input
            ref={fileInputRef}
            type="file"
            accept="image/png,image/jpeg,image/webp"
            multiple
            className="hidden"
            onChange={(e) => { if (e.target.files) addPhotos(e.target.files); e.target.value = ''; }}
            aria-hidden="true"
          />

          {form.previews.length > 0 && (
            <div className="flex gap-2 flex-wrap">
              {form.previews.map((src, idx) => (
                <div key={idx} className="relative group w-12 h-12">
                  <img
                    src={src}
                    alt={`Contact photo ${idx + 1}`}
                    className="w-full h-full rounded-md object-cover border border-gray-700"
                  />
                  <button
                    type="button"
                    onClick={() => removePhoto(idx)}
                    aria-label={`Remove contact photo ${idx + 1}`}
                    className="absolute -top-1 -right-1 w-4 h-4 flex items-center justify-center rounded-full bg-gray-900 border border-gray-700 text-gray-400 hover:text-red-400 opacity-0 group-hover:opacity-100 transition-opacity"
                  >
                    <XIcon className="w-2.5 h-2.5" />
                  </button>
                </div>
              ))}
            </div>
          )}
        </div>

        {/* Add button */}
        <button
          type="button"
          onClick={handleAdd}
          className="w-full rounded-lg border border-gray-600 hover:border-gray-500 bg-gray-800 hover:bg-gray-750 text-gray-200 font-medium py-2 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
        >
          + Add Contact
        </button>
      </div>

      {/* Skip / Continue row */}
      <div className="flex items-center justify-between pt-2">
        <button
          type="button"
          onClick={onSkip}
          className="text-sm text-gray-500 hover:text-gray-300 transition-colors underline underline-offset-2"
        >
          Skip
        </button>
        <button
          type="button"
          onClick={onSkip}
          className="rounded-lg bg-blue-600 hover:bg-blue-500 text-white font-semibold px-5 py-2.5 text-sm transition-colors focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-blue-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-900"
        >
          Continue
        </button>
      </div>
    </div>
  );
}
