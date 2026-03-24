import { useState } from 'react';

import type { PersonEntry } from '../../types/consent';
import { ConsentHistory } from './ConsentHistory';

interface PersonCardProps {
  person: PersonEntry;
  onDelete: (personId: string) => void;
}

const CONSENT_BADGE: Record<string, string> = {
  explicit: 'bg-green-500/15 text-green-400 border-green-500/30',
  assumed: 'bg-yellow-500/15 text-yellow-400 border-yellow-500/30',
  none: 'bg-red-500/15 text-red-400 border-red-500/30',
  unclear: 'bg-gray-700 text-gray-400 border-gray-600',
};

function formatDate(dateStr: string): string {
  try {
    return new Date(dateStr).toLocaleDateString(undefined, {
      year: 'numeric',
      month: 'short',
      day: 'numeric',
    });
  } catch {
    return dateStr;
  }
}

function getInitials(label: string): string {
  return label
    .split(/\s+/)
    .map((w) => w[0] ?? '')
    .slice(0, 2)
    .join('')
    .toUpperCase();
}

const AVATAR_COLORS = [
  'bg-blue-600',
  'bg-purple-600',
  'bg-teal-600',
  'bg-orange-600',
  'bg-pink-600',
  'bg-indigo-600',
];

function avatarColor(label: string): string {
  let hash = 0;
  for (let i = 0; i < label.length; i++) {
    hash = (hash * 31 + label.charCodeAt(i)) >>> 0;
  }
  return AVATAR_COLORS[hash % AVATAR_COLORS.length] ?? 'bg-blue-600';
}

function TrashIcon() {
  return (
    <svg
      className="w-4 h-4"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={1.75}
      aria-hidden="true"
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16"
      />
    </svg>
  );
}

export function PersonCard({ person, onDelete }: PersonCardProps) {
  const [confirmDelete, setConfirmDelete] = useState(false);

  // Derive consent from history since it's not stored directly on PersonEntry
  const derivedConsentKey =
    person.consent_history.times_approved > 0 && person.consent_history.times_appeared > 0
      ? 'assumed'
      : 'unclear';
  const consentBadgeClass =
    CONSENT_BADGE[derivedConsentKey] ?? CONSENT_BADGE['unclear'] ?? 'bg-gray-700 text-gray-400';

  const handleDeleteClick = () => {
    if (confirmDelete) {
      onDelete(person.person_id);
    } else {
      setConfirmDelete(true);
      setTimeout(() => setConfirmDelete(false), 3000);
    }
  };

  return (
    <article
      className="bg-gray-800 rounded-lg p-4 flex flex-col gap-3 hover:bg-gray-750 transition-colors duration-150 border border-transparent hover:border-gray-700"
      aria-label={`Person: ${person.label}`}
    >
      {/* Header: avatar + name + delete */}
      <div className="flex items-start gap-3">
        {/* Initials avatar */}
        <div
          className={`w-10 h-10 rounded-full flex items-center justify-center text-sm font-bold text-white shrink-0 ${avatarColor(person.label)}`}
          aria-hidden="true"
        >
          {getInitials(person.label)}
        </div>

        {/* Name + badges */}
        <div className="flex-1 min-w-0">
          <p className="text-sm font-semibold text-gray-100 truncate">{person.label}</p>
          <div className="flex flex-wrap items-center gap-1.5 mt-1">
            {/* Relationship badge */}
            <span className="inline-flex items-center rounded-full bg-gray-700 px-2 py-0.5 text-xs text-gray-300 capitalize">
              {person.relationship}
            </span>
            {/* Consent status badge (derived from history) */}
            <span
              className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium capitalize ${consentBadgeClass}`}
            >
              {derivedConsentKey}
            </span>
          </div>
        </div>

        {/* Delete button */}
        <button
          type="button"
          onClick={handleDeleteClick}
          aria-label={confirmDelete ? `Confirm delete ${person.label}` : `Delete ${person.label}`}
          title={confirmDelete ? 'Click again to confirm deletion' : 'Delete person'}
          className={`
            shrink-0 flex items-center justify-center rounded-lg px-2 py-1.5 text-xs font-medium
            transition-colors duration-150 focus-visible:outline-none focus-visible:ring-2
            focus-visible:ring-red-500 focus-visible:ring-offset-2 focus-visible:ring-offset-gray-800
            ${confirmDelete
              ? 'bg-red-600 text-white hover:bg-red-500'
              : 'text-gray-500 hover:text-red-400 hover:bg-red-500/10'
            }
          `}
        >
          {confirmDelete ? (
            <span className="flex items-center gap-1">
              <TrashIcon />
              <span>Confirm</span>
            </span>
          ) : (
            <TrashIcon />
          )}
        </button>
      </div>

      {/* Stats row */}
      <div className="grid grid-cols-3 gap-2 text-center">
        <div className="rounded-md bg-gray-900/50 px-2 py-1.5">
          <p className="text-sm font-semibold text-gray-100">
            {person.consent_history.times_appeared}
          </p>
          <p className="text-xs text-gray-500 leading-tight">appearances</p>
        </div>
        <div className="rounded-md bg-gray-900/50 px-2 py-1.5">
          <p className="text-sm font-semibold text-gray-100">
            {person.consent_history.approval_rate !== undefined
              ? `${Math.round(person.consent_history.approval_rate * 100)}%`
              : '—'}
          </p>
          <p className="text-xs text-gray-500 leading-tight">approval</p>
        </div>
        <div className="rounded-md bg-gray-900/50 px-2 py-1.5">
          <p className="text-sm font-semibold text-gray-100">
            {person.consent_history.consent_confidence !== undefined
              ? `${Math.round(person.consent_history.consent_confidence * 100)}%`
              : '—'}
          </p>
          <p className="text-xs text-gray-500 leading-tight">confidence</p>
        </div>
      </div>

      {/* Consent history detail */}
      <ConsentHistory history={person.consent_history} />

      {/* Footer: last seen date */}
      <p className="text-xs text-gray-500">
        Last seen: {formatDate(person.last_seen)}
      </p>
    </article>
  );
}
