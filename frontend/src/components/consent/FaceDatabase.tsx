import { useEffect, useState } from 'react';

import { useConsentStore } from '../../stores/consentStore';
import { PersonCard } from './PersonCard';

function SearchIcon() {
  return (
    <svg
      className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-500 pointer-events-none"
      fill="none"
      viewBox="0 0 24 24"
      stroke="currentColor"
      strokeWidth={2}
      aria-hidden="true"
    >
      <circle cx="11" cy="11" r="8" />
      <path strokeLinecap="round" strokeLinejoin="round" d="M21 21l-4.35-4.35" />
    </svg>
  );
}

function SkeletonCard() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 space-y-3 animate-pulse" aria-hidden="true">
      <div className="flex items-start gap-3">
        <div className="w-10 h-10 rounded-full bg-gray-700 shrink-0" />
        <div className="flex-1 space-y-2">
          <div className="h-3.5 w-1/2 rounded bg-gray-700" />
          <div className="flex gap-1.5">
            <div className="h-3 w-16 rounded-full bg-gray-700" />
            <div className="h-3 w-16 rounded-full bg-gray-700" />
          </div>
        </div>
      </div>
      <div className="grid grid-cols-3 gap-2">
        {[0, 1, 2].map((i) => (
          <div key={i} className="rounded-md bg-gray-900/50 h-10" />
        ))}
      </div>
      <div className="space-y-2">
        <div className="h-2 w-full rounded bg-gray-700" />
        <div className="h-2 w-4/5 rounded bg-gray-700" />
      </div>
    </div>
  );
}

function EmptyState() {
  return (
    <div className="col-span-full flex flex-col items-center justify-center py-16 text-center px-4">
      <div className="w-14 h-14 rounded-full bg-gray-800 flex items-center justify-center mb-4">
        <svg
          className="w-7 h-7 text-gray-600"
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={1.5}
          aria-hidden="true"
        >
          <circle cx="12" cy="8" r="4" />
          <path strokeLinecap="round" strokeLinejoin="round" d="M4 20c0-4 3.582-7 8-7s8 3 8 7" />
        </svg>
      </div>
      <p className="text-sm font-medium text-gray-400">No faces registered.</p>
      <p className="text-xs text-gray-600 mt-1">
        Register someone to enable consent-aware protection.
      </p>
    </div>
  );
}

function NoResultsState({ query }: { query: string }) {
  return (
    <div className="col-span-full flex flex-col items-center justify-center py-12 text-center px-4">
      <p className="text-sm text-gray-400">
        No results for <span className="font-medium text-gray-200">"{query}"</span>
      </p>
      <p className="text-xs text-gray-600 mt-1">Try a different name or clear the search.</p>
    </div>
  );
}

export function FaceDatabase() {
  const { persons, isLoading, error, fetchPersons, removePerson } = useConsentStore();
  const [query, setQuery] = useState('');

  useEffect(() => {
    void fetchPersons();
  }, [fetchPersons]);

  const filtered = query.trim()
    ? persons.filter((p) =>
        p.label.toLowerCase().includes(query.trim().toLowerCase()),
      )
    : persons;

  const handleDelete = (personId: string) => {
    void removePerson(personId);
  };

  return (
    <div className="space-y-4">
      {/* Header + search */}
      <div className="flex items-center justify-between gap-3 flex-wrap">
        <div>
          <h2 className="text-base font-semibold text-gray-100">Registered Faces</h2>
          {!isLoading && (
            <p className="text-xs text-gray-500 mt-0.5">
              {persons.length} {persons.length === 1 ? 'person' : 'persons'} registered
            </p>
          )}
        </div>

        <div className="relative w-full sm:w-64">
          <SearchIcon />
          <input
            type="search"
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Search by name..."
            aria-label="Search registered faces"
            className="w-full rounded-lg pl-9 pr-3 py-2 text-sm bg-gray-800 text-gray-100 placeholder-gray-500 border border-gray-700 hover:border-gray-600 focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 focus:ring-offset-gray-950 transition-colors duration-150"
          />
        </div>
      </div>

      {/* Error banner */}
      {error && (
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
          <p className="text-xs text-red-400">{error}</p>
        </div>
      )}

      {/* Grid */}
      <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
        {isLoading ? (
          Array.from({ length: 3 }).map((_, i) => <SkeletonCard key={i} />)
        ) : persons.length === 0 ? (
          <EmptyState />
        ) : filtered.length === 0 ? (
          <NoResultsState query={query} />
        ) : (
          filtered.map((person) => (
            <PersonCard key={person.person_id} person={person} onDelete={handleDelete} />
          ))
        )}
      </div>
    </div>
  );
}
