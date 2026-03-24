import { useState, useEffect, useCallback } from 'react';
import { useNavigate } from 'react-router-dom';
import { getHistory } from '../../api/history';
import type { SessionSummary } from '../../api/history';
import { SessionCard } from './SessionCard';

type StatusFilter = 'all' | 'completed' | 'failed';
type SortOrder = 'newest' | 'oldest';

const PAGE_SIZE = 20;

interface SessionListProps {
  onSessionSelect?: (session: SessionSummary) => void;
  selectedSessionId?: string | null;
}

function SkeletonCard() {
  return (
    <div className="bg-gray-800 rounded-lg p-4 border border-gray-700/50 animate-pulse">
      <div className="flex items-start justify-between gap-3 mb-3">
        <div className="h-4 bg-gray-700 rounded w-2/3" />
        <div className="flex gap-2">
          <div className="h-5 bg-gray-700 rounded w-16" />
          <div className="h-5 bg-gray-700 rounded w-14" />
        </div>
      </div>
      <div className="grid grid-cols-3 gap-3 mb-3">
        {[0, 1, 2].map((i) => (
          <div key={i} className="text-center space-y-1">
            <div className="h-5 bg-gray-700 rounded mx-auto w-8" />
            <div className="h-3 bg-gray-700 rounded mx-auto w-12" />
          </div>
        ))}
      </div>
      <div className="h-3 bg-gray-700 rounded w-24" />
    </div>
  );
}

export function SessionList({ onSessionSelect, selectedSessionId }: SessionListProps = {}) {
  const navigate = useNavigate();
  const [sessions, setSessions] = useState<SessionSummary[]>([]);
  const [total, setTotal] = useState(0);
  const [page, setPage] = useState(1);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');
  const [sortOrder, setSortOrder] = useState<SortOrder>('newest');

  const fetchHistory = useCallback(async (pageNum: number) => {
    setLoading(true);
    setError(null);
    try {
      const result = await getHistory(pageNum, PAGE_SIZE);
      setSessions(result.items);
      setTotal(result.total);
    } catch {
      setError('Failed to load history. Please try again.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void fetchHistory(page);
  }, [fetchHistory, page]);

  const filteredSessions = sessions
    .filter((s) => statusFilter === 'all' || s.status === statusFilter)
    .sort((a, b) => {
      const dateA = new Date(a.created_at).getTime();
      const dateB = new Date(b.created_at).getTime();
      return sortOrder === 'newest' ? dateB - dateA : dateA - dateB;
    });

  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-3">
        <div className="flex items-center gap-2">
          <label htmlFor="status-filter" className="text-xs text-gray-400 font-medium">
            Status
          </label>
          <select
            id="status-filter"
            value={statusFilter}
            onChange={(e) => {
              setStatusFilter(e.target.value as StatusFilter);
              setPage(1);
            }}
            className="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="all">All</option>
            <option value="completed">Completed</option>
            <option value="failed">Failed</option>
          </select>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="sort-order" className="text-xs text-gray-400 font-medium">
            Sort
          </label>
          <select
            id="sort-order"
            value={sortOrder}
            onChange={(e) => setSortOrder(e.target.value as SortOrder)}
            className="bg-gray-800 border border-gray-700 text-gray-200 text-xs rounded px-2 py-1.5 focus:outline-none focus:ring-1 focus:ring-blue-500"
          >
            <option value="newest">Newest first</option>
            <option value="oldest">Oldest first</option>
          </select>
        </div>

        {!loading && (
          <p className="text-xs text-gray-500 ml-auto">
            {filteredSessions.length} of {total} session{total !== 1 ? 's' : ''}
          </p>
        )}
      </div>

      {error !== null && (
        <div className="rounded-lg bg-red-900/30 border border-red-700/50 p-4 text-sm text-red-300">
          {error}
        </div>
      )}

      {loading ? (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {Array.from({ length: 6 }).map((_, i) => (
            <SkeletonCard key={i} />
          ))}
        </div>
      ) : filteredSessions.length === 0 ? (
        <div className="flex flex-col items-center justify-center py-16 text-center">
          <svg
            className="w-12 h-12 text-gray-600 mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            aria-hidden="true"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z"
            />
          </svg>
          <p className="text-gray-400 font-medium">No processing history yet.</p>
          <p className="text-gray-600 text-sm mt-1">Upload an image to start your first session.</p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          {filteredSessions.map((session) => (
            <div
              key={session.session_id}
              className={selectedSessionId === session.session_id ? 'ring-2 ring-blue-500 rounded-lg' : ''}
            >
              <SessionCard
                session={session}
                onClick={() => {
                  if (onSessionSelect !== undefined) {
                    onSessionSelect(session);
                  } else {
                    navigate(`/process/${session.session_id}`);
                  }
                }}
              />
            </div>
          ))}
        </div>
      )}

      {!loading && totalPages > 1 && (
        <div className="flex items-center justify-center gap-4 pt-2">
          <button
            onClick={() => setPage((p) => Math.max(1, p - 1))}
            disabled={page <= 1}
            className="px-3 py-1.5 text-sm rounded bg-gray-800 border border-gray-700 text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors"
          >
            Previous
          </button>
          <span className="text-sm text-gray-400">
            Page {page} of {totalPages}
          </span>
          <button
            onClick={() => setPage((p) => Math.min(totalPages, p + 1))}
            disabled={page >= totalPages}
            className="px-3 py-1.5 text-sm rounded bg-gray-800 border border-gray-700 text-gray-300 disabled:opacity-40 disabled:cursor-not-allowed hover:bg-gray-700 transition-colors"
          >
            Next
          </button>
        </div>
      )}
    </div>
  );
}
