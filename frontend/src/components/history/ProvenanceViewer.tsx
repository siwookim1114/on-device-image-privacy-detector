import { useState } from 'react';

import apiClient from '../../api/client';

interface ProvenanceRecord {
  timestamp: string;
  action: 'override' | 'approve' | 'reject';
  detection_id: string;
  old_value: string;
  new_value: string;
  reason?: string;
}

interface ProvenanceData {
  session_id: string;
  records: ProvenanceRecord[];
}

interface ProvenanceViewerProps {
  sessionId: string;
}

const ACTION_STYLES: Record<ProvenanceRecord['action'], string> = {
  override: 'text-yellow-400 bg-yellow-900/30 border border-yellow-700/40',
  approve: 'text-green-400 bg-green-900/30 border border-green-700/40',
  reject: 'text-red-400 bg-red-900/30 border border-red-700/40',
};

function formatTimestamp(ts: string): string {
  const date = new Date(ts);
  return date.toLocaleString('en-US', {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}

function RecordRow({ record }: { record: ProvenanceRecord }) {
  const [expanded, setExpanded] = useState(false);

  return (
    <div className="border border-gray-700/50 rounded overflow-hidden">
      <button
        onClick={() => setExpanded((v) => !v)}
        className="w-full flex items-center gap-3 px-3 py-2 text-left hover:bg-gray-700/30 transition-colors"
        aria-expanded={expanded}
      >
        <span
          className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium font-mono flex-shrink-0 ${ACTION_STYLES[record.action]}`}
        >
          {record.action}
        </span>
        <span className="text-xs text-gray-400 font-mono truncate flex-1">
          {record.detection_id}
        </span>
        <span className="text-xs text-gray-500 flex-shrink-0">
          {formatTimestamp(record.timestamp)}
        </span>
        <svg
          className={`w-4 h-4 text-gray-500 flex-shrink-0 transition-transform ${expanded ? 'rotate-180' : ''}`}
          fill="none"
          viewBox="0 0 24 24"
          stroke="currentColor"
          strokeWidth={2}
          aria-hidden="true"
        >
          <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
        </svg>
      </button>

      {expanded && (
        <div className="px-3 py-3 bg-gray-900/50 border-t border-gray-700/50 space-y-2">
          <div className="grid grid-cols-2 gap-3">
            <div>
              <p className="text-xs text-gray-500 mb-1">Old value</p>
              <p className="text-xs text-gray-300 font-mono bg-gray-800 rounded px-2 py-1 break-all">
                {record.old_value || '—'}
              </p>
            </div>
            <div>
              <p className="text-xs text-gray-500 mb-1">New value</p>
              <p className="text-xs text-gray-300 font-mono bg-gray-800 rounded px-2 py-1 break-all">
                {record.new_value || '—'}
              </p>
            </div>
          </div>
          {record.reason !== undefined && record.reason !== '' && (
            <div>
              <p className="text-xs text-gray-500 mb-1">Reason</p>
              <p className="text-xs text-gray-400 font-mono">{record.reason}</p>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

export function ProvenanceViewer({ sessionId }: ProvenanceViewerProps) {
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);
  const [data, setData] = useState<ProvenanceData | null>(null);
  const [error, setError] = useState<string | null>(null);

  const handleToggle = async () => {
    const next = !open;
    setOpen(next);
    if (next && data === null && !loading) {
      setLoading(true);
      setError(null);
      try {
        const response = await apiClient.get<ProvenanceData>(`/history/${sessionId}/provenance`);
        setData(response.data);
      } catch {
        setError('Failed to load provenance data.');
      } finally {
        setLoading(false);
      }
    }
  };

  const records = data?.records ?? [];

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700/50 overflow-hidden text-sm font-mono">
      <button
        onClick={() => void handleToggle()}
        className="w-full flex items-center justify-between px-4 py-3 hover:bg-gray-700/30 transition-colors"
        aria-expanded={open}
      >
        <span className="text-gray-300 font-sans font-medium text-sm">Provenance Log</span>
        <div className="flex items-center gap-2">
          {data !== null && (
            <span className="text-xs text-gray-500 font-sans">
              {records.length} record{records.length !== 1 ? 's' : ''}
            </span>
          )}
          <svg
            className={`w-4 h-4 text-gray-500 transition-transform ${open ? 'rotate-180' : ''}`}
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
            strokeWidth={2}
            aria-hidden="true"
          >
            <path strokeLinecap="round" strokeLinejoin="round" d="M19 9l-7 7-7-7" />
          </svg>
        </div>
      </button>

      {open && (
        <div className="border-t border-gray-700/50 p-4 space-y-2">
          {loading && (
            <div className="flex items-center gap-2 text-gray-500 text-xs font-sans">
              <svg className="w-4 h-4 animate-spin" viewBox="0 0 24 24" fill="none" aria-hidden="true">
                <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" />
                <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4z" />
              </svg>
              Loading provenance...
            </div>
          )}

          {error !== null && (
            <p className="text-xs text-red-400 font-sans">{error}</p>
          )}

          {!loading && error === null && records.length === 0 && (
            <p className="text-xs text-gray-500 font-sans">No provenance records for this session.</p>
          )}

          {records.map((record, idx) => (
            <RecordRow key={`${record.detection_id}-${idx}`} record={record} />
          ))}
        </div>
      )}
    </div>
  );
}
