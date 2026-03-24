import { useState } from 'react';

import type { SessionSummary } from '../api/history';
import { SessionList } from '../components/history/SessionList';
import { DownloadButton } from '../components/history/DownloadButton';
import { ProvenanceViewer } from '../components/history/ProvenanceViewer';

function SelectedSessionPanel({ session }: { session: SessionSummary }) {
  return (
    <aside className="w-80 flex-shrink-0 border-l border-gray-800 p-6 space-y-6 overflow-y-auto">
      <div>
        <h2 className="text-sm font-semibold text-gray-200 mb-1 truncate" title={session.image_filename}>
          {session.image_filename}
        </h2>
        <p className="text-xs text-gray-500 font-mono break-all">{session.session_id}</p>
      </div>

      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Downloads
        </h3>
        <div className="space-y-3">
          <DownloadButton
            sessionId={session.session_id}
            type="protected"
            label="Protected Image"
          />
          <DownloadButton
            sessionId={session.session_id}
            type="risk_json"
            label="Risk Report (JSON)"
          />
          <DownloadButton
            sessionId={session.session_id}
            type="provenance"
            label="Provenance Log (JSON)"
          />
        </div>
      </div>

      <div>
        <h3 className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
          Audit Trail
        </h3>
        <ProvenanceViewer sessionId={session.session_id} />
      </div>
    </aside>
  );
}

export default function History() {
  const [selectedSession, setSelectedSession] = useState<SessionSummary | null>(null);

  return (
    <div className="flex h-full overflow-hidden">
      <div className="flex-1 overflow-y-auto p-8 min-w-0">
        <div className="max-w-4xl mx-auto">
          <div className="mb-8">
            <h1 className="text-2xl font-bold text-gray-100">Processing History</h1>
            <p className="mt-1 text-gray-400">View and download past results</p>
          </div>

          <SessionList onSessionSelect={setSelectedSession} selectedSessionId={selectedSession?.session_id ?? null} />
        </div>
      </div>

      {selectedSession !== null && (
        <SelectedSessionPanel session={selectedSession} />
      )}
    </div>
  );
}
