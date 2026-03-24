import { useState, useEffect, useCallback } from 'react';

import { useStudyLogger } from '../../hooks/useStudyLogger';
import { formatDuration } from '../../lib/formatters';

function getParticipantIdFromUrl(): string | null {
  const params = new URLSearchParams(window.location.search);
  const pid = params.get('pid');
  return pid !== null && pid.trim() !== '' ? pid.trim() : null;
}

export function StudyBar() {
  const participantId = getParticipantIdFromUrl();

  if (participantId === null) return null;

  return <StudyBarInner participantId={participantId} />;
}

function StudyBarInner({ participantId }: { participantId: string }) {
  const { log, getEvents, exportEvents } = useStudyLogger();

  const [startTime] = useState<number>(() => Date.now());
  const [elapsedMs, setElapsedMs] = useState<number>(0);
  const [eventCount, setEventCount] = useState<number>(() => getEvents().length);

  useEffect(() => {
    const id = setInterval(() => {
      setElapsedMs(Date.now() - startTime);
      setEventCount(getEvents().length);
    }, 1000);
    return () => clearInterval(id);
  }, [startTime, getEvents]);

  const handleExport = useCallback(() => {
    log({ event_type: 'export', source: 'mouse' });
    exportEvents();
    setEventCount(getEvents().length);
  }, [log, exportEvents, getEvents]);

  const taskId = sessionStorage.getItem('study_task_id') ?? '—';

  return (
    <div
      role="status"
      aria-label="Study session status bar"
      className="h-8 bg-gray-900 border-t border-gray-800 flex items-center px-3 gap-4 shrink-0"
    >
      {/* Participant */}
      <span className="text-xs text-gray-500 tabular-nums">
        <span className="text-gray-600 uppercase tracking-wider mr-1">PID</span>
        <span className="text-gray-400 font-mono">{participantId}</span>
      </span>

      <Divider />

      {/* Task */}
      <span className="text-xs text-gray-500 tabular-nums">
        <span className="text-gray-600 uppercase tracking-wider mr-1">Task</span>
        <span className="text-gray-400 font-mono">{taskId}</span>
      </span>

      <Divider />

      {/* Elapsed */}
      <span className="text-xs text-gray-500 tabular-nums">
        <span className="text-gray-600 uppercase tracking-wider mr-1">Elapsed</span>
        <span className="text-gray-400 font-mono">{formatDuration(elapsedMs)}</span>
      </span>

      <Divider />

      {/* Event count */}
      <span className="text-xs text-gray-500 tabular-nums">
        <span className="text-gray-600 uppercase tracking-wider mr-1">Events</span>
        <span className="text-gray-400 font-mono">{eventCount}</span>
      </span>

      {/* Spacer */}
      <div className="flex-1" />

      {/* Export button */}
      <button
        type="button"
        onClick={handleExport}
        className="
          h-5 px-2 text-xs text-gray-500 border border-gray-700 rounded
          hover:text-gray-300 hover:border-gray-500
          transition-colors duration-100
          font-medium
        "
      >
        Export Log
      </button>
    </div>
  );
}

function Divider() {
  return <span className="text-gray-800 select-none">|</span>;
}
