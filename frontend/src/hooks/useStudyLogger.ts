import { useRef, useCallback } from 'react';

export interface StudyEvent {
  timestamp: number;
  participant_id: string;
  task_id: string;
  session_id: string;
  event_type:
    | 'page_view'
    | 'image_upload'
    | 'select_element'
    | 'change_severity'
    | 'change_method'
    | 'change_params'
    | 'toggle_protection'
    | 'approve_checkpoint'
    | 'reject_checkpoint'
    | 'chat_submit'
    | 'chat_response'
    | 'accept_result'
    | 'reprocess'
    | 'undo'
    | 'export'
    | 'keyboard_shortcut'
    | 'tab_switch'
    | 'onboarding_step_enter'
    | 'onboarding_step_complete'
    | 'onboarding_identity_save'
    | 'onboarding_identity_skip'
    | 'onboarding_contact_add'
    | 'onboarding_contact_remove'
    | 'onboarding_sensitivity_change'
    | 'onboarding_complete';
  target_detection_id?: string;
  old_value?: string;
  new_value?: string;
  source: 'mouse' | 'keyboard' | 'chat' | 'system';
  metadata?: Record<string, unknown>;
}

export interface StudyLoggerHandle {
  log: (event: Omit<StudyEvent, 'timestamp' | 'participant_id' | 'task_id' | 'session_id'>) => void;
  getEvents: () => StudyEvent[];
  exportEvents: () => void;
  clearEvents: () => void;
}

/** Derive a stable session ID for this browser tab lifetime. */
function resolveSessionId(): string {
  const key = 'study_session_id';
  let id = sessionStorage.getItem(key);
  if (!id) {
    id = `sess_${Date.now()}_${Math.random().toString(36).slice(2, 9)}`;
    sessionStorage.setItem(key, id);
  }
  return id;
}

/**
 * Read a value from URL query params first, then fall back to sessionStorage,
 * then fall back to the provided default.
 */
function resolveParam(queryKey: string, storageKey: string, fallback: string): string {
  const params = new URLSearchParams(window.location.search);
  const fromUrl = params.get(queryKey);
  if (fromUrl !== null && fromUrl.trim() !== '') {
    sessionStorage.setItem(storageKey, fromUrl.trim());
    return fromUrl.trim();
  }
  const fromStorage = sessionStorage.getItem(storageKey);
  if (fromStorage !== null && fromStorage.trim() !== '') {
    return fromStorage.trim();
  }
  return fallback;
}

export function useStudyLogger(): StudyLoggerHandle {
  const participantId = resolveParam('pid', 'study_participant_id', 'unknown');
  const taskId = resolveParam('task', 'study_task_id', '0');
  const sessionId = resolveSessionId();

  const storageKeyRef = useRef<string>(`study_events_${participantId}`);
  const participantIdRef = useRef<string>(participantId);
  const taskIdRef = useRef<string>(taskId);
  const sessionIdRef = useRef<string>(sessionId);

  const eventsRef = useRef<StudyEvent[] | null>(null);

  function ensureLoaded(): StudyEvent[] {
    if (eventsRef.current !== null) return eventsRef.current;
    try {
      const raw = sessionStorage.getItem(storageKeyRef.current);
      eventsRef.current = raw ? (JSON.parse(raw) as StudyEvent[]) : [];
    } catch {
      eventsRef.current = [];
    }
    return eventsRef.current;
  }

  function persist(events: StudyEvent[]): void {
    try {
      sessionStorage.setItem(storageKeyRef.current, JSON.stringify(events));
    } catch {
      // sessionStorage quota exceeded — silently continue; in-memory events
      // are still intact.
    }
  }

  const log = useCallback(
    (event: Omit<StudyEvent, 'timestamp' | 'participant_id' | 'task_id' | 'session_id'>) => {
      const full: StudyEvent = {
        ...event,
        timestamp: Date.now(),
        participant_id: participantIdRef.current,
        task_id: taskIdRef.current,
        session_id: sessionIdRef.current,
      };
      const events = ensureLoaded();
      events.push(full);
      persist(events);
    },
    [],
  );

  const getEvents = useCallback((): StudyEvent[] => {
    return [...ensureLoaded()];
  }, []);

  const exportEvents = useCallback((): void => {
    const events = ensureLoaded();
    const filename = `study_${participantIdRef.current}_task${taskIdRef.current}_${Date.now()}.json`;
    const blob = new Blob([JSON.stringify(events, null, 2)], {
      type: 'application/json',
    });
    const url = URL.createObjectURL(blob);
    const anchor = document.createElement('a');
    anchor.href = url;
    anchor.download = filename;
    anchor.click();
    // Revoke after a short delay to allow the download to begin.
    setTimeout(() => URL.revokeObjectURL(url), 1000);
  }, []);

  const clearEvents = useCallback((): void => {
    eventsRef.current = [];
    try {
      sessionStorage.removeItem(storageKeyRef.current);
    } catch {
      // Ignore
    }
  }, []);

  return { log, getEvents, exportEvents, clearEvents };
}
