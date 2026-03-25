import { useState, useEffect, useRef, useCallback } from 'react';

import type { WsEnvelope } from '../types/api';
import type { PipelineStage, HitlCheckpoint } from '../types/pipeline';
import { usePipelineStore } from '../stores/pipelineStore';

interface UseWebSocketResult {
  connected: boolean;
  lastEvent: WsEnvelope | null;
}

const WS_BASE_URL: string = (import.meta.env.VITE_WS_URL as string | undefined) ?? 'ws://localhost:8000/api/v1';
const PING_INTERVAL_MS = 30_000;
const MAX_BACKOFF_MS = 30_000;
const BASE_BACKOFF_MS = 1_000;

function buildWsUrl(sessionId: string): string {
  const token = sessionStorage.getItem('session_token');
  const base = `${WS_BASE_URL}/pipeline/${sessionId}/stream`;
  return token !== null ? `${base}?token=${encodeURIComponent(token)}` : base;
}

interface StageStartPayload {
  stage: PipelineStage;
}

interface StageCompletePayload {
  stage: PipelineStage;
  elapsed_ms: number;
}

interface HitlCheckpointPayload {
  checkpoint: HitlCheckpoint;
}

interface PipelineCompletePayload {
  stage: PipelineStage;
  total_time_ms: number;
  protections_applied: number;
  results_url: string;
}

export function useWebSocket(sessionId: string | null): UseWebSocketResult {
  const [connected, setConnected] = useState(false);
  const [lastEvent, setLastEvent] = useState<WsEnvelope | null>(null);

  const setStage = usePipelineStore((s) => s.setStage);
  const setHitlCheckpoint = usePipelineStore((s) => s.setHitlCheckpoint);
  const addTiming = usePipelineStore((s) => s.addTiming);

  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimerRef = useRef<ReturnType<typeof setTimeout> | null>(null);
  const pingTimerRef = useRef<ReturnType<typeof setInterval> | null>(null);
  const attemptRef = useRef(0);
  const unmountedRef = useRef(false);

  const clearTimers = useCallback(() => {
    if (reconnectTimerRef.current !== null) {
      clearTimeout(reconnectTimerRef.current);
      reconnectTimerRef.current = null;
    }
    if (pingTimerRef.current !== null) {
      clearInterval(pingTimerRef.current);
      pingTimerRef.current = null;
    }
  }, []);

  const closeSocket = useCallback(() => {
    if (wsRef.current !== null) {
      wsRef.current.onopen = null;
      wsRef.current.onclose = null;
      wsRef.current.onerror = null;
      wsRef.current.onmessage = null;
      wsRef.current.close();
      wsRef.current = null;
    }
  }, []);

  const handleMessage = useCallback(
    (raw: string) => {
      let envelope: WsEnvelope;
      try {
        envelope = JSON.parse(raw) as WsEnvelope;
      } catch {
        return;
      }

      setLastEvent(envelope);

      switch (envelope.type) {
        case 'stage_start': {
          const payload = envelope.payload as StageStartPayload;
          setStage(payload.stage);
          break;
        }
        case 'stage_complete': {
          const payload = envelope.payload as StageCompletePayload;
          addTiming(payload.stage, payload.elapsed_ms);
          break;
        }
        case 'hitl_checkpoint': {
          const payload = envelope.payload as HitlCheckpointPayload;
          setHitlCheckpoint(payload.checkpoint);
          break;
        }
        case 'pipeline_complete': {
          const payload = envelope.payload as PipelineCompletePayload;
          setStage(payload.stage);
          setHitlCheckpoint(null);
          break;
        }
        default:
          break;
      }
    },
    [setStage, setHitlCheckpoint, addTiming],
  );

  const connect = useCallback(
    (sid: string) => {
      closeSocket();

      const ws = new WebSocket(buildWsUrl(sid));
      wsRef.current = ws;

      ws.onopen = () => {
        if (unmountedRef.current) return;
        setConnected(true);
        attemptRef.current = 0;

        pingTimerRef.current = setInterval(() => {
          if (ws.readyState === WebSocket.OPEN) {
            ws.send(JSON.stringify({ type: 'ping' }));
          }
        }, PING_INTERVAL_MS);
      };

      ws.onmessage = (event: MessageEvent<string>) => {
        if (!unmountedRef.current) {
          handleMessage(event.data);
        }
      };

      ws.onclose = () => {
        if (unmountedRef.current) return;
        setConnected(false);
        clearTimers();

        const backoff = Math.min(BASE_BACKOFF_MS * 2 ** attemptRef.current, MAX_BACKOFF_MS);
        attemptRef.current += 1;

        reconnectTimerRef.current = setTimeout(() => {
          if (!unmountedRef.current) {
            connect(sid);
          }
        }, backoff);
      };

      ws.onerror = () => {
        ws.close();
      };
    },
    [closeSocket, clearTimers, handleMessage],
  );

  useEffect(() => {
    unmountedRef.current = false;

    if (sessionId === null) {
      closeSocket();
      clearTimers();
      setConnected(false);
      setLastEvent(null);
      return;
    }

    attemptRef.current = 0;
    connect(sessionId);

    return () => {
      unmountedRef.current = true;
      clearTimers();
      closeSocket();
    };
  }, [sessionId, connect, closeSocket, clearTimers]);

  return { connected, lastEvent };
}
