export interface ApiError {
  code: string;
  message: string;
  details?: Record<string, unknown>;
}

export interface WsEnvelope<T = unknown> {
  type: string;
  session_id: string;
  timestamp: string;
  payload: T;
}
