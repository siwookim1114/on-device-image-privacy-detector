import apiClient from './client';

import type { ChatResponse } from '../types/coordinator';

export async function sendChat(sessionId: string, message: string) {
  const { data } = await apiClient.post<ChatResponse>(`/pipeline/${sessionId}/chat`, { message });
  return data;
}

export async function submitOverrides(sessionId: string, overrides: unknown[]) {
  const { data } = await apiClient.post(`/pipeline/${sessionId}/override`, { overrides });
  return data;
}
