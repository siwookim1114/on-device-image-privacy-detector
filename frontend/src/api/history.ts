import apiClient from './client';

export interface SessionSummary {
  session_id: string;
  image_filename: string;
  status: string;
  created_at: string;
  protections_applied: number;
  total_elements?: number;
  overall_risk?: string;
  total_time_ms?: number;
}

export async function getHistory(page: number = 1, limit: number = 20) {
  const { data } = await apiClient.get('/history', { params: { page, limit } });
  return data as { total: number; items: SessionSummary[] };
}

export async function downloadArtifact(sessionId: string, artifact: 'protected' | 'risk_json' | 'provenance') {
  const { data } = await apiClient.get(`/history/${sessionId}/download`, {
    params: { artifact },
    responseType: 'blob',
  });
  return data;
}
