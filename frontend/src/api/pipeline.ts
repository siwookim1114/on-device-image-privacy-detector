import apiClient from './client';

import type { PipelineStatusResponse, ProcessingMode, EthicalMode } from '../types/pipeline';

export async function startPipeline(image: File, mode: ProcessingMode = 'hybrid', ethicalMode: EthicalMode = 'balanced') {
  const form = new FormData();
  form.append('image', image);
  form.append('config', JSON.stringify({ mode, ethical_mode: ethicalMode }));
  const { data } = await apiClient.post('/pipeline/run', form, { headers: { 'Content-Type': 'multipart/form-data' } });
  return data as { session_id: string; status: string };
}

export async function getPipelineStatus(sessionId: string) {
  const { data } = await apiClient.get<PipelineStatusResponse>(`/pipeline/${sessionId}/status`);
  return data;
}

export async function getPipelineResults(sessionId: string) {
  const { data } = await apiClient.get(`/pipeline/${sessionId}/results`);
  return data;
}

export async function rerunPipeline(sessionId: string, fromStage: string, reason: string) {
  const { data } = await apiClient.post(`/pipeline/${sessionId}/rerun`, { from_stage: fromStage, reason });
  return data;
}

export async function approvePipeline(sessionId: string, checkpoint: string) {
  const { data } = await apiClient.post(`/pipeline/${sessionId}/approve/${checkpoint}`);
  return data;
}

export function getImageUrl(sessionId: string, type: 'original' | 'protected' | 'risk-map') {
  return `${import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1'}/pipeline/${sessionId}/image/${type}`;
}
