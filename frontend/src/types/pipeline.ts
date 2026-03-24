export type ProcessingMode = 'auto' | 'hybrid' | 'manual';
export type EthicalMode = 'strict' | 'balanced' | 'creative';
export type PipelineStatus = 'queued' | 'running' | 'paused_hitl' | 'completed' | 'failed';
export type PipelineStage = 'detection' | 'risk' | 'consent' | 'strategy' | 'sam' | 'execution' | 'export' | 'done';
export type HitlCheckpoint = 'risk_review' | 'strategy_review' | 'execution_verify';

export interface StageProgress {
  stage: PipelineStage;
  step: string;
  elements_processed: number;
  elements_total: number;
  elapsed_ms: number;
}

export interface PipelineStatusResponse {
  session_id: string;
  status: PipelineStatus;
  current_stage: PipelineStage;
  stage_progress: StageProgress | null;
  hitl: {
    waiting: boolean;
    checkpoint: HitlCheckpoint | null;
    checkpoint_reason: string | null;
    elements_requiring_review: string[];
    actions_available: string[];
  };
  timing: {
    queued_at: string | null;
    started_at: string | null;
    completed_at: string | null;
    elapsed_ms: number;
  };
  error: string | null;
}
