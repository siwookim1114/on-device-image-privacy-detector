import type { PipelineStage } from './pipeline';

export type IntentAction = 'process' | 'modify_strategy' | 'strengthen' | 'ignore' | 'query' | 'undo' | 'approve' | 'reject';

export interface ParsedIntent {
  action: IntentAction;
  target_stage: string;
  target_elements: string[] | null;
  confidence: number;
  natural_language: string;
}

export interface ParsedIntentResponse {
  action: IntentAction;
  target_stage: string | null;
  target_elements: string[];
  confidence: number;
  natural_language: string;
}

export interface ChatResponse {
  intent: ParsedIntentResponse;
  response_text: string;
  pipeline_action_taken: { action: string; from_stage: PipelineStage } | null;
  suggestions: string[];
}
