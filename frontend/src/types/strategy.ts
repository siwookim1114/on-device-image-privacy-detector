import type { RiskLevel } from './risk';

export type ObfuscationMethod = 'blur' | 'pixelate' | 'solid_overlay' | 'inpaint' | 'avatar_replace' | 'generative_replace' | 'none';

export interface ProtectionStrategy {
  detection_id: string;
  element: string;
  severity: RiskLevel;
  recommended_action: string;
  recommended_method: ObfuscationMethod | null;
  parameters: Record<string, unknown>;
  reasoning: string;
  alternative_options: Array<{
    method: ObfuscationMethod;
    parameters: Record<string, unknown>;
    reasoning: string;
    score: number;
  }>;
  execution_priority: number;
  requires_user_decision: boolean;
  user_can_override: boolean;
  segmentation_mask_path: string | null;
}

export interface StrategyRecommendations {
  strategies: ProtectionStrategy[];
  total_protections_recommended: number;
  requires_user_confirmation: number;
  estimated_processing_time_ms: number;
}
