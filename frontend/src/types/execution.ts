import type { ObfuscationMethod } from './strategy';

export interface TransformationResult {
  detection_id: string;
  element: string;
  method: ObfuscationMethod;
  parameters: Record<string, unknown>;
  status: 'success' | 'failed' | 'skipped';
  execution_time_ms: number;
  error_message: string | null;
}

export interface ExecutionReport {
  status: 'completed' | 'partial' | 'failed';
  transformations_applied: TransformationResult[];
  elements_unchanged: Array<{ detection_id: string; reason: string }>;
  total_execution_time_ms: number;
  image_path?: string;
  protected_image_path?: string;
}
