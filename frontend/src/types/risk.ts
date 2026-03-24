import type { BoundingBox } from './detection';

export type RiskLevel = 'critical' | 'high' | 'medium' | 'low';
export type RiskType = 'identity_exposure' | 'information_disclosure' | 'location_exposure' | 'context_exposure';

export interface RiskAssessment {
  detection_id: string;
  element_type: string;
  element_description: string;
  risk_type: RiskType;
  severity: RiskLevel;
  color_code: string;
  reasoning: string;
  bbox: BoundingBox;
  requires_protection: boolean;
  screen_state: 'verified_on' | 'verified_off' | null;
  person_id: string | null;
  person_label: string | null;
  classification: 'primary_subject' | 'known_contact' | 'bystander' | null;
  consent_status: 'explicit' | 'assumed' | 'none' | 'unclear' | null;
  consent_confidence: number;
  text_polygon: number[][] | null;
  user_sensitivity_applied: string;
  legal_requirement: boolean;
  screen_bbox: BoundingBox | null;
}

export interface RiskAnalysisResult {
  overall_risk_level: RiskLevel;
  confirmed_risks: number;
  faces_pending_identity: number;
  risk_assessments: RiskAssessment[];
  processing_time_ms: number;
  image_path?: string;
}
