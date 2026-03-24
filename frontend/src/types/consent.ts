export type ConsentStatus = 'explicit' | 'assumed' | 'none' | 'unclear';

export interface ConsentHistory {
  times_appeared: number;
  times_approved: number;
  times_protected: number;
  approval_rate?: number;
  protection_rate?: number;
  last_consent_decision: string | null;
  contexts: string[];
  consent_confidence: number;
}

export interface PersonEntry {
  person_id: string;
  label: string;
  relationship: string;
  consent_history: ConsentHistory;
  first_seen: string;
  last_seen: string;
  notes: string | null;
  risk_decay_factor?: number;
}
