export type SensitivityLevel = 'paranoid' | 'cautious' | 'relaxed';
export type TextSensitivity = 'maximum' | 'standard' | 'minimal';
export type ScreenSensitivity = 'always' | 'smart' | 'never';
export type ObjectSensitivity = 'include' | 'standard' | 'minimal';
export type AutoAdvanceLevel = 'never' | 'low' | 'medium' | 'high';

export interface ContactEntry {
  person_id?: string;
  display_name: string;
  relationship: string;
  consent_level: 'explicit' | 'assumed' | 'none';
  photos: File[];
}

export interface OnboardingState {
  currentStep: number;
  identity: { label: string; photos: File[]; skipped: boolean };
  contacts: ContactEntry[];
  sensitivity: {
    faces: SensitivityLevel;
    text: TextSensitivity;
    screens: ScreenSensitivity;
    objects: ObjectSensitivity;
  };
  advanced: {
    preferred_face_method: string;
    preferred_text_method: string;
    preferred_screen_method: string;
    preferred_object_method: string;
    auto_advance_threshold: AutoAdvanceLevel;
    pause_on_critical: boolean;
  };
  completed: boolean;
}
