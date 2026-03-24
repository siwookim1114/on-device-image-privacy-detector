import { create } from 'zustand';

import type {
  OnboardingState,
  ContactEntry,
  SensitivityLevel,
  TextSensitivity,
  ScreenSensitivity,
  ObjectSensitivity,
  AutoAdvanceLevel,
} from '../types/profile';
import { createProfile } from '../api/profile';

interface OnboardingActions {
  setStep: (step: number) => void;
  setIdentity: (label: string, photos: File[], skipped?: boolean) => void;
  addContact: (contact: ContactEntry) => void;
  removeContact: (index: number) => void;
  setSensitivity: (
    key: keyof OnboardingState['sensitivity'],
    value: SensitivityLevel | TextSensitivity | ScreenSensitivity | ObjectSensitivity,
  ) => void;
  setAdvanced: (key: keyof OnboardingState['advanced'], value: string | boolean | AutoAdvanceLevel) => void;
  submitProfile: () => Promise<void>;
  setCompleted: () => void;
  reset: () => void;
}

type OnboardingStore = OnboardingState & OnboardingActions;

const SENSITIVITY_MAP: Record<string, string> = {
  paranoid: 'high',
  cautious: 'medium',
  relaxed: 'low',
  maximum: 'maximum',
  standard: 'standard',
  minimal: 'minimal',
  always: 'always',
  smart: 'smart',
  never: 'never',
  include: 'include',
};

const initialState: OnboardingState = {
  currentStep: 0,
  identity: { label: '', photos: [], skipped: false },
  contacts: [],
  sensitivity: {
    faces: 'cautious',
    text: 'standard',
    screens: 'smart',
    objects: 'standard',
  },
  advanced: {
    preferred_face_method: 'blur',
    preferred_text_method: 'solid_overlay',
    preferred_screen_method: 'blur',
    preferred_object_method: 'blur',
    auto_advance_threshold: 'medium',
    pause_on_critical: true,
  },
  completed: false,
};

export const useOnboardingStore = create<OnboardingStore>((set, get) => ({
  ...initialState,

  setStep: (step) => set({ currentStep: step }),

  setIdentity: (label, photos, skipped = false) =>
    set({ identity: { label, photos, skipped } }),

  addContact: (contact) =>
    set((state) => ({ contacts: [...state.contacts, contact] })),

  removeContact: (index) =>
    set((state) => ({
      contacts: state.contacts.filter((_, i) => i !== index),
    })),

  setSensitivity: (key, value) =>
    set((state) => ({
      sensitivity: { ...state.sensitivity, [key]: value },
    })),

  setAdvanced: (key, value) =>
    set((state) => ({
      advanced: { ...state.advanced, [key]: value },
    })),

  submitProfile: async () => {
    const state = get();
    const profilePayload = {
      label: state.identity.label || undefined,
      sensitivity: {
        faces: SENSITIVITY_MAP[state.sensitivity.faces] ?? state.sensitivity.faces,
        text: SENSITIVITY_MAP[state.sensitivity.text] ?? state.sensitivity.text,
        screens: SENSITIVITY_MAP[state.sensitivity.screens] ?? state.sensitivity.screens,
        objects: SENSITIVITY_MAP[state.sensitivity.objects] ?? state.sensitivity.objects,
      },
      advanced: {
        preferred_face_method: state.advanced.preferred_face_method,
        preferred_text_method: state.advanced.preferred_text_method,
        preferred_screen_method: state.advanced.preferred_screen_method,
        preferred_object_method: state.advanced.preferred_object_method,
        auto_advance_threshold: state.advanced.auto_advance_threshold,
        pause_on_critical: state.advanced.pause_on_critical,
      },
    };
    await createProfile(profilePayload);
    localStorage.setItem('onboarding_complete', 'true');
    set({ completed: true });
  },

  setCompleted: () => {
    const store = get();
    store.submitProfile().catch(() => {
      // If submit fails, still mark as completed locally so onboarding doesn't loop
      localStorage.setItem('onboarding_complete', 'true');
      set({ completed: true });
    });
  },

  reset: () => set(initialState),
}));
