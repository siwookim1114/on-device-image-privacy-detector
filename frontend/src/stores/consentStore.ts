import { create } from 'zustand';
import type { PersonEntry } from '../types/consent';
import {
  getPersons,
  registerPerson as apiRegisterPerson,
  deletePerson as apiDeletePerson,
} from '../api/consent';

interface ConsentState {
  persons: PersonEntry[];
  isLoading: boolean;
  error: string | null;

  fetchPersons: () => Promise<void>;
  registerPerson: (formData: FormData) => Promise<void>;
  removePerson: (personId: string) => Promise<void>;
  setError: (error: string | null) => void;
}

export const useConsentStore = create<ConsentState>((set) => ({
  persons: [],
  isLoading: false,
  error: null,

  fetchPersons: async () => {
    set({ isLoading: true, error: null });
    try {
      const data = await getPersons();
      set({ persons: data.persons, isLoading: false });
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to load persons.';
      set({ isLoading: false, error: message });
    }
  },

  registerPerson: async (formData: FormData) => {
    set({ isLoading: true, error: null });
    try {
      const newPerson = await apiRegisterPerson(formData);
      set((state) => ({
        persons: [...state.persons, newPerson],
        isLoading: false,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to register person.';
      set({ isLoading: false, error: message });
      throw err;
    }
  },

  removePerson: async (personId: string) => {
    set({ isLoading: true, error: null });
    try {
      await apiDeletePerson(personId);
      set((state) => ({
        persons: state.persons.filter((p) => p.person_id !== personId),
        isLoading: false,
      }));
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to remove person.';
      set({ isLoading: false, error: message });
      throw err;
    }
  },

  setError: (error) => set({ error }),
}));
