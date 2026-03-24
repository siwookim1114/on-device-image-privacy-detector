import apiClient from './client';
import type { PersonEntry } from '../types/consent';

export async function getPersons() {
  const { data } = await apiClient.get<{ total: number; persons: PersonEntry[] }>('/consent/persons');
  return data;
}

export async function registerPerson(formData: FormData) {
  const { data } = await apiClient.post<PersonEntry>('/consent/persons', formData, { headers: { 'Content-Type': 'multipart/form-data' } });
  return data;
}

export async function updatePerson(personId: string, updates: Partial<PersonEntry>) {
  const { data } = await apiClient.put<PersonEntry>(`/consent/persons/${personId}`, updates);
  return data;
}

export async function deletePerson(personId: string) {
  await apiClient.delete(`/consent/persons/${personId}`);
}
