import apiClient from './client';

export interface ProfileData {
  label?: string;
  sensitivity?: {
    faces?: string;
    text?: string;
    screens?: string;
    objects?: string;
  };
  advanced?: {
    preferred_face_method?: string;
    preferred_text_method?: string;
    preferred_screen_method?: string;
    preferred_object_method?: string;
    auto_advance_threshold?: string;
    pause_on_critical?: boolean;
  };
}

export interface QuestionnaireStep {
  id: string;
  text: string;
  type: 'single' | 'multiple' | 'scale';
  options?: Array<{ value: string; label: string }>;
}

export interface QuestionnaireResponse {
  steps: QuestionnaireStep[];
}

export async function getProfile(): Promise<ProfileData> {
  const { data } = await apiClient.get<ProfileData>('/profile');
  return data;
}

export async function createProfile(profileData: ProfileData): Promise<ProfileData> {
  const { data: result } = await apiClient.post<ProfileData>('/profile', profileData);
  return result;
}

export async function updateProfile(profileData: ProfileData): Promise<ProfileData> {
  const { data: result } = await apiClient.put<ProfileData>('/profile', profileData);
  return result;
}

export async function deleteProfile(): Promise<void> {
  await apiClient.delete('/profile');
}

export async function getQuestionnaire(): Promise<QuestionnaireResponse> {
  const { data } = await apiClient.get<QuestionnaireResponse>('/profile/questionnaire');
  return data;
}

export async function enrollFace(file: File): Promise<{ person_id: string }> {
  const formData = new FormData();
  formData.append('file', file);
  formData.append('relationship', 'self');
  formData.append('consent_status', 'explicit');
  const { data } = await apiClient.post<{ person_id: string }>(
    '/profile/enroll-face',
    formData,
    { headers: { 'Content-Type': 'multipart/form-data' } },
  );
  return data;
}
