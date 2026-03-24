import axios from 'axios';

import type { ApiError } from '../types/api';

const apiClient = axios.create({
  baseURL: import.meta.env.VITE_API_URL || 'http://localhost:8000/api/v1',
  timeout: 30000,
});

apiClient.interceptors.request.use((config) => {
  const token = sessionStorage.getItem('session_token');
  if (token) config.headers.Authorization = `Bearer ${token}`;
  return config;
});

apiClient.interceptors.response.use(
  (res) => res,
  (err) => {
    const apiError: ApiError = err.response?.data?.error || { code: 'NETWORK_ERROR', message: err.message };
    return Promise.reject(apiError);
  }
);

export default apiClient;
