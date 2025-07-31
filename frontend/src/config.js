// API Configuration
export const API_BASE_URL = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000';

export const API_ENDPOINTS = {
  CHAT: `${API_BASE_URL}/api/chat`,
  CONTINUE_RESEARCH: `${API_BASE_URL}/api/continue-research`,
  PING: `${API_BASE_URL}/api/ping`
};