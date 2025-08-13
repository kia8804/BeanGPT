// API Configuration
// For production (GitHub Pages), use AWS Lightsail backend
// For development, use local backend
const rawApiUrl = import.meta.env.VITE_API_BASE_URL || 
  (import.meta.env.MODE === 'production' 
    ? 'http://3.237.69.208:8000'  // Your actual Lightsail IP
    : 'http://localhost:8000');

export const API_BASE_URL = rawApiUrl.endsWith('/') ? rawApiUrl.slice(0, -1) : rawApiUrl;

export const API_ENDPOINTS = {
  CHAT: `${API_BASE_URL}/api/chat`,
  CONTINUE_RESEARCH: `${API_BASE_URL}/api/continue-research`,
  PING: `${API_BASE_URL}/api/ping`
};