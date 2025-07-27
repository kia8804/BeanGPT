// Frontend configuration management
const config = {
  // API URL from environment variable, fallback to localhost for development
  API_URL: import.meta.env.VITE_API_URL || 'http://localhost:8000',
  
  // Environment detection
  ENVIRONMENT: import.meta.env.VITE_ENVIRONMENT || 'development',
  
  // Check if running in production
  IS_PRODUCTION: import.meta.env.VITE_ENVIRONMENT === 'production',
  
  // API endpoints
  API_ENDPOINTS: {
    CHAT: '/api/chat',
    CONTINUE_RESEARCH: '/api/continue-research',
    PING: '/api/ping'
  }
};

export default config; 