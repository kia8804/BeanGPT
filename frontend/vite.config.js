import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [react()],
  server: {
    port: 5173,
    proxy: {
      '/api': {
        target: 'http://localhost:8000',
        changeOrigin: true,
      },
    },
  },
  // Use root path for custom domain, subpath for GitHub Pages default domain
  base: process.env.VITE_CUSTOM_DOMAIN === 'true' ? '/' : '/BeanGPT/',
  build: {
    outDir: 'dist',
    sourcemap: false,
    assetsDir: 'assets',
  },
}) 