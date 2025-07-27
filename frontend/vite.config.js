import { defineConfig } from 'vite'
import react from '@vitejs/plugin-react'

// https://vitejs.dev/config/
export default defineConfig(({ mode }) => ({
  plugins: [react()],
  // Set base path for GitHub Pages deployment
  base: mode === 'production' ? '/BeanGPT/' : '/',
  build: {
    outDir: 'dist',
    sourcemap: mode !== 'production',
  },
  server: {
    port: 5173,
    host: true,
  },
  // Handle environment files
  envDir: '.',
})) 