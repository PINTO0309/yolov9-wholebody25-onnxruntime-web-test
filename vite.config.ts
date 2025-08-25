import { defineConfig } from 'vite'
import { resolve } from 'path'

export default defineConfig({
  server: {
    headers: {
      'Cross-Origin-Embedder-Policy': 'require-corp',
      'Cross-Origin-Opener-Policy': 'same-origin'
    }
  },
  build: {
    rollupOptions: {
      external: ['electron']
    }
  },
  optimizeDeps: {
    exclude: ['onnxruntime-web']
  }
})