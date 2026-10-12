import path from "node:path";
import tailwindcss from "@tailwindcss/vite";
import react from "@vitejs/plugin-react";
import { defineConfig } from "vite";

// The API runs separately (uvicorn api.main:app --port 8000). In development
// Vite proxies /api to it, so the browser talks to one origin.
const API_TARGET = process.env.VITE_API_TARGET ?? "http://127.0.0.1:8000";

export default defineConfig({
  plugins: [react(), tailwindcss()],
  resolve: {
    alias: { "@": path.resolve(__dirname, "src") },
  },
  server: {
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: API_TARGET, changeOrigin: false } },
  },
  preview: {
    port: 4173,
    proxy: { "/api": { target: API_TARGET } },
  },
  build: {
    // No manualChunks: forcing recharts into its own chunk also pulled shared
    // helpers (clsx) into it, which made the entry chunk import charts eagerly.
    // Rollup's own splitting keeps recharts in the lazily loaded page chunks.
    target: "es2022",
    chunkSizeWarningLimit: 600,
  },
});
