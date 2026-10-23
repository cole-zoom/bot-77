import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

// dev: the API runs on :8077 (uv run python -m bot77.browser); vite proxies /api to it
export default defineConfig({
  plugins: [react(), tailwindcss()],
  server: { port: 5177, proxy: { "/api": "http://127.0.0.1:8077" } },
});
