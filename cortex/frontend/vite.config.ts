/// <reference types="vitest/config" />
import { defineConfig, type Plugin } from "vite";
import react from "@vitejs/plugin-react";

declare const process: { env: Record<string, string | undefined> };

const API = process.env.CORTEX_API ?? "http://127.0.0.1:8420";

// Production builds only: the dev server's React refresh preamble is an inline script.
// Imported notes are untrusted, so the built app may only talk to its own origin and never
// loads remote images, scripts or frames.
const CSP = [
  "default-src 'self'",
  "img-src 'self' data:",
  "style-src 'self' 'unsafe-inline'",
  "script-src 'self'",
  "connect-src 'self'",
  "object-src 'none'",
  "frame-src 'none'",
  "base-uri 'none'",
  "form-action 'none'",
].join("; ");

function contentSecurityPolicy(): Plugin {
  return {
    name: "cortex-csp",
    apply: "build",
    transformIndexHtml: (html) =>
      html.replace("<head>", `<head>\n    <meta http-equiv="Content-Security-Policy" content="${CSP}" />`),
  };
}

export default defineConfig({
  plugins: [react(), contentSecurityPolicy()],
  server: {
    host: "127.0.0.1",
    port: 5173,
    strictPort: true,
    proxy: { "/api": { target: API, changeOrigin: false } },
  },
  build: { chunkSizeWarningLimit: 2000 },
  test: {
    environment: "jsdom",
    include: ["src/**/*.test.{ts,tsx}"],
  },
});
