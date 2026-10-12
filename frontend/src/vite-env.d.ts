/// <reference types="vite/client" />

interface ImportMetaEnv {
  /** Optional absolute API origin; empty means same origin (Vite proxies /api in development). */
  readonly VITE_API_URL?: string;
}
