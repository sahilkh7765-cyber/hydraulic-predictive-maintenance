/**
 * The only place the app talks to the network. Every response is validated
 * against its zod schema; anything unexpected becomes a typed ApiError.
 */
import type { z } from "zod/mini";
import * as S from "./schemas";
import type { ComponentKey } from "./types";

export type ApiErrorKind = "http" | "network" | "schema";

export class ApiError extends Error {
  readonly kind: ApiErrorKind;
  readonly status: number | null;
  readonly code: string;
  readonly detail: string;

  constructor(kind: ApiErrorKind, status: number | null, code: string, detail: string) {
    super(detail);
    this.name = "ApiError";
    this.kind = kind;
    this.status = status;
    this.code = code;
    this.detail = detail;
  }

  get isWarmingUp(): boolean {
    return this.code === "warming_up";
  }
}

const BASE = import.meta.env.VITE_API_URL ?? "";

async function request<T extends z.ZodMiniType>(path: string, schema: T, init?: RequestInit): Promise<z.infer<T>> {
  let response: Response;
  try {
    response = await fetch(`${BASE}${path}`, { ...init, headers: { Accept: "application/json", ...init?.headers } });
  } catch {
    throw new ApiError(
      "network",
      null,
      "api_unreachable",
      "Cannot reach the API. Start it with: uvicorn api.main:app --port 8000",
    );
  }

  let body: unknown = null;
  const text = await response.text();
  if (text) {
    try {
      body = JSON.parse(text) as unknown;
    } catch {
      body = null;
    }
  }

  if (!response.ok) {
    const parsed = S.ApiErrorBody.safeParse(body);
    if (parsed.success) throw new ApiError("http", response.status, parsed.data.error, parsed.data.detail);
    if (response.status === 502 || response.status === 504) {
      throw new ApiError("network", response.status, "api_unreachable", "The API is not running on port 8000.");
    }
    throw new ApiError("http", response.status, "http_error", `Request failed with status ${response.status}.`);
  }

  const parsed = schema.safeParse(body);
  if (!parsed.success) {
    const first = parsed.error.issues[0];
    const where = first ? `${first.path.join(".")}: ${first.message}` : "unknown field";
    throw new ApiError("schema", response.status, "unexpected_response", `The API response did not match the expected shape (${where}).`);
  }
  return parsed.data;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  health: () => request("/api/health", S.Health),
  overview: () => request("/api/overview", S.Overview),
  sensors: () => request("/api/sensors", S.SensorsResponse),
  labels: () => request("/api/labels", S.LabelsResponse),
  cycle: (id: number) => request(`/api/cycles/${id}`, S.CycleResponse),
  cycleCsvUrl: (id: number) => `${BASE}/api/cycles/${id}/csv`,
  metrics: () => request("/api/metrics", S.MetricsResponse),
  features: () => request("/api/features", S.FeaturesResponse),
  importance: (c: ComponentKey, top = 20) =>
    request(`/api/features/importance?component=${c}&top=${top}`, S.ImportanceResponse),
  predictCycle: (id: number) => request(`/api/predict/cycle/${id}`, S.PredictionResponse, { method: "POST" }),
  predictUpload: (file: File) => {
    const form = new FormData();
    form.append("file", file);
    return request("/api/predict/upload", S.PredictionResponse, { method: "POST", body: form });
  },
  pipeline: (c: ComponentKey) => request(`/api/models/${c}/pipeline`, S.PipelineResponse),
  evaluation: (c: ComponentKey) => request(`/api/evaluation/${c}`, S.EvaluationResponse),
  generalization: () => request("/api/generalization", S.GeneralizationResponse),
  timeline: (c: ComponentKey, start: number, end: number) =>
    request(`/api/generalization/${c}/timeline?start=${start}&end=${end}`, S.TimelineResponse),
  errorCases: (c: ComponentKey) => request(`/api/errors/${c}`, S.ErrorCasesResponse),
  errorProfile: (c: ComponentKey, cycle: number) => request(`/api/errors/${c}/${cycle}`, S.ErrorProfileResponse),
  whatIfOptions: (c: ComponentKey, cycle?: number) =>
    request(`/api/whatif/${c}${cycle === undefined ? "" : `?cycle=${cycle}`}`, S.WhatIfOptions),
  whatIf: (c: ComponentKey, body: { cycle: number; feature: string; value: number }) =>
    request(`/api/whatif/${c}`, S.WhatIfResponse, json(body)),
  noise: (c: ComponentKey) => request(`/api/robustness/${c}/noise`, S.NoiseResponse),
  selective: (c: ComponentKey) => request(`/api/robustness/${c}/selective`, S.SelectiveResponse),
  figures: () => request("/api/figures", S.FiguresResponse),
};

export function errorMessage(error: unknown): string {
  if (error instanceof ApiError) return error.detail;
  if (error instanceof Error) return error.message;
  return "Something went wrong.";
}
