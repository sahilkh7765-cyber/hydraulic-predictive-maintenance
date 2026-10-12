/**
 * TanStack Query hooks. Artefacts never change while the API runs, so every
 * query is cached for the session (staleTime: Infinity). Requests made while
 * the API is warming up are retried quietly instead of failing.
 */
import { QueryClient, keepPreviousData, useMutation, useQuery } from "@tanstack/react-query";
import { ApiError, api } from "./api";
import type { ComponentKey } from "./types";

export const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: Infinity,
      refetchOnWindowFocus: false,
      retry: (failureCount, error) => {
        if (error instanceof ApiError && error.isWarmingUp) return failureCount < 60;
        if (error instanceof ApiError && error.kind === "network") return failureCount < 1;
        return false;
      },
      retryDelay: (_attempt, error) => (error instanceof ApiError && error.isWarmingUp ? 1500 : 1000),
    },
  },
});

export const useHealth = () =>
  useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    staleTime: 0,
    retry: false,
    refetchInterval: (query) => {
      const data = query.state.data;
      if (query.state.error) return 3000;
      return data?.status === "warming" ? 750 : 30000;
    },
  });

export const useOverview = () => useQuery({ queryKey: ["overview"], queryFn: api.overview });
export const useSensors = () => useQuery({ queryKey: ["sensors"], queryFn: api.sensors });
export const useLabels = () => useQuery({ queryKey: ["labels"], queryFn: api.labels });
export const useMetrics = () => useQuery({ queryKey: ["metrics"], queryFn: api.metrics });
export const useFeatures = () => useQuery({ queryKey: ["features"], queryFn: api.features });
export const useGeneralization = () => useQuery({ queryKey: ["generalization"], queryFn: api.generalization });
export const useFigures = () => useQuery({ queryKey: ["figures"], queryFn: api.figures });

export const useCycle = (id: number | null) =>
  useQuery({
    queryKey: ["cycle", id],
    queryFn: () => api.cycle(id as number),
    enabled: id !== null,
    placeholderData: keepPreviousData,
  });

export const useImportance = (c: ComponentKey, top = 20) =>
  useQuery({ queryKey: ["importance", c, top], queryFn: () => api.importance(c, top), placeholderData: keepPreviousData });

export const usePipeline = (c: ComponentKey) =>
  useQuery({ queryKey: ["pipeline", c], queryFn: () => api.pipeline(c), placeholderData: keepPreviousData });

export const useEvaluation = (c: ComponentKey) =>
  useQuery({ queryKey: ["evaluation", c], queryFn: () => api.evaluation(c), placeholderData: keepPreviousData });

export const useTimeline = (c: ComponentKey, start: number, end: number) =>
  useQuery({
    queryKey: ["timeline", c, start, end],
    queryFn: () => api.timeline(c, start, end),
    placeholderData: keepPreviousData,
  });

export const useErrorCases = (c: ComponentKey) =>
  useQuery({ queryKey: ["errors", c], queryFn: () => api.errorCases(c), placeholderData: keepPreviousData });

export const useErrorProfile = (c: ComponentKey, cycle: number | null) =>
  useQuery({
    queryKey: ["error-profile", c, cycle],
    queryFn: () => api.errorProfile(c, cycle as number),
    enabled: cycle !== null,
    placeholderData: keepPreviousData,
  });

export const useWhatIfOptions = (c: ComponentKey, cycle: number | undefined) =>
  useQuery({
    queryKey: ["whatif-options", c, cycle ?? "default"],
    queryFn: () => api.whatIfOptions(c, cycle),
    placeholderData: keepPreviousData,
  });

export const useWhatIf = (c: ComponentKey, body: { cycle: number; feature: string; value: number } | null) =>
  useQuery({
    queryKey: ["whatif", c, body],
    queryFn: () => api.whatIf(c, body as { cycle: number; feature: string; value: number }),
    enabled: body !== null,
    placeholderData: keepPreviousData,
  });

export const useNoise = (c: ComponentKey, enabled = true) =>
  useQuery({ queryKey: ["noise", c], queryFn: () => api.noise(c), enabled });

export const useSelective = (c: ComponentKey) =>
  useQuery({ queryKey: ["selective", c], queryFn: () => api.selective(c), placeholderData: keepPreviousData });

export const usePredictCycle = () => useMutation({ mutationFn: (id: number) => api.predictCycle(id) });
export const usePredictUpload = () => useMutation({ mutationFn: (file: File) => api.predictUpload(file) });
