import { lazy } from "react";
import { createBrowserRouter } from "react-router";
import { AppShell } from "@/components/layout/AppShell";
import { NotFound } from "@/pages/NotFound";
import { OverviewPage } from "@/pages/Overview";

// Overview is in the initial bundle; every chart-heavy page loads on demand.
const DatasetPage = lazy(() => import("@/pages/Dataset"));
const FeaturesPage = lazy(() => import("@/pages/Features"));
const PredictPage = lazy(() => import("@/pages/Predict"));
const ModelsPage = lazy(() => import("@/pages/Models"));
const RobustnessPage = lazy(() => import("@/pages/Robustness"));

export const router = createBrowserRouter([
  {
    path: "/",
    element: <AppShell />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "dataset", element: <DatasetPage /> },
      { path: "features", element: <FeaturesPage /> },
      { path: "predict", element: <PredictPage /> },
      { path: "models", element: <ModelsPage /> },
      { path: "robustness", element: <RobustnessPage /> },
      { path: "*", element: <NotFound /> },
    ],
  },
]);
