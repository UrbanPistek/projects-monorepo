import { get, writable } from "svelte/store";
import * as api from "$lib/api";
import type {
  EstimateReport,
  PreviewGeometry,
  Project,
  ProjectSummary,
  ValidationReport,
} from "$lib/api";

export const project = writable<Project | null>(null);
export const preview = writable<PreviewGeometry | null>(null);
export const summary = writable<ProjectSummary | null>(null);
export const estimate = writable<EstimateReport | null>(null);
export const validation = writable<ValidationReport | null>(null);
export const selectedIds = writable<string[]>([]);
export const errorMessage = writable<string | null>(null);
export const busy = writable(false);

function formatError(err: unknown): string {
  if (typeof err === "object" && err && "message" in err) {
    return String((err as { message: unknown }).message);
  }
  return String(err);
}

export async function refreshAll() {
  const [p, pv, sm, est, val] = await Promise.all([
    api.getProject(),
    api.getPreview(),
    api.getSummary(),
    api.estimateJob().catch(() => null),
    api.validateJob(),
  ]);
  project.set(p);
  preview.set(pv);
  summary.set(sm);
  estimate.set(est);
  validation.set(val);
}

export async function run(action: () => Promise<unknown>) {
  busy.set(true);
  errorMessage.set(null);
  try {
    await action();
    if (get(project) || true) {
      await refreshAll();
    }
  } catch (err) {
    errorMessage.set(formatError(err));
  } finally {
    busy.set(false);
  }
}
