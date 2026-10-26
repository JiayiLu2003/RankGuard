import type { Experiment, RecommendationResponse, UserSummary } from "./types";

interface StaticRecommendations {
  users: Record<string, UserSummary>;
  models: Record<string, Record<string, RecommendationResponse["recommendations"]>>;
}

const staticRoot = `${import.meta.env.BASE_URL}data/`;
const staticMode = import.meta.env.BASE_URL !== "/" || location.hostname.endsWith("github.io");
let staticRecommendations: StaticRecommendations | null = null;

async function json<T>(url: string): Promise<T> {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json() as Promise<T>;
}

async function recommendationsArtifact(): Promise<StaticRecommendations> {
  if (!staticRecommendations) {
    staticRecommendations = await json<StaticRecommendations>(
      `${staticRoot}recommendations.json`
    );
  }
  return staticRecommendations;
}

export async function getExperiment(): Promise<Experiment> {
  if (staticMode) return json<Experiment>(`${staticRoot}experiment.json`);
  try {
    return await json<Experiment>("/api/experiment");
  } catch {
    return json<Experiment>(`${staticRoot}experiment.json`);
  }
}

export async function getUsers(limit = 40): Promise<UserSummary[]> {
  if (staticMode) {
    const payload = await recommendationsArtifact();
    return Object.values(payload.users).slice(0, limit);
  }
  try {
    return await json<UserSummary[]>(`/api/users?limit=${limit}`);
  } catch {
    const payload = await recommendationsArtifact();
    return Object.values(payload.users).slice(0, limit);
  }
}

export async function getRecommendations(
  userId: number,
  model: string
): Promise<RecommendationResponse> {
  if (staticMode) {
    const payload = await recommendationsArtifact();
    const key = String(userId);
    if (!payload.users[key] || !payload.models[model]?.[key]) {
      throw new Error("Recommendation is absent from the static artifact");
    }
    return {
      model,
      user: payload.users[key],
      recommendations: payload.models[model][key],
    };
  }
  try {
    return await json<RecommendationResponse>(
      `/api/users/${userId}/recommendations?model=${model}`
    );
  } catch {
    const payload = await recommendationsArtifact();
    const key = String(userId);
    if (!payload.users[key] || !payload.models[model]?.[key]) {
      throw new Error("Recommendation is absent from the static artifact");
    }
    return {
      model,
      user: payload.users[key],
      recommendations: payload.models[model][key],
    };
  }
}
