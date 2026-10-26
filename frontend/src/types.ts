export type Metrics = Record<string, number>;

export interface CohortMetrics {
  users: number;
  [metric: string]: number;
}

export interface ModelResult {
  id: string;
  label: string;
  metrics: Metrics;
  intervals: Record<string, [number, number]>;
  cohorts: Record<string, CohortMetrics>;
  training_seconds: number;
  loss_tail: number[];
}

export interface Experiment {
  project: string;
  generated_at: string;
  dataset: {
    name: string;
    source: string;
    source_url: string;
    users: number;
    items: number;
    train_interactions: number;
    test_interactions: number;
    positive_rating_threshold: number;
    split: string;
    candidate_protocol: string;
    history_p50: number;
  };
  config: { top_k: number; seed: number };
  models: ModelResult[];
  release_gate: {
    decision: "release" | "review";
    candidate: string;
    candidate_label: string;
    relative_ndcg_lift_vs_popularity: number;
    cohort_regressions: string[];
    reason: string;
    rule: string;
  };
  limitations: string[];
}

export interface UserSummary {
  raw_user_id: number;
  history_length: number;
  held_out_title: string;
}

export interface Recommendation {
  rank: number;
  item_id: number;
  title: string;
  genres: string[];
  training_popularity: number;
}

export interface RecommendationResponse {
  model: string;
  user: UserSummary;
  recommendations: Recommendation[];
}
