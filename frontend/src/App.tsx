import {
  ArrowUpRight,
  Braces,
  Check,
  ChevronRight,
  CircleDot,
  Database,
  Gauge,
  GitCompareArrows,
  Search,
  ShieldCheck,
  TriangleAlert,
} from "lucide-react";
import { useEffect, useMemo, useState } from "react";
import { getExperiment, getRecommendations, getUsers } from "./data";
import type { Experiment, ModelResult, RecommendationResponse, UserSummary } from "./types";

const pct = (value: number) => `${(value * 100).toFixed(1)}%`;
const fixed = (value: number, digits = 3) => value.toFixed(digits);
const cohortLabel: Record<string, string> = {
  short_history: "Short history",
  medium_history: "Medium history",
  deep_history: "Deep history",
};

function MetricBar({ label, value, max, accent }: { label: string; value: number; max: number; accent: boolean }) {
  const width = Math.max(3, (value / Math.max(max, 0.0001)) * 100);
  return (
    <div className="metric-bar-row">
      <span>{label}</span>
      <div className="metric-track">
        <div className={accent ? "metric-fill accent" : "metric-fill"} style={{ width: `${width}%` }} />
      </div>
      <strong>{fixed(value)}</strong>
    </div>
  );
}

function ModelCard({ model, winner, topK }: { model: ModelResult; winner: boolean; topK: number }) {
  const ndcg = model.metrics[`ndcg@${topK}`];
  const interval = model.intervals[`ndcg@${topK}`];
  return (
    <article className={winner ? "model-card winner" : "model-card"}>
      <div className="model-card-head">
        <div>
          <span className="eyebrow">{model.id}</span>
          <h3>{model.label}</h3>
        </div>
        {winner && <span className="winner-chip"><Check size={13} /> metric lead</span>}
      </div>
      <div className="model-score">{fixed(ndcg)}</div>
      <div className="model-score-label">NDCG@{topK} · 95% CI {fixed(interval[0])}–{fixed(interval[1])}</div>
      <div className="model-mini-grid">
        <div><span>Recall</span><strong>{fixed(model.metrics[`recall@${topK}`])}</strong></div>
        <div><span>Coverage</span><strong>{pct(model.metrics[`catalog_coverage@${topK}`])}</strong></div>
        <div><span>Novelty</span><strong>{fixed(model.metrics[`novelty@${topK}`], 2)}</strong></div>
        <div><span>Train</span><strong>{fixed(model.training_seconds, 1)}s</strong></div>
      </div>
    </article>
  );
}

function App() {
  const [experiment, setExperiment] = useState<Experiment | null>(null);
  const [users, setUsers] = useState<UserSummary[]>([]);
  const [selectedUser, setSelectedUser] = useState<number | null>(null);
  const [selectedModel, setSelectedModel] = useState("lightgcn");
  const [recommendations, setRecommendations] = useState<RecommendationResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    Promise.all([getExperiment(), getUsers(40)])
      .then(([experimentData, userData]: [Experiment, UserSummary[]]) => {
        setExperiment(experimentData);
        setUsers(userData);
        setSelectedModel(experimentData.release_gate.candidate);
        setSelectedUser(userData[0]?.raw_user_id ?? null);
      })
      .catch(() => setError("Experiment artifacts are not available. Run the training pipeline first."));
  }, []);

  useEffect(() => {
    if (selectedUser === null) return;
    getRecommendations(selectedUser, selectedModel)
      .then(setRecommendations)
      .catch(() => setRecommendations(null));
  }, [selectedUser, selectedModel]);

  const metricKey = experiment ? `ndcg@${experiment.config.top_k}` : "ndcg@10";
  const maxNdcg = useMemo(
    () => Math.max(...(experiment?.models.map((model) => model.metrics[metricKey]) ?? [1])),
    [experiment, metricKey]
  );

  if (error) {
    return <main className="loading-screen"><TriangleAlert /><h1>RankGuard is waiting for evidence.</h1><p>{error}</p></main>;
  }
  if (!experiment) {
    return <main className="loading-screen"><CircleDot className="pulse" /><h1>Loading the evidence room…</h1></main>;
  }

  const gate = experiment.release_gate;
  const topK = experiment.config.top_k;

  return (
    <div className="app-shell">
      <header className="topbar">
        <a className="brand" href="#top" aria-label="RankGuard home">
          <span className="brand-mark"><ShieldCheck size={19} /></span>
          <span>RANKGUARD</span>
        </a>
        <nav>
          <a href="#compare">Compare</a>
          <a href="#slices">Slices</a>
          <a href="#inspect">Inspect</a>
        </nav>
        <div className="live-state"><span /> artifact live</div>
      </header>

      <main id="top">
        <section className="hero">
          <div className="hero-copy">
            <span className="kicker">Recommendation release evidence · Run {experiment.config.seed}</span>
            <h1>Choose what to ship.<br /><em>Know where it breaks.</em></h1>
            <p>One evaluation contract for four recommenders—ranking quality, uncertainty, catalog behavior, and user-cohort failures in the same frame.</p>
            <div className="hero-actions">
              <a className="button primary" href="#compare">Open model review <ChevronRight size={17} /></a>
              <a
                className="button ghost"
                href={location.hostname.endsWith("github.io")
                  ? "https://github.com/JiayiLu2003/RankGuard/blob/main/src/rankguard/api.py"
                  : "/docs"}
                target="_blank"
                rel="noreferrer"
              >
                API contract <Braces size={16} />
              </a>
            </div>
          </div>
          <aside className={`decision-card ${gate.decision}`}>
            <div className="decision-topline">
              <span>RELEASE GATE / 01</span>
              <Gauge size={18} />
            </div>
            <div className="decision-word">{gate.decision.toUpperCase()}</div>
            <div className="candidate-line"><span>candidate</span><strong>{gate.candidate_label}</strong></div>
            <div className="lift-line">
              <ArrowUpRight size={22} />
              <strong>{pct(gate.relative_ndcg_lift_vs_popularity)}</strong>
              <span>NDCG lift vs popularity</span>
            </div>
            <p>{gate.reason}</p>
          </aside>
        </section>

        <section className="data-ribbon" aria-label="Dataset facts">
          <div><Database size={16} /><span>DATASET</span><strong>{experiment.dataset.name}</strong></div>
          <div><span>USERS</span><strong>{experiment.dataset.users.toLocaleString()}</strong></div>
          <div><span>ITEMS</span><strong>{experiment.dataset.items.toLocaleString()}</strong></div>
          <div><span>TRAIN EVENTS</span><strong>{experiment.dataset.train_interactions.toLocaleString()}</strong></div>
          <div><span>SPLIT</span><strong>leave-last-one-out</strong></div>
          <div><span>CANDIDATES</span><strong>full catalog</strong></div>
        </section>

        <section className="section" id="compare">
          <div className="section-heading">
            <div><span className="section-index">01</span><h2>Model review</h2></div>
            <p>Same users. Same holdout. Same candidate set. Different ranking behavior.</p>
          </div>
          <div className="model-grid">
            {experiment.models.map((model) => (
              <ModelCard key={model.id} model={model} winner={model.id === gate.candidate} topK={topK} />
            ))}
          </div>
          <div className="comparison-panel">
            <div className="panel-title"><GitCompareArrows size={18} /><span>NDCG comparison</span><small>higher is better</small></div>
            <div className="metric-bars">
              {experiment.models.map((model) => (
                <MetricBar key={model.id} label={model.label} value={model.metrics[metricKey]} max={maxNdcg} accent={model.id === gate.candidate} />
              ))}
            </div>
          </div>
        </section>

        <section className="section slice-section" id="slices">
          <div className="section-heading inverse">
            <div><span className="section-index">02</span><h2>Failure slices</h2></div>
            <p>Overall averages can hide the users a release leaves behind.</p>
          </div>
          <div className="slice-grid">
            {Object.keys(cohortLabel).map((cohort) => {
              const rows = experiment.models.map((model) => ({
                model: model.label,
                id: model.id,
                users: model.cohorts[cohort].users,
                ndcg: model.cohorts[cohort][metricKey],
              }));
              const best = Math.max(...rows.map((row) => row.ndcg));
              return (
                <article className="slice-card" key={cohort}>
                  <div className="slice-head">
                    <div><span>{cohortLabel[cohort]}</span><small>{rows[0].users} users</small></div>
                    <strong>{cohort === "short_history" ? "≤20" : cohort === "medium_history" ? "21–50" : ">50"}</strong>
                  </div>
                  {rows.map((row) => (
                    <div className="slice-row" key={row.id}>
                      <span>{row.model}</span>
                      <div><i style={{ width: `${Math.max(4, row.ndcg / Math.max(best, 0.001) * 100)}%` }} /></div>
                      <strong>{fixed(row.ndcg)}</strong>
                    </div>
                  ))}
                </article>
              );
            })}
          </div>
          <div className="gate-rule"><ShieldCheck size={19} /><span>{gate.rule}</span></div>
        </section>

        <section className="section inspect-section" id="inspect">
          <div className="section-heading">
            <div><span className="section-index">03</span><h2>Recommendation inspector</h2></div>
            <p>Move from aggregate metrics to the ranked items a user actually receives.</p>
          </div>
          <div className="inspect-layout">
            <aside className="inspect-controls">
              <label>
                <span><Search size={14} /> user</span>
                <select value={selectedUser ?? ""} onChange={(event) => setSelectedUser(Number(event.target.value))}>
                  {users.map((user) => <option key={user.raw_user_id} value={user.raw_user_id}>User {user.raw_user_id} · {user.history_length} events</option>)}
                </select>
              </label>
              <label>
                <span><CircleDot size={14} /> model</span>
                <select value={selectedModel} onChange={(event) => setSelectedModel(event.target.value)}>
                  {experiment.models.map((model) => <option key={model.id} value={model.id}>{model.label}</option>)}
                </select>
              </label>
              {recommendations && (
                <div className="user-context">
                  <span>Held-out truth</span>
                  <strong>{recommendations.user.held_out_title}</strong>
                  <small>The final positive event is never shown during training.</small>
                </div>
              )}
            </aside>
            <div className="recommendation-list">
              {recommendations?.recommendations.map((item) => (
                <article key={item.item_id}>
                  <span className="rank">{String(item.rank).padStart(2, "0")}</span>
                  <div className="movie-copy"><strong>{item.title}</strong><span>{item.genres.join(" / ") || "Unclassified"}</span></div>
                  <div className="popularity"><span>train events</span><strong>{item.training_popularity}</strong></div>
                </article>
              ))}
            </div>
          </div>
        </section>

        <section className="evidence-ledger">
          <div className="ledger-title"><span>04</span><h2>Evidence ledger</h2></div>
          <div className="ledger-grid">
            <div><span>Split policy</span><strong>{experiment.dataset.split}</strong></div>
            <div><span>Evaluation universe</span><strong>{experiment.dataset.candidate_protocol}</strong></div>
            <div><span>Positive signal</span><strong>rating ≥ {experiment.dataset.positive_rating_threshold}</strong></div>
            <div><span>Reproducibility seed</span><strong>{experiment.config.seed}</strong></div>
          </div>
          <div className="limitations">
            <span>WHAT THIS RUN DOES NOT PROVE</span>
            {experiment.limitations.map((limitation) => <p key={limitation}><TriangleAlert size={14} />{limitation}</p>)}
          </div>
        </section>
      </main>

      <footer>
        <div><span>RANKGUARD</span><small>Train → compare → diagnose → serve.</small></div>
        <a href={experiment.dataset.source_url} target="_blank" rel="noreferrer">Dataset source <ArrowUpRight size={14} /></a>
      </footer>
    </div>
  );
}

export default App;
