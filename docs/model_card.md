# Model Card: RankGuard Benchmark

## Intended use

RankGuard compares recommendation approaches under one offline protocol and demonstrates how an
engineering team could block a model release when aggregate gains hide a cohort regression.

## Compared models

- **Popularity:** global log interaction counts.
- **Item-kNN:** cosine item similarity over the binary user-item matrix, trimmed to 80 neighbors.
- **BPR-MF:** 48-dimensional user/item embeddings trained with Bayesian Personalized Ranking.
- **LightGCN:** two rounds of normalized bipartite message passing with a BPR objective.

## Evaluation

For each user, one held-out item is relevant. Recall@10 is therefore also the hit rate. NDCG@10
discounts hits by rank. Coverage, novelty, and long-tail share describe catalog behavior. User-level
ranking metrics receive percentile bootstrap confidence intervals with 500 resamples.

## Release policy

The highest-NDCG model is released only when it improves NDCG@10 by at least 5% over popularity and
does not regress in short-, medium-, or deep-history cohorts. The verified run selects Item-kNN as the
metric leader but returns `REVIEW` because the deep-history cohort regresses.

## Non-goals

- Predicting online CTR, retention, revenue, or causal business lift.
- Claiming production throughput from MovieLens scale.
- Treating bootstrap intervals as protection against temporal or population shift.
- Presenting the cohort gate as a full safety or fairness evaluation.
