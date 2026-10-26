# RankGuard Interview Guide

## 60-second project explanation

“I built RankGuard because a recommender that wins on one average metric may still be unsafe to ship.
It trains four models on real MovieLens interactions under a chronological leave-last-one-out split,
evaluates the full catalog with ranking, coverage, novelty, confidence intervals, and history-length
cohorts, then exposes the result through FastAPI and a React dashboard. Item-kNN won overall NDCG by
57.7% over popularity, but it regressed for deep-history users, so the release gate returned REVIEW.
The main value is the complete decision loop: model training, fair comparison, failure analysis, and
serving—not a claim that one architecture always wins.”

## Questions to be ready for

### Why chronological holdout?

Random splitting can let future preferences influence training. Holding out the last positive event
more closely matches recommending the next relevant item.

### Why full-catalog evaluation?

Sampled negatives make evaluation cheaper but can inflate and destabilize ranking metrics depending on
the sampler. MovieLens is small enough to rank every unseen item.

### Why did Item-kNN beat LightGCN?

The dataset is compact and dense enough for item co-occurrence to be strong. A more expressive model
does not guarantee better generalization under a fixed training budget. This is why RankGuard compares
models rather than treating architecture novelty as evidence.

### What would you add before production?

Exposure-aware logs, temporal backtests, feature pipelines, approximate retrieval, online A/B testing,
latency/load tests, drift monitoring, and business guardrails. A production extension should use a
separate public dataset with logged exposure and richer feedback, without reusing private research assets.

### What is the most important failure in the verified run?

Item-kNN wins overall but loses to popularity for users with more than 50 historical positives. A
global release would improve the average while degrading the most active cohort.
