# Data Card: MovieLens 100K in RankGuard

## Source and use

RankGuard downloads MovieLens 100K from GroupLens at runtime. The dataset is not committed to this
repository. HTTPS verification stays enabled: the downloader uses the operating system trust store
when available and falls back to the bundled `certifi` roots. The verified artifact uses ratings of
at least 4 as implicit positive interactions.

## Transformation

1. Retain users with at least five positive events.
2. Sort every user's events by timestamp, then item ID for deterministic tie-breaking.
3. Hold out the final positive event as the test target.
4. Use all preceding positive events as training history.
5. Evaluate against the full mapped catalog after masking training items.

## Verified scale

| Field | Value |
|---|---:|
| Eligible users | 938 |
| Mapped items | 1,447 |
| Training interactions | 54,423 |
| Test interactions | 938 |
| Median training history | 39 |

## Known limits

- Ratings are explicit historical preferences, not modern feed impressions or watch-time events.
- The data does not contain exposure logs; missing interactions cannot be interpreted as dislikes.
- MovieLens is useful for reproducibility and debugging, not production-scale throughput claims.
- The user-cohort analysis uses history length only and is not a demographic fairness analysis.
