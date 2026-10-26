# RankGuard Resume Bullets

## 中文简历版

**RankGuard｜推荐系统评测与发布决策平台**

技术栈：PyTorch、LightGCN、BPR、FastAPI、React、TypeScript、Docker、GitHub Actions

- 基于 MovieLens 100K 构建可复现的推荐系统闭环，实现 Popularity、Item-kNN、BPR-MF 与
  LightGCN，并采用逐用户时间留一法和全物品排序，避免随机划分与负样本采样造成的评测偏差。
- 统一计算 NDCG@10、Recall@10、MRR、覆盖率、长尾占比及用户 bootstrap 置信区间；实验中
  Item-kNN 的 NDCG@10 相对热门度基线提升 57.7%，但发现深历史用户切片回退，因此发布门输出
  `REVIEW`，避免被总体均值掩盖的用户退化。
- 将版本化实验结果封装为 FastAPI 服务，并实现 React/TypeScript 评审台，支持模型对比、失败
  切片和逐用户推荐检查；配套单元测试、移动端/桌面端浏览器测试、CI 与静态在线演示。

## English resume version

**RankGuard — Recommendation Evaluation & Release Workbench**

PyTorch, LightGCN, BPR, FastAPI, React, TypeScript, Docker, GitHub Actions

- Built a reproducible recommender benchmark on MovieLens 100K with Popularity, Item-kNN, BPR-MF,
  and LightGCN under chronological leave-one-out and full-catalog ranking.
- Unified ranking quality, catalog behavior, user-bootstrap intervals, and history-cohort diagnostics;
  measured a 57.7% NDCG@10 lift for Item-kNN while detecting a deep-history cohort regression that
  changed the automated release decision to `REVIEW`.
- Served versioned experiment artifacts through FastAPI and built a responsive React/TypeScript review
  console with model comparison, slice analysis, and user-level recommendation inspection; added CI,
  unit tests, browser tests, Docker, and a static live demo.

## One-line GitHub description

Evidence-first recommender benchmark with LightGCN, BPR, cohort diagnostics, release gates, FastAPI,
and a React review console.
