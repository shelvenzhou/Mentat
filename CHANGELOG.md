# Changelog

## 0.2.0

- Added configurable reranker backends for search result reranking:
  local `sentence-transformers` cross-encoder and external HTTP reranker service.
- Added reranker configuration knobs for provider selection, model, endpoint,
  candidate limits, and score weighting.
- Added independent heat-based ranking bias support after reranking.
- Added reranker provider tests plus Chinese and English reranking regression
  cases.
- Added benchmark scripts for reranker quality and latency, including full
  `mentat.search()` latency comparisons across no reranker, local reranker, and
  external reranker modes.
- Added `.env.example` entries and README documentation for local and external
  reranker setups.
