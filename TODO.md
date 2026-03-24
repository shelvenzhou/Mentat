# TODO: OpenClaw Integration Remaining Features

This document tracks features needed to complete Mentat's integration as OpenClaw's memory system, per the Design 3.0 spec (`design-docs/design3.0.md`).

## Completed (P0)

- [x] **#20** `search(toc_only=True)` — ToC-only search mode returning document summaries + matched sections without chunk content
- [x] **#21** `read_segment(doc_id, section_path)` — Targeted section reading by doc_id + section name
- [x] **#23** Heat map persistence — `heat_map.json` saved with debounced writes, loaded on init, flushed on shutdown
- [x] **#31** Skill integration layer — `mentat/skill.py` with OpenAI tool schemas, system prompt, `GET /skill` endpoint, `mentat skill` CLI
- [x] **#32** Path-based dedup — `PathIndex` replaces old documents when same file path is re-indexed with changed content (stubs + chunks cleaned up). `_JsonMap` base class shared with `ContentHashCache`. `add_content()` uses synthetic `__content__:{filename}` keys for URL/content dedup.

## P1 — Should Have

### #22 `get_long_term_summary()` — LPM Retrieval
**Goal**: Dedicated API to retrieve summaries of hot (frequently accessed) documents.
**Context**: Currently summaries are stored in LanceDB chunks table `summary` field. No way to query "give me all hot document summaries" without knowing doc_ids.
**Approach**:
- Add `Mentat.get_hot_summaries(top_k=10)` that queries access_tracker hot queue → fetches stubs + chunk summaries
- Expose via `GET /summaries/hot` endpoint and `mentat summaries` CLI
- Consider a separate LPM storage (e.g., `summaries/` directory with markdown files) per Design 3.0

### #25 ToC "Optimized" Marking
**Goal**: After a document is auto-summarized via hot promotion, mark its ToC entries as `optimized: true` so agents preferentially read summaries.
**Context**: `_on_access_promote()` already triggers `summarize_doc()`. Missing: updating the stub/ToC metadata to reflect this.
**Approach**:
- Add `optimized: bool = False` field to stub schema or probe_json metadata
- After `summarize_doc()` succeeds, update the stub's probe_json to mark `optimized=True`
- In `search(toc_only=True)` and `read_segment()`, surface this flag
- Agents can then prefer `get_summary` over `read_segment` for optimized sections

### #17 Weighted Heat Formula
**Goal**: Replace simple 2-access promotion with configurable `Heat = Frequency × w1 + Recency × w2`.
**Context**: Current `AccessTracker` promotes on second access regardless of time gap. A document accessed twice in 6 months shouldn't be equally hot as one accessed twice in 5 minutes.
**Approach**:
- Track per-key access count and timestamps in `_recent` (replace `float` timestamp with a dataclass: `{count, first_access, last_access}`)
- Compute heat score on each `track()` call
- Promote when heat exceeds `threshold_high` (configurable via `MENTAT_HEAT_THRESHOLD_HIGH`)
- Persist the enriched state in `heat_map.json`

### #27 Hierarchical toc_path
**Goal**: Build `parent/child` paths from flat TocEntry list, enabling path-based navigation like `"Chapter 1/Installation/Prerequisites"`.
**Context**: Current section matching in `read_segment()` is flat string matching. Design 3.0 envisions `toc_path` as `"projects/凤凰项目/架构草案"`.
**Approach**:
- Add utility `build_toc_paths(toc_entries) → Dict[str, TocEntry]` that walks entries maintaining a level stack
- Store `toc_path` in chunk metadata during `add()`
- Support path-based lookups in `read_segment()`

## P2 — Can Do Later

### #24 Cold Data Eviction
**Goal**: Remove low-heat chunks from LanceDB to reduce search pressure, keep only minimal summaries.
**Context**: No eviction mechanism exists. Vector DB grows unboundedly.
**Approach**:
- Add `threshold_low` to heat config
- Periodic sweep (e.g., hourly in BackgroundProcessor) checks heat scores
- For cold docs: archive chunk content to summary, delete vectors from LanceDB
- Keep stubs with `evicted: true` flag

### #26 Physical Line Number Mapping
**Goal**: Map chunks to source file line ranges for Agent-directed reads.
**Context**: Probes know chunk positions but don't store line numbers.
**Approach**: Add `line_start`, `line_end` to Chunk metadata during probe, persist in chunk records.

### #28 Storage Directory Restructuring
**Goal**: Restructure storage to match Design 3.0: `memory/raw/`, `memory/summaries/`, `memory/.mentat/`.
**Context**: Current layout is `mentat_files/{doc_id}.ext` + `mentat_db/`. This is internal and doesn't affect API. Code module structure was refactored (hub.py decomposed into indexer/searcher/reader, BaseVectorStorage ABC, metadata filtering, service layer) but storage directory layout unchanged.
**Approach**: Reconfigure `MentatConfig` paths, add migration utility.

### #19 Timestamp Metadata Extraction
**Goal**: Extract timestamps from filenames and content (e.g., `2024-02.md` → date context).
**Context**: Design 3.0 chunk schema is `{content, toc_path, timestamp}`. Currently no temporal metadata.
**Approach**: Add regex-based date extraction in probes, store in chunk.metadata.

### #33 Chunk-Level Diff on Re-Index
**Goal**: When a file is re-indexed with changed content, only re-embed changed chunks instead of full re-processing.
**Context**: Current path-based dedup (#32) deletes the entire old document and re-indexes from scratch. For large files where only a few lines changed, this wastes embedding API calls on unchanged chunks.
**Approach**:
- After probing the new version, compare new chunk hashes against old chunk hashes (stored in LanceDB)
- Keep unchanged chunks (same hash → reuse existing vectors)
- Only embed new/modified chunks
- Delete stale chunks that no longer exist in the new version
- Estimated savings: for a 100-chunk file with 5 changed lines, ~95% fewer embedding calls

### #18 Fine-Grained Processing Status
**Goal**: Distinguish "embedding done but summary pending" vs "both in progress" in status API.
**Context**: Current status is coarse (pending/processing/completed). Agent can't tell if it can already search but summaries aren't ready yet.
**Approach**: Split `ProcessingTask.status` into `embedding_status` + `summary_status`.

## OpenClaw Source Modifications Required

When integrating Mentat into OpenClaw, the following changes are needed on the OpenClaw side:

1. **Replace memory read/write** — Swap raw file reads with Mentat API calls (`search_memory` → `read_segment`)
2. **Inject Mentat skill** — Add tool definitions from `mentat.export_skill()` into agent's tool list + system prompt
3. **STM overflow** — When LLM context window approaches token limit, call `mentat.add_content()` to persist overflow to MTM
4. **Startup/shutdown** — Call `mentat.start_processor()` / `mentat.shutdown()` in app lifecycle
