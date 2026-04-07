# Mentat Wiki Sync

You are maintaining a filesystem-based wiki workspace.

1. Read `log.md`. Find the most recent `## [...] sync |` line. Collect all `ingest` lines after it. These are the pending docs.
2. For each pending doc, read `pages/<sid>.md`. Use the title, brief intro, and contents list to decide where it belongs.
3. Read existing `topics/*.md` and decide whether each pending doc extends an existing topic or needs a new one.
4. Update affected `topics/<slug>.md` files. Use YAML frontmatter with `name`, `slug`, `source_docs`, and `updated_at`.
5. Every factual sentence in a topic page should end with citations like `[^sid]` or `[^sid#section-slug]`.
6. Rewrite `index.md` from scratch with categories for Entities, Concepts, Sources, Memories, and Conversations.
7. Append a `sync` event to `log.md` when you finish.

Important:
- `pages/*.md`, `_memories.md`, `_conversations.md`, and `_page_map.json` are Mentat-owned inputs.
- `index.md`, `topics/*.md`, and `topics/*.verified.json` are agent-owned outputs.
- Prefer synthesis from intros and tables of contents; use full page details only when needed.
