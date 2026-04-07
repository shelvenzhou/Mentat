# Mentat Wiki Verify

You are verifying claims in synthesized topic pages.

1. Scan `topics/*.md`.
2. Any topic without `topics/<slug>.verified.json`, or whose sidecar is older than the topic markdown, needs verification.
3. For each cited claim, read the relevant `pages/<sid>.md` source and judge whether the claim is `supported`, `partial`, `contradicted`, or `not_in_source`.
4. Write `topics/<slug>.verified.json` with `{checked_at, claims:[...]}`.
5. Append a `verify` event to `log.md` when you finish.

Important:
- Preserve citations and be explicit when evidence is only partial.
- Prefer quoting or paraphrasing the exact source passage in the sidecar evidence field.
