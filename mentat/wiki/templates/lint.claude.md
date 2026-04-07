# Mentat Wiki Lint

You are doing housekeeping on the wiki workspace.

Check for:
- dead `[^sid]` citations that do not resolve through `_page_map.json`
- orphan topic pages not linked from `index.md`
- stale topics whose cited source pages are missing

Fix what you safely can, then append a `lint` event to `log.md`.
