"""File watcher: auto-reindex when files change in watched directories.

Each collection can have `watch_paths` and `watch_ignore`. The watcher
monitors these directories and re-indexes files when their content
changes (verified via SHA-256 hash), with built-in throttling (5s per file).

Requires the `watchfiles` package (Rust backend for efficient FS monitoring).
"""

import asyncio
import mentat


async def main():
    await mentat.start_processor()

    # --- Create a collection with watch config ---
    mentat.create_collection(
        "project_src",
        watch_paths=["/home/user/project/src"],
        watch_ignore=["node_modules", "*.lock", "__pycache__", "*.pyc"],
        metadata={"type": "project", "name": "my-project"},
    )

    mentat.create_collection(
        "project_docs",
        watch_paths=["/home/user/project/docs"],
        watch_ignore=["_build"],
        auto_add_sources=["watcher:*"],  # auto-route watcher-indexed files
    )

    # --- Start the watcher ---
    # The watcher reads watch configs from all collections and starts
    # one async watch task per collection. It's integrated into Hub.start().
    # When files change:
    #   1. Throttle check (skip if processed within 5s)
    #   2. SHA-256 hash check (skip if content unchanged)
    #   3. Re-index with force=True
    #   4. Add to owning collection

    hub = mentat.Mentat.get_instance()
    await hub.watcher.start()
    print("Watcher started. Monitoring directories...")

    # --- Sync watcher after collection changes ---
    # If you create/update/delete collections at runtime, call sync()
    mentat.create_collection(
        "notes",
        watch_paths=["/home/user/notes"],
    )
    await hub.watcher.sync()  # picks up the new "notes" collection
    print("Watcher synced with new collection config")

    # --- Let it run (in a real app, this runs alongside your main loop) ---
    try:
        print("Watching for changes... (Ctrl+C to stop)")
        await asyncio.sleep(3600)  # or until your app exits
    except KeyboardInterrupt:
        pass

    # --- Cleanup ---
    await hub.watcher.stop()
    await mentat.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
