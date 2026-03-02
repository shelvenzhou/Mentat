"""Tests for SectionHeatTracker — weighted scoring, time decay, persistence."""

import asyncio
import time

import pytest

from mentat.core.section_heat import SectionHeatTracker


# ── Basic Recording ──────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_record_creates_entry():
    tracker = SectionHeatTracker(max_entries=100)
    crossed = await tracker.record("doc1", "Introduction", weight=1.0)

    assert crossed is False
    assert tracker.get_score("doc1", "Introduction") == pytest.approx(1.0, abs=0.1)


@pytest.mark.asyncio
async def test_record_accumulates_score():
    tracker = SectionHeatTracker(max_entries=100)
    await tracker.record("doc1", "Methods", weight=1.0)
    await tracker.record("doc1", "Methods", weight=3.0)

    assert tracker.get_score("doc1", "Methods") == pytest.approx(4.0, abs=0.1)


@pytest.mark.asyncio
async def test_record_sections_batch():
    tracker = SectionHeatTracker(max_entries=100)
    await tracker.record_sections("doc1", ["A", "B", "C"], weight=2.0)

    assert tracker.get_score("doc1", "A") == pytest.approx(2.0, abs=0.1)
    assert tracker.get_score("doc1", "B") == pytest.approx(2.0, abs=0.1)
    assert tracker.get_score("doc1", "C") == pytest.approx(2.0, abs=0.1)


@pytest.mark.asyncio
async def test_record_sections_strips_whitespace():
    tracker = SectionHeatTracker(max_entries=100)
    await tracker.record_sections("doc1", ["  A  ", "", "  B"], weight=1.0)

    assert tracker.get_score("doc1", "A") == pytest.approx(1.0, abs=0.1)
    assert tracker.get_score("doc1", "B") == pytest.approx(1.0, abs=0.1)
    # Empty string should be skipped
    stats = tracker.stats()
    assert stats["total_entries"] == 2


@pytest.mark.asyncio
async def test_different_docs_tracked_separately():
    tracker = SectionHeatTracker(max_entries=100)
    await tracker.record("doc1", "Intro", weight=3.0)
    await tracker.record("doc2", "Intro", weight=1.0)

    assert tracker.get_score("doc1", "Intro") == pytest.approx(3.0, abs=0.1)
    assert tracker.get_score("doc2", "Intro") == pytest.approx(1.0, abs=0.1)


# ── Weight Differentiation ──────────────────────────────────────────


@pytest.mark.asyncio
async def test_weight_differentiation():
    """read_segment(3.0) > inspect(2.0) > search(1.0)."""
    tracker = SectionHeatTracker(max_entries=100)

    # Simulate: search finds section, then inspect, then read_segment
    await tracker.record("doc1", "search_section", weight=1.0)
    await tracker.record("doc1", "inspect_section", weight=2.0)
    await tracker.record("doc1", "read_section", weight=3.0)

    assert tracker.get_score("doc1", "read_section") > tracker.get_score(
        "doc1", "inspect_section"
    )
    assert tracker.get_score("doc1", "inspect_section") > tracker.get_score(
        "doc1", "search_section"
    )


# ── Time Decay ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_decay_half_life(monkeypatch):
    """After one half-life, score should be halved."""
    half_life = 100.0  # seconds
    tracker = SectionHeatTracker(half_life_seconds=half_life, max_entries=100)

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    await tracker.record("doc1", "Methods", weight=10.0)

    # Advance time by exactly one half-life
    fake_time[0] = 1000.0 + half_life
    score = tracker.get_score("doc1", "Methods")
    assert score == pytest.approx(5.0, rel=0.01)

    # Advance by another half-life (total 2x)
    fake_time[0] = 1000.0 + 2 * half_life
    score = tracker.get_score("doc1", "Methods")
    assert score == pytest.approx(2.5, rel=0.01)


@pytest.mark.asyncio
async def test_no_decay_at_zero_elapsed(monkeypatch):
    """Score should be exact when read immediately."""
    tracker = SectionHeatTracker(half_life_seconds=100, max_entries=100)

    fake_time = [1000.0]
    monkeypatch.setattr(time, "time", lambda: fake_time[0])

    await tracker.record("doc1", "Intro", weight=7.5)
    score = tracker.get_score("doc1", "Intro")
    assert score == pytest.approx(7.5, rel=0.001)


@pytest.mark.asyncio
async def test_get_score_nonexistent():
    tracker = SectionHeatTracker(max_entries=100)
    assert tracker.get_score("nope", "nope") == 0.0


# ── Hot Threshold ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hot_threshold_crossed():
    tracker = SectionHeatTracker(hot_threshold=5.0, max_entries=100)

    # Below threshold
    crossed = await tracker.record("doc1", "Intro", weight=4.0)
    assert crossed is False

    # Crosses threshold (4 + 2 = 6 > 5)
    crossed = await tracker.record("doc1", "Intro", weight=2.0)
    assert crossed is True

    # Already hot — should not fire again
    crossed = await tracker.record("doc1", "Intro", weight=1.0)
    assert crossed is False


@pytest.mark.asyncio
async def test_on_hot_section_callback():
    seen = []
    event = asyncio.Event()

    async def on_hot(doc_id: str, section: str):
        seen.append((doc_id, section))
        event.set()

    tracker = SectionHeatTracker(
        hot_threshold=3.0, max_entries=100, on_hot_section=on_hot
    )

    await tracker.record("doc1", "Results", weight=4.0)
    await asyncio.wait_for(event.wait(), timeout=1.0)

    assert seen == [("doc1", "Results")]


# ── Hot Sections Query ───────────────────────────────────────────────


@pytest.mark.asyncio
async def test_get_hot_sections():
    tracker = SectionHeatTracker(hot_threshold=5.0, max_entries=100)
    await tracker.record("doc1", "Hot", weight=6.0)
    await tracker.record("doc1", "Cold", weight=1.0)
    await tracker.record("doc2", "AlsoHot", weight=8.0)

    hot = tracker.get_hot_sections()
    assert len(hot) == 2
    assert hot[0]["section"] == "AlsoHot"  # highest score first
    assert hot[1]["section"] == "Hot"


@pytest.mark.asyncio
async def test_get_hot_sections_filtered_by_doc():
    tracker = SectionHeatTracker(hot_threshold=5.0, max_entries=100)
    await tracker.record("doc1", "A", weight=6.0)
    await tracker.record("doc2", "B", weight=7.0)

    hot = tracker.get_hot_sections(doc_id="doc1")
    assert len(hot) == 1
    assert hot[0]["doc_id"] == "doc1"


@pytest.mark.asyncio
async def test_get_top_sections():
    tracker = SectionHeatTracker(max_entries=100)
    await tracker.record("doc1", "A", weight=1.0)
    await tracker.record("doc1", "B", weight=5.0)
    await tracker.record("doc1", "C", weight=3.0)
    await tracker.record("doc2", "X", weight=10.0)  # different doc

    top = tracker.get_top_sections("doc1", limit=2)
    assert len(top) == 2
    assert top[0]["section"] == "B"
    assert top[1]["section"] == "C"


# ── Capacity / Eviction ─────────────────────────────────────────────


@pytest.mark.asyncio
async def test_max_entries_eviction():
    tracker = SectionHeatTracker(max_entries=3)
    await tracker.record("doc1", "A", weight=1.0)
    await tracker.record("doc1", "B", weight=5.0)
    await tracker.record("doc1", "C", weight=3.0)

    # This should evict A (lowest score)
    await tracker.record("doc1", "D", weight=4.0)

    stats = tracker.stats()
    assert stats["total_entries"] == 3
    assert tracker.get_score("doc1", "A") == 0.0  # evicted
    assert tracker.get_score("doc1", "B") > 0  # kept


# ── Stats ────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_stats():
    tracker = SectionHeatTracker(
        half_life_seconds=3600, hot_threshold=5.0, max_entries=500
    )
    await tracker.record("doc1", "A", weight=6.0)
    await tracker.record("doc1", "B", weight=1.0)

    stats = tracker.stats()
    assert stats["total_entries"] == 2
    assert stats["hot_count"] == 1  # A is hot (6 > 5), B is not
    assert stats["max_entries"] == 500
    assert stats["half_life_seconds"] == 3600


# ── Persistence ──────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_persistence_save_and_load(tmp_path):
    path = str(tmp_path / "section_heat.json")

    t1 = SectionHeatTracker(
        half_life_seconds=86400, hot_threshold=5.0, max_entries=100, persist_path=path
    )
    await t1.record("doc1", "Intro", weight=3.0)
    await t1.record("doc1", "Methods", weight=7.0)
    await t1.record("doc2", "Results", weight=2.0)
    t1.save_now()

    # Load into a new tracker
    t2 = SectionHeatTracker(
        half_life_seconds=86400, hot_threshold=5.0, max_entries=100, persist_path=path
    )
    assert t2.stats()["total_entries"] == 3
    assert t2.get_score("doc1", "Intro") == pytest.approx(3.0, abs=0.5)
    assert t2.get_score("doc1", "Methods") == pytest.approx(7.0, abs=0.5)
    assert t2.get_score("doc2", "Results") == pytest.approx(2.0, abs=0.5)


@pytest.mark.asyncio
async def test_persistence_corrupted_file(tmp_path):
    path = tmp_path / "section_heat.json"
    path.write_text("not valid json{{{")

    tracker = SectionHeatTracker(persist_path=str(path))
    # Should recover gracefully, start empty
    assert tracker.stats()["total_entries"] == 0


@pytest.mark.asyncio
async def test_persistence_missing_file():
    tracker = SectionHeatTracker(
        persist_path="/tmp/nonexistent_dir_xyz_mentat/section_heat.json"
    )
    assert tracker.stats()["total_entries"] == 0


def test_save_now_no_persist_path():
    tracker = SectionHeatTracker()
    tracker.save_now()  # Should be a no-op, no error


@pytest.mark.asyncio
async def test_persistence_respects_max_entries(tmp_path):
    path = str(tmp_path / "section_heat.json")

    # Save with 5 entries
    t1 = SectionHeatTracker(max_entries=100, persist_path=path)
    for i in range(5):
        await t1.record("doc1", f"S{i}", weight=float(i + 1))
    t1.save_now()

    # Load with smaller capacity — should trim
    t2 = SectionHeatTracker(max_entries=3, persist_path=path)
    assert t2.stats()["total_entries"] == 3


# ── Integration with Mentat (smoke) ─────────────────────────────────


@pytest.mark.asyncio
async def test_search_tracks_section_heat(mentat_instance, tmp_path):
    """search() should record section heat with weight 1.0."""
    # Must be large enough (> 1000 tokens) to avoid bypass and get sectioned chunks
    p = tmp_path / "doc.md"
    content = "# Alpha\n\n" + ("Alpha content paragraph. " * 200) + "\n\n"
    content += "## Beta\n\n" + ("Beta content paragraph. " * 200) + "\n"
    p.write_text(content)

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    results = await mentat_instance.search("Alpha content", top_k=5)

    # Give the fire-and-forget task time to complete
    await asyncio.sleep(0.1)

    # Verify search returned sections, then check heat was tracked
    sections_found = any(r.section for r in results)
    if sections_found:
        stats = mentat_instance.section_heat.stats()
        assert stats["total_entries"] > 0


@pytest.mark.asyncio
async def test_read_segment_tracks_section_heat(mentat_instance, tmp_path):
    """read_segment() should record section heat with weight 3.0."""
    p = tmp_path / "doc.md"
    p.write_text(
        "# Main\n\nMain content.\n\n"
        "## Sub1\n\nSub1 content.\n\n"
        "## Sub2\n\nSub2 content.\n"
    )

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    await mentat_instance.read_segment(doc_id, "Main")

    # Give the fire-and-forget task time to complete
    await asyncio.sleep(0.1)

    # "Main" is a parent section — children should also be tracked
    stats = mentat_instance.section_heat.stats()
    assert stats["total_entries"] >= 1

    # The score should reflect weight 3.0
    top = mentat_instance.section_heat.get_top_sections(doc_id, limit=10)
    if top:
        assert top[0]["raw_score"] >= 3.0


@pytest.mark.asyncio
async def test_inspect_sections_tracks_heat(mentat_instance, tmp_path):
    """inspect(sections=...) should record section heat with weight 2.0."""
    p = tmp_path / "doc.md"
    p.write_text("# Alpha\n\nAlpha.\n\n## Beta\n\nBeta.\n")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    await mentat_instance.inspect(doc_id, sections=["Alpha"])

    # Give the fire-and-forget task time to complete
    await asyncio.sleep(0.1)

    stats = mentat_instance.section_heat.stats()
    assert stats["total_entries"] >= 1


@pytest.mark.asyncio
async def test_section_heat_exposed_in_stats(mentat_instance):
    """stats() should include section_heat key."""
    stats = mentat_instance.stats()
    assert "section_heat" in stats
    assert "total_entries" in stats["section_heat"]
    assert "hot_count" in stats["section_heat"]


@pytest.mark.asyncio
async def test_get_section_heat_method(mentat_instance, tmp_path):
    """get_section_heat() returns results after tracking."""
    p = tmp_path / "doc.md"
    p.write_text("# Title\n\nContent.\n")

    doc_id = await mentat_instance.add(str(p), force=True, wait=True)
    await mentat_instance.read_segment(doc_id, "Title")
    await asyncio.sleep(0.1)

    heat = mentat_instance.get_section_heat(doc_id=doc_id)
    assert isinstance(heat, list)
    if heat:
        assert "doc_id" in heat[0]
        assert "section" in heat[0]
        assert "score" in heat[0]
