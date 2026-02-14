from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes.instruction_templates import (
    CALENDAR_BRIEF_INTRO,
    CALENDAR_INSTRUCTIONS,
)

try:
    from icalendar import Calendar

    _ICAL_AVAILABLE = True
except ImportError:
    _ICAL_AVAILABLE = False


class CalendarProbe(BaseProbe):
    """Probe for iCalendar files (.ics)."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        if not _ICAL_AVAILABLE:
            return False
        return filename.lower().endswith(".ics") or content_type == "text/calendar"

    def run(self, file_path: str) -> ProbeResult:
        with open(file_path, "rb") as f:
            cal = Calendar.from_ical(f.read())

        events: List[Dict[str, Any]] = []
        attendees_set: set = set()
        recurring_count = 0
        one_time_count = 0

        for component in cal.walk():
            if component.name != "VEVENT":
                continue

            summary = str(component.get("SUMMARY", "Untitled"))
            dtstart = component.get("DTSTART")
            dtend = component.get("DTEND")
            description = str(component.get("DESCRIPTION", ""))
            location = str(component.get("LOCATION", ""))
            rrule = component.get("RRULE")

            start_str = str(dtstart.dt) if dtstart else None
            end_str = str(dtend.dt) if dtend else None

            if rrule:
                recurring_count += 1
            else:
                one_time_count += 1

            # Collect attendees
            attendee_list = component.get("ATTENDEE")
            if attendee_list:
                if isinstance(attendee_list, list):
                    for a in attendee_list:
                        attendees_set.add(str(a).replace("mailto:", ""))
                else:
                    attendees_set.add(str(attendee_list).replace("mailto:", ""))

            events.append(
                {
                    "summary": summary,
                    "start": start_str,
                    "end": end_str,
                    "description": description[:200],
                    "location": location,
                    "recurring": bool(rrule),
                }
            )

        # Sort events by start time
        events.sort(key=lambda e: e["start"] or "")

        # --- Time range ---
        starts = [e["start"] for e in events if e["start"]]
        time_range_start = starts[0] if starts else None
        time_range_end = starts[-1] if starts else None

        # --- ToC: events as entries ---
        toc_entries: List[TocEntry] = []
        for event in events[:30]:  # Cap at 30 entries
            annot_parts = []
            if event["start"]:
                annot_parts.append(event["start"])
            if event["recurring"]:
                annot_parts.append("Recurring")
            annotation = " | ".join(annot_parts) if annot_parts else None

            preview = event["location"] or event["description"][:120] or None

            toc_entries.append(
                TocEntry(
                    level=1,
                    title=event["summary"],
                    annotation=annotation,
                    preview=preview,
                )
            )

        # --- Stats ---
        stats: Dict[str, Any] = {
            "event_count": len(events),
            "recurring_count": recurring_count,
            "one_time_count": one_time_count,
            "attendee_count": len(attendees_set),
        }
        if time_range_start and time_range_end:
            stats["time_range"] = {
                "start": time_range_start,
                "end": time_range_end,
            }
        if attendees_set:
            stats["attendees"] = sorted(attendees_set)[:20]

        # --- Topic ---
        cal_name = str(cal.get("X-WR-CALNAME", "")) or Path(file_path).stem
        time_desc = ""
        if time_range_start and time_range_end:
            time_desc = f", {time_range_start} to {time_range_end}"

        topic = TopicInfo(
            title=cal_name,
            first_paragraph=(
                f"Calendar with {len(events)} events{time_desc}, "
                f"{len(attendees_set)} attendees"
            ),
        )

        structure = StructureInfo(toc=toc_entries)

        # --- Chunks: one per event ---
        chunks: List[Chunk] = []
        for i, event in enumerate(events):
            parts = [event["summary"]]
            if event["start"]:
                parts.append(f"Start: {event['start']}")
            if event["end"]:
                parts.append(f"End: {event['end']}")
            if event["location"]:
                parts.append(f"Location: {event['location']}")
            if event["description"]:
                parts.append(f"Description: {event['description']}")

            chunks.append(
                Chunk(
                    content="\n".join(parts),
                    index=i,
                    section=event["summary"],
                )
            )

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="calendar",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=chunks[0].content[:500] if chunks else None,
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate Calendar-specific instructions."""
        # Brief intro
        brief_intro = CALENDAR_BRIEF_INTRO

        # Full instructions
        instructions = CALENDAR_INSTRUCTIONS.format(filename=probe_result.filename)

        return brief_intro, instructions
