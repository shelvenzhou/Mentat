import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    Chunk,
)
from mentat.probes._utils import format_size
from mentat.probes.instruction_templates import (
    IMAGE_BRIEF_INTRO,
    IMAGE_INSTRUCTIONS,
)

try:
    from PIL import Image
    from PIL.ExifTags import TAGS, GPSTAGS

    _PILLOW_AVAILABLE = True
except ImportError:
    _PILLOW_AVAILABLE = False


class ImageProbe(BaseProbe):
    """Probe for image files (JPEG, PNG, GIF, etc.)."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        if not _PILLOW_AVAILABLE:
            return False
        return filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".tiff", ".tif", ".webp")
        )

    def run(self, file_path: str) -> ProbeResult:
        file_size = os.path.getsize(file_path)

        with Image.open(file_path) as img:
            width, height = img.size
            img_format = img.format or Path(file_path).suffix.upper().lstrip(".")
            mode = img.mode

            # --- EXIF extraction ---
            exif_data = self._extract_exif(img)

        # --- Build descriptive first paragraph ---
        parts = [f"{img_format} image", f"{width}x{height}", mode]
        if exif_data.get("DateTime"):
            parts.append(f"taken {exif_data['DateTime']}")
        if exif_data.get("Make") or exif_data.get("Model"):
            camera = " ".join(
                filter(None, [exif_data.get("Make"), exif_data.get("Model")])
            )
            parts.append(f"with {camera}")
        first_paragraph = ", ".join(parts)

        # --- Stats ---
        stats: Dict[str, Any] = {
            "width": width,
            "height": height,
            "format": img_format,
            "mode": mode,
            "file_size_bytes": file_size,
            "file_size_human": format_size(file_size),
        }
        if exif_data:
            stats["exif"] = exif_data

        # --- Topic ---
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=first_paragraph,
        )

        # --- Single metadata chunk ---
        chunk_content = f"Image: {first_paragraph}\nFile size: {format_size(file_size)}"
        if exif_data:
            for k, v in exif_data.items():
                chunk_content += f"\n{k}: {v}"

        chunks = [Chunk(content=chunk_content, index=0, section="metadata")]

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="image",
            topic=topic,
            structure=StructureInfo(),
            stats=stats,
            chunks=chunks,
            raw_snippet=None,
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate Image-specific instructions."""
        # Brief intro
        brief_intro = IMAGE_BRIEF_INTRO

        # Full instructions
        instructions = IMAGE_INSTRUCTIONS.format(filename=probe_result.filename)

        return brief_intro, instructions

    def _extract_exif(self, img: "Image.Image") -> Dict[str, str]:
        """Extract human-readable EXIF data from an image."""
        result: Dict[str, str] = {}

        try:
            exif = img.getexif()
        except Exception:
            return result

        if not exif:
            return result

        # Key EXIF tags we care about
        interesting_tags = {
            "DateTime",
            "DateTimeOriginal",
            "Make",
            "Model",
            "Software",
            "ImageDescription",
            "Artist",
            "Copyright",
            "ExposureTime",
            "FNumber",
            "ISOSpeedRatings",
            "FocalLength",
        }

        for tag_id, value in exif.items():
            tag_name = TAGS.get(tag_id, str(tag_id))
            if tag_name in interesting_tags:
                result[tag_name] = str(value)[:200]

        # GPS info
        gps_info = exif.get(0x8825)
        if gps_info:
            try:
                gps_data = {}
                for key, val in gps_info.items():
                    tag_name = GPSTAGS.get(key, str(key))
                    gps_data[tag_name] = str(val)
                if "GPSLatitude" in gps_data and "GPSLongitude" in gps_data:
                    result["GPS"] = (
                        f"Lat: {gps_data['GPSLatitude']}, "
                        f"Lon: {gps_data['GPSLongitude']}"
                    )
            except Exception:
                pass

        return result
