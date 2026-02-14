import pandas as pd
import numpy as np
from pathlib import Path
from typing import Any, Dict, List, Tuple
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    TocEntry,
    Chunk,
)
from mentat.probes._utils import estimate_tokens
from mentat.probes.instruction_templates import (
    CSV_BRIEF_INTRO,
    CSV_INSTRUCTIONS,
    CSV_SAMPLING_NOTE_FULL,
    CSV_SAMPLING_NOTE_SAMPLED,
)


class CSVProbe(BaseProbe):
    """Probe for CSV files."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".csv") or content_type == "text/csv"

    def run(self, file_path: str) -> ProbeResult:
        df = pd.read_csv(file_path)
        columns = df.columns.tolist()

        # --- Per-column type hints ---
        column_types = self._infer_column_types(df)

        # --- Cardinality ---
        cardinality = {col: int(df[col].nunique()) for col in columns}

        # --- Null rates ---
        null_rates = df.isnull().mean().to_dict()
        null_counts = df.isnull().sum().to_dict()

        # --- Outliers for numeric columns (Z-score > 3) ---
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1 and col_data.std() > 0:
                z_scores = (col_data - col_data.mean()) / col_data.std()
                outlier_count = (abs(z_scores) > 3).sum()
                if outlier_count > 0:
                    outliers[col] = int(outlier_count)

        # --- String length stats for text columns ---
        text_cols = df.select_dtypes(include=["object"]).columns
        strlen_stats = {}
        for col in text_cols:
            lengths = df[col].dropna().astype(str).str.len()
            if len(lengths) > 0:
                strlen_stats[col] = {
                    "min": int(lengths.min()),
                    "max": int(lengths.max()),
                    "mean": round(float(lengths.mean()), 1),
                }

        # --- Describe (pandas summary) ---
        describe = df.describe(include="all").to_dict()

        # --- Stats ---
        raw_csv = df.to_csv(index=False)
        approx_tokens = estimate_tokens(raw_csv)
        stats: Dict[str, Any] = {
            "row_count": len(df),
            "col_count": len(columns),
            "approx_tokens": approx_tokens,
            "column_types": column_types,
            "cardinality": cardinality,
            "describe": describe,
            "null_rates": null_rates,
            "outlier_counts": outliers,
            "strlen_stats": strlen_stats,
        }

        # --- Topic ---
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=(
                f"CSV dataset with {len(df)} rows and {len(columns)} columns: "
                f"{', '.join(columns)}."
            ),
        )

        # --- Column ToC ---
        toc_entries: List[TocEntry] = []
        for col in columns:
            parts = [column_types.get(col, "?")]
            parts.append(f"{cardinality.get(col, 0)} unique")
            nc = null_counts.get(col, 0)
            if nc > 0:
                parts.append(f"{nc} nulls")
            annotation = " | ".join(parts)
            toc_entries.append(
                TocEntry(level=1, title=col, annotation=annotation)
            )

        structure = StructureInfo(columns=columns, toc=toc_entries)

        # --- Small-file bypass ---
        is_small = len(df) * len(columns) < 50
        if is_small:
            stats["is_full_content"] = True
            full_csv = df.to_csv(index=False)
            result = ProbeResult(
                filename=Path(file_path).name,
                file_type="csv",
                topic=topic,
                structure=structure,
                stats=stats,
                chunks=[
                    Chunk(
                        content=full_csv,
                        index=0,
                        section="full_data",
                        metadata={"total_rows": len(df)},
                    )
                ],
                raw_snippet=full_csv,
            )
            # Generate format-specific instructions
            brief_intro, instructions = self.generate_instructions(result)
            result.brief_intro = brief_intro
            result.instructions = instructions
            return result

        # --- Representative sampling: first + middle + last row ---
        stats["is_full_content"] = False
        indices = [0]
        if len(df) > 2:
            indices.append(len(df) // 2)
        if len(df) > 1:
            indices.append(len(df) - 1)
        sample_df = df.iloc[indices]
        sample_csv = sample_df.to_csv(index=False)

        chunks = [
            Chunk(
                content=sample_csv,
                index=0,
                section="representative_sample",
                metadata={
                    "sample_rows": len(indices),
                    "total_rows": len(df),
                    "sample_indices": indices,
                },
            )
        ]

        result = ProbeResult(
            filename=Path(file_path).name,
            file_type="csv",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=sample_csv,
        )

        # Generate format-specific instructions
        brief_intro, instructions = self.generate_instructions(result)
        result.brief_intro = brief_intro
        result.instructions = instructions

        return result

    def generate_instructions(self, probe_result: ProbeResult) -> Tuple[str, str]:
        """Generate CSV-specific instructions."""
        stats = probe_result.stats

        # Brief intro
        brief_intro = CSV_BRIEF_INTRO

        # Sampling strategy description
        if stats.get('is_full_content'):
            sampling_strategy = CSV_SAMPLING_NOTE_FULL
        else:
            sampling_strategy = CSV_SAMPLING_NOTE_SAMPLED

        # Full instructions
        instructions = CSV_INSTRUCTIONS.format(
            sampling_strategy=sampling_strategy,
            filename=probe_result.filename,
        )

        return brief_intro, instructions

    def _infer_column_types(self, df: pd.DataFrame) -> Dict[str, str]:
        type_map = {}
        for col in df.columns:
            dtype = df[col].dtype
            if pd.api.types.is_bool_dtype(dtype):
                type_map[col] = "Bool"
            elif pd.api.types.is_integer_dtype(dtype):
                type_map[col] = "Int"
            elif pd.api.types.is_float_dtype(dtype):
                type_map[col] = "Float"
            elif pd.api.types.is_datetime64_any_dtype(dtype):
                type_map[col] = "DateTime"
            else:
                # Try to detect date-like strings
                sample = df[col].dropna().head(5).astype(str)
                if len(sample) > 0 and self._looks_like_date(sample):
                    type_map[col] = "Date"
                else:
                    type_map[col] = "String"
        return type_map

    def _looks_like_date(self, series: pd.Series) -> bool:
        try:
            pd.to_datetime(series, format="mixed")
            return True
        except (ValueError, TypeError):
            return False
