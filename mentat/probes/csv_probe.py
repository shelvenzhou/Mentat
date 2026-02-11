import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from mentat.probes.base import (
    BaseProbe,
    ProbeResult,
    TopicInfo,
    StructureInfo,
    Chunk,
)


class CSVProbe(BaseProbe):
    """Probe for CSV files."""

    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".csv") or content_type == "text/csv"

    def run(self, file_path: str) -> ProbeResult:
        df = pd.read_csv(file_path)

        columns = df.columns.tolist()

        # --- Stats: pandas.describe() + custom outliers (design requirement) ---
        describe = df.describe(include="all").to_dict()

        # Null rates
        null_rates = df.isnull().mean().to_dict()

        # Outliers for numeric columns (Z-score > 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 1 and col_data.std() > 0:
                z_scores = (col_data - col_data.mean()) / col_data.std()
                outlier_count = (abs(z_scores) > 3).sum()
                if outlier_count > 0:
                    outliers[col] = int(outlier_count)

        # String length stats for text columns
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

        stats = {
            "row_count": len(df),
            "col_count": len(columns),
            "describe": describe,
            "null_rates": null_rates,
            "outlier_counts": outliers,
            "strlen_stats": strlen_stats,
        }

        # --- Topic: inferred from filename + column summary ---
        topic = TopicInfo(
            title=Path(file_path).stem,
            first_paragraph=f"CSV dataset with {len(df)} rows and {len(columns)} columns: {', '.join(columns)}.",
        )

        # --- Structure: header row ---
        structure = StructureInfo(columns=columns)

        # --- Chunks: header + sample rows ---
        # One chunk with header + first N rows as example
        sample = df.head(10).to_csv(index=False)
        chunks = [
            Chunk(
                content=sample,
                index=0,
                section="header_and_sample",
                metadata={"sample_rows": min(10, len(df)), "total_rows": len(df)},
            )
        ]

        return ProbeResult(
            filename=Path(file_path).name,
            file_type="csv",
            topic=topic,
            structure=structure,
            stats=stats,
            chunks=chunks,
            raw_snippet=sample,
        )
