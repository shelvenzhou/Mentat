import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any
from mentat.probes.base import BaseProbe, ProbeResult


class CSVProbe(BaseProbe):
    def can_handle(self, filename: str, content_type: str) -> bool:
        return filename.lower().endswith(".csv") or content_type == "text/csv"

    def run(self, file_path: str) -> ProbeResult:
        df = pd.read_csv(file_path)

        # 1. Structure: Headers and basic shape
        structure = {
            "columns": df.columns.tolist(),
            "row_count": len(df),
            "col_count": len(df.columns),
        }

        # 2. Stats: pandas.describe() and custom outliers
        # We convert to dict for JSON serialization
        stats = df.describe(include="all").to_dict()

        # Custom: Null rates
        null_rates = df.isnull().mean().to_dict()
        stats["null_rates"] = null_rates

        # Custom: Outliers for numeric columns (Z-score > 3)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        outliers = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            if len(col_data) > 0:
                z_scores = (col_data - col_data.mean()) / col_data.std()
                outlier_count = (abs(z_scores) > 3).sum()
                if outlier_count > 0:
                    outliers[col] = int(outlier_count)
        stats["outlier_counts"] = outliers

        # 3. Summary Hint
        summary_hint = f"CSV with {structure['row_count']} rows and {structure['col_count']} columns. "
        summary_hint += f"Columns: {', '.join(structure['columns'])}."

        return ProbeResult(
            doc_id="",  # To be filled by Hub
            filename=Path(file_path).name,
            file_type="csv",
            structure=structure,
            stats=stats,
            summary_hint=summary_hint,
            raw_snippet=df.head(10).to_csv(),  # Sample data
        )
