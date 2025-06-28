#!/usr/bin/env python3
"""merge_csv_columns.py

Copy columns AxL_mean, AxL_diff, SphEq_mean, SphEq_diff **from**
``data/Data_M1D6_Preproc.csv`` **to** ``data/Data_M1D5_Preproc.csv`` by default,
inserting them immediately before the column ``MRRsOD_Binary``.

You can still override the paths from the command line, but if you simply
run

```bash
python merge_csv_columns.py
```

the script will do exactly the requested operation and overwrite
``data/Data_M1D5_Preproc.csv`` in‑place.

If you prefer to write the merged result elsewhere, use the ``-o`` or
``--output`` option.

```
python merge_csv_columns.py -o data/merged_output.csv
```

Assumptions
-----------
* Rows in the two CSVs correspond 1‑to‑1 (same order). If that is not the
  case, merge them on a key before using this script.
* The four columns to copy exist in the source file, and the column
  ``MRRsOD_Binary`` exists in the destination file.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Configuration
# ─────────────────────────────────────────────────────────────────────────────
COLUMNS_TO_COPY = [
    "AxL_mean",
    "AxL_diff",
    "SphEq_mean",
    "SphEq_diff",
]
TARGET_COLUMN = "MRRsOD_Binary"

DEFAULT_SRC = Path("data/Data_M1D6_Preproc.csv")
DEFAULT_DST = Path("data/Data_M1D5_Preproc.csv")

# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Copy specified columns from CSV A to CSV B, inserting them before "
            "a target column. Defaults are set up for the current project."
        ),
    )

    parser.add_argument(
        "file_a",
        nargs="?",
        type=Path,
        default=DEFAULT_SRC,
        help=f"Source CSV (default: {DEFAULT_SRC})",
    )
    parser.add_argument(
        "file_b",
        nargs="?",
        type=Path,
        default=DEFAULT_DST,
        help=f"Destination CSV (default: {DEFAULT_DST})",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,  # decide later: if None, we overwrite file_b
        help="Path to write merged CSV (default: overwrite destination)",
    )
    return parser.parse_args()

# ─────────────────────────────────────────────────────────────────────────────
# Core logic
# ─────────────────────────────────────────────────────────────────────────────

def merge_columns(src: Path, dst: Path, out: Path | None = None) -> Path:
    """Merge *in‑memory* and write CSV.

    Returns the path written to (useful for logging).
    """
    # 1. Read both CSV files --------------------------------------------------
    try:
        df_a = pd.read_csv(src)
    except FileNotFoundError as e:
        sys.exit(f"Error: Source file not found: {e.filename}")

    try:
        df_b = pd.read_csv(dst)
    except FileNotFoundError as e:
        sys.exit(f"Error: Destination file not found: {e.filename}")

    # 2. Validate columns -----------------------------------------------------
    missing_cols = [c for c in COLUMNS_TO_COPY if c not in df_a.columns]
    if missing_cols:
        sys.exit(
            f"Error: Columns {missing_cols} not found in {src}. "
            "Check spelling or column names."
        )

    if TARGET_COLUMN not in df_b.columns:
        sys.exit(
            f"Error: Target column '{TARGET_COLUMN}' not found in {dst}."
        )

    # 3. Optional row‑count sanity check --------------------------------------
    if len(df_a) != len(df_b):
        sys.exit(
            "Error: Row counts differ between the two files. "
            "Align or merge them before using this script."
        )

    # 4. Insert columns -------------------------------------------------------
    insert_position = df_b.columns.get_loc(TARGET_COLUMN)

    for col in reversed(COLUMNS_TO_COPY):
        df_b.insert(insert_position, col, df_a[col].values)

    # 5. Write result ---------------------------------------------------------
    output_path = out or dst  # in‑place by default
    df_b.to_csv(output_path, index=False)
    return output_path

# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    args = parse_args()
    output_path = merge_columns(args.file_a, args.file_b, args.output)
    print(f"✅ Merged CSV written to {output_path}")


if __name__ == "__main__":
    main()
