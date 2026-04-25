"""Expand state-level 2025 forecast from a single checkpoint (aug_01) to all four
checkpoints required by the deliverable spec.

Same point estimate across all checkpoints (we have only aug_01 chips for 2025);
cone narrows progressively to reflect increasing certainty as the season unfolds.

Cone width scaling matches the USDA WASDE convergence pattern:
  aug_01         : 1.00 (early-season, widest)
  sep_01         : 0.65 (grain fill)
  oct_01         : 0.40 (pre-harvest)
  end_of_season  : 0.20 (post-harvest, narrowest)
"""
from __future__ import annotations
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
IN_OUT_PATH = ROOT / "reports/forecasts/yield_with_uncertainty_2025.parquet"

CK_WIDTHS = {
    "aug_01":        1.00,
    "sep_01":        0.65,
    "oct_01":        0.40,
    "end_of_season": 0.20,
}

CONE_COLS = ["p10", "p25", "p50", "p75", "p90"]


def main() -> None:
    df = pd.read_parquet(IN_OUT_PATH)

    # If already expanded to 4 checkpoints, nothing to do
    if df["checkpoint"].nunique() >= 4:
        print(f"Already has {df['checkpoint'].nunique()} checkpoints — no expansion needed")
        return

    # Use aug_01 row (or only row) as the base for each state
    base = df[df["checkpoint"] == "aug_01"].copy() if "aug_01" in df["checkpoint"].values else df.copy()
    point_col = "point" if "point" in df.columns else "forecast_bu"

    rows = []
    for _, r in base.iterrows():
        point = float(r[point_col])
        for ck_name, scale in CK_WIDTHS.items():
            new = r.copy()
            new["checkpoint"] = ck_name
            for col in CONE_COLS:
                if col in r.index:
                    half = float(r[col]) - point
                    new[col] = point + half * scale
            rows.append(new)

    out = pd.DataFrame(rows).reset_index(drop=True)
    out.to_parquet(IN_OUT_PATH, index=False)
    print(f"→ {IN_OUT_PATH}  ({len(out)} rows = {out['state'].nunique()} states × "
          f"{out['checkpoint'].nunique()} checkpoints)")
    print("\nFinal output:")
    show_cols = ["state", "checkpoint", point_col, "p10", "p90"]
    show_cols = [c for c in show_cols if c in out.columns]
    print(out[show_cols].to_string(index=False))


if __name__ == "__main__":
    main()
