"""One-shot orchestrator: download every dataset needed to train.

Usage:
    python scripts/download_all.py                       # full pipeline (~overnight, includes HLS chips)
    python scripts/download_all.py --tier mini           # NASS + USDM only (~5 min, no creds)
    python scripts/download_all.py --tier earthdata      # NASA HLS direct, NO Google Earth Engine (~15 min)
    python scripts/download_all.py --tier sample         # NASS+USDM+HLS NDVI summaries via GEE (~20 min)
    python scripts/download_all.py --tier quick          # analog-feature pipeline via GEE (~2 hr)
    python scripts/download_all.py --tier full           # everything incl. Prithvi chips (~overnight)
    python scripts/download_all.py --force               # re-run even if outputs exist
    python scripts/download_all.py --skip hls            # skip specific step(s)
    python scripts/download_all.py --only nass           # run only specific step(s)
    python scripts/download_all.py --dry-run             # print plan, run nothing

Tiers:
    mini      = nass, usdm                                       → labels + drought, no creds (~5 min)
    earthdata = mini + hls_nasa                                  → real HLS scenes, NO GEE NEEDED (~15 min)
    sample    = mini + hls_ndvi (narrow via HLS_YEARS)           → satellite NDVI via GEE (~20 min)
    quick     = mini + gridmet, landsat, labels, features, check → analog XGBoost via GEE (~2 hr)
    full      = quick + gnatsgo, hls_ndvi, hls                   → Prithvi+LoRA via GEE (~overnight)

Credential paths:
    earthdata tier  → only needs NASS_API_KEY + NASA Earthdata account (~30s signup,
                      https://urs.earthdata.nasa.gov/users/new)
    sample/quick/full → also need EE_PROJECT (Google Cloud project with Earth Engine)

Tunable knobs (env vars, set before running):
    HLS_YEARS=2024,2025      narrow GEE HLS NDVI to specific years (sample tier)
    HLS_BBOX, HLS_DATE_*     narrow direct NASA HLS download (earthdata tier — see fetch_hls_earthaccess.py)

Each step writes to a known parquet/zarr path; if that path exists and --force is
not passed, the step is skipped (idempotent).
"""
from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


@dataclass
class Step:
    key: str
    label: str
    script: str                          # path relative to ROOT
    output: Path                         # canonical output to test for completeness
    needs_env: tuple[str, ...] = ()      # required env vars
    est: str = "?"                       # rough wall-clock estimate
    deps: list[str] = field(default_factory=list)  # step keys that must finish first

STEPS: list[Step] = [
    Step("nass",     "NASS QuickStats yield + crop progress",
         "scripts/data/fetch_nass.py",
         ROOT / "data/raw/nass/yield_county.parquet",
         needs_env=("NASS_API_KEY",), est="~3 min"),
    Step("usdm",     "US Drought Monitor weekly county stats",
         "scripts/data/fetch_usdm.py",
         ROOT / "data/raw/usdm/county_weekly.parquet",
         est="~2 min"),
    Step("gnatsgo",  "gNATSGO soil county aggregates (GEE)",
         "scripts/data/fetch_gnatsgo.py",
         ROOT / "data/raw/gnatsgo/county_soil.parquet",
         needs_env=("EE_PROJECT",), est="~10 min"),
    Step("gridmet",  "gridMET county-day weather (GEE)",
         "scripts/data/export_gridmet.py",
         ROOT / "data/raw/gridmet/county_daily.parquet",
         needs_env=("EE_PROJECT",), est="~30 min"),
    Step("landsat",  "Landsat C2 NDVI backfill 2005-2014 (GEE)",
         "scripts/data/export_landsat_c2_backfill.py",
         ROOT / "data/raw/landsat/county_month_vi.parquet",
         needs_env=("EE_PROJECT",), est="~1 hr"),
    Step("hls_ndvi", "HLS county-month NDVI 2015-2025 (GEE)",
         "scripts/data/export_hls_ndvi.py",
         ROOT / "data/processed/features/hls_county_month_ndvi.parquet",
         needs_env=("EE_PROJECT",), est="~1 hr"),
    Step("hls",      "HLS Prithvi chips 2015-2025 (GEE, HEAVY)",
         "scripts/data/export_hls_chips.py",
         ROOT / "data/processed/chips",          # directory; idempotent inside script
         needs_env=("EE_PROJECT",), est="~overnight"),
    Step("hls_nasa", "HLS scenes direct from NASA LP DAAC (no GEE)",
         "scripts/data/fetch_hls_earthaccess.py",
         ROOT / "data/raw/hls/manifest.parquet",
         est="~10-20 min"),
    Step("labels",   "Build chip metadata + label tables",
         "scripts/training/build_labels_metadata.py",
         ROOT / "data/processed/labels/chip_metadata.parquet",
         est="~2 min",
         deps=["nass", "hls"]),
    Step("features", "Build state-checkpoint analog-matching features",
         "scripts/training/build_features.py",
         ROOT / "data/processed/features/state_checkpoint_features.parquet",
         est="~5 min",
         deps=["gridmet", "usdm", "landsat", "hls_ndvi"]),
    Step("check",    "Sanity-check all data joins",
         "notebooks/01_sanity_check.py",
         ROOT / "data/processed/features/state_checkpoint_features.parquet",  # re-uses
         est="~1 min",
         deps=["labels", "features"]),
]

# Tier presets: which step keys belong to each tier.
# mini  = absolute minimum to have yield labels + drought signal (no Earth Engine).
# quick = enough to train an analog-features XGBoost model end-to-end.
# full  = everything including the overnight HLS chip export for Prithvi fine-tuning.
TIERS: dict[str, list[str]] = {
    "mini":     ["nass", "usdm"],
    "earthdata":["nass", "usdm", "hls_nasa"],   # satellite imagery without GEE setup
    "sample":   ["nass", "usdm", "hls_ndvi"],   # satellite NDVI summaries via GEE
    "quick":    ["nass", "usdm", "gridmet", "landsat", "labels", "features", "check"],
    "full":     [s.key for s in STEPS],
}


# ---------- pretty printing ----------
def hr(s: str) -> None: print(f"\n{'═' * 8} {s} {'═' * 8}")
def fmt(t: float) -> str: return f"{int(t // 60)}m{int(t % 60):02d}s"


def output_present(step: Step) -> bool:
    if step.output.is_dir():
        # For chip dir, "present" = at least one zarr exists
        return any(step.output.rglob("*.zarr"))
    return step.output.exists()


def check_env(step: Step) -> list[str]:
    return [v for v in step.needs_env if not os.getenv(v)]


def run_step(step: Step, dry: bool) -> bool:
    hr(f"{step.key}  {step.label}  ({step.est})")
    missing = check_env(step)
    if missing:
        print(f"  SKIP — missing env: {missing}. See .env.example")
        return True  # skipped, not failed — don't cascade-block dependents
    cmd = [sys.executable, str(ROOT / step.script)]
    print(f"  $ {' '.join(cmd)}")
    if dry:
        print("  (dry-run, not executing)")
        return True
    t0 = time.time()
    rc = subprocess.call(cmd, cwd=str(ROOT))
    dt = time.time() - t0
    if rc != 0:
        print(f"  FAILED in {fmt(dt)} (exit {rc})")
        return False
    print(f"  done in {fmt(dt)}")
    return True


def run_parallel(selected: list[Step], dry: bool, workers: int) -> list[str]:
    """Run steps in parallel, respecting deps. Returns list of failed step keys."""
    step_map = {s.key: s for s in selected}
    selected_keys = set(step_map)
    done: set[str] = set()
    failed: set[str] = set()
    futures = {}
    failed_keys: list[str] = []

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pending = list(selected)
        while pending or futures:
            # Launch any step whose deps are satisfied and hasn't failed
            still_pending = []
            for step in pending:
                blocking_deps = [d for d in step.deps if d in selected_keys and d not in done]
                failed_deps = [d for d in step.deps if d in failed]
                if failed_deps:
                    print(f"\n[SKIP] {step.key} — dep(s) failed: {failed_deps}")
                    failed.add(step.key)
                    failed_keys.append(step.key)
                elif not blocking_deps:
                    fut = pool.submit(run_step, step, dry)
                    futures[fut] = step
                else:
                    still_pending.append(step)
            pending = still_pending

            if not futures:
                break  # nothing running, nothing launchable → stuck (dep cycle or all done)

            # Wait for the next step to finish
            finished = next(as_completed(futures))
            step = futures.pop(finished)
            ok = finished.result()
            if ok:
                done.add(step.key)
            else:
                failed.add(step.key)
                failed_keys.append(step.key)

    return failed_keys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=list(TIERS), default="full",
                    help="preset bundle of steps. Steps needing missing creds are "
                         "skipped automatically. mini=NASS+USDM; quick=XGBoost-ready; full=everything")
    ap.add_argument("--force", action="store_true", help="re-run steps even if outputs exist")
    ap.add_argument("--skip", action="append", default=[], help="step key(s) to skip")
    ap.add_argument("--only", action="append", default=[], help="run only these step key(s)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--workers", type=int, default=4,
                    help="max parallel download steps (default: 4). "
                         "GEE allows ~5 concurrent requests before throttling.")
    args = ap.parse_args()

    tier_keys = set(TIERS[args.tier])

    # Pre-flight: print plan
    hr(f"PLAN  (tier={args.tier})")
    selected = []
    for s in STEPS:
        if args.only and s.key not in args.only:    status = "skip (--only)"
        elif s.key in args.skip:                    status = "skip (--skip)"
        elif s.key not in tier_keys:                status = f"skip (not in tier={args.tier})"
        elif not args.force and output_present(s):  status = "skip (output exists)"
        else:
            status = "run"
            selected.append(s)
        print(f"  [{status:28s}] {s.key:9s} {s.label}")

    if not selected:
        print("\nNothing to do.")
        return

    # Pre-flight: verify GEE auth once — only if EE_PROJECT is actually set
    ee_project = os.getenv("EE_PROJECT")
    if ee_project and any("EE_PROJECT" in s.needs_env for s in selected):
        try:
            import ee  # noqa: F401
            ee.Initialize(project=ee_project)
            print(f"\nGEE OK (project={ee_project})")
        except Exception as e:
            print(f"\nGEE init failed: {e}")
            print("Run: earthengine authenticate")
            sys.exit(2)
    elif any("EE_PROJECT" in s.needs_env for s in selected):
        print("\nEE_PROJECT not set — GEE steps will be skipped automatically.")

    # Execute
    overall_t0 = time.time()
    print(f"\nRunning up to {args.workers} steps in parallel (dep-ordered).")
    failed = run_parallel(selected, args.dry_run, workers=args.workers)
    hr("SUMMARY")
    print(f"  total wall: {fmt(time.time() - overall_t0)}")
    print(f"  ran: {len(selected) - len(failed)}/{len(selected)}")
    if failed:
        print(f"  failed: {failed}")
        sys.exit(1)
    print("  All steps complete.")


if __name__ == "__main__":
    main()
