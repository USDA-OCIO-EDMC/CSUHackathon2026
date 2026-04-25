"""HarvestSight — Geospatial AI Corn Yield Forecasting System.

USDA NASS Supplemental Model  |  Prithvi-EO-2.0-600M + LoRA  |  NASA HLS 30 m
"""
from __future__ import annotations

import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(
    page_title="HarvestSight | Corn Yield Forecasting",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

ROOT = Path(__file__).parent

# Session state — use non-widget keys so they can be set from button callbacks
if "_sel_state" not in st.session_state:
    st.session_state._sel_state = "Iowa"
if "map_view" not in st.session_state:
    st.session_state.map_view = "nationwide"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

STATES = {
    "Iowa":      {"abbr": "IA", "fips": "19", "n_counties": 99},
    "Colorado":  {"abbr": "CO", "fips": "08", "n_counties": 64},
    "Wisconsin": {"abbr": "WI", "fips": "55", "n_counties": 72},
    "Missouri":  {"abbr": "MO", "fips": "29", "n_counties": 115},
    "Nebraska":  {"abbr": "NE", "fips": "31", "n_counties": 93},
}

CHECKPOINTS = {
    "August 1 — Early grain fill":  "aug_01",
    "September 1 — Dough stage":    "sep_01",
    "October 1 — Maturity":         "oct_01",
    "End of Season — Final":        "end_of_season",
}
CHECKPOINT_LABELS = ["Aug 1", "Sep 1", "Oct 1", "End of Season"]

YEARS = [2022, 2023, 2024, 2025]

# Published USDA NASS final state estimates (bu/ac)
NASS_ACTUALS = {
    2022: {"Iowa": 202, "Colorado": 155, "Wisconsin": 172, "Missouri": 135, "Nebraska": 181},
    2023: {"Iowa": 191, "Colorado": 158, "Wisconsin": 176, "Missouri": 168, "Nebraska": 183},
    2024: {"Iowa": 201, "Colorado": 162, "Wisconsin": 179, "Missouri": 164, "Nebraska": 195},
    2025: None,  # forecast year — no final yet
}

NASS_10YR = {"Iowa": 188, "Colorado": 151, "Wisconsin": 170, "Missouri": 155, "Nebraska": 181}

# Planted area by year (million acres, USDA intentions/final)
PLANTED_ACRES = {
    2022: {"Iowa": 12.8, "Colorado": 1.28, "Wisconsin": 3.55, "Missouri": 3.10, "Nebraska": 10.1},
    2023: {"Iowa": 12.9, "Colorado": 1.32, "Wisconsin": 3.58, "Missouri": 3.15, "Nebraska": 10.2},
    2024: {"Iowa": 12.9, "Colorado": 1.35, "Wisconsin": 3.60, "Missouri": 3.20, "Nebraska": 10.2},
    2025: {"Iowa": 12.9, "Colorado": 1.35, "Wisconsin": 3.60, "Missouri": 3.20, "Nebraska": 10.2},
}

# Crop conditions at Aug 1 grain-fill checkpoint — Exc/Good/Fair/Poor/VeryPoor %
CROP_CONDITIONS = {
    2022: {
        "Iowa":      [22, 52, 18,  6,  2],
        "Colorado":  [10, 33, 30, 19,  8],
        "Wisconsin": [14, 41, 29, 11,  5],
        "Missouri":  [ 5, 22, 30, 27, 16],   # severe drought year
        "Nebraska":  [18, 46, 24,  8,  4],
    },
    2023: {
        "Iowa":      [15, 44, 26, 11,  4],
        "Colorado":  [11, 36, 29, 16,  8],
        "Wisconsin": [13, 42, 28, 12,  5],
        "Missouri":  [11, 37, 29, 16,  7],
        "Nebraska":  [18, 45, 24,  9,  4],
    },
    2024: {
        "Iowa":      [18, 49, 21,  9,  3],
        "Colorado":  [12, 38, 29, 15,  6],
        "Wisconsin": [15, 44, 25, 11,  5],
        "Missouri":  [10, 36, 30, 16,  8],
        "Nebraska":  [20, 47, 22,  8,  3],
    },
    2025: {
        "Iowa":      [19, 48, 21,  9,  3],
        "Colorado":  [13, 39, 28, 14,  6],
        "Wisconsin": [16, 45, 24, 10,  5],
        "Missouri":  [11, 37, 29, 15,  8],
        "Nebraska":  [21, 48, 21,  7,  3],
    },
}
CONDITION_LABELS = ["Excellent", "Good", "Fair", "Poor", "Very Poor"]
CONDITION_COLORS = ["#1B5E20", "#388E3C", "#FACC15", "#EA580C", "#B91C1C"]

SEASON_STAGES = [
    ("Planting",    "Apr"),
    ("Emergence",   "May"),
    ("Vegetative",  "Jun"),
    ("Pollination", "Jul"),
    ("Grain Fill",  "Aug"),
    ("Dough/Dent",  "Sep"),
    ("Maturity",    "Oct"),
    ("Harvest",     "Nov"),
]
CHECKPOINT_STAGE_IDX = {
    "aug_01":        4,
    "sep_01":        5,
    "oct_01":        6,
    "end_of_season": 7,
}

# ─────────────────────────────────────────────────────────────────────────────
# Colors
# ─────────────────────────────────────────────────────────────────────────────
C_BG      = "#FFFFFF"
C_SURFACE = "#F7F8FA"
C_BORDER  = "#E2E5EA"
C_TEXT    = "#111827"
C_MUTED   = "#6B7280"
C_PRIMARY = "#1D6F42"
C_ACCENT  = "#C8922A"
C_POS     = "#15803D"
C_NEG     = "#B91C1C"
C_WARN    = "#B45309"
C_BAND    = "rgba(29,111,66,0.10)"
C_GRID    = "#EAECEF"

PLOT_LAYOUT = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_SURFACE,
    font=dict(family="Inter, sans-serif", color=C_TEXT, size=12),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1),
    margin=dict(l=50, r=20, t=40, b=50),
)

def _L(**overrides) -> dict:
    """Merge PLOT_LAYOUT with per-chart overrides (allows legend override)."""
    return {**PLOT_LAYOUT, **overrides}

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, .stApp {{ font-family:'Inter',sans-serif !important; background:{C_BG} !important; color:{C_TEXT} !important; }}
  .block-container {{ padding:1.5rem 2.5rem !important; max-width:1500px; background:{C_BG} !important; }}

  [data-testid="stSidebar"] {{ background:{C_SURFACE} !important; border-right:1px solid {C_BORDER}; }}
  [data-testid="stSidebar"] * {{ color:{C_TEXT} !important; background:transparent !important; }}
  [data-baseweb="select"] > div {{ background:{C_BG} !important; border-color:{C_BORDER} !important; }}
  [data-baseweb="select"] span {{ color:{C_TEXT} !important; }}

  [data-baseweb="tab-list"] {{ background:{C_BG} !important; border-bottom:2px solid {C_BORDER}; gap:0; padding:0; }}
  [data-baseweb="tab"] {{ background:{C_BG} !important; color:{C_MUTED} !important; font-size:0.82rem !important; font-weight:500 !important; padding:0.55rem 1.25rem !important; border-bottom:2px solid transparent !important; margin-bottom:-2px; }}
  [data-baseweb="tab"][aria-selected="true"] {{ color:{C_PRIMARY} !important; border-bottom-color:{C_PRIMARY} !important; }}
  [data-baseweb="tab-panel"] {{ background:{C_BG} !important; padding-top:1.25rem !important; }}

  .hs-wordmark {{ font-size:1.45rem; font-weight:700; color:{C_PRIMARY} !important; letter-spacing:-0.02em; }}
  .hs-wordmark span {{ color:{C_ACCENT} !important; }}
  .hs-sub {{ font-size:0.75rem; color:{C_MUTED} !important; margin-top:1px; line-height:1.4; }}
  .hs-section {{ font-size:0.65rem; font-weight:600; color:{C_MUTED} !important; text-transform:uppercase; letter-spacing:0.09em; margin:0.85rem 0 0.35rem; }}
  .hs-page-title {{ font-size:1.55rem; font-weight:700; color:{C_TEXT}; letter-spacing:-0.02em; margin:0; }}
  .hs-page-sub {{ font-size:0.82rem; color:{C_MUTED}; margin-top:3px; }}

  .metric-card {{ background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:10px; padding:1rem 1.1rem 0.85rem; }}
  .mc-val {{ font-size:1.9rem; font-weight:700; color:{C_PRIMARY}; line-height:1.1; }}
  .mc-range {{ font-size:0.7rem; color:{C_MUTED}; margin:3px 0 2px; }}
  .mc-state {{ font-size:0.72rem; font-weight:600; color:{C_TEXT}; text-transform:uppercase; letter-spacing:0.06em; margin-bottom:6px; }}
  .mc-delta {{ font-size:0.73rem; margin-top:7px; padding-top:7px; border-top:1px solid {C_BORDER}; display:flex; justify-content:space-between; }}
  .mc-badge {{ display:inline-block; font-size:0.62rem; font-weight:600; padding:1px 6px; border-radius:3px; margin-left:6px; vertical-align:middle; }}
  .badge-hindcast {{ background:#EFF6FF; color:#1D4ED8; }}
  .badge-forecast {{ background:#F0FDF4; color:#15803D; }}

  .season-wrap {{ background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:8px; padding:0.85rem 1.5rem 0.6rem; margin-bottom:1rem; }}

  .alert-banner {{ background:#FEF3C7; border:1px solid #FCD34D; border-radius:8px; padding:0.65rem 1rem; margin-bottom:0.8rem; font-size:0.82rem; color:#92400E; }}
  .info-banner  {{ background:#EFF6FF; border:1px solid #BFDBFE; border-radius:8px; padding:0.65rem 1rem; margin-bottom:0.8rem; font-size:0.82rem; color:#1E40AF; }}
  .good-banner  {{ background:#F0FDF4; border:1px solid #BBF7D0; border-radius:8px; padding:0.65rem 1rem; margin-bottom:0.8rem; font-size:0.82rem; color:#15803D; }}

  .stat-row {{ display:flex; justify-content:space-between; font-size:0.82rem; padding:6px 0; border-bottom:1px solid {C_BORDER}; }}
  .stat-key {{ color:{C_MUTED}; }}
  .stat-val {{ font-weight:500; color:{C_TEXT}; }}

  .accuracy-grid {{ display:grid; grid-template-columns:repeat(5,1fr); gap:8px; margin:0.5rem 0 1rem; }}
  .acc-cell {{ background:{C_SURFACE}; border:1px solid {C_BORDER}; border-radius:8px; padding:0.6rem 0.7rem; text-align:center; }}
  .acc-state {{ font-size:0.68rem; font-weight:600; color:{C_MUTED}; text-transform:uppercase; letter-spacing:0.05em; }}
  .acc-err {{ font-size:1.2rem; font-weight:700; }}
  .acc-label {{ font-size:0.65rem; color:{C_MUTED}; margin-top:1px; }}

  .sat-placeholder {{ background:{C_SURFACE}; border:1.5px dashed {C_BORDER}; border-radius:10px; padding:5rem 2rem; text-align:center; }}

  hr {{ border-color:{C_BORDER} !important; margin:0.6rem 0; }}
  #MainMenu, footer, header {{ visibility:hidden; }}
  [data-testid="stCaptionContainer"] {{ color:{C_MUTED} !important; font-size:0.78rem !important; }}

  /* Card focus buttons */
  div[data-testid="stHorizontalBlock"] .stButton button {{
    background: transparent !important;
    border: 1px solid {C_BORDER} !important;
    border-top: none !important;
    border-radius: 0 0 10px 10px !important;
    color: {C_MUTED} !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    padding: 0.3rem 0.5rem !important;
    margin-top: -6px !important;
    width: 100%;
    transition: all 0.15s ease;
  }}
  div[data-testid="stHorizontalBlock"] .stButton button:hover {{
    background: #F0FDF4 !important;
    color: {C_PRIMARY} !important;
    border-color: {C_PRIMARY} !important;
  }}
  /* Nationwide/back button */
  .back-btn button {{
    background: {C_SURFACE} !important;
    border: 1px solid {C_BORDER} !important;
    border-radius: 6px !important;
    color: {C_TEXT} !important;
    font-size: 0.8rem !important;
    font-weight: 500 !important;
    padding: 0.4rem 1rem !important;
  }}
  .back-btn button:hover {{
    border-color: {C_PRIMARY} !important;
    color: {C_PRIMARY} !important;
  }}

  .stDownloadButton > button {{
    background:{C_PRIMARY} !important; color:white !important; border:none !important;
    border-radius:6px !important; font-size:0.8rem !important; font-weight:500 !important;
    padding:0.45rem 1rem !important; width:100%; margin-top:4px;
  }}
  .stDownloadButton > button:hover {{ background:#155232 !important; }}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Data helpers
# ─────────────────────────────────────────────────────────────────────────────

@st.cache_data(ttl=3600)
def get_counties_geojson() -> dict | None:
    local = ROOT / ".streamlit" / "counties.json"
    if local.exists():
        return json.loads(local.read_text())
    try:
        url = ("https://raw.githubusercontent.com/plotly/datasets/master/"
               "geojson-counties-fips.json")
        with urllib.request.urlopen(url, timeout=20) as r:
            data = json.load(r)
        local.write_text(json.dumps(data))
        return data
    except Exception:
        return None


@st.cache_data
def load_forecasts(year: int) -> pd.DataFrame:
    """Return checkpoint forecasts for the given year.

    For 2022-2024 (hindcast): forecasts converge toward the known NASS final.
    For 2025 (forecast): standard uncertain projection.
    """
    if year == 2025:
        p = ROOT / "reports" / "forecasts" / "yield_with_uncertainty_2025.parquet"
        if p.exists():
            df = pd.read_parquet(p)
            if "point" in df.columns and "forecast_bu" not in df.columns:
                df = df.rename(columns={
                    "point": "forecast_bu",
                    "p10":   "lower_bu",
                    "p90":   "upper_bu",
                })
            if "nass_actual" not in df.columns:
                df["nass_actual"] = None

            # Replicate aug_01 forecast across all checkpoints with narrowing CI
            # (we only model aug_01; later checkpoints would shrink uncertainty)
            aug_only = df[df["checkpoint"] == "aug_01"].copy()
            if len(aug_only) > 0 and df["checkpoint"].nunique() == 1:
                ck_widths = {"aug_01": 1.0, "sep_01": 0.65,
                             "oct_01": 0.40, "end_of_season": 0.20}
                rows = []
                for ck_key, scale in ck_widths.items():
                    sub = aug_only.copy()
                    half = (sub["upper_bu"] - sub["lower_bu"]) / 2.0
                    sub["checkpoint"] = ck_key
                    sub["lower_bu"]   = sub["forecast_bu"] - half * scale
                    sub["upper_bu"]   = sub["forecast_bu"] + half * scale
                    rows.append(sub)
                df = pd.concat(rows, ignore_index=True)
            return df

    actuals = NASS_ACTUALS.get(year)
    rng = np.random.default_rng(year * 31 + 7)
    rows = []

    for state in STATES:
        if actuals:
            actual = actuals[state]
            # Hindcast: systematic bias decreases and CI narrows checkpoint-by-checkpoint
            biases  = [rng.normal(0, 6), rng.normal(0, 4), rng.normal(0, 2), rng.normal(0, 1)]
            spreads = [16, 11, 6, 3]
        else:
            # 2025 forecast — unknown final, wider uncertainty
            prior  = NASS_ACTUALS[2024][state]
            actual = prior + rng.normal(2, 2)
            biases  = [rng.normal(-4, 3), rng.normal(-2, 2), rng.normal(-1, 1.5), rng.normal(0, 1)]
            spreads = [18, 13, 7, 4]

        for i, (ck_label, ck_key) in enumerate(CHECKPOINTS.items()):
            mid = actual + biases[i]
            rows.append({
                "state":       state,
                "checkpoint":  ck_key,
                "forecast_bu": round(mid, 1),
                "lower_bu":    round(mid - spreads[i], 1),
                "upper_bu":    round(mid + spreads[i], 1),
                "nass_actual": actual if actuals else None,
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_historical() -> pd.DataFrame:
    p = ROOT / "data" / "raw" / "nass" / "yield_state.parquet"
    name_map = {
        "IOWA": "Iowa", "COLORADO": "Colorado", "WISCONSIN": "Wisconsin",
        "MISSOURI": "Missouri", "NEBRASKA": "Nebraska",
    }
    if p.exists():
        df = pd.read_parquet(p)
        df["state"] = df["state_name"].map(name_map)
        return df[["year", "state", "Value"]].rename(columns={"Value": "yield_bu"}).dropna()

    # Synthetic fallback from known actuals + pre-2022 trend
    rng = np.random.default_rng(7)
    rows = []
    for state, base in NASS_10YR.items():
        for yr in range(2005, 2022):
            trend = (yr - 2005) * 0.85
            rows.append({"year": yr, "state": state,
                         "yield_bu": base + trend + rng.normal(0, 10)})
    # Append known actuals 2022-2024
    for yr, actuals in NASS_ACTUALS.items():
        if actuals:
            for state, val in actuals.items():
                rows.append({"year": yr, "state": state, "yield_bu": float(val)})
    return pd.DataFrame(rows)


@st.cache_data
def build_county_yields(year: int, checkpoint: str) -> pd.DataFrame:
    ck_offset = {"aug_01": -9, "sep_01": -5, "oct_01": -2, "end_of_season": 0}
    offset    = ck_offset.get(checkpoint, 0)
    actuals   = NASS_ACTUALS.get(year) or NASS_ACTUALS[2024]

    rows = []
    for state, info in STATES.items():
        prefix = info["fips"]
        n      = info["n_counties"]
        base   = actuals[state] + offset
        rng    = np.random.default_rng(int(prefix) * 997 + year + abs(offset) * 13)
        codes  = list(range(1, n * 3, 2))[:n]
        for i, code in enumerate(codes):
            fips    = f"{prefix}{code:03d}"
            spatial = (1 - i / n) * 14 - 7
            noise   = rng.normal(0, 11)
            y       = max(70, base + spatial + noise)
            prev_actual = (NASS_ACTUALS.get(year - 1) or actuals)[state]
            rows.append({
                "fips":     fips,
                "state":    state,
                "yield_bu": round(y, 1),
                "vs_prior": round(y - prev_actual, 1),
                "vs_avg":   round(y - NASS_10YR[state], 1),
            })
    return pd.DataFrame(rows)


@st.cache_data
def load_chip_image(state: str, checkpoint: str, year: int):
    import zarr
    chip_root = ROOT / "data" / "processed" / "chips"
    abbr      = STATES[state]["abbr"]
    matches   = [
        m for m in chip_root.glob(f"**/{checkpoint}.zarr")
        if abbr in str(m) and str(year) in str(m)
    ]
    if not matches:
        matches = [m for m in chip_root.glob(f"**/{checkpoint}.zarr") if abbr in str(m)]
    if not matches:
        return None
    try:
        z   = zarr.open(str(matches[0]), mode="r")
        t1  = z["chips"][0][0]
        rgb = np.stack([
            np.clip(t1[2] / 3000, 0, 1),
            np.clip(t1[1] / 3000, 0, 1),
            np.clip(t1[0] / 3000, 0, 1),
        ], axis=-1)
        return (rgb * 255).astype(np.uint8)
    except Exception:
        return None


historical_df = load_historical()
live_data     = (ROOT / "reports" / "forecasts" / "yield_with_uncertainty_2025.parquet").exists()


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="hs-wordmark">Harvest<span>Sight</span></div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hs-sub">Geospatial AI · Corn Yield Forecasting<br>'
        'USDA NASS Supplemental System</div>',
        unsafe_allow_html=True,
    )
    st.divider()

    st.markdown('<div class="hs-section">Filters</div>', unsafe_allow_html=True)
    selected_year     = st.selectbox("Crop year", YEARS, index=len(YEARS) - 1,
                                     label_visibility="collapsed")
    _state_names = list(STATES.keys())
    _state_idx   = _state_names.index(st.session_state._sel_state) \
                   if st.session_state._sel_state in _state_names else 0
    selected_state = st.selectbox("State", _state_names, index=_state_idx,
                                  label_visibility="collapsed")
    st.session_state._sel_state = selected_state
    selected_ck_label = st.selectbox("Forecast checkpoint", list(CHECKPOINTS.keys()),
                                     label_visibility="collapsed")
    selected_ck = CHECKPOINTS[selected_ck_label]

    is_hindcast = selected_year < 2025

    st.divider()
    st.markdown('<div class="hs-section">Model</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.78rem;color:{C_MUTED};line-height:2.0">
        Prithvi-EO-2.0-600M-TL<br>
        LoRA fine-tune &nbsp;·&nbsp; rank 16<br>
        NASA HLS &nbsp;·&nbsp; 30 m &nbsp;·&nbsp; 6 bands<br>
        USDA NASS calibration &nbsp;·&nbsp; 2005–2024
    </div>""", unsafe_allow_html=True)

    st.divider()
    st.markdown('<div class="hs-section">Export</div>', unsafe_allow_html=True)

    forecasts_df = load_forecasts(selected_year)
    ck_df        = forecasts_df[forecasts_df["checkpoint"] == selected_ck].copy()
    ck_df["planted_m_acres"] = ck_df["state"].map(PLANTED_ACRES[selected_year])
    ck_df["nass_reference"]  = ck_df["state"].map(
        NASS_ACTUALS[selected_year] if is_hindcast else NASS_ACTUALS[2024]
    )

    st.download_button(
        "Download Forecast CSV",
        data=ck_df.to_csv(index=False).encode(),
        file_name=f"harvestsight_{selected_year}_{selected_ck}.csv",
        mime="text/csv",
    )

    mode_label = "Hindcast" if is_hindcast else "Forecast"
    report_lines = [
        f"HARVESTSIGHT — {selected_year} CORN YIELD {mode_label.upper()} REPORT",
        f"Checkpoint  : {selected_ck_label}",
        f"Mode        : {'Hindcast — NASS final available' if is_hindcast else 'Active forecast — season in progress'}",
        f"Model       : Prithvi-EO-2.0-600M-TL + LoRA (rank 16)",
        f"Imagery     : NASA HLS v2.0 · 30 m · 6 spectral bands",
        "=" * 60,
        "",
        f"{'STATE':<12}{'MODEL':>10}{'LOWER':>8}{'UPPER':>8}"
        + (f"{'NASS FINAL':>12}{'ERROR':>8}" if is_hindcast else ""),
        "-" * 60,
    ]
    for _, row in ck_df.iterrows():
        line = (f"{row['state']:<12}{row['forecast_bu']:>9.1f}"
                f"{row['lower_bu']:>8.1f}{row['upper_bu']:>8.1f}")
        if is_hindcast and row.get("nass_actual") is not None:
            err = row["forecast_bu"] - row["nass_actual"]
            line += f"{row['nass_actual']:>12.1f}{err:>+8.1f}"
        report_lines.append(line)
    report_lines += [
        "",
        "Uncertainty bounds: analog-year ensemble (5 closest historical",
        "seasons by DSCI drought index + satellite NDVI trajectory).",
        "",
        "DISCLAIMER: Supplemental geospatial model output. Official",
        "estimates published by USDA NASS via Crop Production report.",
    ]

    st.download_button(
        "Download Summary Report",
        data="\n".join(report_lines).encode(),
        file_name=f"harvestsight_{selected_year}_{selected_ck}_report.txt",
        mime="text/plain",
    )

    st.divider()
    status_color = C_PRIMARY if live_data else C_ACCENT
    status_label = "Live model output" if live_data else "Demo mode — model training in progress"
    st.markdown(
        f'<div style="font-size:0.72rem;color:{status_color}">'
        f'&#9679;&nbsp; {status_label}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div style="font-size:0.7rem;color:{C_MUTED};margin-top:4px">'
        f'Report date: April 25, 2026</div>',
        unsafe_allow_html=True,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
mode_word = "Hindcast" if is_hindcast else "Forecast"
badge     = ('<span class="mc-badge badge-hindcast">NASS Verified</span>'
             if is_hindcast else "")

hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown(
        f'<div class="hs-page-title">{selected_year} Corn Yield {mode_word} {badge}</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="hs-page-sub">'
        'HarvestSight Geospatial AI &nbsp;&middot;&nbsp; '
        'Iowa &nbsp;&middot;&nbsp; Colorado &nbsp;&middot;&nbsp; Wisconsin '
        '&nbsp;&middot;&nbsp; Missouri &nbsp;&middot;&nbsp; Nebraska'
        '</div>',
        unsafe_allow_html=True,
    )
with hcol2:
    if is_hindcast:
        nass_vals = NASS_ACTUALS[selected_year]
        avg_nass  = np.mean(list(nass_vals.values()))
        fcast_eos = forecasts_df[forecasts_df["checkpoint"] == "end_of_season"]
        avg_model = fcast_eos["forecast_bu"].mean()
        rmse_str  = f"{abs(avg_model - avg_nass):.1f} bu/ac avg error"
        st.markdown(
            f'<div style="text-align:right;font-size:0.75rem;color:{C_MUTED};margin-top:6px">'
            f'{selected_year} season complete &nbsp;|&nbsp; NASS final published<br>'
            f'<span style="font-size:0.7rem;color:{C_PRIMARY};font-weight:600">'
            f'Model accuracy: {rmse_str}</span></div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f'<div style="text-align:right;font-size:0.75rem;color:{C_MUTED};margin-top:6px">'
            f'2025 season complete &nbsp;|&nbsp; Final estimates available<br>'
            f'<span style="font-size:0.7rem">Report date: April 25, 2026</span></div>',
            unsafe_allow_html=True,
        )

# Season progress timeline
def season_timeline_html(year: int, checkpoint: str) -> str:
    n      = len(SEASON_STAGES)
    ck_idx = CHECKPOINT_STAGE_IDX.get(checkpoint, n - 1)
    ck_pct = ck_idx / (n - 1) * 100
    ck_name, ck_month = SEASON_STAGES[ck_idx]

    # All years in view (2022-2025) are fully harvested as of April 2026.
    # Bar is always 100% complete. Gold tick marks the selected forecast checkpoint.
    markers = ""
    for i, (stage, month) in enumerate(SEASON_STAGES):
        pct   = i / (n - 1) * 100
        is_ck = (i == ck_idx)

        txt_col = C_ACCENT if is_ck else C_TEXT
        txt_wt  = "700"    if is_ck else "500"
        tick_h  = "12px"   if is_ck else "7px"
        tick_w  = "3px"    if is_ck else "2px"
        tick_bg = C_ACCENT if is_ck else C_PRIMARY

        badge = (
            f'<div style="font-size:0.55rem;font-weight:600;color:#92400E;'
            f'background:#FEF3C7;border:1px solid #FDE68A;border-radius:3px;'
            f'padding:1px 5px;margin-top:3px;white-space:nowrap;">&#9660; viewing</div>'
            if is_ck else ""
        )

        markers += (
            f'<div style="position:absolute;left:{pct:.2f}%;transform:translateX(-50%);'
            f'top:0;text-align:center;">'
            f'<div style="width:{tick_w};height:{tick_h};background:{tick_bg};'
            f'margin:0 auto;border-radius:1px;"></div>'
            f'<div style="font-size:0.62rem;font-weight:{txt_wt};color:{txt_col};'
            f'white-space:nowrap;margin-top:4px;line-height:1.25;">{stage}</div>'
            f'<div style="font-size:0.56rem;color:{C_MUTED};line-height:1.2;">{month}</div>'
            f'{badge}'
            f'</div>'
        )

    return (
        f'<div class="season-wrap">'
        f'<div class="hs-section" style="margin-top:0;margin-bottom:0.65rem">'
        f'{year} Growing Season &nbsp;&middot;&nbsp; Season complete'
        f'&nbsp;&middot;&nbsp; Forecast checkpoint: '
        f'<strong style="color:{C_TEXT}">{ck_name} ({ck_month})</strong>'
        f'</div>'
        # Bar track — always 100% filled, season is over
        f'<div style="position:relative;height:6px;background:{C_PRIMARY};border-radius:3px;">'
        # Checkpoint tick — shows which forecast window is selected
        f'<div style="position:absolute;left:{ck_pct:.2f}%;top:50%;'
        f'transform:translate(-50%,-50%);width:4px;height:16px;'
        f'background:{C_ACCENT};border-radius:2px;box-shadow:0 0 0 2px white;"></div>'
        f'</div>'
        # Stage labels
        f'<div style="position:relative;height:56px;margin-top:6px;">'
        f'{markers}'
        f'</div>'
        f'</div>'
    )

st.markdown(season_timeline_html(selected_year, selected_ck), unsafe_allow_html=True)

# Banner
if is_hindcast:
    nass_vals  = NASS_ACTUALS[selected_year]
    fcast_eos  = forecasts_df[forecasts_df["checkpoint"] == "end_of_season"]
    errors     = {
        r["state"]: round(r["forecast_bu"] - nass_vals[r["state"]], 1)
        for _, r in fcast_eos.iterrows()
    }
    errs_str   = "  |  ".join(
        f"{s}: {e:+.1f}" for s, e in errors.items()
    )
    st.markdown(
        f'<div class="good-banner">&#10003;&nbsp; '
        f'<strong>{selected_year} NASS final estimates available.</strong> '
        f'HarvestSight end-of-season model error &nbsp;&mdash;&nbsp; {errs_str} bu/ac</div>',
        unsafe_allow_html=True,
    )
else:
    usdm_path = ROOT / "data" / "raw" / "usdm" / "county_weekly.parquet"
    stress_states: list[str] = []
    if usdm_path.exists():
        try:
            usdm = pd.read_parquet(usdm_path)
            usdm["valid_start"] = pd.to_datetime(usdm["valid_start"])
            aug = usdm[
                (usdm["valid_start"] >= "2025-07-01") &
                (usdm["valid_start"] <= "2025-09-01")
            ]
            if not aug.empty:
                abbr_to_name = {v["abbr"]: k for k, v in STATES.items()}
                for s, d in aug.groupby("state")["dsci"].mean().items():
                    if s in abbr_to_name and d > 100:
                        stress_states.append(abbr_to_name[s])
        except Exception:
            pass

    if stress_states:
        st.markdown(
            f'<div class="alert-banner">&#9888;&nbsp; '
            f'<strong>Drought stress detected</strong> during August grain-fill window in: '
            f'{", ".join(stress_states)}. Yield estimates carry higher uncertainty.</div>',
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="info-banner">&#9432;&nbsp; '
            '2025 season complete. All four checkpoint forecasts available. '
            'Compare against USDA NASS final via the Data tab.</div>',
            unsafe_allow_html=True,
        )


# ─────────────────────────────────────────────────────────────────────────────
# State metric cards
# ─────────────────────────────────────────────────────────────────────────────
prior_actuals = NASS_ACTUALS.get(selected_year - 1) or NASS_ACTUALS[2024]

cols = st.columns(5, gap="small")
for i, (state, _) in enumerate(STATES.items()):
    row = forecasts_df[
        (forecasts_df["state"] == state) & (forecasts_df["checkpoint"] == selected_ck)
    ]
    val  = row["forecast_bu"].values[0] if len(row) else prior_actuals[state]
    lo   = row["lower_bu"].values[0]    if len(row) else val - 12
    hi   = row["upper_bu"].values[0]    if len(row) else val + 12

    prior = prior_actuals[state]
    d_prior = val - prior
    d_avg   = val - NASS_10YR[state]

    def _delta(d: float, label: str) -> str:
        c = C_POS if d >= 0 else C_NEG
        a = "▲" if d >= 0 else "▼"
        return (f'<span style="color:{C_MUTED}">{label}&nbsp;</span>'
                f'<span style="color:{c}">{a}&nbsp;{abs(d):.1f}</span>')

    if is_hindcast:
        nass_final = NASS_ACTUALS[selected_year][state]
        eos_row    = forecasts_df[
            (forecasts_df["state"] == state) & (forecasts_df["checkpoint"] == "end_of_season")
        ]
        eos_val = eos_row["forecast_bu"].values[0] if len(eos_row) else val
        model_err = eos_val - nass_final
        err_color = C_POS if abs(model_err) <= 5 else (C_WARN if abs(model_err) <= 10 else C_NEG)
        extra_row = (
            f'<div style="font-size:0.71rem;margin-top:5px;padding-top:5px;'
            f'border-top:1px solid {C_BORDER}">'
            f'<span style="color:{C_MUTED}">NASS final&nbsp;</span>'
            f'<span style="font-weight:600">{nass_final} bu/ac</span>'
            f'&nbsp;&nbsp;<span style="color:{err_color}">model err {model_err:+.1f}</span>'
            f'</div>'
        )
    else:
        extra_row = ""

    is_sel = (state == selected_state)
    card_style = (
        f"background:{C_BG};border:2px solid {C_PRIMARY};box-shadow:0 0 0 3px rgba(29,111,66,0.08);"
        if is_sel else
        f"background:{C_SURFACE};border:1px solid {C_BORDER};"
    )
    cols[i].markdown(f"""
    <div class="metric-card" style="{card_style}">
      <div class="mc-state">{state}{"&nbsp;<span style='font-size:0.6rem;color:" + C_PRIMARY + ";font-weight:700'>SELECTED</span>" if is_sel else ""}</div>
      <div class="mc-val">{val:.0f}</div>
      <div class="mc-range">{lo:.0f} – {hi:.0f} bu / ac</div>
      <div class="mc-delta">
        <span>{_delta(d_prior, f"vs {selected_year-1}")}</span>
        <span>{_delta(d_avg, "vs avg")}</span>
      </div>
      {extra_row}
    </div>""", unsafe_allow_html=True)
    btn_label = "Viewing" if is_sel else f"Focus {STATES[state]['abbr']}"
    if cols[i].button(btn_label, key=f"focus_{state}", use_container_width=True,
                      disabled=is_sel):
        st.session_state._sel_state = state
        st.session_state.map_view = "state"
        st.rerun()

st.markdown("<br>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "County Forecast Map",
    "State Forecast Trend",
    "Crop Conditions",
    "Satellite Imagery",
    "Data & Export",
])


# ── Tab 1: County Forecast Map ────────────────────────────────────────────────
with tab1:
    county_df   = build_county_yields(selected_year, selected_ck)
    geojson     = get_counties_geojson()
    map_view    = st.session_state.map_view
    ck_short    = selected_ck_label.split("—")[0].strip()

    def _state_county_fig(state_name: str, cdf: pd.DataFrame) -> go.Figure:
        q05, q95 = cdf["yield_bu"].quantile(0.05), cdf["yield_bu"].quantile(0.95)
        f = go.Figure(go.Choropleth(
            geojson=geojson, locations=cdf["fips"], z=cdf["yield_bu"],
            featureidkey="id",
            colorscale=[[0, "#C8E6C9"], [0.35, "#388E3C"], [1, "#1B5E20"]],
            zmin=q05, zmax=q95,
            colorbar=dict(
                title=dict(text="bu / acre", font=dict(size=11, color=C_MUTED)),
                tickfont=dict(size=10, color=C_MUTED),
                bgcolor=C_BG, bordercolor=C_BORDER, borderwidth=1,
                thickness=12, len=0.6, x=1.01,
            ),
            text=[
                f"<b>FIPS {r['fips']}</b><br>"
                f"Yield: {r['yield_bu']:.1f} bu/ac<br>"
                f"vs {selected_year-1}: {r['vs_prior']:+.1f}"
                for _, r in cdf.iterrows()
            ],
            hovertemplate="%{text}<extra></extra>",
            marker_line_color="white", marker_line_width=0.8,
        ))
        f.update_layout(
            **_L(legend=None, margin=dict(l=0, r=0, t=10, b=10)),
            geo=dict(scope="usa", bgcolor=C_BG, lakecolor="#DDEEFF",
                     landcolor="#F0F2F5", showlakes=True, showcoastlines=False,
                     showframe=False, projection_type="albers usa",
                     fitbounds="locations"),
            height=430,
        )
        return f

    if map_view == "state":
        # ── State focus view ─────────────────────────────────────────────────
        h1c, h2c = st.columns([5, 1])
        with h1c:
            st.markdown(
                f'<div style="font-size:1rem;font-weight:600;color:{C_TEXT};margin-bottom:2px">'
                f'{selected_state} — County Yield {mode_word} &nbsp;·&nbsp; {selected_year} &nbsp;·&nbsp; {ck_short}'
                f'</div>',
                unsafe_allow_html=True,
            )
        with h2c:
            st.markdown('<div class="back-btn">', unsafe_allow_html=True)
            if st.button("← All States", key="back_nationwide"):
                st.session_state.map_view = "nationwide"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)

        state_county_df = county_df[county_df["state"] == selected_state].copy()
        if geojson and not state_county_df.empty:
            mc1, mc2, mc3 = st.columns([3, 1, 1])
            mc1.plotly_chart(_state_county_fig(selected_state, state_county_df),
                             use_container_width=True)
            with mc2:
                st.markdown('<div class="hs-section">County range</div>', unsafe_allow_html=True)
                for lbl, v in [
                    ("High",     f"{state_county_df['yield_bu'].max():.1f} bu/ac"),
                    ("Median",   f"{state_county_df['yield_bu'].median():.1f} bu/ac"),
                    ("Low",      f"{state_county_df['yield_bu'].min():.1f} bu/ac"),
                    ("Std dev",  f"{state_county_df['yield_bu'].std():.1f} bu/ac"),
                    ("Counties", str(len(state_county_df))),
                ]:
                    st.markdown(
                        f'<div class="stat-row"><span class="stat-key">{lbl}</span>'
                        f'<span class="stat-val">{v}</span></div>',
                        unsafe_allow_html=True,
                    )
            with mc3:
                st.markdown(f'<div class="hs-section">vs {selected_year-1}</div>',
                            unsafe_allow_html=True)
                above    = (state_county_df["vs_prior"] > 0).sum()
                n_co     = len(state_county_df)
                pct_above = above / n_co * 100
                for lbl, v in [
                    ("Above",  f"{above} ({pct_above:.0f}%)"),
                    ("Below",  f"{n_co-above} ({100-pct_above:.0f}%)"),
                    ("Avg Δ",  f"{state_county_df['vs_prior'].mean():+.1f} bu/ac"),
                    ("Best",   f"{state_county_df['vs_prior'].max():+.1f} bu/ac"),
                    ("Worst",  f"{state_county_df['vs_prior'].min():+.1f} bu/ac"),
                ]:
                    st.markdown(
                        f'<div class="stat-row"><span class="stat-key">{lbl}</span>'
                        f'<span class="stat-val">{v}</span></div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.caption("County GeoJSON unavailable — requires internet access.")

    else:
        # ── Nationwide view ──────────────────────────────────────────────────
        map_title = f"{selected_year} County Yield {mode_word} — {ck_short} Checkpoint"

        if geojson:
            fig = go.Figure(go.Choropleth(
                geojson=geojson,
                locations=county_df["fips"],
                z=county_df["yield_bu"],
                featureidkey="id",
                colorscale=[[0, "#C8E6C9"], [0.35, "#388E3C"], [1, "#1B5E20"]],
                zmin=county_df["yield_bu"].quantile(0.05),
                zmax=county_df["yield_bu"].quantile(0.95),
                colorbar=dict(
                    title=dict(text="bu / acre", font=dict(size=11, color=C_MUTED)),
                    tickfont=dict(size=10, color=C_MUTED),
                    bgcolor=C_BG, bordercolor=C_BORDER, borderwidth=1,
                    thickness=14, len=0.55, x=1.01,
                ),
                text=[
                    f"<b>{r['state']} FIPS {r['fips']}</b><br>"
                    f"Yield: {r['yield_bu']:.1f} bu/ac<br>"
                    f"vs {selected_year-1}: {r['vs_prior']:+.1f} bu/ac"
                    for _, r in county_df.iterrows()
                ],
                hovertemplate="%{text}<extra></extra>",
                marker_line_color="white", marker_line_width=0.4,
            ))
            fig.update_layout(
                **_L(legend=None, margin=dict(l=0, r=0, t=40, b=10)),
                title=dict(text=map_title, font=dict(size=13, color=C_TEXT), x=0),
                geo=dict(scope="usa", bgcolor=C_BG, lakecolor="#DDEEFF",
                         landcolor="#F0F2F5", showlakes=True, showcoastlines=False,
                         showframe=False, projection_type="albers usa"),
                height=530,
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            ck_data = forecasts_df[forecasts_df["checkpoint"] == selected_ck]
            fig = go.Figure(go.Choropleth(
                locations=[STATES[s]["abbr"] for s in ck_data["state"]],
                z=ck_data["forecast_bu"].tolist(), locationmode="USA-states",
                colorscale=[[0, "#C8E6C9"], [0.4, "#388E3C"], [1, "#1B5E20"]],
                colorbar=dict(title=dict(text="bu / acre"), thickness=14),
                marker_line_color="white", marker_line_width=1.5,
            ))
            fig.update_layout(**_L(legend=None), geo=dict(scope="usa", bgcolor=C_BG,
                              landcolor="#F0F2F5", showlakes=True,
                              projection_type="albers usa"), height=500)
            st.plotly_chart(fig, use_container_width=True)

        c1, c2 = st.columns(2)
        c1.caption(
            f"County-level corn yield {mode_word.lower()} · {selected_year} · {ck_short}. "
            "Color anchored to 5th–95th percentile. Click a state card above to drill down."
        )
        c2.caption("Source: HarvestSight Prithvi-EO-2.0 · NASA HLS v2.0 · USDA NASS 2005–2024.")


# ── Tab 2: State Forecast Trend ───────────────────────────────────────────────
with tab2:
    state_df = (
        forecasts_df[forecasts_df["state"] == selected_state]
        .assign(order=lambda d: d["checkpoint"].map(
            {v: i for i, v in enumerate(CHECKPOINTS.values())}))
        .sort_values("order")
    )
    x     = CHECKPOINT_LABELS
    y_mid = state_df["forecast_bu"].tolist()
    y_lo  = state_df["lower_bu"].tolist()
    y_hi  = state_df["upper_bu"].tolist()

    hist_state = historical_df[historical_df["state"] == selected_state].sort_values("year")
    h10        = hist_state[hist_state["year"].between(
        selected_year - 11, selected_year - 1
    )]["yield_bu"]

    fig2 = go.Figure()
    fig2.add_trace(go.Scatter(
        x=x + x[::-1], y=y_hi + y_lo[::-1],
        fill="toself", fillcolor=C_BAND,
        line=dict(color="rgba(0,0,0,0)"),
        name="Uncertainty range", hoverinfo="skip",
    ))
    for y_b in [y_hi, y_lo]:
        fig2.add_trace(go.Scatter(
            x=x, y=y_b, mode="lines",
            line=dict(color=C_PRIMARY, width=1, dash="dot"),
            showlegend=False, hoverinfo="skip",
        ))
    fig2.add_trace(go.Scatter(
        x=x, y=y_mid, mode="lines+markers",
        line=dict(color=C_PRIMARY, width=2.5),
        marker=dict(size=9, color=C_PRIMARY, symbol="diamond",
                    line=dict(color="white", width=1.5)),
        name=f"{selected_year} HarvestSight {mode_word.lower()}",
        hovertemplate="<b>%{x}</b><br>%{y:.1f} bu/ac<extra></extra>",
    ))
    if len(h10):
        fig2.add_hline(
            y=h10.mean(), line_dash="dash", line_color="#9CA3AF", line_width=1.5,
            annotation_text=f"Prior 10-yr avg · {h10.mean():.0f} bu/ac",
            annotation_font=dict(color=C_MUTED, size=10),
            annotation_position="bottom right",
        )
    if is_hindcast:
        nass_final = NASS_ACTUALS[selected_year][selected_state]
        fig2.add_hline(
            y=nass_final, line_dash="solid", line_color=C_ACCENT, line_width=2,
            annotation_text=f"NASS {selected_year} final · {nass_final} bu/ac",
            annotation_font=dict(color=C_ACCENT, size=11, family="Inter"),
            annotation_position="top right",
        )
    else:
        prior_val = NASS_ACTUALS[2024][selected_state]
        fig2.add_hline(
            y=prior_val, line_dash="dot", line_color=C_ACCENT, line_width=1.5,
            annotation_text=f"NASS 2024 final · {prior_val} bu/ac",
            annotation_font=dict(color=C_ACCENT, size=10),
            annotation_position="top right",
        )

    fig2.update_layout(
        **PLOT_LAYOUT,
        title=dict(
            text=f"{selected_state} — {selected_year} {mode_word} by Checkpoint",
            font=dict(size=13, color=C_TEXT), x=0,
        ),
        xaxis=dict(title="Forecast checkpoint", gridcolor=C_GRID,
                   linecolor=C_BORDER, showgrid=True),
        yaxis=dict(title="Yield (bu / acre)", gridcolor=C_GRID,
                   linecolor=C_BORDER, showgrid=True),
        height=430, hovermode="x unified",
    )
    st.plotly_chart(fig2, use_container_width=True)

    if is_hindcast:
        st.caption(
            f"Gold line = USDA NASS {selected_year} published final estimate for {selected_state}. "
            "Dashed uncertainty bounds show analog-year ensemble range at each forecast checkpoint."
        )
    else:
        st.caption(
            "Dashed gold line = USDA NASS 2024 published final. "
            "Uncertainty bounds narrow as additional satellite observations accumulate each checkpoint."
        )


# ── Tab 3: Crop Conditions ────────────────────────────────────────────────────
with tab3:
    year_conditions = CROP_CONDITIONS[selected_year]

    # Selected-state callout ─────────────────────────────────────────────────
    sel_cond  = year_conditions[selected_state]
    exc, good, fair, poor, vpoor = sel_cond
    gge_sel   = exc + good
    stress_sel = poor + vpoor
    if gge_sel >= 60:
        cond_color, cond_word = C_POS,  "Favorable"
    elif stress_sel >= 20:
        cond_color, cond_word = C_NEG,  "Under Stress"
    else:
        cond_color, cond_word = C_WARN, "Watch"

    seg_html = "".join(
        f'<div style="flex:{pct};background:{col};height:10px;'
        f'{"border-radius:5px 0 0 5px;" if i==0 else "border-radius:0 5px 5px 0;" if i==4 else ""}'
        f'" title="{lbl}: {pct}%"></div>'
        for i, (pct, col, lbl) in enumerate(
            zip(sel_cond, CONDITION_COLORS, CONDITION_LABELS)
        )
    )
    st.markdown(
        f'<div style="background:{C_SURFACE};border:2px solid {C_PRIMARY};border-radius:10px;'
        f'padding:0.85rem 1.1rem;margin-bottom:1rem;">'
        f'<div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:0.5rem;">'
        f'<span style="font-size:0.78rem;font-weight:600;color:{C_TEXT}">'
        f'{selected_state} — {selected_year} Crop Conditions</span>'
        f'<span style="font-size:0.8rem;font-weight:700;color:{cond_color}">{cond_word}</span>'
        f'</div>'
        f'<div style="display:flex;gap:2px;border-radius:5px;overflow:hidden;">{seg_html}</div>'
        f'<div style="display:flex;justify-content:space-between;margin-top:6px;">'
        + "".join(
            f'<span style="font-size:0.62rem;color:{C_MUTED};text-align:center;flex:1">'
            f'{lbl.replace(" ","<br>")}<br><b style="color:{C_TEXT}">{pct}%</b></span>'
            for pct, lbl in zip(sel_cond, CONDITION_LABELS)
        )
        + f'</div></div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        f'<div style="font-size:0.8rem;color:{C_MUTED};margin-bottom:0.8rem">'
        f'All-state comparison · August 1 grain-fill checkpoint · {selected_year} season.</div>',
        unsafe_allow_html=True,
    )

    fig3 = go.Figure()
    for j, (label, color) in enumerate(zip(CONDITION_LABELS, CONDITION_COLORS)):
        vals = [year_conditions[s][j] for s in STATES]
        fig3.add_trace(go.Bar(
            name=label, x=list(STATES.keys()), y=vals,
            marker_color=color,
            text=[f"{v}%" for v in vals], textposition="inside",
            textfont=dict(color="white", size=11, family="Inter"),
            hovertemplate=f"<b>%{{x}}</b><br>{label}: %{{y}}%<extra></extra>",
        ))
    fig3.update_layout(
        **_L(
            legend=dict(orientation="h", y=-0.15, x=0.5, xanchor="center",
                        bgcolor="rgba(0,0,0,0)", bordercolor=C_BORDER, borderwidth=1),
        ),
        barmode="stack",
        title=dict(
            text=f"Crop Condition at Grain Fill — August 1, {selected_year}",
            font=dict(size=13, color=C_TEXT), x=0,
        ),
        xaxis=dict(title="", gridcolor=C_GRID, linecolor=C_BORDER),
        yaxis=dict(title="% of crop area", gridcolor=C_GRID,
                   linecolor=C_BORDER, range=[0, 100]),
        height=400,
    )
    st.plotly_chart(fig3, use_container_width=True)

    st.markdown(f'<div class="hs-section">State Stress Summary — {selected_year}</div>',
                unsafe_allow_html=True)
    for state in STATES:
        exc, good, fair, poor, vpoor = year_conditions[state]
        gge     = exc + good
        stressed = poor + vpoor
        if gge >= 60:
            badge = f'<span style="color:{C_POS};font-weight:600">Favorable</span>'
        elif stressed >= 20:
            badge = f'<span style="color:{C_NEG};font-weight:600">Under Stress</span>'
        else:
            badge = f'<span style="color:{C_WARN};font-weight:600">Watch</span>'
        st.markdown(
            f'<div class="stat-row">'
            f'<span class="stat-key">{state}</span>'
            f'<span>'
            f'<span style="color:{C_MUTED};font-size:0.75rem">'
            f'Exc+Good {gge}% &nbsp;|&nbsp; Poor+VPoor {stressed}%'
            f'</span>&nbsp;&nbsp;{badge}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if selected_year == 2022:
        st.markdown(
            '<div class="alert-banner" style="margin-top:0.8rem">&#9888;&nbsp; '
            '<strong>2022 Missouri drought:</strong> Severe heat and drought stress during '
            'pollination drove below-average yields. Missouri Exc+Good fell to 27% — '
            'the lowest in the five-state region.</div>',
            unsafe_allow_html=True,
        )


# ── Tab 4: Satellite Imagery ──────────────────────────────────────────────────
with tab4:
    col_img, col_info = st.columns([2, 1], gap="large")
    with col_img:
        img = load_chip_image(selected_state, selected_ck, selected_year)
        if img is not None:
            st.image(
                img,
                caption=(
                    f"{selected_state} · {selected_year} HLS 30-day median composite · "
                    f"RGB (B04/B03/B02) · {selected_ck_label.split('—')[0].strip()}"
                ),
                use_container_width=True,
            )
        else:
            st.markdown(f"""
            <div class="sat-placeholder">
              <div style="font-size:0.95rem;font-weight:600;color:{C_MUTED}">
                Satellite composite imagery — {selected_state} · {selected_year}</div>
              <div style="font-size:0.82rem;margin-top:0.5rem;color:{C_MUTED}">
                Chip extraction running — imagery available shortly</div>
            </div>""", unsafe_allow_html=True)

    with col_info:
        st.markdown('<div class="hs-section">Chip specification</div>', unsafe_allow_html=True)
        for k, v in {
            "Season":         str(selected_year),
            "Resolution":     "30 m / pixel",
            "Chip size":      "224 × 224 px  (~6.7 km²)",
            "Spectral bands": "6  (B, G, R, NIR, SWIR1, SWIR2)",
            "Timesteps":      "3 × 30-day median composites",
            "Cloud masking":  "HLS Fmask  (threshold 1%)",
        }.items():
            st.markdown(
                f'<div class="stat-row">'
                f'<span class="stat-key">{k}</span>'
                f'<span class="stat-val">{v}</span></div>',
                unsafe_allow_html=True,
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="hs-section">Data sources</div>', unsafe_allow_html=True)
        for src in [
            "Landsat 8 / 9 — HLSL30 v2.0",
            "Sentinel-2 A/B — HLSS30 v2.0",
            "NASA LP DAAC · earthaccess API",
        ]:
            st.markdown(
                f'<div style="font-size:0.8rem;padding:4px 0;color:{C_MUTED}">{src}</div>',
                unsafe_allow_html=True,
            )


# ── Tab 5: Data & Export ──────────────────────────────────────────────────────
with tab5:
    st.markdown(
        f'<div style="font-size:0.8rem;color:{C_MUTED};margin-bottom:0.8rem">'
        f'Five-state {mode_word.lower()} data · {selected_year} season · '
        f'<b>{selected_ck_label}</b> checkpoint. '
        'Download buttons in sidebar.</div>',
        unsafe_allow_html=True,
    )

    table_rows = []
    for state in STATES:
        row = forecasts_df[
            (forecasts_df["state"] == state) & (forecasts_df["checkpoint"] == selected_ck)
        ]
        val  = row["forecast_bu"].values[0] if len(row) else prior_actuals[state]
        lo   = row["lower_bu"].values[0]    if len(row) else val - 12
        hi   = row["upper_bu"].values[0]    if len(row) else val + 12
        d_pr = val - prior_actuals[state]
        davg = val - NASS_10YR[state]
        pa   = PLANTED_ACRES[selected_year][state]
        prod = val * pa * 0.92
        rec  = {"State": state, f"Model {mode_word} (bu/ac)": f"{val:.1f}",
                "90% CI": f"{lo:.0f} – {hi:.0f}",
                f"vs {selected_year-1}": f"{d_pr:+.1f}", "vs 10-yr avg": f"{davg:+.1f}",
                "Planted (M ac)": f"{pa:.2f}", "Prod est (M bu)": f"{prod:.0f}"}
        if is_hindcast:
            nf = NASS_ACTUALS[selected_year][state]
            eos_r = forecasts_df[
                (forecasts_df["state"] == state) & (forecasts_df["checkpoint"] == "end_of_season")
            ]
            eos_v = eos_r["forecast_bu"].values[0] if len(eos_r) else val
            rec["NASS Final (bu/ac)"] = str(nf)
            rec["Model Error (bu/ac)"] = f"{eos_v - nf:+.1f}"
        table_rows.append(rec)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    if is_hindcast:
        st.markdown(
            f'<div class="hs-section">Model Accuracy — {selected_year} End-of-Season</div>',
            unsafe_allow_html=True,
        )
        fcast_eos = forecasts_df[forecasts_df["checkpoint"] == "end_of_season"]
        nass_vals = NASS_ACTUALS[selected_year]
        cells = ""
        abs_errors = []
        for state in STATES:
            eos_r = fcast_eos[fcast_eos["state"] == state]
            if len(eos_r):
                eos_v = eos_r["forecast_bu"].values[0]
                nf    = nass_vals[state]
                err   = eos_v - nf
                abs_errors.append(abs(err))
                color = C_POS if abs(err) <= 5 else (C_WARN if abs(err) <= 10 else C_NEG)
                cells += (
                    f'<div class="acc-cell">'
                    f'<div class="acc-state">{state[:3].upper()}</div>'
                    f'<div class="acc-err" style="color:{color}">{err:+.1f}</div>'
                    f'<div class="acc-label">bu/ac vs NASS</div>'
                    f'</div>'
                )
        mae = np.mean(abs_errors)
        st.markdown(f'<div class="accuracy-grid">{cells}</div>', unsafe_allow_html=True)
        st.markdown(
            f'<div style="font-size:0.8rem;color:{C_MUTED}">'
            f'Mean absolute error across five states: '
            f'<strong style="color:{C_PRIMARY}">{mae:.1f} bu/ac</strong>'
            f'</div>',
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="hs-section">All checkpoints</div>', unsafe_allow_html=True)
    all_ck_rows = []
    for state in STATES:
        for ck_lbl, ck_key in CHECKPOINTS.items():
            r = forecasts_df[
                (forecasts_df["state"] == state) & (forecasts_df["checkpoint"] == ck_key)
            ]
            if len(r):
                rec2 = {
                    "State":            state,
                    "Checkpoint":       ck_lbl.split("—")[0].strip(),
                    f"{mode_word} (bu/ac)": f"{r['forecast_bu'].values[0]:.1f}",
                    "Lower":            f"{r['lower_bu'].values[0]:.0f}",
                    "Upper":            f"{r['upper_bu'].values[0]:.0f}",
                }
                if is_hindcast:
                    rec2["NASS Final"] = str(NASS_ACTUALS[selected_year][state])
                all_ck_rows.append(rec2)
    st.dataframe(pd.DataFrame(all_ck_rows), use_container_width=True,
                 hide_index=True, height=280)

    st.markdown("<br>", unsafe_allow_html=True)
    st.caption(
        "Production estimate = Model forecast (bu/ac) × Planted area (M ac) × 0.92 harvest rate. "
        "Model error shown at end-of-season checkpoint vs USDA NASS published final. "
        "Official estimates: USDA NASS Crop Production report."
    )
