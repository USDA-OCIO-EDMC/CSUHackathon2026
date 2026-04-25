"""
APHIS Fruit Fly Pathway Risk Monitor — dashboard + full report.

Run:
    python main.py prototype
    python main.py aws --bucket ffed-hackathon-mahanyas
"""

from __future__ import annotations

import csv
import html
import json
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
EXPORTS = ROOT / "data" / "exports"
OUT_FILE = EXPORTS / "dashboard.html"
REPORT_FILE = EXPORTS / "report.html"
REPORTS_FILE = EXPORTS / "ai_intelligence_reports.json"
API_SUMMARY_FILE = EXPORTS / "api" / "dashboard_summary.json"
SEASONAL_EVIDENCE_FILE = EXPORTS / "api" / "seasonal_evidence.json"
SEASONAL_EVIDENCE_SUMMARY_FILE = EXPORTS / "api" / "seasonal_evidence_summary.json"

MONTH_NAMES = ["", "Jan", "Feb", "Mar", "Apr", "May", "Jun",
               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


SOURCE_LINKS = {
    "aphis_fruit_flies": "https://www.aphis.usda.gov/plant-pests-diseases/fruit-flies",
    "aphis_host_lists": "https://www.aphis.usda.gov/plant-pests-diseases/fruit-flies/fruit-fly-host-lists",
    "aphis_medfly": "https://www.aphis.usda.gov/plant-pests-diseases/medfly",
    "aphis_emergency_funds": "https://www.aphis.usda.gov/news/agency-announcements/usda-secures-emergency-funds-combat-exotic-fruit-fly",
    "bts_t100": "https://www.transtats.bts.gov/DatabaseInfo.asp?QO_VQ=EEE",
    "census_porths": "https://api.census.gov/data/timeseries/intltrade/imports/porthsimport.html",
    "gbif": "https://www.gbif.org/occurrence/search?taxon_key=3520",
    "gbif_taxon": "https://www.gbif.org/species/3520",
    "ippc_reporting": "https://www.ippc.int/en/countries/reportingsystem-summary/all/",
    "bloomberg_fight": "https://news.bloomberglaw.com/bloomberg-government-news/inside-the-fight-against-a-billion-dollar-fruit-fly-invasion-1?context=search&index=5",
    "south_florida_loss": "https://www.growingproduce.com/vegetables/at-least-4-1-million-lost-in-south-florida-fruit-fly-outbreak/",
}


def _load_reports() -> list[dict]:
    if not REPORTS_FILE.exists():
        return []
    data = json.loads(REPORTS_FILE.read_text())
    return data.get("reports", data if isinstance(data, list) else [])


def _load_api_summary() -> dict:
    if not API_SUMMARY_FILE.exists():
        return {}
    return json.loads(API_SUMMARY_FILE.read_text(encoding="utf-8"))


def _load_seasonal_evidence() -> tuple[list[dict], dict]:
    evidence: list[dict] = []
    summary: dict = {}
    if SEASONAL_EVIDENCE_FILE.exists():
        evidence = json.loads(SEASONAL_EVIDENCE_FILE.read_text(encoding="utf-8"))
        for row in evidence:
            if isinstance(row.get("evidence_json"), str):
                row["evidence"] = json.loads(row.get("evidence_json") or "[]")
                row.pop("evidence_json", None)
            if isinstance(row.get("caveats_json"), str):
                row["caveats"] = json.loads(row.get("caveats_json") or "[]")
                row.pop("caveats_json", None)
    if SEASONAL_EVIDENCE_SUMMARY_FILE.exists():
        summary = json.loads(SEASONAL_EVIDENCE_SUMMARY_FILE.read_text(encoding="utf-8"))
    return evidence, summary


def _safe(value: object) -> str:
    return html.escape("" if value is None else str(value))


def _public_species_note(value: object) -> str:
    text = "" if value is None else str(value)
    if not text or "Species profile not mapped yet" in text:
        return "Not in curated watchlist; scored by pathway + GBIF overlap"
    return text


def _csv_records(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Seasonal narrative data
# ---------------------------------------------------------------------------
SEASONAL_NARRATIVES = [
    {
        "months": [7, 8, 9],
        "commodity": "fresh_fruit",
        "origin": "MEX",
        "headline": "Peak mango & stone fruit import window",
        "narrative": (
            "July–September is peak mango and stone fruit harvest in Sinaloa and Sonora. "
            "US domestic production is winding down, so import volume from Mexico surges. "
            "Anastrepha ludens (Mexican fruit fly) emergence peaks in summer heat. "
            "LAX receives approximately 40% of US–Mexico fresh fruit volume by sea cargo."
        ),
        "fly": "Anastrepha ludens",
        "action": "100% manifest review on SEA cargo. Secondary hold on any damaged or overripe fruit.",
    },
    {
        "months": [9, 10, 11, 12],
        "commodity": "fresh_fruit",
        "origin": "MEX",
        "headline": "Avocado import peak — US domestic off-season",
        "narrative": (
            "September–December: US avocado production is off-season. "
            "Mexico (Michoacán) supplies over 80% of US avocado imports during this window. "
            "Anastrepha ludens is active in Michoacán year-round but peaks in fall. "
            "California detection rates historically spike during this window."
        ),
        "fly": "Anastrepha ludens",
        "action": "Surge secondary inspection at LAX, SAN, CAL for SEA avocado shipments.",
    },
    {
        "months": [7, 8, 9],
        "commodity": "fresh_fruit",
        "origin": "THA",
        "headline": "Longan & rambutan harvest — B. dorsalis active",
        "narrative": (
            "July–September is Thailand's longan and rambutan harvest season. "
            "Bactrocera dorsalis (Oriental fruit fly) is active year-round in central Thailand "
            "but population peaks align with this harvest window. "
            "SEA cargo takes 18–22 days transit — larvae can survive in sealed containers."
        ),
        "fly": "Bactrocera dorsalis",
        "action": "Flag SEA fresh fruit manifests from THA. Check for mixed shipments with tropical fruit.",
    },
    {
        "months": [1, 2],
        "commodity": "cut_flowers",
        "origin": "ECU",
        "headline": "Valentine's season — 3–4× normal flower volume",
        "narrative": (
            "January–February: Ecuador's cut flower exports to MIA peak 3–4× normal volume "
            "for Valentine's Day. Flowers are low-risk hosts but mixed shipments with "
            "tropical fruit are the interception vector. "
            "Anastrepha fraterculus is present in Ecuador's coastal growing regions."
        ),
        "fly": "Anastrepha fraterculus",
        "action": "Inspect mixed-commodity shipments at MIA. Co-loaded fruit is the target.",
    },
    {
        "months": [5, 6, 7, 8],
        "commodity": "fresh_fruit",
        "origin": "COL",
        "headline": "Colombian mango season peaks at MIA",
        "narrative": (
            "May–August: Colombia's mango export season peaks. MIA is the primary entry point. "
            "Anastrepha obliqua (West Indian fruit fly) is established in Colombia's Caribbean coast. "
            "Florida's subtropical climate makes establishment risk high if interception fails."
        ),
        "fly": "Anastrepha obliqua",
        "action": "Prioritize secondary inspection at MIA for COL fresh fruit. Florida establishment risk is HIGH.",
    },
    {
        "months": [6, 7, 8],
        "commodity": "passenger_baggage",
        "origin": "PHL",
        "headline": "Summer travel peak — undeclared fruit in baggage",
        "narrative": (
            "June–August is peak travel season from the Philippines. "
            "Bactrocera dorsalis is widespread across the Philippine archipelago. "
            "Undeclared mango, guava, and tropical fruit in passenger baggage is the primary vector. "
            "LAX and HNL are the main entry points."
        ),
        "fly": "Bactrocera dorsalis",
        "action": "Increase baggage secondary inspection rate at LAX and HNL for PHL-origin flights.",
    },
    {
        "months": [7, 8, 9],
        "commodity": "fresh_fruit",
        "origin": "VNM",
        "headline": "Dragon fruit & lychee export peak",
        "narrative": (
            "July–September: Vietnam's dragon fruit and lychee export season. "
            "Bactrocera dorsalis and B. correcta are both active in the Mekong Delta. "
            "SEA cargo routes to LAX carry the highest volume during this window."
        ),
        "fly": "Bactrocera dorsalis",
        "action": "Flag VNM SEA cargo at LAX. Dragon fruit is a confirmed B. dorsalis host.",
    },
    {
        "months": [1, 2, 3],
        "commodity": "fresh_fruit",
        "origin": "BRA",
        "headline": "Southern Hemisphere summer — counterintuitive peak",
        "narrative": (
            "January–March is summer in Brazil. While US inspection staffing is in low-season mode, "
            "Brazilian mango and papaya exports peak. Anastrepha fraterculus and A. obliqua "
            "are both active. MIA receives the bulk of this volume — "
            "the window most likely to be under-inspected."
        ),
        "fly": "Anastrepha fraterculus",
        "action": "Maintain MIA inspection capacity in January. Brazilian summer creates a US winter inspection gap.",
    },
]

NARRATIVES_JSON = json.dumps(SEASONAL_NARRATIVES, ensure_ascii=True)

# Crop/agriculture exposure profiles for counties represented in the sample
# detection set. Values are planning proxies from public crop reports or county
# agriculture summaries, not a replacement for parcel-level crop maps.
AG_EXPOSURE_PROFILES = {
    "CA|Ventura": {
        "value_m": 2170.2,
        "value_label": "$2.17B gross agriculture value",
        "crops": ["strawberries", "lemons", "raspberries", "avocados", "peppers", "tomatoes"],
        "host_alignment": 0.95,
        "confidence": "HIGH",
        "source": "Ventura County 2023 Crop and Livestock Report",
        "url": "https://news.venturacounty.gov/en/20240724-crop-and-livestock-report-released/",
    },
    "CA|Riverside": {
        "value_m": 1540.3,
        "value_label": "$1.54B gross crop value",
        "crops": ["nursery stock", "dates", "avocados", "lemons", "bell peppers", "table grapes"],
        "host_alignment": 0.88,
        "confidence": "HIGH",
        "source": "Riverside County 2023 Agriculture Production Report",
        "url": "https://storymaps.arcgis.com/stories/d11869b86f094d86aa79529966ec33bc",
    },
    "FL|Miami-Dade": {
        "value_m": 838.0,
        "value_label": "$838M agriculture proxy",
        "crops": ["ornamentals", "winter vegetables", "tropical fruit", "avocados", "mangoes"],
        "host_alignment": 0.90,
        "confidence": "MEDIUM",
        "source": "Miami-Dade agriculture profile and 2017 USDA census reference",
        "url": "https://www.miamidade.gov/global/economy/agriculture.page",
    },
    "CA|San Bernardino": {
        "value_m": 381.2,
        "value_label": "$381M gross agriculture value",
        "crops": ["citrus", "avocados", "apples", "strawberries", "oriental vegetables"],
        "host_alignment": 0.70,
        "confidence": "HIGH",
        "source": "San Bernardino County 2023 Crop Report",
        "url": "https://awm.sbcounty.gov/wp-content/uploads/sites/84/2024/09/2023-AWM-CROP-REPORT-FINAL-090324.pdf",
    },
    "CA|San Diego": {
        "value_m": 292.6,
        "value_label": "$293M fruit and nut crop value",
        "crops": ["avocados", "citrus", "nursery", "cut flowers", "subtropical fruit"],
        "host_alignment": 0.84,
        "confidence": "MEDIUM",
        "source": "San Diego County 2023 crop statistics",
        "url": "https://ucanr.edu/sites/default/files/2025-08/AWM-2023-Crop-Statistics-and-Annual-Report-compressed-compressed.pdf",
    },
    "CA|Los Angeles": {
        "value_m": 200.5,
        "value_label": "$201M agriculture production archive proxy",
        "crops": ["nursery", "vegetables", "citrus", "avocados", "urban fruit trees"],
        "host_alignment": 0.66,
        "confidence": "MEDIUM",
        "source": "Los Angeles County crop report archive",
        "url": "https://lacfb.org/crop-reports-2/",
    },
    "CA|Orange": {
        "value_m": 75.7,
        "value_label": "$75.7M gross agriculture value",
        "crops": ["nursery", "citrus", "avocados", "vegetables"],
        "host_alignment": 0.62,
        "confidence": "HIGH",
        "source": "Orange County 2023 Crop Report",
        "url": "https://ocerac.ocpublicworks.com/sites/ocpwocerac/files/2024-12/2023%20Crop%20Report%20%28Final%29.pdf",
    },
    "TX|Hidalgo": {
        "value_m": 78.5,
        "value_label": "$78.5M irrigated crop value proxy",
        "crops": ["citrus", "vegetables", "sugarcane", "row crops"],
        "host_alignment": 0.72,
        "confidence": "MEDIUM",
        "source": "Texas Comptroller 2023 Hidalgo productivity values",
        "url": "https://comptroller.texas.gov/auto-data/PT2/PVS/2023P/1080000001B.php",
    },
    "CA|Santa Clara": {
        "value_m": 359.0,
        "value_label": "$359M agriculture value proxy",
        "crops": ["nursery", "mushrooms", "vegetables", "fruit and nuts"],
        "host_alignment": 0.55,
        "confidence": "MEDIUM",
        "source": "Santa Clara County crop report summary",
        "url": "https://news.santaclaracounty.gov/nursery-crops-and-mushrooms-remain-champs-santa-clara-county-agriculture",
    },
}

AG_EXPOSURE_JSON = json.dumps(AG_EXPOSURE_PROFILES, ensure_ascii=True)

# ---------------------------------------------------------------------------
# CSS
# ---------------------------------------------------------------------------
SHARED_CSS = """
  /* ── Theme tokens ── */
  :root {
    --bg:       #f0f4f0;
    --surface:  #ffffff;
    --surface2: #f7faf7;
    --border:   #d0d9d0;
    --border2:  #e4ece4;
    --text:     #1a2e1a;
    --text2:    #3d5a3d;
    --muted:    #6b8068;
    --primary:      #2d6a4f;
    --primary-lt:   #e4f0ea;
    --primary-mid:  #3d8b62;
    --high:     #b83232;
    --high-bg:  #fdf0f0;
    --high-bd:  #f0c0c0;
    --med:      #c06010;
    --med-bg:   #fef5ec;
    --med-bd:   #f0d0a0;
    --low:      #2e8050;
    --low-bg:   #edf8f2;
    --low-bd:   #b0dfc0;
    --air:      #1a6fa8;
    --air-bg:   #e8f2fa;
    --sea:      #0e8070;
    --sea-bg:   #e4f5f2;
    --land:     #6040a0;
    --land-bg:  #f0ebfa;
    --shadow:   0 1px 4px rgba(0,0,0,.10), 0 0 0 1px rgba(0,0,0,.04);
    --shadow-md:0 4px 16px rgba(0,0,0,.12);
    --r:        8px;
    --r-sm:     5px;
  }
  body.dark {
    --bg:       #0d1117;
    --surface:  #161b22;
    --surface2: #0d1117;
    --border:   #21262d;
    --border2:  #161b22;
    --text:     #e6edf3;
    --text2:    #8b949e;
    --muted:    #6e7681;
    --primary:      #3fb950;
    --primary-lt:   #122118;
    --primary-mid:  #2ea043;
    --high:     #f85149;
    --high-bg:  #1a0b0a;
    --high-bd:  #4a1515;
    --med:      #e3b341;
    --med-bg:   #1a140a;
    --med-bd:   #4a3510;
    --low:      #3fb950;
    --low-bg:   #0a1a10;
    --low-bd:   #154520;
    --air:      #58a6ff;
    --air-bg:   #0a1525;
    --sea:      #39d353;
    --sea-bg:   #0a1a10;
    --land:     #bc8cff;
    --land-bg:  #15103a;
    --shadow:   0 1px 4px rgba(0,0,0,.4);
    --shadow-md:0 4px 16px rgba(0,0,0,.5);
  }

  /* ── Reset ── */
  *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
  html, body { height: 100%; font-size: 14px; }
  body {
    background: var(--bg);
    color: var(--text);
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", "Inter", Helvetica, sans-serif;
    line-height: 1.5;
    transition: background .2s, color .2s;
  }
  a { color: var(--primary); text-decoration: none; }
  a:hover { text-decoration: underline; }
  button { font-family: inherit; }

  /* ── Typography helpers ── */
  .kpi-number { font-size: 38px; font-weight: 800; letter-spacing: -.5px; line-height: 1; }
  .kpi-label  { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-top: 4px; }
  .section-label { font-size: 10px; font-weight: 700; text-transform: uppercase; letter-spacing: .08em; color: var(--muted); margin-bottom: 6px; display: block; }
  .body-text { font-size: 13px; color: var(--text2); line-height: 1.6; }
  .fine { font-size: 11px; color: var(--muted); font-style: italic; }

  /* ── Badges ── */
  .badge {
    display: inline-block; padding: 2px 8px; border-radius: 999px;
    font-size: 11px; font-weight: 700; letter-spacing: .04em; text-transform: uppercase;
  }
  .bHIGH { background: var(--high-bg); color: var(--high); border: 1px solid var(--high-bd); }
  .bMED  { background: var(--med-bg);  color: var(--med);  border: 1px solid var(--med-bd); }
  .bLOW  { background: var(--low-bg);  color: var(--low);  border: 1px solid var(--low-bd); }
  .bAIR  { background: var(--air-bg);  color: var(--air);  border: 1px solid rgba(88,166,255,.3); }
  .bSEA  { background: var(--sea-bg);  color: var(--sea);  border: 1px solid rgba(57,211,83,.3); }
  .bLAND { background: var(--land-bg); color: var(--land); border: 1px solid rgba(188,140,255,.3); }

  /* ── Buttons ── */
  .btn {
    display: inline-flex; align-items: center; gap: 5px;
    padding: 7px 14px; border-radius: var(--r-sm); font-size: 12px; font-weight: 600;
    cursor: pointer; border: 1px solid var(--border); background: var(--surface);
    color: var(--text2); transition: all .15s; white-space: nowrap;
  }
  .btn:hover { background: var(--surface2); border-color: var(--primary); color: var(--primary); }
  .btn-primary { background: var(--primary); border-color: var(--primary); color: #fff; }
  .btn-primary:hover { background: var(--primary-mid); border-color: var(--primary-mid); color: #fff; }
  .btn-sm { padding: 5px 10px; font-size: 11px; }
  .btn-icon { padding: 6px 8px; font-size: 15px; }
  .btn-active { background: var(--primary-lt); border-color: var(--primary); color: var(--primary); }

  /* ── Segmented control ── */
  .seg-control { display: flex; border: 1px solid var(--border); border-radius: var(--r-sm); overflow: hidden; }
  .seg-btn {
    flex: 1; padding: 6px 4px; font-size: 11px; font-weight: 600; text-align: center;
    cursor: pointer; border: none; background: var(--surface); color: var(--muted);
    border-right: 1px solid var(--border); transition: all .15s;
  }
  .seg-btn:last-child { border-right: none; }
  .seg-btn:hover { background: var(--surface2); color: var(--text); }
  .seg-btn.active { background: var(--primary); color: #fff; }

  /* ── Cards ── */
  .card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 14px 16px; box-shadow: var(--shadow);
  }

  /* ── Tables ── */
  .data-table { width: 100%; border-collapse: collapse; font-size: 12px; }
  .data-table thead th {
    background: var(--surface2); color: var(--muted); font-size: 10px; font-weight: 700;
    text-transform: uppercase; letter-spacing: .06em; padding: 8px 12px;
    border-bottom: 2px solid var(--border); border-right: 1px solid var(--border);
    text-align: left; white-space: nowrap;
  }
  .data-table thead th:last-child { border-right: none; }
  .data-table tbody tr { border-bottom: 1px solid var(--border2); transition: background .1s; }
  .data-table tbody tr:nth-child(even) { background: rgba(45,106,79,.035); }
  .data-table tbody tr:hover { background: var(--surface2); }
  .data-table tbody td {
    padding: 8px 12px; color: var(--text2); vertical-align: middle;
    border-right: 1px solid var(--border2);
  }
  .data-table tbody td:last-child { border-right: none; }
  .data-table .num { text-align: right; font-variant-numeric: tabular-nums; font-weight: 600; }

  /* ── Select ── */
  select {
    width: 100%; background: var(--surface); color: var(--text);
    border: 1px solid var(--border); border-radius: var(--r-sm);
    padding: 7px 10px; font-size: 12px; font-family: inherit;
    cursor: pointer; appearance: none;
    background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='12' height='12' viewBox='0 0 12 12'%3E%3Cpath fill='%236b8068' d='M6 8L1 3h10z'/%3E%3C/svg%3E");
    background-repeat: no-repeat; background-position: right 10px center; padding-right: 28px;
  }
  select:focus { outline: none; border-color: var(--primary); box-shadow: 0 0 0 3px rgba(45,106,79,.15); }

  /* ── Range slider ── */
  input[type=range] {
    width: 100%; accent-color: var(--primary); cursor: pointer; height: 4px;
    border-radius: 2px;
  }

  /* ── Route card ── */
  .route-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 12px 14px; margin-bottom: 8px;
    cursor: pointer; transition: all .15s; position: relative;
  }
  .route-card:hover { border-color: var(--primary); box-shadow: var(--shadow); }
  .route-card.selected { border-color: var(--primary); background: var(--primary-lt); }
  .route-card-rank {
    position: absolute; top: 12px; left: 14px;
    font-size: 10px; font-weight: 800; color: var(--muted);
    text-transform: uppercase; letter-spacing: .05em;
  }

  /* ── KPI card ── */
  .kpi-card {
    background: var(--surface); border: 1px solid var(--border);
    border-radius: var(--r); padding: 12px 16px; box-shadow: var(--shadow);
    display: flex; flex-direction: column; align-items: flex-start; min-width: 120px;
  }

  /* ── Insight card ── */
  .insight-card {
    background: var(--primary-lt); border: 1px solid rgba(45,106,79,.25);
    border-radius: var(--r); padding: 12px 14px;
  }
  .insight-head { font-size: 13px; font-weight: 700; color: var(--primary); margin-bottom: 5px; }
  .insight-body { font-size: 12px; color: var(--text2); line-height: 1.6; }
  .insight-action {
    margin-top: 8px; font-size: 11px; font-weight: 600; color: var(--med);
    padding-top: 6px; border-top: 1px solid rgba(45,106,79,.2);
  }

  /* ── Score bars ── */
  .sbar { margin: 5px 0; }
  .sbar-row { display: flex; justify-content: space-between; font-size: 11px; color: var(--text2); margin-bottom: 3px; }
  .sbar-track { height: 6px; border-radius: 3px; background: var(--border2); overflow: hidden; }
  .sbar-fill  { height: 100%; border-radius: 3px; }

  /* ── Danger calendar ── */
  .dcal { display: grid; grid-template-columns: repeat(12,1fr); gap: 3px; margin: 8px 0; }
  .dcal-cell {
    height: 22px; border-radius: 4px; display: flex; align-items: center;
    justify-content: center; font-size: 9px; font-weight: 700;
    cursor: pointer; transition: opacity .15s; border: 1px solid transparent;
  }
  .dcal-cell:hover { opacity: .8; border-color: var(--primary); }

  /* ── Misc ── */
  .dot-live {
    display: inline-block; width: 7px; height: 7px; border-radius: 50%;
    background: var(--low); box-shadow: 0 0 0 3px rgba(46,128,80,.2); margin-right: 5px;
  }
  .divider { border: none; border-top: 1px solid var(--border); margin: 10px 0; }
  .text-high { color: var(--high); } .text-med { color: var(--med); }
  .text-low  { color: var(--low); } .text-muted { color: var(--muted); }
  .text-primary { color: var(--primary); }
  .flex { display: flex; } .flex-center { display: flex; align-items: center; }
  .flex-between { display: flex; align-items: center; justify-content: space-between; }
  .gap4 { gap: 4px; } .gap6 { gap: 6px; } .gap8 { gap: 8px; } .gap12 { gap: 12px; }
  .mt6 { margin-top: 6px; } .mt10 { margin-top: 10px; } .mt14 { margin-top: 14px; }
  .mb6 { margin-bottom: 6px; } .mb10 { margin-bottom: 10px; }
"""


# ---------------------------------------------------------------------------
# Build dashboard
# ---------------------------------------------------------------------------

def build_html() -> str:
    summary = _load_api_summary()
    seasonal_evidence, seasonal_evidence_summary = _load_seasonal_evidence()
    outputs = summary.get("outputs", {})
    now_str = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    ts = summary.get("generated_at", now_str)

    country_intel = outputs.get("country_intelligence", [])
    state_comp = outputs.get("state_comparison", [])
    species_list = outputs.get("species_watchlist", [])
    county_detection_rows = _csv_records(EXPORTS / "county_detection_rollup.csv")

    # Server-rendered country table rows (used in bottom panel)
    country_rows_html = ""
    for c in country_intel[:25]:
        mn = c.get("peak_month", 0)
        pk = MONTH_NAMES[mn] if 0 < mn < 13 else "—"
        spc = _public_species_note(c.get("key_species", "")).split(";")[0].strip()
        fly_short = "Pathway + GBIF scored" if spc.startswith("Not in curated watchlist") else (
            spc.split("(")[-1].replace(")", "").strip() if "(" in spc else spc[:30]
        )
        risk = c.get("max_risk", 0)
        tier_cls = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        country_rows_html += (
            f"<tr>"
            f"<td><b>{c['origin_country']}</b></td>"
            f"<td class='num {tier_cls}'>{risk:.1f}</td>"
            f"<td class='num'>{c.get('high_routes', 0)}</td>"
            f"<td>{pk}</td>"
            f"<td style='font-size:11px'>{(c.get('primary_pathway','') or '')[:40]}</td>"
            f"<td style='font-size:11px;font-style:italic'>{fly_short[:35]}</td>"
            f"</tr>"
        )

    # Species rows
    species_rows_html = ""
    for s in species_list:
        risk = s.get("max_pathway_risk", 0)
        tier_cls = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        species_rows_html += (
            f"<tr>"
            f"<td style='font-style:italic'><b>{s.get('species','')}</b></td>"
            f"<td>{s.get('common_name','')}</td>"
            f"<td>{s.get('state','')}</td>"
            f"<td style='font-size:11px'>{s.get('watch_ports','')}</td>"
            f"<td style='font-size:11px'>{(s.get('origin_countries','') or '')[:40]}</td>"
            f"<td class='num {tier_cls}'>{risk:.1f}</td>"
            f"</tr>"
        )

    # State rows
    state_rows_html = ""
    for s in state_comp:
        risk = s.get("max_risk", 0)
        tier_cls = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        state_rows_html += (
            f"<tr>"
            f"<td><b>{s['state']}</b></td>"
            f"<td class='num {tier_cls}'>{risk:.1f}</td>"
            f"<td class='num'>{s.get('high_routes', 0)}</td>"
            f"<td style='font-size:11px'>{s.get('top_ports','')}</td>"
            f"<td style='font-size:11px'>{s.get('modes','')}</td>"
            f"</tr>"
        )

    narratives_js = NARRATIVES_JSON
    ag_exposure_js = AG_EXPOSURE_JSON
    county_detection_js = json.dumps(county_detection_rows, ensure_ascii=True)
    seasonal_evidence_js = json.dumps(seasonal_evidence[:500], ensure_ascii=False)
    seasonal_evidence_summary_js = json.dumps(seasonal_evidence_summary, ensure_ascii=False)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>APHIS Fruit Fly Pathway Risk Monitor</title>
<link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css">
<script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
<style>
{SHARED_CSS}

/* ── App shell ── */
html, body {{ height: 100%; overflow: hidden; }}
#app {{ display: flex; flex-direction: column; height: 100vh; overflow-y: auto; }}

/* ── Header ── */
#header {{
  flex: 0 0 auto;
  background: var(--surface);
  border-bottom: 2px solid var(--primary);
  padding: 0 16px;
  display: flex; align-items: center; gap: 0;
  height: 52px; z-index: 200;
  box-shadow: 0 2px 8px rgba(0,0,0,.08);
}}
#header-brand {{
  display: flex; align-items: center; gap: 8px;
  font-size: 14px; font-weight: 800; color: var(--primary);
  letter-spacing: -.2px; white-space: nowrap; margin-right: 20px;
}}
#header-brand svg {{ flex: 0 0 auto; }}

/* Top tabs */
#top-tabs {{
  display: flex; align-items: stretch; height: 100%; gap: 0; flex: 1;
}}
.top-tab {{
  display: flex; align-items: center; padding: 0 16px;
  font-size: 13px; font-weight: 600; color: var(--muted);
  cursor: pointer; border-bottom: 3px solid transparent;
  margin-bottom: -2px; transition: all .15s; white-space: nowrap;
  border-top: 3px solid transparent;
}}
.top-tab:hover {{ color: var(--primary); background: var(--primary-lt); }}
.top-tab.active {{ color: var(--primary); border-bottom-color: var(--primary); }}

/* Header actions */
#header-actions {{ display: flex; align-items: center; gap: 8px; margin-left: 12px; }}

/* ── Main layout ── */
#main-area {{
  flex: 0 0 calc(100vh - 92px); display: flex; overflow: hidden; position: relative;
}}

/* ── Operational briefing hero ── */
#hero-section {{
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: repeat(3, minmax(0, 1fr));
  gap: 12px;
  padding: 12px 14px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
}}
#briefing-area {{
  flex: 0 0 auto;
  background: var(--bg);
  border-top: 1px solid var(--border);
}}
#briefing-intro {{
  padding: 14px 14px 4px;
}}
#briefing-intro h2 {{
  margin: 0 0 4px;
  font-size: 15px;
  font-weight: 800;
  color: var(--text);
}}
#briefing-intro p {{
  max-width: 980px;
  font-size: 12px;
  color: var(--text2);
  line-height: 1.55;
}}
.hero-card {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  padding: 13px 14px;
  box-shadow: var(--shadow);
  min-width: 0;
}}
.hero-label {{
  font-size: 10px;
  font-weight: 800;
  text-transform: uppercase;
  letter-spacing: .08em;
  color: var(--muted);
  margin-bottom: 7px;
}}
.hero-headline {{
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  line-height: 1.28;
  margin-bottom: 6px;
}}
.hero-body {{
  font-size: 12px;
  color: var(--text2);
  line-height: 1.5;
}}
.hero-action {{
  margin-top: 8px;
  padding-top: 7px;
  border-top: 1px solid var(--border2);
  font-size: 11px;
  font-weight: 700;
  color: var(--med);
  line-height: 1.45;
}}
.hero-route {{
  padding: 7px 0;
  border-bottom: 1px solid var(--border2);
  cursor: pointer;
}}
.hero-route:last-child {{ border-bottom: none; padding-bottom: 0; }}
.hero-route-top {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 2px;
}}
.hero-route-title {{
  font-size: 13px;
  font-weight: 800;
  color: var(--text);
  min-width: 0;
}}
.hero-route-score {{
  font-size: 13px;
  font-weight: 800;
  font-variant-numeric: tabular-nums;
}}
.hero-route-why {{
  font-size: 12px;
  color: var(--text2);
  line-height: 1.45;
}}
.hero-route-fly {{
  margin-top: 2px;
  font-size: 11px;
  color: var(--muted);
  font-style: italic;
}}
.country-risk-item {{
  padding: 8px 0;
  border-bottom: 1px solid var(--border2);
}}
.country-risk-item:last-child {{ border-bottom: none; padding-bottom: 0; }}
.country-risk-header {{
  display: flex;
  justify-content: space-between;
  align-items: baseline;
  gap: 8px;
  margin-bottom: 4px;
}}
.country-name {{
  font-size: 13px;
  font-weight: 800;
  color: var(--text);
}}
.country-risk-score {{
  font-size: 16px;
  font-weight: 800;
  letter-spacing: -.2px;
  font-variant-numeric: tabular-nums;
}}
.country-risk-meta {{
  display: flex;
  align-items: center;
  gap: 7px;
  font-size: 11px;
  color: var(--muted);
}}
.country-species {{
  min-width: 0;
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
  font-style: italic;
}}

#briefing-sections {{
  flex: 0 0 auto;
  display: grid;
  grid-template-columns: repeat(5, minmax(0, 1fr));
  gap: 8px;
  padding: 0 14px 10px;
  background: var(--bg);
  border-bottom: 1px solid var(--border);
}}
.collapsible-section {{
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: var(--r);
  overflow: hidden;
  box-shadow: var(--shadow);
}}
.collapsible-header {{
  min-height: 38px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 9px 11px;
  cursor: pointer;
  transition: background .15s;
}}
.collapsible-header:hover {{ background: var(--surface2); }}
.collapsible-title {{
  font-size: 15px;
  font-weight: 700;
  color: var(--text);
  overflow: hidden;
  text-overflow: ellipsis;
  white-space: nowrap;
}}
.collapsible-icon {{
  font-size: 12px;
  color: var(--muted);
  transition: transform .2s;
}}
.collapsible-section.open .collapsible-icon {{ transform: rotate(180deg); }}
.collapsible-content {{
  display: none;
  padding: 0 11px 11px;
}}
.collapsible-section.open .collapsible-content {{ display: block; }}
.mode-breakdown-chart {{ display: grid; gap: 8px; margin-top: 2px; }}
.mode-bar {{
  min-height: 31px;
  display: flex;
  align-items: center;
  justify-content: space-between;
  gap: 8px;
  padding: 7px 9px;
  border: 1px solid var(--border);
  border-radius: var(--r-sm);
  background:
    linear-gradient(to right, color-mix(in srgb, var(--mode-color) 22%, transparent) var(--pct), var(--surface2) var(--pct));
}}
.mode-label, .mode-pct {{
  font-size: 13px;
  font-weight: 700;
  color: var(--text);
}}
.mode-totals, .briefing-note {{
  margin-top: 8px;
  padding-top: 7px;
  border-top: 1px solid var(--border2);
  font-size: 11px;
  color: var(--muted);
  font-style: italic;
  line-height: 1.45;
}}
.compact-country-row {{
  display: flex;
  justify-content: space-between;
  gap: 8px;
  padding: 5px 0;
  border-bottom: 1px solid var(--border2);
  font-size: 12px;
  color: var(--text2);
}}
.compact-country-row:last-child {{ border-bottom: none; }}
.compact-months {{
  display: grid;
  grid-template-columns: repeat(6, 1fr);
  gap: 4px;
}}
.compact-month {{
  min-height: 24px;
  border-radius: 4px;
  display: grid;
  place-items: center;
  font-size: 10px;
  font-weight: 800;
  border: 1px solid var(--border2);
}}
.ag-forecast-controls {{
  display: flex;
  align-items: center;
  gap: 8px;
  margin: 2px 0 10px;
}}
.ag-forecast-controls input {{ flex: 1; }}
.ag-year-value {{
  min-width: 42px;
  text-align: right;
  font-size: 14px;
  font-weight: 800;
  color: var(--primary);
}}
.ag-forecast-bars {{ display: grid; gap: 7px; }}
.ag-bar-row {{
  display: grid;
  grid-template-columns: 86px 1fr 58px;
  gap: 7px;
  align-items: center;
  font-size: 11px;
  color: var(--text2);
}}
.ag-bar-track {{
  height: 9px;
  border-radius: 99px;
  background: var(--border2);
  overflow: hidden;
}}
.ag-bar-fill {{
  height: 100%;
  border-radius: 99px;
  background: var(--high);
}}
.ag-year-axis {{
  display: grid;
  grid-template-columns: repeat(5, 1fr);
  gap: 4px;
  margin-top: 8px;
  font-size: 10px;
  color: var(--muted);
  text-align: center;
}}
.ag-source-line {{
  margin-top: 7px;
  font-size: 11px;
  color: var(--text2);
  line-height: 1.45;
}}

@media (max-width: 1100px) {{
  #hero-section {{ grid-template-columns: 1fr; max-height: 36vh; overflow-y: auto; }}
  #briefing-sections {{ grid-template-columns: 1fr 1fr; }}
}}
@media (max-width: 760px) {{
  #briefing-sections {{ grid-template-columns: 1fr; }}
}}

/* KPI bar (floats above map) */
#kpi-bar {{
  position: absolute; top: 12px; left: 12px; z-index: 1300;
  display: flex; gap: 8px; pointer-events: none;
}}
.kpi-card {{ pointer-events: auto; }}

/* Map */
#map-wrap {{ flex: 1; position: relative; overflow: hidden; isolation: isolate; }}
#leaflet-map {{ position: absolute; inset: 0; width: 100%; height: 100%; z-index: 1; }}
.leaflet-control-zoom {{
  margin-top: 118px !important;
  margin-left: 12px !important;
  box-shadow: var(--shadow) !important;
  border: 1px solid var(--border) !important;
}}
.leaflet-control-scale {{
  margin-left: 12px !important;
  margin-bottom: 38px !important;
}}
.leaflet-control-scale-line {{
  background: rgba(255,255,255,.92) !important;
  border-color: var(--border) !important;
  color: var(--text) !important;
  font-weight: 700;
}}
body.dark .leaflet-control-scale-line {{
  background: rgba(13,17,23,.92) !important;
  color: var(--text) !important;
}}

#map-readout {{
  position: absolute; left: 12px; bottom: 64px; z-index: 1300;
  display: grid; gap: 4px; pointer-events: none;
}}
.map-readout-pill {{
  width: max-content; max-width: min(320px, 60vw);
  background: rgba(255,255,255,.94); border: 1px solid var(--border);
  border-radius: var(--r-sm); padding: 5px 8px; box-shadow: var(--shadow);
  font-size: 11px; font-weight: 700; color: var(--text);
}}
body.dark .map-readout-pill {{ background: rgba(13,17,23,.94); }}

/* Drag handle */
#drag-handle {{
  flex: 0 0 6px; cursor: col-resize;
  background: var(--border);
  display: flex; align-items: center; justify-content: center;
  transition: background .15s; z-index: 100;
  position: relative;
}}
#drag-handle:hover, #drag-handle.dragging {{ background: var(--primary); }}
#drag-handle::after {{
  content: ''; position: absolute;
  width: 2px; height: 32px; background: var(--surface);
  border-radius: 1px; opacity: .7;
}}

/* Right panel */
#right-panel {{
  width: 340px; min-width: 240px; max-width: 520px;
  display: flex; flex-direction: column;
  background: var(--bg); border-left: 1px solid var(--border);
  overflow: hidden; flex: 0 0 auto;
  position: relative;
}}
#right-panel-inner {{
  flex: 1; overflow-y: auto; padding: 12px;
  scrollbar-width: thin; scrollbar-color: var(--border) transparent;
}}
#right-panel-inner::-webkit-scrollbar {{ width: 4px; }}
#right-panel-inner::-webkit-scrollbar-track {{ background: transparent; }}
#right-panel-inner::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 2px; }}
.detail-section {{
  display: none;
  height: 100%;
  overflow-y: auto;
  padding: 12px;
}}
aside.detail-mode .briefing-section {{ display: none; }}
aside.detail-mode .detail-section {{ display: block; }}
.back-btn {{
  width: 100%;
  display: inline-flex;
  align-items: center;
  justify-content: flex-start;
  padding: 8px 10px;
  margin-bottom: 10px;
  border-radius: var(--r-sm);
  border: 1px solid var(--border);
  background: var(--surface);
  color: var(--primary);
  font-size: 12px;
  font-weight: 700;
  cursor: pointer;
}}
.back-btn:hover {{
  background: var(--primary-lt);
  border-color: var(--primary);
}}

/* Filter group */
.filter-label {{
  font-size: 10px; font-weight: 700; text-transform: uppercase;
  letter-spacing: .08em; color: var(--muted); display: block; margin-bottom: 5px;
}}
.filter-group {{ margin-bottom: 10px; }}

/* Month slider area */
.slider-row {{ display: flex; align-items: center; gap: 8px; }}
.slider-month-label {{
  font-size: 14px; font-weight: 800; color: var(--primary);
  min-width: 32px; text-align: right; letter-spacing: -.3px;
}}
.slider-ticks {{
  display: grid; grid-template-columns: repeat(12, 1fr);
  font-size: 9px; color: var(--muted); margin-top: 2px; padding: 0 39px 0 0;
  text-align: center;
}}

/* Map legend */
#map-legend {{
  position: absolute; bottom: 16px; right: 16px; z-index: 1300;
  background: var(--surface); border: 1px solid var(--border);
  border-radius: var(--r); padding: 10px 12px; box-shadow: var(--shadow);
  font-size: 11px;
}}
.legend-row {{ display: flex; align-items: center; gap: 7px; margin-bottom: 5px; }}
.legend-row:last-child {{ margin-bottom: 0; }}
.leg-line {{ width: 22px; height: 0; border-top: 3px solid; border-radius: 2px; }}
.leg-dot  {{ width: 10px; height: 10px; border-radius: 50%; flex: 0 0 22px; margin-left: 0; }}

/* ── Bottom panel ── */
#bottom-bar {{
  flex: 0 0 auto;
  background: var(--surface); border-top: 1px solid var(--border);
  display: flex; align-items: center; padding: 0 14px;
  height: 40px; gap: 4px; z-index: 150; box-shadow: 0 -2px 8px rgba(0,0,0,.06);
}}
.bottom-tab {{
  padding: 8px 14px; font-size: 12px; font-weight: 600; color: var(--muted);
  cursor: pointer; border-radius: var(--r-sm); transition: all .15s; white-space: nowrap;
}}
.bottom-tab:hover {{ background: var(--surface2); color: var(--text); }}
.bottom-tab.active {{ background: var(--primary-lt); color: var(--primary); }}
#bottom-toggle {{
  margin-left: auto;
  display: flex; align-items: center; gap: 5px;
  padding: 5px 12px; border-radius: var(--r-sm); font-size: 11px; font-weight: 600;
  cursor: pointer; border: 1px solid var(--border); background: var(--surface);
  color: var(--text2); transition: all .15s; white-space: nowrap;
}}
#bottom-toggle:hover {{ border-color: var(--primary); color: var(--primary); background: var(--primary-lt); }}
#bottom-panel {{
  flex: 0 0 auto; height: 0; overflow: hidden;
  background: var(--bg); border-top: 1px solid var(--border);
  transition: height .25s ease;
}}
#bottom-panel.open {{ height: 260px; overflow-y: auto; }}
#bottom-panel-inner {{ padding: 14px 16px; }}

/* ── Tab content views (top tab switching) ── */
.tab-view {{ display: none; }}
.tab-view.active {{ display: block; }}

/* ── Seasonal evidence modal ── */
#seasonal-evidence-modal {{
  position: fixed; inset: 0; z-index: 6000; display: none;
  background: rgba(10,24,18,.32); backdrop-filter: blur(2px);
}}
#seasonal-evidence-modal.open {{ display: grid; place-items: center; }}
.evidence-card {{
  width: min(760px, 94vw); max-height: 86vh; overflow-y: auto;
  background: var(--surface); border: 1px solid var(--border); border-radius: 18px;
  box-shadow: 0 24px 70px rgba(0,0,0,.22); padding: 20px;
}}
.evidence-layer {{
  border: 1px solid var(--border); border-radius: var(--r);
  padding: 10px 12px; margin: 8px 0; background: var(--surface2);
}}
.evidence-pill {{
  display: inline-flex; align-items: center; padding: 3px 8px; border-radius: 999px;
  font-size: 10px; font-weight: 800; letter-spacing: .04em; text-transform: uppercase;
  background: var(--primary-lt); color: var(--primary); border: 1px solid var(--primary-bd);
}}
.evidence-caveat {{
  padding: 10px 12px; background: #fff7e6; border: 1px solid #f2d29b;
  color: #694100; border-radius: var(--r); font-size: 12px; line-height: 1.55;
}}
body.dark .evidence-caveat {{ background: rgba(244,162,97,.14); border-color: rgba(244,162,97,.35); color: #ffd7a8; }}

/* ── Theme toggle ── */
#theme-toggle {{
  width: 36px; height: 36px; border-radius: 50%;
  background: var(--surface2); border: 1px solid var(--border);
  cursor: pointer; display: flex; align-items: center; justify-content: center;
  font-size: 16px; transition: all .15s;
}}
#theme-toggle:hover {{ border-color: var(--primary); background: var(--primary-lt); }}

/* ── Footer / stat bar ── */
#stat-bar {{
  position: absolute; bottom: 0; left: 0;
  background: rgba(255,255,255,.9); border-top: 1px solid var(--border);
  padding: 4px 12px; font-size: 11px; color: var(--muted);
  display: flex; gap: 14px; backdrop-filter: blur(4px);
  z-index: 1250;
}}
body.dark #stat-bar {{ background: rgba(13,17,23,.9); }}

/* ── Seasonal window card ── */
#seasonal-window-content .month-grid {{
  display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; margin-top: 10px;
}}
.month-chip {{
  padding: 8px 6px; border-radius: var(--r-sm); text-align: center;
  font-size: 11px; font-weight: 600; cursor: pointer; border: 1px solid transparent;
  transition: all .15s;
}}
.month-chip:hover {{ border-color: var(--primary); }}
.month-chip.active {{ border-color: var(--primary); box-shadow: 0 0 0 2px rgba(45,106,79,.2); }}

/* scrollbar global */
* {{ scrollbar-width: thin; scrollbar-color: var(--border) transparent; }}
</style>
</head>
<body>
<div id="app">

<!-- ═══════════════════════════ HEADER ═══════════════════════════ -->
<div id="header">
  <div id="header-brand">
    <svg width="20" height="20" viewBox="0 0 20 20" fill="none">
      <circle cx="10" cy="10" r="9" fill="#2d6a4f" opacity=".15"/>
      <path d="M10 3c0 0-4 3-4 7s4 7 4 7 4-3 4-7-4-7-4-7z" fill="#2d6a4f" opacity=".4"/>
      <circle cx="10" cy="10" r="3" fill="#2d6a4f"/>
    </svg>
    APHIS Fruit Fly Pathway Risk Monitor
  </div>

  <div id="top-tabs">
    <div class="top-tab active" data-tab="map" onclick="switchTopTab('map')">Risk Map</div>
    <div class="top-tab" data-tab="country" onclick="switchTopTab('country')">Country Analysis</div>
    <div class="top-tab" data-tab="species" onclick="switchTopTab('species')">Species Watch</div>
    <div class="top-tab" data-tab="seasonal" onclick="switchTopTab('seasonal')">Seasonal Window</div>
    <div class="top-tab" data-tab="methods" onclick="switchTopTab('methods')">Data &amp; Methods</div>
  </div>

  <div id="header-actions">
    <button id="theme-toggle" onclick="toggleTheme()" title="Toggle light/dark">☀</button>
    <a href="report.html" target="_blank" class="btn btn-primary btn-sm">Full Report →</a>
  </div>
</div>

<!-- ═══════════════════════════ MAIN ════════════════════════════ -->
<div id="main-area">

  <!-- MAP TAB (default) -->
  <div id="map-wrap">
    <!-- KPI cards -->
    <div id="kpi-bar">
      <div class="kpi-card">
        <div class="kpi-number text-high" id="kpi-high">—</div>
        <div class="kpi-label">HIGH Routes</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-number" style="color:var(--primary)" id="kpi-countries">—</div>
        <div class="kpi-label">Countries at Risk</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-number" style="color:var(--air)" id="kpi-peak">—</div>
        <div class="kpi-label">Peak Month</div>
      </div>
      <div class="kpi-card">
        <div class="kpi-number text-med" id="kpi-port">—</div>
        <div class="kpi-label">Top Port</div>
      </div>
    </div>

    <div id="leaflet-map"></div>

    <!-- Always-visible map measurements -->
    <div id="map-readout" aria-live="polite">
      <div class="map-readout-pill" id="zoom-readout">Zoom 3 · world view</div>
      <div class="map-readout-pill" id="coord-readout">Move cursor over map for lat/lon</div>
    </div>

    <!-- Stat bar -->
    <div id="stat-bar">
      <span id="sb-routes">—</span>
      <span id="sb-high" class="text-high">—</span>
      <span id="sb-med" class="text-med">—</span>
      <span id="sb-low" class="text-low">—</span>
      <span id="sb-filter" style="color:var(--primary)"></span>
    </div>

    <!-- Map legend -->
    <div id="map-legend">
      <div class="section-label" style="margin-bottom:8px">Route mode</div>
      <div class="legend-row"><div class="leg-line" style="border-color:var(--air);border-style:dashed"></div><span>AIR</span></div>
      <div class="legend-row"><div class="leg-line" style="border-color:var(--sea)"></div><span>SEA</span></div>
      <div class="legend-row"><div class="leg-line" style="border-color:var(--land);border-style:dotted"></div><span>LAND</span></div>
      <div class="divider"></div>
      <div class="section-label" style="margin-bottom:8px">Risk tier</div>
      <div class="legend-row"><div class="leg-dot" style="background:var(--high)"></div><span>HIGH</span></div>
      <div class="legend-row"><div class="leg-dot" style="background:var(--med)"></div><span>MED</span></div>
      <div class="legend-row"><div class="leg-dot" style="background:var(--low)"></div><span>LOW</span></div>
    </div>

  </div>

  <!-- Drag handle -->
  <div id="drag-handle" title="Drag to resize panel"></div>

  <!-- Right panel -->
  <aside id="right-panel">
    <div id="right-panel-inner" class="briefing-section operator-brief-section">

      <!-- Month + play -->
      <div class="filter-group">
        <div class="flex-between mb6">
          <span class="filter-label" style="margin-bottom:0">Time window</span>
          <div style="display:flex;gap:6px">
            <button id="all-months-btn" class="btn btn-sm btn-active" onclick="setAllMonths()" title="Show all months">All</button>
            <button id="play-btn" class="btn btn-sm" onclick="togglePlay()" title="Animate January through December in a loop">▶ Play</button>
          </div>
        </div>
        <div class="slider-row">
          <input type="range" id="month-slider" min="1" max="12" step="1" value="1" oninput="onMonthChange()">
          <span class="slider-month-label" id="month-label">ALL</span>
        </div>
        <div class="slider-ticks">
          <span>J</span><span>F</span><span>M</span><span>A</span>
          <span>M</span><span>J</span><span>J</span><span>A</span><span>S</span>
          <span>O</span><span>N</span><span>D</span>
        </div>
      </div>

      <div class="divider"></div>

      <!-- Transport mode -->
      <div class="filter-group">
        <span class="filter-label">Transport mode</span>
        <div class="seg-control">
          <div class="seg-btn active" data-mode="ALL" onclick="setMode('ALL')">All</div>
          <div class="seg-btn" data-mode="AIR" onclick="setMode('AIR')">✈ AIR</div>
          <div class="seg-btn" data-mode="SEA" onclick="setMode('SEA')">⛴ SEA</div>
          <div class="seg-btn" data-mode="LAND" onclick="setMode('LAND')">🚛 LAND</div>
        </div>
      </div>

      <!-- Risk tier -->
      <div class="filter-group">
        <span class="filter-label">Risk tier</span>
        <div class="seg-control">
          <div class="seg-btn active" data-tier="ALL" onclick="setTier('ALL')">All</div>
          <div class="seg-btn" data-tier="HIGH" onclick="setTier('HIGH')" style="color:var(--high)">HIGH</div>
          <div class="seg-btn" data-tier="MEDIUM" onclick="setTier('MEDIUM')" style="color:var(--med)">MED</div>
          <div class="seg-btn" data-tier="LOW" onclick="setTier('LOW')" style="color:var(--low)">LOW</div>
        </div>
      </div>

      <!-- State + Host -->
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px" class="mb10">
        <div>
          <span class="filter-label">State focus</span>
          <select id="fil-state" onchange="applyFilters()">
            <option value="ALL">National</option>
            <option value="CA">California</option>
            <option value="TX">Texas</option>
            <option value="FL">Florida</option>
          </select>
        </div>
        <div>
          <span class="filter-label">Host / commodity</span>
          <select id="fil-host" onchange="applyFilters()">
            <option value="ALL">All hosts</option>
          </select>
        </div>
      </div>

      <div class="divider"></div>

      <!-- Top 3 routes -->
      <div>
        <div class="flex-between mb6">
          <span class="filter-label" style="margin-bottom:0">Top surge routes</span>
          <button id="div-btn" class="btn btn-sm" onclick="toggleDiversity()">★ Diverse</button>
        </div>
        <div id="top3-cards"></div>
        <div style="font-size:10px;color:var(--muted);margin-top:4px;line-height:1.4" id="top3-note">
          When month = ALL, corridors are deduplicated to best-scoring route per origin × port × commodity.
        </div>
      </div>

      <div class="divider" style="margin-top:12px"></div>

      <!-- Seasonal insight -->
      <div id="window-card" class="mt10"></div>

    </div>
    <div id="routeDetailSection" class="detail-section route-detail-section">
      <button class="back-btn" onclick="hideRouteDetail()">← Back to briefing</button>
      <div class="card mb10">
        <div class="section-label">Selected route</div>
        <div id="detail-route-title" style="font-size:18px;font-weight:800;color:var(--text)">Route detail</div>
      </div>
      <div id="scoreBreakdown" class="card mb10"></div>
      <div id="whyThisRoute" class="card"></div>
    </div>
  </aside>

</div><!-- /main-area -->

<!-- ═══════════════════════════ BELOW-MAP BRIEFING ═════════════ -->
<section id="briefing-area" aria-label="Operational briefing">
  <div id="briefing-intro">
    <h2>Operational Briefing</h2>
    <p>Scroll below the map for narrative decision support. These cards update with the active filters, while the map remains the primary first-screen operational surface.</p>
  </div>

  <section id="hero-section" aria-label="Route briefing cards">
    <div class="hero-card alert-card">
      <div class="hero-label">THIS MONTH'S ALERT</div>
      <div class="hero-headline" id="hero-alert-headline">Loading current pathway alert...</div>
      <div class="hero-body" id="hero-alert-body">Risk briefing will update as route data loads.</div>
      <div class="hero-action" id="hero-alert-action">Apply filters to focus the briefing on a state, month, host, or mode.</div>
    </div>

    <div class="hero-card routes-card">
      <div class="hero-label">TOP 3 SURGE ROUTES</div>
      <div id="hero-routes-list">
        <div class="hero-body">Loading surge routes...</div>
      </div>
    </div>

    <div class="hero-card corridor-card">
      <div class="hero-label">WATCH THIS CORRIDOR</div>
      <div id="hero-country-risk">
        <div class="hero-body">Loading country risk...</div>
      </div>
    </div>
  </section>

  <section id="briefing-sections" aria-label="Supporting dashboard details">
    <div class="collapsible-section" id="mode-breakdown-section">
      <div class="collapsible-header" onclick="toggleSection('mode-breakdown')" role="button" tabindex="0">
        <span class="collapsible-title">Transport Mode Breakdown</span>
        <span class="collapsible-icon">▼</span>
      </div>
      <div class="collapsible-content" id="mode-breakdown-content">
        <div id="mode-breakdown-chart" class="mode-breakdown-chart"></div>
        <div id="mode-breakdown-totals" class="mode-totals"></div>
      </div>
    </div>

    <div class="collapsible-section" id="country-detail-section">
      <div class="collapsible-header" onclick="toggleSection('country-detail')" role="button" tabindex="0">
        <span class="collapsible-title">Top 5 Countries</span>
        <span class="collapsible-icon">▼</span>
      </div>
      <div class="collapsible-content" id="country-detail-content">
        <div id="country-detail-list"></div>
        <div class="briefing-note">Country bubbles remain visible on the map; this list updates with active route filters.</div>
      </div>
    </div>

    <div class="collapsible-section" id="seasonal-calendar-section">
      <div class="collapsible-header" onclick="toggleSection('seasonal-calendar')" role="button" tabindex="0">
        <span class="collapsible-title">Seasonal Calendar</span>
        <span class="collapsible-icon">▼</span>
      </div>
      <div class="collapsible-content" id="seasonal-calendar-content">
        <div id="briefing-month-strip" class="compact-months"></div>
        <div class="briefing-note">Use the full bottom-panel calendar for country-specific evidence cells and detection confidence.</div>
      </div>
    </div>

    <div class="collapsible-section" id="ag-exposure-section">
      <div class="collapsible-header" onclick="toggleSection('ag-exposure')" role="button" tabindex="0">
        <span class="collapsible-title">Agricultural Exposure Forecast</span>
        <span class="collapsible-icon">▼</span>
      </div>
      <div class="collapsible-content" id="ag-exposure-content">
        <div class="ag-forecast-controls">
          <input type="range" id="ag-year-slider" min="2026" max="2030" step="1" value="2026" oninput="renderAgExposureForecast()">
          <span class="ag-year-value" id="ag-year-label">2026</span>
        </div>
        <div id="ag-forecast-bars" class="ag-forecast-bars"></div>
        <div class="ag-year-axis"><span>2026</span><span>2027</span><span>2028</span><span>2029</span><span>2030</span></div>
        <div id="ag-forecast-note" class="briefing-note"></div>
      </div>
    </div>

    <div class="collapsible-section" id="data-limits-section">
      <div class="collapsible-header" onclick="toggleSection('data-limits')" role="button" tabindex="0">
        <span class="collapsible-title">Data Limits: USPS/courier gap</span>
        <span class="collapsible-icon">▼</span>
      </div>
      <div class="collapsible-content" id="data-limits-content">
        <div class="briefing-note" style="border-top:none;margin-top:0;padding-top:0">
          USPS and express courier pathways are not modeled because credible public operational volume data was not available. The schema can accept those pathways as additional transport modes when data becomes available. This prototype uses monthly public-data cadence and does not claim real-time feeds or sub-national origin analysis.
        </div>
      </div>
    </div>
  </section>
</section>

<!-- ═══════════════════════════ BOTTOM BAR ════════════════════════ -->
<div id="bottom-bar">
  <div class="bottom-tab active" data-btab="country" onclick="switchBottomTab('country')">Country Risk</div>
  <div class="bottom-tab" data-btab="species" onclick="switchBottomTab('species')">Species Watch</div>
  <div class="bottom-tab" data-btab="seasonal" onclick="switchBottomTab('seasonal')">Seasonal Calendar</div>
  <div class="bottom-tab" data-btab="methods" onclick="switchBottomTab('methods')">Data &amp; Methods</div>
  <button id="bottom-toggle" onclick="toggleBottomPanel()">▲ Open Panel</button>
</div>

<div id="bottom-panel">
  <div id="bottom-panel-inner">

    <!-- Country Risk -->
    <div class="btab-content active" id="btab-country">
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr>
            <th>Country</th><th class="num">Max Risk</th><th class="num">HIGH Routes</th>
            <th>Peak Month</th><th>Primary Pathway</th><th>Key Species</th>
          </tr></thead>
          <tbody>{country_rows_html}</tbody>
        </table>
        <p class="fine mt6">"Pathway + GBIF scored" means the country is not in the curated watchlist species profile, so this row is ranked from route exposure, host commodities, and GBIF pathway evidence rather than a named species card.</p>
        <p class="fine mt6">USPS and express courier are acknowledged data gaps — not scored in this model due to absence of credible public operational volume data.</p>
      </div>
    </div>

    <!-- Species Watch -->
    <div class="btab-content" id="btab-species" style="display:none">
      <div style="overflow-x:auto">
        <table class="data-table">
          <thead><tr>
            <th>Species</th><th>Common name</th><th>State focus</th>
            <th>Watch ports</th><th>Origin countries</th><th class="num">Max route risk</th>
          </tr></thead>
          <tbody>{species_rows_html}</tbody>
        </table>
      </div>
    </div>

    <!-- Seasonal Calendar -->
    <div class="btab-content" id="btab-seasonal" style="display:none">
      <div id="seasonal-evidence-summary" style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-bottom:12px"></div>
      <div id="danger-cal-wrap">
        <div class="flex-between mb6">
          <span class="filter-label" style="margin-bottom:0">Country danger calendar</span>
          <select id="cal-country" onchange="renderDangerCalendar()" style="width:auto;display:inline-block;padding:4px 28px 4px 8px">
            <option value="">Select country...</option>
          </select>
        </div>
        <div id="danger-cal"></div>
        <p class="fine" style="margin-top:4px">Click a month cell to filter the map. Color intensity = average risk score.</p>
      </div>
      <div class="evidence-caveat mt10">
        <b>Evidence caveat:</b> calendar colors are route-risk estimates. The evidence drill-down separates direct APHIS sample detections from pathway proxies, fly-host overlap, and seasonal priors. Inferred cells have no direct detection record for that exact species/state/month.
      </div>
    </div>

    <!-- Data & Methods -->
    <div class="btab-content" id="btab-methods" style="display:none">
      <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px;font-size:12px;color:var(--text2)">
        <div>
          <b style="color:var(--text);font-size:13px">Score components</b>
          <p style="margin-top:6px">Five weighted indicators: fly species–host co-occurrence (35%), cargo volume exposure (25%), host commodity fraction (20%), route frequency (10%), and detection proximity within 200 km (10%). A seasonal multiplier (×0.6–×1.5) adjusts for origin phenology.</p>
        </div>
        <div>
          <b style="color:var(--text);font-size:13px">Primary data sources</b>
          <ul style="margin-top:6px;padding-left:16px;line-height:1.8">
            <li>BTS T-100 International Segment (passenger volumes)</li>
            <li>US Census FT-920 (cargo by commodity + country)</li>
            <li>GBIF occurrence records (species co-occurrence)</li>
            <li>APHIS interception records (detection proximity)</li>
            <li>APHIS Fruit Fly Host Lists (commodity mapping)</li>
          </ul>
        </div>
        <div>
          <b style="color:var(--text);font-size:13px">Validation</b>
          <p style="margin-top:6px">Top 10% of ranked routes accounts for 34.1% of historical detection signal. HIGH-tier routes show 9.01× the detection density of LOW-tier routes. Temporal holdout Precision@10: 90%. Best model ROC AUC: 0.980.</p>
          <p style="margin-top:8px"><a href="report.html" target="_blank">Full methodology →</a></p>
        </div>
      </div>
    </div>

  </div>
</div>

</div><!-- /app -->

<div id="seasonal-evidence-modal" onclick="if(event.target.id==='seasonal-evidence-modal')closeSeasonalEvidence()">
  <div class="evidence-card">
    <div class="flex-between mb10">
      <div>
        <div class="section-label">Seasonal evidence trace</div>
        <h2 id="seasonal-evidence-title" style="margin:2px 0 0;font-size:20px;color:var(--text)">Evidence</h2>
      </div>
      <button class="btn btn-sm" onclick="closeSeasonalEvidence()">Close</button>
    </div>
    <div id="seasonal-evidence-body"></div>
  </div>
</div>

<script>
// ══════════════════════════════════════════════════════════════
// DATA & STATE
// ══════════════════════════════════════════════════════════════
const NARRATIVES = {narratives_js};
const AG_EXPOSURE_PROFILES = {ag_exposure_js};
const COUNTY_DETECTIONS = {county_detection_js};
const SEASONAL_EVIDENCE = {seasonal_evidence_js};
const SEASONAL_EVIDENCE_SUMMARY = {seasonal_evidence_summary_js};
const MONTH_NAMES = ['ALL','Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
const STATE_PORT_MAP = {{CA:['LAX','SAN','SFO','CAL'],TX:['HOU','DAL','LAR'],FL:['MIA','ORL']}};
const MODE_COLORS = {{AIR:'#1a6fa8', SEA:'#0e8070', LAND:'#6040a0'}};
const TIER_COLORS = {{HIGH:'#b83232', MEDIUM:'#c06010', LOW:'#2e8050'}};

let routeData=null, countryData=null, portData=null, apiSummary=null;
let apiSummaryData=null;
let activeMonth=0, activeTier='ALL', activeMode='ALL', briefDiversity=false;
let playInterval=null, leafletMap=null, routeLayer=null, countryLayer=null, portLayer=null;
let isDark=false, isPanelOpen=false, activeBottomTab='country';
let sectionStates={{'mode-breakdown':false,'country-detail':false,'seasonal-calendar':false,'ag-exposure':false,'data-limits':false}};

function esc(value) {{
  return String(value ?? '').replace(/[&<>"']/g, ch => ({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}}[ch]));
}}

function tierTextClass(risk) {{
  risk=+risk||0;
  return risk>=70?'text-high':risk>=55?'text-med':'text-low';
}}

function badgeClass(tier) {{
  return tier==='MEDIUM'?'MED':(tier||'LOW');
}}

function routeProps(featureOrProps) {{
  return featureOrProps?.properties || featureOrProps || {{}};
}}

function sourceToLinks(source) {{
  const text=String(source||'');
  const urls=text.match(/https?:\/\/[^\s,]+/g) || [];
  if (!urls.length) return esc(text || 'current run');
  const leading=esc(text.replace(/https?:\/\/[^\s,]+/g,'').replace(/\s+and\s+|\s*,\s*/g,' ').trim());
  return `${{leading?leading+' ':''}}${{urls.map((u,i)=>`<a target="_blank" rel="noopener" href="${{esc(u)}}">source ${{i+1}}</a>`).join(' · ')}}`;
}}

// ══════════════════════════════════════════════════════════════
// INIT
// ══════════════════════════════════════════════════════════════
function initMap() {{
  const tiles_light = 'https://{{s}}.basemaps.cartocdn.com/light_all/{{z}}/{{x}}/{{y}}{{r}}.png';
  const tiles_dark  = 'https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png';
  leafletMap = L.map('leaflet-map', {{center:[20,-20],zoom:3,zoomControl:true,attributionControl:false}});
  window._tileLight = L.tileLayer(tiles_light, {{maxZoom:19,attribution:'&copy; CartoDB &copy; OSM'}});
  window._tileDark  = L.tileLayer(tiles_dark,  {{maxZoom:19,attribution:'&copy; CartoDB &copy; OSM'}});
  window._tileLight.addTo(leafletMap);
  L.control.scale({{position:'bottomleft', metric:true, imperial:true, maxWidth:160}}).addTo(leafletMap);
  L.control.attribution({{position:'bottomright'}}).addTo(leafletMap);
  leafletMap.on('zoomend moveend', updateMapReadout);
  leafletMap.on('mousemove', e => {{
    const el=document.getElementById('coord-readout');
    if (el) el.textContent=`Lat ${{e.latlng.lat.toFixed(2)}} · Lon ${{e.latlng.lng.toFixed(2)}}`;
  }});
  leafletMap.on('mouseout', () => {{
    const el=document.getElementById('coord-readout');
    if (el) el.textContent='Move cursor over map for lat/lon';
  }});
  updateMapReadout();
  window._leafletMap = leafletMap;
}}

function updateMapReadout() {{
  if (!leafletMap) return;
  const el=document.getElementById('zoom-readout');
  if (!el) return;
  const z=leafletMap.getZoom();
  const c=leafletMap.getCenter();
  el.textContent=`Zoom ${{z}} · center ${{c.lat.toFixed(1)}}°, ${{c.lng.toFixed(1)}}°`;
}}

async function loadData() {{
  try {{
    const [rRes,cRes,pRes,aRes] = await Promise.all([
      fetch('geojson/top_routes_lines.geojson'),
      fetch('geojson/country_risk_points.geojson'),
      fetch('geojson/port_bubbles.geojson'),
      fetch('api/dashboard_summary.json'),
    ]);
    routeData   = await rRes.json();
    countryData = await cRes.json();
    portData    = await pRes.json();
    apiSummary  = await aRes.json();
    apiSummaryData = apiSummary?.outputs || {{}};
  }} catch(e) {{ console.warn('GeoJSON load:', e); }}
  buildDynamicFilters();
  buildCountryCalSelect();
  drawAll();
  updateKPIs();
  renderOperationalBriefing();
  renderTop3();
  renderWindowCard();
  renderEvidenceSummary();
  updateStatBar();
}}

// ══════════════════════════════════════════════════════════════
// DYNAMIC FILTERS
// ══════════════════════════════════════════════════════════════
function buildDynamicFilters() {{
  if (!routeData) return;
  const all = [...new Set(routeData.features.map(f=>(f.properties||{{}}).commodity_group).filter(Boolean))].sort();
  const sel = document.getElementById('fil-host');
  sel.innerHTML = '<option value="ALL">All hosts</option>';
  all.forEach(c => {{
    const o = document.createElement('option');
    o.value=c; o.textContent=c.replace(/_/g,' ');
    sel.appendChild(o);
  }});
}}

function buildCountryCalSelect() {{
  if (!routeData) return;
  const cs = [...new Set(routeData.features.map(f=>(f.properties||{{}}).origin_country).filter(Boolean))].sort();
  const sel = document.getElementById('cal-country');
  cs.forEach(c => {{
    const o = document.createElement('option');
    o.value=c; o.textContent=c;
    sel.appendChild(o);
  }});
}}

function renderEvidenceSummary() {{
  const el=document.getElementById('seasonal-evidence-summary');
  if (!el) return;
  const s=SEASONAL_EVIDENCE_SUMMARY||{{}};
  const counts=s.confidence_counts||{{}};
  const cards=[
    ['Cells', s.total_cells||0, 'species × state × month'],
    ['Direct', s.direct_detection_cells||0, 'cells with detections'],
    ['Inferred', s.inferred_cells||0, 'no direct detection'],
    ['Records', s.total_interception_records||0, 'sample detections used'],
  ];
  el.innerHTML=cards.map(c=>`<div style="background:var(--surface);border:1px solid var(--border);border-radius:var(--r);padding:10px 12px;box-shadow:var(--shadow)">
    <div style="font-size:22px;font-weight:800;color:var(--primary);line-height:1">${{c[1]}}</div>
    <div style="font-size:10px;text-transform:uppercase;letter-spacing:.07em;color:var(--muted);font-weight:700">${{c[0]}}</div>
    <div style="font-size:11px;color:var(--text2);margin-top:4px">${{c[2]}}</div>
  </div>`).join('') + `<div style="grid-column:1/-1;font-size:11px;color:var(--muted)">Confidence mix: HIGH ${{counts.HIGH||0}} · MODERATE ${{counts.MODERATE||0}} · LOW ${{counts.LOW||0}} · INFERRED ${{counts.INFERRED||0}}</div>`;
}}

// ══════════════════════════════════════════════════════════════
// FILTER LOGIC
// ══════════════════════════════════════════════════════════════
function getFiltered() {{
  if (!routeData) return [];
  let fs = routeData.features;
  if (activeMonth>0) fs=fs.filter(f=>+(f.properties||{{}}).month===activeMonth);
  if (activeTier!=='ALL') fs=fs.filter(f=>(f.properties||{{}}).risk_tier===activeTier);
  if (activeMode!=='ALL') fs=fs.filter(f=>(f.properties||{{}}).transport_mode===activeMode);
  const state=document.getElementById('fil-state').value;
  if (state!=='ALL') {{
    const ports=STATE_PORT_MAP[state]||[];
    fs=fs.filter(f=>ports.includes((f.properties||{{}}).us_port));
  }}
  const host=document.getElementById('fil-host').value;
  if (host!=='ALL') fs=fs.filter(f=>(f.properties||{{}}).commodity_group===host);
  return fs;
}}

function deduplicateCorridors(features) {{
  if (activeMonth>0) return features;
  const m=new Map();
  features.forEach(f=>{{
    const p=f.properties||{{}};
    const k=`${{p.origin_country}}|${{p.us_port}}|${{p.commodity_group}}|${{p.transport_mode}}`;
    const prev=m.get(k);
    if (!prev||+p.risk_score>+(prev.properties||{{}}).risk_score) m.set(k,f);
  }});
  return [...m.values()];
}}

function top3Routes(features) {{
  const deduped=deduplicateCorridors(features);
  const sorted=[...deduped].sort((a,b)=>+(b.properties||{{}}).risk_score-+(a.properties||{{}}).risk_score);
  if (!briefDiversity) return sorted.slice(0,3);
  const seen=new Set(), res=[];
  for (const f of sorted) {{
    const o=(f.properties||{{}}).origin_country;
    if (!seen.has(o)) {{seen.add(o); res.push(f);}}
    if (res.length===3) break;
  }}
  return res;
}}

// ══════════════════════════════════════════════════════════════
// OPERATIONAL BRIEFING
// ══════════════════════════════════════════════════════════════
function getNarrativeForRoute(route, monthOverride=null) {{
  const p=routeProps(route);
  const month=monthOverride || (activeMonth>0?activeMonth:(+p.month||new Date().getMonth()+1));
  const commodity=p.commodity_group || '';
  const origin=p.origin_country || '';
  const narr=getNarrative(month, commodity, origin);
  if (narr) return narr;
  const commodityText=(commodity || 'host commodity').replace(/_/g,' ');
  const mode=p.transport_mode || 'AIR';
  const port=p.us_port || 'selected port';
  const risk=+(p.risk_score||0);
  const fly=p.key_species || 'Pathway-scored fruit fly species';
  return {{
    headline:`${{origin || 'Origin'}} ${{commodityText}} via ${{mode}}`,
    narrative:`${{MONTH_NAMES[month] || 'Current window'}}: ${{origin || 'This origin'}} to ${{port}} combines ${{commodityText}} exposure, transport volume, and seasonal pathway pressure. Composite risk is ${{risk.toFixed(1)}}.`,
    fly,
    action:`Prioritize manifest review and secondary inspection for ${{mode}} arrivals at ${{port}}.`
  }};
}}

function countryLookup() {{
  const rows=(apiSummary?.outputs?.country_intelligence)||[];
  const m=new Map();
  rows.forEach(c=>m.set(c.origin_country,c));
  return m;
}}

function topCountriesForFeatures(features, limit=5) {{
  const lookup=countryLookup();
  const grouped=new Map();
  deduplicateCorridors(features).forEach(f=>{{
    const p=f.properties||{{}};
    const origin=p.origin_country;
    if (!origin) return;
    const row=grouped.get(origin)||{{origin_country:origin,max_risk:0,high_routes:0,route_count:0}};
    row.max_risk=Math.max(row.max_risk,+p.risk_score||0);
    row.high_routes += p.risk_tier==='HIGH'?1:0;
    row.route_count += 1;
    grouped.set(origin,row);
  }});
  let rows=[...grouped.values()];
  if (!rows.length) rows=[...lookup.values()].map(c=>({{...c}}));
  return rows.map(row=>({{...(lookup.get(row.origin_country)||{{}}),...row}}))
    .sort((a,b)=>(+b.max_risk||0)-(+a.max_risk||0) || (+b.high_routes||0)-(+a.high_routes||0))
    .slice(0,limit);
}}

function renderCountryList(rows, compact=false) {{
  if (!rows.length) return '<div class="hero-body">No country risk rows match current filters.</div>';
  return rows.map(c=>{{
    const risk=+c.max_risk||0;
    const species=String(c.key_species || 'Pathway + GBIF scored').split(';')[0].trim();
    if (compact) {{
      return `<div class="compact-country-row">
        <span><b>${{esc(c.origin_country)}}</b> · ${{(+c.high_routes||0)}} HIGH</span>
        <span class="${{tierTextClass(risk)}}"><b>${{risk.toFixed(1)}}</b></span>
      </div>`;
    }}
    return `<div class="country-risk-item">
      <div class="country-risk-header">
        <span class="country-name">${{esc(c.origin_country)}}</span>
        <span class="country-risk-score ${{tierTextClass(risk)}}">${{risk.toFixed(1)}}</span>
      </div>
      <div class="country-risk-meta">
        <span class="badge bHIGH">${{+c.high_routes||0}} HIGH routes</span>
        <span class="country-species" title="${{esc(species)}}">${{esc(species)}}</span>
      </div>
    </div>`;
  }}).join('');
}}

function renderModeBreakdown(features) {{
  const deduped=deduplicateCorridors(features);
  const modes=['AIR','SEA','LAND'];
  const labels={{AIR:'AIR',SEA:'SEA',LAND:'LAND'}};
  const colors={{AIR:'var(--air)',SEA:'var(--sea)',LAND:'var(--land)'}};
  const totals={{AIR:0,SEA:0,LAND:0}};
  const highs={{AIR:0,SEA:0,LAND:0}};
  deduped.forEach(f=>{{
    const p=f.properties||{{}};
    const mode=p.transport_mode||'AIR';
    if (!(mode in totals)) return;
    totals[mode]+=1;
    if (p.risk_tier==='HIGH') highs[mode]+=1;
  }});
  const totalHigh=modes.reduce((sum,m)=>sum+highs[m],0);
  const bars=modes.map(mode=>{{
    const pct=totalHigh?highs[mode]/totalHigh*100:0;
    return `<div class="mode-bar" style="--pct:${{pct.toFixed(0)}}%;--mode-color:${{colors[mode]}}">
      <span class="mode-label">${{labels[mode]}}</span>
      <span class="mode-pct">${{pct.toFixed(0)}}%</span>
    </div>`;
  }}).join('');
  const chart=document.getElementById('mode-breakdown-chart');
  const totalsEl=document.getElementById('mode-breakdown-totals');
  if (chart) chart.innerHTML=bars;
  if (totalsEl) totalsEl.textContent=`HIGH-tier mix: AIR ${{totalHigh?Math.round(highs.AIR/totalHigh*100):0}}%, SEA ${{totalHigh?Math.round(highs.SEA/totalHigh*100):0}}%, LAND ${{totalHigh?Math.round(highs.LAND/totalHigh*100):0}}%. Total routes: AIR ${{totals.AIR}} / SEA ${{totals.SEA}} / LAND ${{totals.LAND}}.`;
}}

function renderBriefingMonthStrip(features) {{
  const el=document.getElementById('briefing-month-strip');
  if (!el) return;
  const byMonth=Array(13).fill(null).map(()=>({{sum:0,count:0}}));
  deduplicateCorridors(features).forEach(f=>{{
    const p=f.properties||{{}};
    const m=+p.month;
    if (m>=1&&m<=12) {{byMonth[m].sum+=+p.risk_score||0; byMonth[m].count+=1;}}
  }});
  el.innerHTML=Array.from({{length:12}},(_,i)=>{{
    const m=i+1;
    const avg=byMonth[m].count?byMonth[m].sum/byMonth[m].count:0;
    const cls=avg>=70?'var(--high)':avg>=55?'var(--med)':avg>0?'var(--low)':'var(--border2)';
    const color=avg>=55?'#fff':'var(--text)';
    return `<div class="compact-month" style="background:${{cls}};color:${{color}}" title="${{MONTH_NAMES[m]}} average risk ${{avg.toFixed(1)}}">${{MONTH_NAMES[m]}}</div>`;
  }}).join('');
}}

function detectionRowsForActiveState() {{
  const state=document.getElementById('fil-state')?.value || 'ALL';
  return (COUNTY_DETECTIONS||[]).filter(r => state==='ALL' || r.state===state);
}}

function routePressureByPort(features) {{
  const out={{}};
  deduplicateCorridors(features).forEach(f=>{{
    const p=f.properties||{{}};
    const port=p.us_port;
    if (!port) return;
    const tierMult=p.risk_tier==='HIGH'?1.0:p.risk_tier==='MEDIUM'?0.45:0.15;
    out[port]=(out[port]||0)+(+p.risk_score||0)*tierMult;
  }});
  return out;
}}

function fallbackAgProfile(row) {{
  const state=row.state || 'US';
  const county=row.county || 'County';
  const defaults={{
    CA:['citrus','avocados','grapes','nursery','vegetables'],
    FL:['tropical fruit','winter vegetables','ornamentals','avocados','mangoes'],
    TX:['citrus','vegetables','row crops','nursery'],
    HI:['tropical fruit','nursery','papaya','mango','vegetables'],
    AZ:['citrus','vegetables','melons','nursery'],
  }};
  return {{
    value_m:35,
    value_label:'local host-crop proxy',
    crops:defaults[state]||['host crops','nursery','fruit trees'],
    host_alignment:0.50,
    confidence:'LOW',
    source:`${{county}} County proxy from detection + pathway context`,
    url:''
  }};
}}

function agExposureRows(features, year) {{
  const pressure=routePressureByPort(features);
  const byCounty=new Map();
  detectionRowsForActiveState().forEach(r=>{{
    const key=`${{r.state}}|${{r.county}}`;
    const row=byCounty.get(key)||{{state:r.state,county:r.county,detections:0,ports:new Set(),species:new Set()}};
    row.detections += +(r.detection_count||0);
    if (r.nearest_port) row.ports.add(r.nearest_port);
    String(r.species_list||'').split(';').forEach(s=>{{if(s.trim())row.species.add(s.trim());}});
    byCounty.set(key,row);
  }});
  const trend=1 + Math.max(0, year-2026)*0.06;
  const rows=[...byCounty.entries()].map(([key,row])=>{{
    const profile=AG_EXPOSURE_PROFILES[key] || fallbackAgProfile(row);
    const portPressure=[...row.ports].reduce((sum,p)=>sum+(pressure[p]||0),0);
    const pathwayMult=1 + Math.min(portPressure/1800,0.45);
    const detectionMult=1 + Math.min(row.detections,6)*0.10;
    const hostMult=0.55 + (+profile.host_alignment||0.5)*0.45;
    const exposedValue=(+profile.value_m||0)*hostMult*detectionMult*pathwayMult*trend;
    return {{...row, key, profile, portPressure, exposedValue, ports:[...row.ports], species:[...row.species]}};
  }}).sort((a,b)=>b.exposedValue-a.exposedValue).slice(0,5);
  return rows;
}}

function renderAgExposureForecast() {{
  const slider=document.getElementById('ag-year-slider');
  const label=document.getElementById('ag-year-label');
  const bars=document.getElementById('ag-forecast-bars');
  const note=document.getElementById('ag-forecast-note');
  if (!slider||!label||!bars||!note) return;
  const year=+slider.value||2026;
  label.textContent=String(year);
  const rows=agExposureRows(getFiltered(),year);
  if (!rows.length) {{
    bars.innerHTML='<div class="hero-body">No detection counties match the active state filter.</div>';
    note.textContent='Exposure view needs detection counties plus crop/agriculture profiles.';
    return;
  }}
  const maxVal=Math.max(...rows.map(r=>r.exposedValue),1);
  bars.innerHTML=rows.map(r=>{{
    const pct=Math.max(4,Math.min(100,r.exposedValue/maxVal*100));
    const crops=(r.profile.crops||[]).slice(0,3).join(', ');
    const title=`${{r.state}} ${{r.county}}: ${{r.profile.value_label}}; crops: ${{crops}}; source: ${{r.profile.source}}`;
    return `<div class="ag-bar-row" title="${{esc(title)}}">
      <span><b>${{esc(r.county)}}</b></span>
      <div class="ag-bar-track"><div class="ag-bar-fill" style="width:${{pct.toFixed(0)}}%"></div></div>
      <span class="${{r.exposedValue>=500?'text-high':r.exposedValue>=150?'text-med':'text-low'}}"><b>$${{r.exposedValue.toFixed(0)}}M</b></span>
      <span style="grid-column:1/-1;font-size:10px;color:var(--muted);line-height:1.35">
        ${{r.detections}} detections · ${{esc(crops)}} · ${{esc(r.profile.confidence)}} confidence · ${{esc(r.ports.join('/')||'nearby port')}}
      </span>
    </div>`;
  }}).join('');
  const top=rows[0];
  note.innerHTML=`Planning scenario, not a spread forecast: exposed value combines observed detection counties, nearby filtered pathway pressure, host-crop alignment, and a 6% annual pressure scenario. Spotted lanternfly is used only as a response precedent, not a genetic comparator; fruit flies are Diptera and SLF is Hemiptera. Top source: <a target="_blank" href="${{esc(top.profile.url||'#')}}">${{esc(top.profile.source)}}</a>.`;
}}

function renderOperationalBriefing() {{
  const filtered=getFiltered();
  const routes=top3Routes(filtered);
  const top=routes[0];
  const month=activeMonth>0?activeMonth:(top?+(top.properties||{{}}).month:new Date().getMonth()+1);
  const alert=getNarrativeForRoute(top, month);
  document.getElementById('hero-alert-headline').textContent=`${{MONTH_NAMES[month]}}: ${{alert.headline}}`;
  document.getElementById('hero-alert-body').innerHTML=`${{esc(alert.narrative)}} <span class="hero-route-fly">${{esc(alert.fly||'Multiple species')}}</span>`;
  document.getElementById('hero-alert-action').textContent=alert.action || 'Focus inspection on the highest-risk matching route.';

  const routeList=document.getElementById('hero-routes-list');
  routeList.innerHTML=routes.length?routes.map((f,i)=>{{
    const p=f.properties||{{}};
    const narr=getNarrativeForRoute(f);
    const mode=p.transport_mode||'AIR';
    return `<div class="hero-route" onclick="showRouteDetailById('${{esc(p.route_id||'')}}')">
      <div class="hero-route-top">
        <span class="hero-route-title">#${{i+1}} ${{esc(p.origin_country)}} → ${{esc(p.us_port)}} · ${{esc(mode)}}</span>
        <span class="hero-route-score ${{tierTextClass(p.risk_score)}}">${{(+p.risk_score||0).toFixed(1)}}</span>
      </div>
      <div class="hero-route-why">${{esc(narr.headline)}} — ${{esc((narr.narrative||'').split('. ')[0])}}.</div>
      <div class="hero-route-fly">${{esc(narr.fly||'Multiple species')}} · ${{esc((p.commodity_group||'').replace(/_/g,' '))}}</div>
    </div>`;
  }}).join(''):'<div class="hero-body">No surge routes match current filters.</div>';

  const countries=topCountriesForFeatures(filtered,5);
  const countryHtml=renderCountryList(countries);
  document.getElementById('hero-country-risk').innerHTML=countryHtml;
  document.getElementById('country-detail-list').innerHTML=renderCountryList(countries,true);
  renderModeBreakdown(filtered);
  renderBriefingMonthStrip(filtered);
  renderAgExposureForecast();
}}

function toggleSection(sectionId) {{
  const section=document.getElementById(sectionId+'-section');
  if (!section) return;
  const isOpen=section.classList.toggle('open');
  sectionStates[sectionId]=isOpen;
}}

function restoreSectionStates() {{
  Object.entries(sectionStates).forEach(([id,open])=>{{
    const section=document.getElementById(id+'-section');
    if (section) section.classList.toggle('open',!!open);
  }});
}}

// ══════════════════════════════════════════════════════════════
// MAP DRAWING
// ══════════════════════════════════════════════════════════════
function routeStyle(f) {{
  const p=f.properties||{{}};
  const mode=p.transport_mode||'AIR';
  const tier=p.risk_tier||'LOW';
  const col=MODE_COLORS[mode]||'#1a6fa8';
  const w=tier==='HIGH'?2.8:tier==='MEDIUM'?2:1.4;
  const s={{color:col,weight:w,opacity:tier==='HIGH'?0.85:tier==='MEDIUM'?0.7:0.5}};
  if (mode==='AIR') s.dashArray='7,4';
  if (mode==='LAND') s.dashArray='2,5';
  return s;
}}

function drawAll() {{
  drawRoutes();
  drawCountries();
  drawPorts();
}}

function drawRoutes() {{
  if (!leafletMap) return;
  if (routeLayer) leafletMap.removeLayer(routeLayer);
  const filtered=getFiltered();
  const deduped=deduplicateCorridors(filtered);
  const top=([...deduped].sort((a,b)=>+(b.properties||{{}}).risk_score-+(a.properties||{{}}).risk_score)).slice(0,150);
  routeLayer=L.geoJSON({{type:'FeatureCollection',features:top}},{{
    style:routeStyle,
    onEachFeature:(f,layer)=>{{
      const p=f.properties||{{}};
      const mode=p.transport_mode||'AIR';
      const mEmoji=mode==='AIR'?'✈':mode==='SEA'?'⛴':'🚛';
      const tCol=TIER_COLORS[p.risk_tier]||TIER_COLORS.LOW;
      const narr=getNarrative(+p.month||activeMonth,p.commodity_group,p.origin_country);
      layer.bindPopup(`
        <div style="font:13px/1.5 -apple-system,sans-serif;min-width:200px;max-width:260px">
          <div style="font-size:16px;font-weight:800;margin-bottom:6px;color:#1a2e1a">
            ${{p.origin_country}} → ${{p.us_port}}
          </div>
          <div style="display:flex;align-items:center;gap:8px;margin-bottom:6px">
            <span style="font-size:22px;font-weight:800;color:${{tCol}}">${{(+p.risk_score||0).toFixed(1)}}</span>
            <span class="badge b${{p.risk_tier==='MEDIUM'?'MED':p.risk_tier}}">${{p.risk_tier}}</span>
            <span style="font-size:12px;color:#6b8068">${{mEmoji}} ${{mode}}</span>
          </div>
          <div style="font-size:12px;color:#3d5a3d;margin-bottom:4px">
            ${{(p.commodity_group||'').replace(/_/g,' ')}} · ${{MONTH_NAMES[+p.month||0]||''}}
          </div>
          ${{narr?`<div style="font-size:11px;font-style:italic;color:#2d6a4f;margin-bottom:4px">${{narr.fly}}</div>`:''}}
          ${{narr?`<div style="font-size:11px;color:#3d5a3d">${{narr.headline}}</div>`:''}}
          <div style="margin-top:8px;padding-top:6px;border-top:1px solid #d0d9d0">
            <a href="javascript:void(0)" onclick="showRouteDetailById('${{p.route_id||''}}')" style="font-size:12px;font-weight:600;color:#2d6a4f">View route details →</a>
          </div>
        </div>`);
    }}
  }}).addTo(leafletMap);
  window._leafletMap=leafletMap;
}}

function drawCountries() {{
  if (!countryData||!leafletMap) return;
  if (countryLayer) leafletMap.removeLayer(countryLayer);
  countryLayer=L.geoJSON(countryData,{{
    pointToLayer:(f,ll)=>{{
      const p=f.properties||{{}};
      const risk=+p.max_risk||+p.risk_score||50;
      const col=risk>=70?'#b83232':risk>=55?'#c06010':'#2e8050';
      const r=Math.max(5,Math.min(18,risk/7));
      return L.circleMarker(ll,{{radius:r,fillColor:col,color:'rgba(255,255,255,.5)',weight:1.5,fillOpacity:0.75}});
    }},
    onEachFeature:(f,layer)=>{{
      const p=f.properties||{{}};
      layer.bindTooltip(`<b>${{p.country||p.origin_country}}</b><br>Max risk: ${{(+p.max_risk||+p.risk_score||0).toFixed(1)}}`,{{sticky:true}});
    }}
  }}).addTo(leafletMap);
}}

function drawPorts() {{
  if (!portData||!leafletMap) return;
  if (portLayer) leafletMap.removeLayer(portLayer);
  portLayer=L.geoJSON(portData,{{
    pointToLayer:(f,ll)=>{{
      return L.circleMarker(ll,{{radius:6,fillColor:'#1a6fa8',color:'#0d4f7a',weight:2,fillOpacity:0.9}});
    }},
    onEachFeature:(f,layer)=>{{
      const p=f.properties||{{}};
      layer.bindTooltip(`<b>${{p.port_code||p.port}}</b> — ${{p.route_count||''}} routes`,{{sticky:true}});
    }}
  }}).addTo(leafletMap);
}}

// ══════════════════════════════════════════════════════════════
// KPI CARDS
// ══════════════════════════════════════════════════════════════
function updateKPIs() {{
  const filtered=getFiltered();
  const deduped=deduplicateCorridors(filtered);
  const high=deduped.filter(f=>(f.properties||{{}}).risk_tier==='HIGH').length;
  const countries=[...new Set(deduped.map(f=>(f.properties||{{}}).origin_country).filter(Boolean))].length;
  // Peak month from currently filtered routes. If the user selected a month,
  // display that month instead of recomputing from the single-month subset.
  const monthCounts=Array(13).fill(0);
  deduped.forEach(f=>{{
    const m=+(f.properties||{{}}).month;
    if (m>=1&&m<=12) monthCounts[m]+=1;
  }});
  let peakMonth=activeMonth>0?activeMonth:0;
  if (!peakMonth) {{
    const maxCount=Math.max(...monthCounts.slice(1));
    peakMonth=maxCount>0?monthCounts.indexOf(maxCount):0;
  }}
  // Top port
  const portCount={{}};
  deduped.forEach(f=>{{const p=(f.properties||{{}}).us_port; if(p) portCount[p]=(portCount[p]||0)+1;}});
  const topPort=Object.entries(portCount).sort((a,b)=>b[1]-a[1])[0]?.[0]||'—';
  document.getElementById('kpi-high').textContent=high;
  document.getElementById('kpi-countries').textContent=countries;
  document.getElementById('kpi-peak').textContent=MONTH_NAMES[peakMonth]||'—';
  document.getElementById('kpi-port').textContent=topPort;
}}

// ══════════════════════════════════════════════════════════════
// TOP 3 ROUTES
// ══════════════════════════════════════════════════════════════
function renderTop3() {{
  const container=document.getElementById('top3-cards');
  const routes=top3Routes(getFiltered());
  if (!routes.length) {{
    container.innerHTML='<p style="font-size:12px;color:var(--muted);padding:4px 0">No routes match current filters.</p>';
    return;
  }}
  container.innerHTML=routes.map((f,i)=>{{
    const p=f.properties||{{}};
    const mode=p.transport_mode||'AIR';
    const mEmoji=mode==='AIR'?'✈':mode==='SEA'?'⛴':'🚛';
    const tCol=TIER_COLORS[p.risk_tier]||TIER_COLORS.LOW;
    const narr=getNarrative(+p.month||activeMonth,p.commodity_group,p.origin_country);
    const bCls=p.risk_tier==='MEDIUM'?'MED':p.risk_tier;
    return `<div class="route-card" onclick="showRouteDetailById('${{p.route_id||''}}')">
      <div style="display:flex;justify-content:space-between;align-items:flex-start;margin-bottom:5px">
        <div>
          <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.06em;color:var(--muted);margin-bottom:3px">#${{i+1}} surge route</div>
          <div style="font-size:15px;font-weight:800;color:var(--text)">${{p.origin_country}} → ${{p.us_port}}</div>
        </div>
        <div style="text-align:right">
          <div style="font-size:24px;font-weight:800;color:${{tCol}};letter-spacing:-.5px">${{(+p.risk_score||0).toFixed(1)}}</div>
          <span class="badge b${{bCls}}">${{p.risk_tier}}</span>
        </div>
      </div>
      <div style="font-size:12px;color:var(--text2)">
        ${{(p.commodity_group||'').replace(/_/g,' ')}}
        <span style="color:var(--muted)">·</span>
        <span style="color:${{MODE_COLORS[mode]||'var(--air)'}}"> ${{mEmoji}} ${{mode}}</span>
        ${{activeMonth===0&&p.month?'<span style="color:var(--muted)"> · Peak '+MONTH_NAMES[p.month]+'</span>':''}}
      </div>
      ${{narr?`<div style="font-size:11px;font-style:italic;color:var(--muted);margin-top:4px">${{narr.fly}}</div>`:''}}
    </div>`;
  }}).join('');
  const btn=document.getElementById('div-btn');
  if (briefDiversity) {{btn.classList.add('btn-active'); btn.textContent='▼ Score view';}}
  else {{btn.classList.remove('btn-active'); btn.textContent='★ Diverse';}}
}}

// ══════════════════════════════════════════════════════════════
// SEASONAL NARRATIVE
// ══════════════════════════════════════════════════════════════
function getNarrative(month,commodity,origin) {{
  for (const n of NARRATIVES) {{
    if (n.months.includes(month)&&n.commodity===commodity&&n.origin===origin) return n;
  }}
  for (const n of NARRATIVES) {{ if (n.months.includes(month)&&n.origin===origin) return n; }}
  for (const n of NARRATIVES) {{ if (n.months.includes(month)&&n.commodity===commodity) return n; }}
  return null;
}}

function renderWindowCard() {{
  const el=document.getElementById('window-card');
  const m=activeMonth>0?activeMonth:new Date().getMonth()+1;
  const top=top3Routes(getFiltered())[0];
  const origin=top?(top.properties||{{}}).origin_country:'MEX';
  const commodity=top?(top.properties||{{}}).commodity_group:'fresh_fruit';
  const narr=getNarrative(m,commodity,origin)||getNarrative(m,'fresh_fruit','MEX');
  if (!narr) {{el.innerHTML=''; return;}}
  el.innerHTML=`<div class="insight-card">
    <div class="insight-head">${{MONTH_NAMES[m]}}: ${{narr.headline}}</div>
    <div class="insight-body">${{narr.narrative}}</div>
    <div class="insight-action">▶ ${{narr.action}}</div>
  </div>`;
}}

// ══════════════════════════════════════════════════════════════
// ROUTE DETAIL PANEL
// ══════════════════════════════════════════════════════════════
function showRouteDetailById(routeId) {{
  if (!routeData||!routeId) return;
  const feature=routeData.features.find(f=>(f.properties||{{}}).route_id===routeId);
  if (!feature) return;
  showRouteDetail(feature.properties||{{}});
}}

function renderScoreBreakdown(features) {{
  const p=routeProps((features||[])[0]);
  const target=document.getElementById('scoreBreakdown');
  if (!target) return;
  if (!p || !Object.keys(p).length) {{
    target.innerHTML='<div class="section-label">Score breakdown</div><div class="hero-body">No route selected.</div>';
    return;
  }}

  const mode=p.transport_mode||'AIR';
  const mEmoji=mode==='AIR'?'✈':mode==='SEA'?'⛴':'🚛';
  const tCol=TIER_COLORS[p.risk_tier]||TIER_COLORS.LOW;
  const bCls=p.risk_tier==='MEDIUM'?'MED':p.risk_tier;
  const narr=getNarrative(+p.month||activeMonth,p.commodity_group,p.origin_country);
  const comps=[
    {{label:'Fly–host co-occurrence',val:+p.contrib_fly_overlap||0,col:'#b83232',desc:'Species present in origin + commodity is a known host'}},
    {{label:'Volume / exposure',       val:+p.contrib_volume||0,      col:'#1a6fa8',desc:'Cargo weight + passenger flow normalized to 0–1'}},
    {{label:'Host commodity fraction', val:+p.contrib_host||0,        col:'#c06010',desc:'Share of commodity that are confirmed fly hosts'}},
    {{label:'Route frequency',         val:+p.contrib_frequency||0,   col:'#6b8068',desc:'How often this origin–port corridor operates'}},
    {{label:'Detection proximity',     val:+p.contrib_detection||0,   col:'#2e8050',desc:'Interception records within 200 km of destination port'}},
  ];
  const maxV=Math.max(...comps.map(c=>c.val),0.01);
  const bars=comps.map(c=>`<div class="sbar">
    <div class="sbar-row"><span>${{c.label}}</span><span style="font-weight:700;color:${{c.col}}">${{(c.val*100).toFixed(1)}}%</span></div>
    <div class="sbar-track"><div class="sbar-fill" style="width:${{(c.val/maxV*100).toFixed(0)}}%;background:${{c.col}}"></div></div>
    <div style="font-size:10px;color:var(--muted);margin-top:2px">${{c.desc}}</div>
  </div>`).join('');

  target.innerHTML=`
    <div style="margin-bottom:14px">
      <div class="section-label">Route detail</div>
      <div style="font-size:20px;font-weight:800;color:var(--text);margin-bottom:6px">${{p.origin_country}} → ${{p.us_port}}</div>
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px">
        <span style="font-size:32px;font-weight:800;color:${{tCol}};letter-spacing:-.5px">${{(+p.risk_score||0).toFixed(1)}}</span>
        <div>
          <span class="badge b${{bCls}}">${{p.risk_tier}}</span><br>
          <span style="font-size:12px;color:var(--muted)">${{mEmoji}} ${{mode}} · ${{(p.commodity_group||'').replace(/_/g,' ')}} · ${{MONTH_NAMES[+p.month||0]}}</span>
        </div>
      </div>
      ${{narr?`<div style="font-size:13px;font-weight:600;color:var(--primary);margin-bottom:4px">${{narr.fly}}</div>`:''}}
    </div>
    <div style="margin-bottom:14px">
      <div class="section-label">Score components</div>
      <div style="font-size:11px;color:var(--muted);margin-bottom:8px">Surge modifier ×${{(+p.operational_surge_modifier||1).toFixed(2)}} · Seasonal ×${{(+p.seasonal_multiplier||1).toFixed(2)}} · Confidence: ${{p.confidence_level||'—'}}</div>
      ${{bars}}
    </div>
    ${{p.caveat?`<div style="font-size:11px;padding:8px 10px;background:var(--med-bg);border:1px solid var(--med-bd);border-radius:var(--r-sm);color:var(--med)">⚠ ${{p.caveat}}</div>`:''}}`;
}}

function renderWhyThisRoute(features, apiSummaryData) {{
  const p=routeProps((features||[])[0]);
  const target=document.getElementById('whyThisRoute');
  if (!target) return;
  if (!p || !Object.keys(p).length) {{
    target.innerHTML='<div class="section-label">Why this route</div><div class="hero-body">No route selected.</div>';
    return;
  }}

  const narr=getNarrative(+p.month||activeMonth,p.commodity_group,p.origin_country);
  const seasonal=+p.seasonal_multiplier||1;
  const sLabel=seasonal>=1.3?'Above-baseline — surge window':seasonal>=1.0?'Baseline activity':'Below-baseline — off-peak';
  const detNote=(+p.contrib_detection||0)<0.05?'Detection support is weak and this corridor may be under-trapped.':'Detection records nearby support this as an active pathway.';
  const summary=(apiSummaryData?.summary_stats)||{{}};
  const contextBits=[];
  if (summary.total_routes) contextBits.push(`${{summary.total_routes}} modeled routes`);
  if (summary.high_routes) contextBits.push(`${{summary.high_routes}} high-tier routes`);
  const contextLine=contextBits.length?`Dataset context: ${{contextBits.join(' · ')}}.`:'';

  let whyParagraph='';
  if (narr) {{
    whyParagraph=`<b>Why now:</b> ${{narr.narrative}} Seasonal multiplier ×${{seasonal.toFixed(2)}} indicates ${{sLabel.toLowerCase()}}. ${{detNote}}`;
  }} else {{
    whyParagraph=`Seasonal multiplier ×${{seasonal.toFixed(2)}} indicates ${{sLabel.toLowerCase()}}. ${{detNote}}`;
  }}

  target.innerHTML=`
    <div class="section-label">Why this route</div>
    <p style="font-size:12px;color:var(--text2);line-height:1.7">${{whyParagraph}}</p>
    ${{contextLine?`<p style="font-size:11px;color:var(--muted);margin-top:8px">${{contextLine}}</p>`:''}}
    ${{narr?`<div style="margin-top:10px;font-size:11px;font-weight:700;color:var(--med);padding:8px 10px;background:var(--med-bg);border-radius:var(--r-sm)">▶ Recommended action: ${{narr.action}}</div>`:''}}`;
}}

function showRouteDetail(routeProps) {{
  const sidebar=document.querySelector('aside#right-panel') || document.querySelector('aside');
  if (!sidebar || !routeProps || !Object.keys(routeProps).length) return;
  sidebar.classList.add('detail-mode');
  const title=document.getElementById('detail-route-title');
  if (title) title.textContent=`${{routeProps.origin_country||'Origin'}} → ${{routeProps.us_port||'Port'}}`;
  renderScoreBreakdown([{{properties: routeProps}}]);
  renderWhyThisRoute([{{properties: routeProps}}], apiSummaryData);
}}

function hideRouteDetail() {{
  const sidebar=document.querySelector('aside#right-panel') || document.querySelector('aside');
  if (!sidebar) return;
  sidebar.classList.remove('detail-mode');
}}

// ══════════════════════════════════════════════════════════════
// DANGER CALENDAR
// ══════════════════════════════════════════════════════════════
function renderDangerCalendar() {{
  const country=document.getElementById('cal-country').value;
  const el=document.getElementById('danger-cal');
  if (!country||!routeData) {{el.innerHTML=''; return;}}
  const byMonth=Array(13).fill(null).map(()=>{{return {{sum:0,count:0}};}});
  routeData.features.forEach(f=>{{
    const p=f.properties||{{}};
    if (p.origin_country!==country) return;
    const m=+p.month;
    if (m>=1&&m<=12) {{byMonth[m].sum+=(+p.risk_score||0); byMonth[m].count++;}}
  }});
  const cells=Array.from({{length:12}},(_,i)=>{{
    const m=i+1;
    const d=byMonth[m];
    const avg=d.count?d.sum/d.count:0;
    let bg,textCol;
    if (avg>=70) {{bg='var(--high)'; textCol='#fff';}}
    else if (avg>=55) {{bg='var(--med)'; textCol='#fff';}}
    else if (avg>=40) {{bg='var(--low)'; textCol='#fff';}}
    else {{bg='var(--border2)'; textCol='var(--muted)';}}
    const evidence=seasonalEvidenceForCountryMonth(country,m);
    const conf=evidence.length?evidence[0].confidence_level:'INFERRED';
    const badge=conf==='HIGH'?'★★★':conf==='MODERATE'?'★★':conf==='LOW'?'★':'▫';
    return `<div class="dcal-cell" style="background:${{bg}};color:${{textCol}}"
      title="${{MONTH_NAMES[m]}}: avg ${{avg.toFixed(0)}} · evidence ${{badge}}"
      onclick="document.getElementById('month-slider').value=${{m}};onMonthChange();showSeasonalEvidence('${{country}}',${{m}})">
      <span>${{MONTH_NAMES[m]}}</span><span style="font-size:8px;margin-left:2px">${{badge}}</span></div>`;
  }}).join('');
  el.innerHTML=`<div class="dcal">${{cells}}</div>`;
}}

function seasonalEvidenceForCountryMonth(country, month) {{
  const rows=(SEASONAL_EVIDENCE||[]).filter(r => {{
    const origins=String(r.origin_countries||'').split(',').map(x=>x.trim());
    return +r.month===month && origins.includes(country);
  }});
  return rows.sort((a,b)=>(+b.max_risk||0)-(+a.max_risk||0)).slice(0,5);
}}

function showSeasonalEvidence(country, month) {{
  const rows=seasonalEvidenceForCountryMonth(country, month);
  const modal=document.getElementById('seasonal-evidence-modal');
  const title=document.getElementById('seasonal-evidence-title');
  const body=document.getElementById('seasonal-evidence-body');
  title.textContent=`${{country}} · ${{MONTH_NAMES[month]}} evidence`;
  if (!rows.length) {{
    body.innerHTML=`<div class="evidence-caveat">No species-specific evidence cell was generated for this country-month. The calendar color still reflects route-level risk from the scored pathway table.</div>`;
    modal.classList.add('open');
    return;
  }}
  body.innerHTML=rows.map(r=>{{
    const layers=Array.isArray(r.evidence)?r.evidence:[];
    const caveats=Array.isArray(r.caveats)?r.caveats:[];
    const layerHtml=layers.map(l=>`<div class="evidence-layer">
      <div class="evidence-pill">${{l.layer||'evidence'}}</div>
      <div style="font-size:13px;color:var(--text);margin-top:6px">${{l.description||''}}</div>
      <div style="font-size:11px;color:var(--muted);margin-top:4px">Source: ${{sourceToLinks(l.source)}} · sample: ${{l.sample_size??'proxy'}}</div>
    </div>`).join('');
    const caveatHtml=caveats.map(c=>`<li>${{c}}</li>`).join('');
    return `<div style="border-bottom:1px solid var(--border);padding:0 0 14px;margin-bottom:14px">
      <div class="flex-between gap12">
        <div>
          <div style="font-size:15px;font-weight:800;color:var(--text);font-style:italic">${{r.species_name}}</div>
          <div style="font-size:12px;color:var(--text2)">${{r.common_name||''}} · ${{r.region}} · ${{r.host_groups||'host pathways'}}</div>
        </div>
        <div style="text-align:right">
          <div class="evidence-pill">${{r.confidence_level}}</div>
          <div style="font-size:18px;font-weight:800;color:var(--high);margin-top:4px">${{(+r.max_risk||0).toFixed(1)}}</div>
        </div>
      </div>
      <p class="fine mt6">${{r.confidence_rationale||''}}</p>
      ${{layerHtml}}
      <div class="evidence-caveat mt10"><b>Caveats:</b><ul style="margin:6px 0 0 18px">${{caveatHtml}}</ul></div>
    </div>`;
  }}).join('');
  modal.classList.add('open');
}}

function closeSeasonalEvidence() {{
  document.getElementById('seasonal-evidence-modal').classList.remove('open');
}}

// ══════════════════════════════════════════════════════════════
// CONTROLS
// ══════════════════════════════════════════════════════════════
function onMonthChange() {{
  activeMonth=+document.getElementById('month-slider').value;
  document.getElementById('month-label').textContent=MONTH_NAMES[activeMonth];
  document.getElementById('all-months-btn').classList.remove('btn-active');
  applyFilters();
}}

function setAllMonths() {{
  activeMonth=0;
  document.getElementById('month-label').textContent='ALL';
  document.getElementById('all-months-btn').classList.add('btn-active');
  applyFilters();
}}

function setMode(m) {{
  activeMode=m;
  document.querySelectorAll('.seg-btn[data-mode]').forEach(b=>b.classList.toggle('active',b.dataset.mode===m));
  applyFilters();
}}

function setTier(t) {{
  activeTier=t;
  document.querySelectorAll('.seg-btn[data-tier]').forEach(b=>b.classList.toggle('active',b.dataset.tier===t));
  applyFilters();
}}

function applyFilters() {{
  drawRoutes();
  updateKPIs();
  renderOperationalBriefing();
  renderTop3();
  renderWindowCard();
  updateStatBar();
  restoreSectionStates();
}}

function toggleDiversity() {{
  briefDiversity=!briefDiversity;
  renderTop3();
}}

// Play animation
function togglePlay() {{
  const btn=document.getElementById('play-btn');
  if (playInterval) {{
    clearInterval(playInterval);
    playInterval=null;
    btn.textContent='▶ Play';
    return;
  }}
  btn.textContent='⏸ Pause';
  document.getElementById('all-months-btn').classList.remove('btn-active');
  if (activeMonth===0) {{activeMonth=1; document.getElementById('month-slider').value=1; document.getElementById('month-label').textContent=MONTH_NAMES[1]; applyFilters();}}
  playInterval=setInterval(()=>{{
    activeMonth=activeMonth>=12?1:activeMonth+1;
    document.getElementById('month-slider').value=activeMonth;
    document.getElementById('month-label').textContent=MONTH_NAMES[activeMonth];
    applyFilters();
  }},1000);
}}

// ══════════════════════════════════════════════════════════════
// STAT BAR
// ══════════════════════════════════════════════════════════════
function updateStatBar() {{
  const filtered=getFiltered();
  const deduped=deduplicateCorridors(filtered);
  const high=deduped.filter(f=>(f.properties||{{}}).risk_tier==='HIGH').length;
  const med=deduped.filter(f=>(f.properties||{{}}).risk_tier==='MEDIUM').length;
  const low=deduped.filter(f=>(f.properties||{{}}).risk_tier==='LOW').length;
  document.getElementById('sb-routes').textContent=`${{deduped.length}} routes`;
  document.getElementById('sb-high').textContent=`HIGH: ${{high}}`;
  document.getElementById('sb-med').textContent=`MED: ${{med}}`;
  document.getElementById('sb-low').textContent=`LOW: ${{low}}`;
  const parts=[];
  if (activeMonth>0) parts.push(MONTH_NAMES[activeMonth]);
  if (activeTier!=='ALL') parts.push(activeTier);
  if (activeMode!=='ALL') parts.push(activeMode);
  const state=document.getElementById('fil-state').value;
  const host=document.getElementById('fil-host').value;
  if (state!=='ALL') parts.push(state);
  if (host!=='ALL') parts.push(host.replace(/_/g,' '));
  document.getElementById('sb-filter').textContent=parts.length?'Filter: '+parts.join(' · '):'';
}}

// ══════════════════════════════════════════════════════════════
// TAB NAVIGATION (top)
// ══════════════════════════════════════════════════════════════
function switchTopTab(tab) {{
  document.querySelectorAll('.top-tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===tab));
  // Tab views are all on the map; just update bottom panel for non-map tabs
  if (tab!=='map') {{
    const tabToBottom={{country:'country',species:'species',seasonal:'seasonal',methods:'methods'}};
    if (tabToBottom[tab]) {{
      switchBottomTab(tabToBottom[tab]);
      if (!isPanelOpen) toggleBottomPanel();
    }}
  }}
}}

// ══════════════════════════════════════════════════════════════
// BOTTOM PANEL
// ══════════════════════════════════════════════════════════════
function switchBottomTab(tab) {{
  activeBottomTab=tab;
  document.querySelectorAll('.bottom-tab').forEach(t=>t.classList.toggle('active',t.dataset.btab===tab));
  document.querySelectorAll('.btab-content').forEach(c=>{{c.style.display='none';}});
  const el=document.getElementById('btab-'+tab);
  if (el) el.style.display='block';
}}

function toggleBottomPanel() {{
  isPanelOpen=!isPanelOpen;
  const panel=document.getElementById('bottom-panel');
  const btn=document.getElementById('bottom-toggle');
  panel.classList.toggle('open',isPanelOpen);
  btn.textContent=isPanelOpen?'▼ Close Panel':'▲ Open Panel';
  if (isPanelOpen&&activeBottomTab==='seasonal') renderDangerCalendar();
}}

// ══════════════════════════════════════════════════════════════
// DRAG-TO-RESIZE
// ══════════════════════════════════════════════════════════════
(function() {{
  const handle=document.getElementById('drag-handle');
  const panel=document.getElementById('right-panel');
  let dragging=false, startX=0, startW=0;
  handle.addEventListener('mousedown',e=>{{
    dragging=true; startX=e.clientX; startW=panel.offsetWidth;
    handle.classList.add('dragging');
    document.body.style.cursor='col-resize';
    document.body.style.userSelect='none';
  }});
  document.addEventListener('mousemove',e=>{{
    if (!dragging) return;
    const delta=startX-e.clientX;
    const newW=Math.min(520,Math.max(240,startW+delta));
    panel.style.width=newW+'px';
    if (leafletMap) leafletMap.invalidateSize();
  }});
  document.addEventListener('mouseup',()=>{{
    if (!dragging) return;
    dragging=false;
    handle.classList.remove('dragging');
    document.body.style.cursor='';
    document.body.style.userSelect='';
  }});
}})();

// ══════════════════════════════════════════════════════════════
// THEME TOGGLE
// ══════════════════════════════════════════════════════════════
function toggleTheme() {{
  isDark=!isDark;
  document.body.classList.toggle('dark',isDark);
  document.getElementById('theme-toggle').textContent=isDark?'🌙':'☀';
  if (isDark) {{
    window._tileDark.addTo(leafletMap);
    leafletMap.removeLayer(window._tileLight);
  }} else {{
    window._tileLight.addTo(leafletMap);
    leafletMap.removeLayer(window._tileDark);
  }}
}}

// ══════════════════════════════════════════════════════════════
// BOOT
// ══════════════════════════════════════════════════════════════
initMap();
loadData();
</script>
</body>
</html>"""


# ---------------------------------------------------------------------------
# Build report page
# ---------------------------------------------------------------------------

def build_report_html() -> str:
    reports = _load_reports()
    summary = _load_api_summary()
    seasonal_evidence, seasonal_evidence_summary = _load_seasonal_evidence()
    outputs = summary.get("outputs", {})
    ts = summary.get("generated_at", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"))

    species_list = outputs.get("species_watchlist", [])
    country_intel = outputs.get("country_intelligence", [])
    state_comp = outputs.get("state_comparison", [])
    emerging = _csv_records(EXPORTS / "emerging_species_risk.csv") or outputs.get("emerging_species", [])
    surv_gaps = _csv_records(EXPORTS / "surveillance_gap_routes.csv") or outputs.get("surveillance_gaps", [])
    validation_lift = _csv_records(EXPORTS / "validation_tier_lift.csv")
    model_metrics = _csv_records(EXPORTS / "model_bakeoff_metrics.csv")

    def _rows(items, fn):
        return "".join(fn(i) for i in items) if items else \
               '<tr><td colspan="10" style="color:var(--muted);font-style:italic;padding:12px">No data in current run.</td></tr>'

    def species_row(s):
        risk = s.get("max_pathway_risk", 0)
        tc = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        return (f"<tr><td style='font-style:italic'><b>{s.get('species','')}</b></td>"
                f"<td>{s.get('common_name','')}</td><td>{s.get('state','')}</td>"
                f"<td>{s.get('watch_ports','')}</td><td>{(s.get('origin_countries','') or '')[:45]}</td>"
                f"<td class='num {tc}'>{risk:.1f}</td>"
                f"<td style='font-size:11px;font-style:italic'>{(s.get('example_hosts','') or '')[:40]}</td></tr>")

    def country_row(c):
        mn = c.get("peak_month", 0)
        pk = MONTH_NAMES[mn] if 0 < mn < 13 else "—"
        risk = c.get("max_risk", 0)
        tc = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        key_species = _public_species_note(c.get("key_species", ""))
        return (f"<tr><td><b>{c['origin_country']}</b></td>"
                f"<td class='num {tc}'>{risk:.1f}</td>"
                f"<td class='num'>{c.get('high_routes',0)}</td>"
                f"<td class='num'>{c.get('route_count',0)}</td><td>{pk}</td>"
                f"<td style='font-size:11px'>{(c.get('primary_pathway','') or '')[:50]}</td>"
                f"<td style='font-size:11px;font-style:italic'>{key_species[:70]}</td></tr>")

    def emerg_row(e):
        risk = float(e.get("pathway_risk_score") or e.get("max_route_risk") or e.get("risk_score") or 0)
        early = float(e.get("early_warning_score") or 0)
        tc = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        return (f"<tr><td style='font-style:italic'>{e.get('species','')}</td>"
                f"<td>{e.get('origin_country','')}</td><td>{e.get('us_port','')}</td>"
                f"<td>{str(e.get('commodity_group','')).replace('_',' ')}</td>"
                f"<td class='num'>{int(float(e.get('month') or 0)) or '—'}</td>"
                f"<td class='num text-high'>{early:.1f}</td>"
                f"<td class='num {tc}'>{risk:.1f}</td>"
                f"<td style='font-size:11px'>{(e.get('why_flagged') or e.get('reason') or '')[:95]}</td></tr>")

    def gap_row(g):
        risk = float(g.get("max_risk") or g.get("risk_score") or 0)
        mean = float(g.get("mean_risk") or 0)
        det = float(g.get("min_detection_count") or g.get("detection_count_200km") or 0)
        tc = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        return (f"<tr><td>{g.get('origin_country','')}</td><td>{g.get('us_port','')}</td>"
                f"<td>{str(g.get('commodity_group','')).replace('_',' ')}</td>"
                f"<td class='num {tc}'>{risk:.1f}</td>"
                f"<td class='num'>{mean:.1f}</td>"
                f"<td class='num'>{det:.0f}</td></tr>")

    def evidence_row(e):
        risk = float(e.get("max_risk") or 0)
        tc = "text-high" if risk >= 70 else ("text-med" if risk >= 55 else "text-low")
        return (f"<tr><td style='font-style:italic'>{_safe(e.get('species_name'))}</td>"
                f"<td>{_safe(e.get('region'))}</td><td>{_safe(e.get('month_label'))}</td>"
                f"<td class='num {tc}'>{risk:.1f}</td>"
                f"<td class='num'>{int(float(e.get('interception_count') or 0))}</td>"
                f"<td class='num'>{int(float(e.get('evidence_layer_count') or 0))}</td>"
                f"<td>{_safe(e.get('confidence_level'))}</td>"
                f"<td style='font-size:11px'>{_safe(e.get('confidence_rationale'))[:110]}</td></tr>")

    ai_html = ""
    for r in reports:
        ai_html += f"""<div style="margin-bottom:20px;padding:14px 16px;background:var(--surface);
          border:1px solid var(--border);border-radius:var(--r);box-shadow:var(--shadow)">
          <div style="font-size:13px;font-weight:700;color:var(--primary);margin-bottom:8px">
            Q: {r.get('question','')}</div>
          <div style="font-size:13px;color:var(--text2);line-height:1.65">{r.get('answer','')}</div>
        </div>"""

    seasonal_html = "".join(f"""<div style="margin-bottom:12px;padding:12px 14px;
      background:var(--surface);border:1px solid var(--border);border-radius:var(--r)">
      <div style="font-size:10px;font-weight:700;text-transform:uppercase;letter-spacing:.07em;
        color:var(--muted);margin-bottom:4px">{n['origin']} · {n['commodity'].replace('_',' ')} · Months: {', '.join(MONTH_NAMES[m] for m in n['months'])}</div>
      <div style="font-size:13px;font-weight:700;color:var(--primary);margin-bottom:5px">{n['headline']}</div>
      <div style="font-size:12px;color:var(--text2);line-height:1.6;margin-bottom:6px">{n['narrative']}</div>
      <div style="font-size:11px;font-weight:600;color:var(--med)">▶ {n['action']}</div>
    </div>""" for n in SEASONAL_NARRATIVES)

    mode_pressure = outputs.get("mode_pressure", [])
    mode_total = sum(float(m.get("route_count", 0) or 0) for m in mode_pressure) or 1
    mode_bars = "".join(
        f"""<div class="viz-row"><span>{_safe(m.get('transport_mode'))}</span>
        <div class="viz-track"><div style="width:{float(m.get('route_count',0) or 0)/mode_total*100:.1f}%;background:var(--primary)"></div></div>
        <b>{int(float(m.get('route_count',0) or 0))}</b></div>"""
        for m in mode_pressure
    )
    lift_bars = "".join(
        f"""<div class="viz-row"><span>{_safe(r.get('risk_tier'))}</span>
        <div class="viz-track"><div style="width:{min(float(r.get('mean_det',0) or 0)/5*100,100):.1f}%;background:{'var(--high)' if r.get('risk_tier')=='HIGH' else 'var(--med)' if r.get('risk_tier')=='MEDIUM' else 'var(--low)'}"></div></div>
        <b>{float(r.get('mean_det',0) or 0):.2f}</b></div>"""
        for r in validation_lift
    )
    best_model = next((m for m in model_metrics if m.get("model") == "logistic_regression"), model_metrics[0] if model_metrics else {})
    top_countries_viz = country_intel[:6]
    country_bars = "".join(
        f"""<div class="viz-row"><span>{_safe(c.get('origin_country'))}</span>
        <div class="viz-track"><div style="width:{min(float(c.get('max_risk',0) or 0),100):.1f}%;background:var(--high)"></div></div>
        <b>{float(c.get('max_risk',0) or 0):.1f}</b></div>"""
        for c in top_countries_viz
    )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>FFED Full Pathway Risk Report</title>
<style>
{SHARED_CSS}
  html, body {{ height: auto; overflow: auto; }}
  body {{ max-width: 1440px; margin: 0 auto; padding: 24px 28px 80px; }}
  h1 {{ font-size: 26px; font-weight: 800; letter-spacing: -.3px; margin-bottom: 4px; }}
  h2 {{ font-size: 18px; font-weight: 700; color: var(--primary); margin: 32px 0 12px;
       border-bottom: 2px solid var(--primary-lt); padding-bottom: 8px; }}
  h3 {{ font-size: 14px; font-weight: 700; margin: 16px 0 8px; }}
  p {{ font-size: 13px; color: var(--text2); line-height: 1.65; margin-bottom: 10px; }}
  .metric {{ display: inline-block; margin-right: 28px; margin-bottom: 12px; }}
  .report-grid {{ display: grid; grid-template-columns: minmax(0, 7fr) minmax(280px, 3fr); gap: 24px; align-items: start; }}
  .report-main {{ min-width: 0; }}
  .report-side {{ position: sticky; top: 16px; display: grid; gap: 14px; }}
  .report-search {{ position: sticky; top: 0; z-index: 20; background: var(--bg); padding: 10px 0 14px; }}
  #report-search {{ width: 100%; border: 1px solid var(--border); border-radius: var(--r); padding: 11px 14px; font-size: 14px; background: var(--surface); color: var(--text); box-shadow: var(--shadow); }}
  .visual-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: var(--r); padding: 14px 16px; box-shadow: var(--shadow); }}
  .visual-card h3 {{ margin: 0 0 10px; color: var(--primary); }}
  .viz-row {{ display: grid; grid-template-columns: 62px 1fr 50px; gap: 8px; align-items: center; margin: 7px 0; font-size: 11px; color: var(--text2); }}
  .viz-track {{ height: 8px; border-radius: 99px; background: var(--border2); overflow: hidden; }}
  .viz-track div {{ height: 100%; border-radius: 99px; }}
  .source-list a {{ display: block; font-size: 12px; margin: 5px 0; }}
  .data-section.is-hidden {{ display: none; }}
  mark {{ background: #fff2a8; color: #1a2e1a; padding: 0 2px; border-radius: 2px; }}
  .evidence-caveat {{ padding: 10px 12px; background: #fff7e6; border: 1px solid #f2d29b; color: #694100; border-radius: var(--r); font-size: 12px; line-height: 1.55; }}
  @media (max-width: 980px) {{ .report-grid {{ grid-template-columns: 1fr; }} .report-side {{ position: static; }} }}
</style>
</head>
<body>

<div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:24px;
  padding-bottom:16px;border-bottom:2px solid var(--primary)">
  <div>
    <div style="font-size:12px;color:var(--muted);margin-bottom:6px">
      <a href="dashboard.html">← Back to Dashboard</a>
    </div>
    <h1 style="color:var(--primary)">APHIS Fruit Fly Pathway Risk — Full Report</h1>
    <div style="font-size:12px;color:var(--muted);margin-top:4px">
      <span class="dot-live"></span>Generated: {ts} · USDA APHIS Plant Protection &amp; Quarantine
    </div>
  </div>
</div>

<div class="report-search">
  <input id="report-search" type="search" placeholder="Search country, route, host, baggage, avocado, fruit fly, metric..." oninput="filterReport(this.value)">
</div>

<div class="report-grid">
<main class="report-main">

<section class="data-section">
<h2>Key Metrics</h2>
<div>
  <div class="metric"><div class="kpi-number text-high">573</div><div class="kpi-label">HIGH-tier routes</div></div>
  <div class="metric"><div class="kpi-number" style="color:var(--primary)">2,856</div><div class="kpi-label">Total routes scored</div></div>
  <div class="metric"><div class="kpi-number" style="color:var(--air)">90%</div><div class="kpi-label">Holdout Precision@10</div></div>
  <div class="metric"><div class="kpi-number" style="color:var(--sea)">0.980</div><div class="kpi-label">Best model ROC AUC</div></div>
  <div class="metric"><div class="kpi-number text-med">9.01×</div><div class="kpi-label">HIGH vs LOW detection density</div></div>
</div>
<p>Top 10% of ranked routes accounts for <b>34.1%</b> of historical detection signal. Top 20% covers <b>60.8%</b>. HIGH-tier routes show 9.01× the detection density of LOW-tier routes. Temporal holdout Precision@10: 90%.</p>
<p class="fine">Metric definitions: Precision@10 is the share of top-10 scored routes with APHIS detections near the destination port in the holdout window. ROC AUC is the model's holdout ability to rank detection-positive corridors above detection-negative corridors. Lift compares mean nearby detections on HIGH-tier routes against LOW-tier routes.</p>
</section>

<section class="data-section">
<h2>Species Watchlist</h2>
<p>Quarantine fruit fly species aligned to high-risk pathway routes by receiving state.</p>
<div style="overflow-x:auto"><table class="data-table">
  <thead><tr><th>Species</th><th>Common name</th><th>State</th><th>Watch ports</th><th>Origin countries</th><th class="num">Max route risk</th><th>Example hosts</th></tr></thead>
  <tbody>{_rows(species_list, species_row)}</tbody>
</table></div>
</section>

<section class="data-section">
<h2>Country Risk Intelligence</h2>
<div style="overflow-x:auto"><table class="data-table">
  <thead><tr><th>Country</th><th class="num">Max Risk</th><th class="num">HIGH</th><th class="num">Total Routes</th><th>Peak</th><th>Primary Pathway</th><th>Key Species</th></tr></thead>
  <tbody>{_rows(country_intel[:25], country_row)}</tbody>
</table></div>
<p class="fine">“Not in curated watchlist” means the country is not yet mapped in the small dashboard species-profile table; route risk still uses pathway exposure, host matching, GBIF-derived fly overlap, and APHIS detection context.</p>
</section>

<section class="data-section">
<h2>Emerging Species Early Warning</h2>
<div style="overflow-x:auto"><table class="data-table">
  <thead><tr><th>Species</th><th>Origin</th><th>Port</th><th>Host</th><th class="num">Month</th><th class="num">Early warning</th><th class="num">Pathway risk</th><th>Watch reason</th></tr></thead>
  <tbody>{_rows(emerging[:25], emerg_row)}</tbody>
</table></div>
<p class="fine">This is a forward-looking pathway plausibility screen, not a confirmed interception claim. It combines species watchlist logic with the already scored route exposure.</p>
</section>

<section class="data-section">
<h2>Surveillance Coverage Gaps</h2>
<p><b>Recommended use:</b> Treat this table as a trap-placement and review queue. These are high pathway-risk corridors with weak nearby detection support, so officers should review coverage before interpreting low detections as low risk.</p>
<div style="overflow-x:auto"><table class="data-table">
  <thead><tr><th>Origin</th><th>Port</th><th>Host</th><th class="num">Max Risk</th><th class="num">Mean Risk</th><th class="num">Min detections</th></tr></thead>
  <tbody>{_rows(surv_gaps[:25], gap_row)}</tbody>
</table></div>
</section>

<section class="data-section">
<h2>Seasonal Intelligence Calendar</h2>
<div class="visual-card" style="box-shadow:none;margin-bottom:12px">
  <h3>Evidence Health</h3>
  <p>This seasonal layer contains <b>{seasonal_evidence_summary.get('total_cells', 0)}</b> species-state-month cells. Direct detection cells: <b>{seasonal_evidence_summary.get('direct_detection_cells', 0)}</b>. Inferred cells: <b>{seasonal_evidence_summary.get('inferred_cells', 0)}</b>. Total project detection records represented: <b>{seasonal_evidence_summary.get('total_interception_records', 0)}</b>.</p>
  <p class="fine">Cells marked INFERRED have no direct detection record for that exact species/state/month. They are retained as planning hypotheses because they are backed by route exposure, fly-host overlap, and seasonality proxies.</p>
</div>
{seasonal_html}
<p class="fine">Seasonal reference combines route-month risk outputs with APHIS host biology, APHIS fruit fly program context, and public trade/import seasonality assumptions. Source links are listed in the evidence panel.</p>

<h3>Traceable Seasonal Evidence Sample</h3>
<div style="overflow-x:auto"><table class="data-table">
  <thead><tr><th>Species</th><th>Region</th><th>Month</th><th class="num">Max Risk</th><th class="num">Detections</th><th class="num">Layers</th><th>Confidence</th><th>Rationale</th></tr></thead>
  <tbody>{_rows(sorted(seasonal_evidence, key=lambda e: (float(e.get('max_risk') or 0), int(e.get('interception_count') or 0)), reverse=True)[:20], evidence_row)}</tbody>
</table></div>

<div class="evidence-caveat mt10">
  <b>Permanent seasonal caveats:</b>
  <ul style="margin:6px 0 0 18px">
    <li>Surveillance bias: months with more inspections or traps can produce more detections.</li>
    <li>Reporting lag: recent months may undercount detections.</li>
    <li>Inferred cells are hypotheses, not confirmed findings.</li>
    <li>Courier and USPS pathways are acknowledged data gaps in this public prototype.</li>
  </ul>
</div>
</section>

<section class="data-section">
<h2>Score Methodology</h2>
<p>The risk score is a weighted composite of five indicators applied to each origin–port–commodity–month corridor:</p>
<table class="data-table">
  <thead><tr><th>Component</th><th class="num">Weight</th><th>Definition</th></tr></thead>
  <tbody>
    <tr><td style="font-weight:600;color:var(--high)">Fly–host co-occurrence</td><td class="num">35%</td><td>Verified presence of quarantine fruit fly species in origin country × commodity is a confirmed host (APHIS Fruit Fly Host Lists)</td></tr>
    <tr><td style="font-weight:600;color:var(--air)">Volume / exposure</td><td class="num">25%</td><td>Cargo weight (kg) and passenger volume normalized to 0–1 index. Source: BTS T-100, Census FT-920.</td></tr>
    <tr><td style="font-weight:600;color:var(--med)">Host commodity fraction</td><td class="num">20%</td><td>Proportion of the commodity group that are confirmed fly hosts per APHIS host lists.</td></tr>
    <tr><td style="font-weight:600;color:var(--muted)">Route frequency</td><td class="num">10%</td><td>Normalized count of route-months in the dataset. Frequent corridors carry sustained pressure.</td></tr>
    <tr><td style="font-weight:600;color:var(--low)">Detection proximity</td><td class="num">10%</td><td>Count of APHIS interception records within 200 km of the destination port. Under-trapped corridors may appear artificially low.</td></tr>
  </tbody>
</table>
<p style="margin-top:10px">A <b>seasonal multiplier</b> (×0.6–×1.5) is applied post-scoring based on origin-country phenology and US import calendar. An <b>operational surge modifier</b> adjusts for recent detection spikes and port alert levels.</p>
</section>

<section class="data-section">
<h2>Data Sources &amp; Honest Gaps</h2>
<table class="data-table">
  <thead><tr><th>Source</th><th>Used for</th><th>Known limitation</th></tr></thead>
  <tbody>
    <tr><td>BTS T-100 International Segment</td><td>Passenger volume by route</td><td>Last-segment only — passenger origin is a proxy, not true OD.</td></tr>
    <tr><td>US Census FT-920 (Foreign Trade)</td><td>Cargo weight by commodity + country + port</td><td>HS code aggregations — commodity groups are approximate.</td></tr>
    <tr><td>GBIF occurrence records</td><td>Species × country co-occurrence</td><td>Observation bias; not official regulatory pest status. IPPC/NAPPO should be used for status updates.</td></tr>
    <tr><td>APHIS interception records</td><td>Detection proximity signal</td><td>Detection-effort bias — under-trapped corridors appear low-risk.</td></tr>
    <tr><td>APHIS Fruit Fly Host Lists</td><td>Commodity-to-host mapping</td><td>Specific commodities (avocado, melons, blueberries) are tracked individually in route data but mapped to broad groups.</td></tr>
    <tr><td>USPS / express courier</td><td>Not modeled</td><td>Acknowledged gap. APHIS AQI has authority but credible public operational volume data is unavailable.</td></tr>
  </tbody>
</table>
<p class="fine mt6">Upgrade path: Census porthsimport API (port × HS × country × mode) · BTS DB1B/OD40 (passenger origin–destination) · IPPC/NAPPO official pest reports · USDA FATUS for commodity-level agricultural import volumes.</p>
</section>

<section class="data-section">
<h2>Analytical Findings</h2>
{ai_html if ai_html else '<p style="color:var(--muted);font-style:italic">No pre-computed analytical findings in current run.</p>'}
</section>

</main>

<aside class="report-side">
  <div class="visual-card">
    <h3>Top Country Risk</h3>
    {country_bars or '<p class="fine">No country bars available.</p>'}
  </div>
  <div class="visual-card">
    <h3>Pathway Mix</h3>
    {mode_bars or '<p class="fine">No pathway mix available.</p>'}
  </div>
  <div class="visual-card">
    <h3>Detection Lift</h3>
    {lift_bars or '<p class="fine">No tier lift data available.</p>'}
    <p class="fine mt6">Mean detections per route by tier; HIGH / LOW = 9.01× in the current validation export.</p>
  </div>
  <div class="visual-card">
    <h3>Model Validation</h3>
    <div class="metric"><div class="kpi-number" style="color:var(--sea)">{float(best_model.get('roc_auc_holdout', 0) or 0):.3f}</div><div class="kpi-label">ROC AUC</div></div>
    <div class="metric"><div class="kpi-number text-high">{float(best_model.get('precision_at_10_holdout', 0) or 0)*100:.0f}%</div><div class="kpi-label">Precision@10</div></div>
    <p class="fine">Best current model: {_safe(best_model.get('model', 'n/a'))}. Compared against shared holdout labels in the model bake-off export.</p>
  </div>
  <div class="visual-card source-list">
    <h3>Evidence Links</h3>
    <a target="_blank" href="{SOURCE_LINKS['aphis_fruit_flies']}">APHIS Exotic Fruit Flies</a>
    <a target="_blank" href="{SOURCE_LINKS['aphis_host_lists']}">APHIS Fruit Fly Host Lists</a>
    <a target="_blank" href="{SOURCE_LINKS['aphis_medfly']}">APHIS Medfly host examples</a>
    <a target="_blank" href="{SOURCE_LINKS['aphis_emergency_funds']}">APHIS $129.2M emergency funds</a>
    <a target="_blank" href="{SOURCE_LINKS['bts_t100']}">BTS T-100 route data</a>
    <a target="_blank" href="{SOURCE_LINKS['census_porths']}">Census port HS imports API</a>
    <a target="_blank" href="{SOURCE_LINKS['gbif']}">GBIF Tephritidae occurrence search</a>
    <a target="_blank" href="{SOURCE_LINKS['gbif_taxon']}">GBIF Tephritidae taxon record</a>
    <a target="_blank" href="{SOURCE_LINKS['ippc_reporting']}">IPPC official pest reporting summary</a>
  </div>
  <div class="visual-card source-list">
    <h3>Outbreak Cost Context</h3>
    <a target="_blank" href="{SOURCE_LINKS['bloomberg_fight']}">Bloomberg Government invasion context</a>
    <a target="_blank" href="{SOURCE_LINKS['south_florida_loss']}">South Florida $4.1M outbreak loss report</a>
    <a target="_blank" href="{SOURCE_LINKS['aphis_emergency_funds']}">APHIS emergency funding announcement</a>
    <p class="fine mt6">These sources explain why pathway triage matters. They provide operational and economic context, not direct inputs to the route score.</p>
  </div>
</aside>
</div>

<div style="margin-top:40px;padding-top:16px;border-top:1px solid var(--border);
  font-size:11px;color:var(--muted);display:flex;justify-content:space-between;align-items:center">
  <span>USDA APHIS Plant Protection &amp; Quarantine · Fruit Fly Exclusion &amp; Detection Program</span>
  <span>
    <a href="dashboard.html">← Dashboard</a> ·
    <a href="https://ffed-hackathon-mahanyas.s3.us-west-2.amazonaws.com/dashboard.html" target="_blank">Live version</a>
  </span>
</div>

<script>
function filterReport(q) {{
  const query = (q || '').trim().toLowerCase();
  document.querySelectorAll('.data-section').forEach(sec => {{
    const hit = !query || sec.textContent.toLowerCase().includes(query);
    sec.classList.toggle('is-hidden', !hit);
  }});
}}
</script>

</body>
</html>"""


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def run() -> Path:
    EXPORTS.mkdir(parents=True, exist_ok=True)
    OUT_FILE.write_text(build_html(), encoding="utf-8")
    REPORT_FILE.write_text(build_report_html(), encoding="utf-8")
    print(f"Dashboard: {OUT_FILE} ({OUT_FILE.stat().st_size // 1024} KB)")
    print(f"Report:    {REPORT_FILE} ({REPORT_FILE.stat().st_size // 1024} KB)")
    return OUT_FILE


if __name__ == "__main__":
    run()
