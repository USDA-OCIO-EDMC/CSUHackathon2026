from __future__ import annotations

import math
from collections import defaultdict
from datetime import date
from typing import Iterable

STATE_ABBREV = {
    "AL": "Alabama", "AK": "Alaska", "AZ": "Arizona", "AR": "Arkansas", "CA": "California",
    "CO": "Colorado", "CT": "Connecticut", "DE": "Delaware", "FL": "Florida", "GA": "Georgia",
    "HI": "Hawaii", "ID": "Idaho", "IL": "Illinois", "IN": "Indiana", "IA": "Iowa",
    "KS": "Kansas", "KY": "Kentucky", "LA": "Louisiana", "ME": "Maine", "MD": "Maryland",
    "MA": "Massachusetts", "MI": "Michigan", "MN": "Minnesota", "MS": "Mississippi", "MO": "Missouri",
    "MT": "Montana", "NE": "Nebraska", "NV": "Nevada", "NH": "New Hampshire", "NJ": "New Jersey",
    "NM": "New Mexico", "NY": "New York", "NC": "North Carolina", "ND": "North Dakota", "OH": "Ohio",
    "OK": "Oklahoma", "OR": "Oregon", "PA": "Pennsylvania", "RI": "Rhode Island", "SC": "South Carolina",
    "SD": "South Dakota", "TN": "Tennessee", "TX": "Texas", "UT": "Utah", "VT": "Vermont",
    "VA": "Virginia", "WA": "Washington", "WV": "West Virginia", "WI": "Wisconsin", "WY": "Wyoming",
    "DC": "District of Columbia", "PR": "Puerto Rico", "VI": "U.S. Virgin Islands", "GU": "Guam",
}

SEASONS = {
    "winter": {12, 1, 2},
    "spring": {3, 4, 5},
    "summer": {6, 7, 8},
    "fall": {9, 10, 11},
    "autumn": {9, 10, 11},
}

# Countries inferred from EPPO datasheets as native/origin ranges for the modeled fruit fly species.
FRUIT_FLY_ORIGIN_COUNTRIES = {
    "Angola", "Bangladesh", "Belize", "Bhutan", "Botswana", "Brunei Darussalam",
    "Cambodia", "China", "Costa Rica", "Ethiopia", "Guatemala", "Honduras",
    "India", "Indonesia", "Kenya", "Laos", "Madagascar", "Malawi", "Malaysia",
    "Mexico", "Mozambique", "Myanmar", "Namibia", "Nepal", "Pakistan", "Panama",
    "Philippines", "South Africa", "Sri Lanka", "Tanzania", "Thailand", "Uganda",
    "Vietnam", "Zambia", "Zimbabwe",
}

def parse_number(value) -> float:
    if value is None:
        return 0.0
    s = str(value).strip().replace(",", "")
    if not s or s.lower() in {"nan", "none", "null"}:
        return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0

def parse_month_date(value: str | None, year=None, month=None) -> date | None:
    if year and month:
        try:
            return date(int(float(year)), int(float(month)), 1)
        except Exception:
            pass
    if not value:
        return None
    s = str(value).strip()
    import re
    # Handles 2024-06, 2024/06, 06/2024, 2024-06-01, etc.
    m = re.search(r"(20\d{2}|19\d{2})[-/](\d{1,2})", s)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    m = re.search(r"(\d{1,2})[-/](20\d{2}|19\d{2})", s)
    if m:
        return date(int(m.group(2)), int(m.group(1)), 1)
    for fmt in ("%b %Y", "%B %Y", "%m/%d/%Y", "%Y-%m-%d"):
        try:
            from datetime import datetime
            d = datetime.strptime(s, fmt).date()
            return date(d.year, d.month, 1)
        except ValueError:
            continue
    return None

def month_shift(d: date, delta: int) -> date:
    total = d.year * 12 + (d.month - 1) + delta
    return date(total // 12, total % 12 + 1, 1)

def pearson(xs: Iterable[float], ys: Iterable[float]) -> float | None:
    x = list(xs); y = list(ys)
    if len(x) < 3 or len(x) != len(y):
        return None
    mx = sum(x) / len(x); my = sum(y) / len(y)
    vx = sum((v - mx) ** 2 for v in x)
    vy = sum((v - my) ** 2 for v in y)
    if vx <= 0 or vy <= 0:
        return None
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    return cov / math.sqrt(vx * vy)

def state_from_city_name(city_name: str | None) -> str:
    if not city_name:
        return "Unknown"
    parts = str(city_name).split(",")
    if len(parts) >= 2:
        abbrev = parts[-1].strip().upper()
        return STATE_ABBREV.get(abbrev, abbrev)
    return "Unknown"

def season_filter(month: int, season: str | None) -> bool:
    if not season or season == "all":
        return True
    return month in SEASONS.get(season.lower(), set())

def normalize_metric(metric: str | None) -> str:
    allowed = {"freight", "passengers", "mail", "payload", "flights"}
    return metric if metric in allowed else "freight"


def haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in miles."""
    r_miles = 3958.8
    p1 = math.radians(lat1)
    p2 = math.radians(lat2)
    d_phi = math.radians(lat2 - lat1)
    d_lambda = math.radians(lon2 - lon1)
    a = math.sin(d_phi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(d_lambda / 2) ** 2
    return 2 * r_miles * math.asin(math.sqrt(a))
