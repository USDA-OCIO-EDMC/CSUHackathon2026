#!/usr/bin/env python3
"""
city_map_generator.py
─────────────────────
Reads a JSON list of city strings, geocodes each one using ArcGIS (primary)
with Photon as a fallback — both free, no API key required — then draws a
circular zone polygon around every city and writes an interactive Folium map.

Each city zone is coloured by its risk factor (0–10):
  • 0        → fully transparent / invisible
  • 1–4      → yellow
  • 4–7      → orange
  • 7–10     → red

The map includes a date navigator (← / →) that steps through months.
Changing the date live-updates which zones are visible and their styling.

Usage
─────
  1. Install deps:  pip install folium geopy
  2. Place track_cities.json in the same directory (or update CITIES_JSON).
  3. Call main() with a nested risk_factors dict, e.g.:

       main({
           "detroit, mi": {
               "03/2024": {"Aedes aegypti": 7.5, "Culex pipiens": 0.0},
               "04/2024": {"Aedes aegypti": 0.0, "Culex pipiens": 4.2},
           },
           "miami, fl": {
               "03/2024": {"Aedes albopictus": 6.1, "Aedes aegypti": 0.0},
           },
       })

     Structure:
       • Layer 1 key  — city name, lowercase, matching track_cities.json
       • Layer 2 key  — date string "MM/YYYY"
       • Layer 3 key  — insect species name (string)
       • Value        — risk factor float (0–10)

     For each city+date, exactly one species should carry a non-zero risk
     value; all others are ignored.  Cities / dates absent from the dict
     default to risk 0 (invisible).

  4. Open cities_map.html in your browser.

Geocoding is cached in geocache.json so subsequent runs are instant.
"""

import json
import math
import os
import random
import sys
import time

import folium
from geopy.exc import GeocoderQuotaExceeded, GeocoderServiceError, GeocoderTimedOut
from geopy.geocoders import ArcGIS, Photon

# ══════════════════════════════════════════════════════════════════════════════
#  CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

CITIES_JSON = "track_cities.json"
CACHE_FILE  = "geocache.json"
OUTPUT_HTML = "cities_map.html"

ZONE_RADIUS_KM = 22
CIRCLE_POINTS  = 48

MAP_CENTER = [39.5, -98.35]
MAP_ZOOM   = 4
MAP_TILES  = "CartoDB positron"

REQUEST_DELAY  = 0.3
MAX_RETRIES    = 4
BACKOFF_BASE   = 2.0
BACKOFF_JITTER = 0.5


# ══════════════════════════════════════════════════════════════════════════════
#  RISK → COLOUR MAPPING  (Python-side, mirrored in JS below)
# ══════════════════════════════════════════════════════════════════════════════

def risk_to_style(risk: float) -> dict:
    """
    Convert a risk value (0–10) into a Folium/Leaflet style dict.

    Colour ramp (risk > 0):
      1.0  →  #facc15  (yellow)
      5.5  →  #f97316  (orange)
      10.0 →  #b91c1c  (deep red)

    Opacity also scales with risk: low-risk zones are more transparent.
    """
    risk = max(0.0, min(float(risk), 10.0))

    if risk == 0.0:
        return {
            "fillColor":   "#000000",
            "color":       "#000000",
            "weight":       0,
            "fillOpacity":  0.0,
            "opacity":      0.0,
        }

    t = risk / 10.0  # normalise to [0, 1]

    if t <= 0.5:
        s = t / 0.5
        r = int(250 + (249 - 250) * s)
        g = int(204 + (115 - 204) * s)
        b = int( 21 + ( 22 -  21) * s)
    else:
        s = (t - 0.5) / 0.5
        r = int(249 + (185 - 249) * s)
        g = int(115 + ( 28 - 115) * s)
        b = int( 22 + ( 28 -  22) * s)

    fill_hex   = f"#{r:02x}{g:02x}{b:02x}"
    border_hex = f"#{max(r-30,0):02x}{max(g-20,0):02x}{max(b-20,0):02x}"

    fill_opacity   = 0.25 + 0.50 * t
    border_opacity = 0.60 + 0.40 * t

    return {
        "fillColor":   fill_hex,
        "color":       border_hex,
        "weight":       1 if risk < 5 else 1.5,
        "fillOpacity":  round(fill_opacity, 3),
        "opacity":      round(border_opacity, 3),
    }


# ══════════════════════════════════════════════════════════════════════════════
#  GEOCODER HELPERS
# ══════════════════════════════════════════════════════════════════════════════

def make_geocoders() -> list:
    return [
        ("ArcGIS", ArcGIS(timeout=10)),
        ("Photon", Photon(user_agent="city_zone_mapper/1.0", timeout=10)),
    ]


def load_cities(path: str) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_cache(path: str) -> dict:
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_cache(path: str, cache: dict) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, indent=2)


def query_variants(city: str) -> list:
    base = city.split("/")[0].strip()
    return list(dict.fromkeys([
        city + ", USA",
        base + ", USA",
        city,
        base,
    ]))


def geocode_with_retry(geocoders: list, city: str) -> dict | None:
    for geo_name, geolocator in geocoders:
        for query in query_variants(city):
            for attempt in range(1, MAX_RETRIES + 1):
                try:
                    loc = geolocator.geocode(query)
                    if loc:
                        return {"lat": loc.latitude, "lon": loc.longitude}
                    break
                except GeocoderQuotaExceeded:
                    wait = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
                    print(f"\n  ⚠  [{geo_name}] quota exceeded — waiting {wait:.1f}s …")
                    time.sleep(wait)
                except GeocoderTimedOut:
                    wait = BACKOFF_BASE ** attempt + random.uniform(0, BACKOFF_JITTER)
                    print(f"\n  ⚠  [{geo_name}] timed out (attempt {attempt}) — retrying in {wait:.1f}s …")
                    time.sleep(wait)
                except GeocoderServiceError as exc:
                    print(f"\n  ⚠  [{geo_name}] service error: {exc} — skipping to fallback")
                    break
            time.sleep(REQUEST_DELAY)
    return None


def geocode_all(cities: list, cache: dict) -> dict:
    geocoders = make_geocoders()
    results   = {}
    total     = len(cities)

    for idx, city in enumerate(cities, 1):
        if city in cache:
            results[city] = cache[city]
            sys.stdout.write(f"\r  [cache] {idx:>3}/{total}  {city[:60]:<60}")
            sys.stdout.flush()
            continue

        coords = geocode_with_retry(geocoders, city)
        results[city] = coords
        cache[city]   = coords

        tag = f"({coords['lat']:.4f}, {coords['lon']:.4f})" if coords else "NOT FOUND"
        print(f"  {'✓' if coords else '✗'} [{idx:>3}/{total}] {city}  →  {tag}")

    print()
    return results


# ══════════════════════════════════════════════════════════════════════════════
#  GEOMETRY HELPER
# ══════════════════════════════════════════════════════════════════════════════

def circle_ring(lat: float, lon: float, radius_km: float, n: int = 48) -> list:
    """Return a closed GeoJSON coordinate ring [[lon, lat], …] for a circle."""
    R    = 6371.0
    ring = []
    for i in range(n + 1):
        theta  = math.radians(360 * i / n)
        d_lat  = (radius_km / R) * math.cos(theta)
        d_lon  = (radius_km / R) * math.sin(theta) / math.cos(math.radians(lat))
        ring.append([lon + math.degrees(d_lon), lat + math.degrees(d_lat)])
    return ring


# ══════════════════════════════════════════════════════════════════════════════
#  NESTED RISK-FACTOR PARSER
# ══════════════════════════════════════════════════════════════════════════════

def parse_nested_risks(risk_factors: dict) -> tuple[dict, list]:
    """
    Flatten the three-level nested risk_factors dict into a two-level structure
    keyed by (city_lower, date_str) and collect a sorted list of all dates.

    Input structure
    ───────────────
      {
        "detroit, mi": {
          "03/2024": {"Aedes aegypti": 7.5, "Culex pipiens": 0.0},
          "04/2024": {"Aedes aegypti": 0.0, "Culex pipiens": 4.2},
        },
        ...
      }

    Output
    ──────
      parsed = {
        "detroit, mi": {
          "03/2024": {"risk": 7.5, "species": "Aedes aegypti"},
          "04/2024": {"risk": 4.2, "species": "Culex pipiens"},
        },
        ...
      }
      sorted_dates = ["03/2024", "04/2024"]   # chronological order

    Rules
    ─────
      • For each city+date exactly one species should have a non-zero value;
        the first non-zero species found is used and the rest are ignored.
      • If all species are zero the entry is stored with risk=0 and species="—".
    """
    parsed    = {}
    all_dates: set[str] = set()

    for city, date_dict in risk_factors.items():
        city_lower = city.lower()
        parsed[city_lower] = {}

        for date_str, species_dict in date_dict.items():
            all_dates.add(date_str)

            best_species = "—"
            best_risk    = 0.0

            for species, risk in species_dict.items():
                r = float(risk)
                if r > 0:
                    best_species = species
                    best_risk    = r
                    break  # only one non-zero per city+date

            parsed[city_lower][date_str] = {
                "risk":    best_risk,
                "species": best_species,
            }

    def _date_key(d: str) -> tuple[int, int]:
        mm, yyyy = d.split("/")
        return (int(yyyy), int(mm))

    sorted_dates = sorted(all_dates, key=_date_key)
    return parsed, sorted_dates


# ══════════════════════════════════════════════════════════════════════════════
#  MAP BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_map(
    city_coords:  dict,
    parsed_risks: dict,
    all_dates:    list,
    output_path:  str,
) -> None:
    """
    Render an interactive Folium/Leaflet map with per-date city risk zones.

    Args:
        city_coords:  {city_string: {"lat": float, "lon": float} | None}
        parsed_risks: output of parse_nested_risks()
        all_dates:    sorted list of "MM/YYYY" date strings
        output_path:  path to write the HTML file
    """
    m = folium.Map(
        location=MAP_CENTER,
        zoom_start=MAP_ZOOM,
        tiles=MAP_TILES,
    )
    map_var = m.get_name()   # Leaflet map JS variable name injected by Folium

    # ── Build per-city JS payload ─────────────────────────────────────────────
    cities_js = []
    for city, coords in city_coords.items():
        if not coords:
            continue

        city_lower = city.lower()
        parts      = city.rsplit(", ", 1)
        city_label = parts[0] if len(parts) == 2 else city
        state      = parts[1] if len(parts) == 2 else ""

        # Pre-compute circle ring once per city
        ring = circle_ring(coords["lat"], coords["lon"], ZONE_RADIUS_KM, CIRCLE_POINTS)

        # Gather data for every date (default to zero for missing dates)
        dates_data: dict = {}
        for d in all_dates:
            entry = parsed_risks.get(city_lower, {}).get(
                d, {"risk": 0.0, "species": "—"}
            )
            dates_data[d] = entry

        cities_js.append({
            "city":      city_label,
            "state":     state,
            "full_name": city,
            "ring":      ring,
            "dates":     dates_data,
        })

    cities_json = json.dumps(cities_js,  separators=(",", ":"))
    dates_json  = json.dumps(all_dates,  separators=(",", ":"))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_html = """
    <div id="risk-legend" style="
        position:fixed; bottom:40px; right:40px; z-index:1000;
        background:white; border-radius:8px; padding:14px 18px;
        box-shadow:0 2px 8px rgba(0,0,0,0.18);
        font-family:system-ui,sans-serif; font-size:13px; line-height:1.6;
    ">
      <div style="font-weight:700;margin-bottom:8px;color:#1e293b;">Risk Factor</div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <div style="width:16px;height:16px;border-radius:3px;background:#facc15;opacity:0.7;"></div>
        <span>Low &nbsp; (1–3)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <div style="width:16px;height:16px;border-radius:3px;background:#f97316;opacity:0.8;"></div>
        <span>Medium (4–6)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:4px;">
        <div style="width:16px;height:16px;border-radius:3px;background:#dc2626;opacity:0.9;"></div>
        <span>High &nbsp; (7–10)</span>
      </div>
      <div style="display:flex;align-items:center;gap:8px;">
        <div style="width:16px;height:16px;border-radius:3px;background:#e2e8f0;"></div>
        <span>None &nbsp; (0)</span>
      </div>
    </div>
    """

    # ── Date navigator ────────────────────────────────────────────────────────
    date_nav_html = """
    <div id="date-nav" style="
        position:fixed; bottom:40px; left:50%; transform:translateX(-50%);
        z-index:1000;
        background:white; border-radius:8px; padding:10px 18px;
        box-shadow:0 2px 8px rgba(0,0,0,0.18);
        font-family:system-ui,sans-serif; font-size:14px;
        display:flex; align-items:center; gap:14px;
    ">
      <button id="btn-prev" onclick="cityMapChangeDate(-1)" style="
        background:#f1f5f9; border:none; border-radius:6px;
        padding:6px 14px; cursor:pointer; font-size:18px; line-height:1;
        color:#334155; transition:background 0.15s;
      " title="Previous month">&#8592;</button>

      <div style="text-align:center; min-width:90px;">
        <div style="font-size:11px;color:#64748b;margin-bottom:2px;letter-spacing:.04em;">DATE</div>
        <div id="date-display" style="font-weight:700;color:#1e293b;font-size:16px;letter-spacing:.03em;">—</div>
      </div>

      <button id="btn-next" onclick="cityMapChangeDate(1)" style="
        background:#f1f5f9; border:none; border-radius:6px;
        padding:6px 14px; cursor:pointer; font-size:18px; line-height:1;
        color:#334155; transition:background 0.15s;
      " title="Next month">&#8594;</button>
    </div>
    """

    m.get_root().html.add_child(folium.Element(legend_html))
    m.get_root().html.add_child(folium.Element(date_nav_html))

    # ── JavaScript ────────────────────────────────────────────────────────────
    # The colour ramp logic is mirrored from risk_to_style() above so that the
    # browser can compute fill colours without a round-trip to Python.
    js = f"""
<script>
(function () {{
  /* ── Data injected from Python ─────────────────────────────────────────── */
  var CITY_DATA = {cities_json};
  var ALL_DATES = {dates_json};

  /* ── State ─────────────────────────────────────────────────────────────── */
  var currentIdx = 0;
  var riskLayer  = null;   // Leaflet LayerGroup for current date's polygons

  /* ── Colour helpers (mirrors Python risk_to_style) ─────────────────────── */
  function toHex2(n) {{
    return ('0' + Math.max(0, Math.min(255, Math.round(n))).toString(16)).slice(-2);
  }}

  function riskToStyle(risk) {{
    if (risk <= 0) return null;
    var t = Math.min(Math.max(risk, 0), 10) / 10.0;
    var r, g, b, s;

    if (t <= 0.5) {{
      s = t / 0.5;
      r = 250 + (249 - 250) * s;
      g = 204 + (115 - 204) * s;
      b =  21 + ( 22 -  21) * s;
    }} else {{
      s = (t - 0.5) / 0.5;
      r = 249 + (185 - 249) * s;
      g = 115 + ( 28 - 115) * s;
      b =  22 + ( 28 -  22) * s;
    }}

    var fillHex   = '#' + toHex2(r) + toHex2(g) + toHex2(b);
    var borderHex = '#' + toHex2(r - 30) + toHex2(g - 20) + toHex2(b - 20);
    var fillOp    = Math.round((0.25 + 0.50 * t) * 1000) / 1000;
    var borderOp  = Math.round((0.60 + 0.40 * t) * 1000) / 1000;

    return {{
      fillColor:   fillHex,
      color:       borderHex,
      weight:      risk < 5 ? 1 : 1.5,
      fillOpacity: fillOp,
      opacity:     borderOp,
    }};
  }}

  function hoverStyle(baseStyle) {{
    return Object.assign({{}}, baseStyle, {{
      weight:      2.5,
      fillOpacity: Math.min(baseStyle.fillOpacity + 0.20, 0.95),
    }});
  }}

  /* ── Tooltip / popup builders ───────────────────────────────────────────── */
  function makeTooltip(city, info) {{
    var loc = city.city + (city.state ? ', ' + city.state : '');
    return (
      '<div style="font-family:system-ui,sans-serif;font-size:13px;padding:2px 4px;">' +
      '<b>' + loc + '</b><br/>' +
      'Risk: <b>' + info.risk.toFixed(1) + '</b><br/>' +
      'Species: <i>' + info.species + '</i>' +
      '</div>'
    );
  }}

  function makePopup(city, info, dateStr) {{
    var loc = city.city + (city.state ? ', ' + city.state : '');
    return (
      '<div style="font-family:system-ui,sans-serif;font-size:13px;min-width:210px;">' +
      '<div style="font-size:15px;font-weight:700;margin-bottom:10px;color:#1e293b;">' +
        loc +
      '</div>' +
      '<table style="border-collapse:collapse;width:100%;">' +
        '<tr>' +
          '<td style="color:#64748b;padding:3px 10px 3px 0;white-space:nowrap;">Date</td>' +
          '<td style="font-weight:600;">' + dateStr + '</td>' +
        '</tr>' +
        '<tr>' +
          '<td style="color:#64748b;padding:3px 10px 3px 0;white-space:nowrap;">Risk Factor</td>' +
          '<td style="font-weight:600;">' + info.risk.toFixed(2) + '</td>' +
        '</tr>' +
        '<tr>' +
          '<td style="color:#64748b;padding:3px 10px 3px 0;white-space:nowrap;">Species</td>' +
          '<td style="font-style:italic;color:#0f766e;">' + info.species + '</td>' +
        '</tr>' +
      '</table>' +
      '</div>'
    );
  }}

  /* ── Core render function ───────────────────────────────────────────────── */
  function renderDate(dateStr) {{
    var map = window['{map_var}'];

    // Clear previous layer
    if (riskLayer) {{ map.removeLayer(riskLayer); }}
    riskLayer = L.layerGroup().addTo(map);

    CITY_DATA.forEach(function (city) {{
      var info = city.dates[dateStr];
      if (!info || info.risk <= 0) return;

      var style = riskToStyle(info.risk);
      if (!style) return;

      // GeoJSON ring is [lon, lat]; Leaflet needs [lat, lon]
      var latlngs = city.ring.map(function (p) {{ return [p[1], p[0]]; }});
      var poly    = L.polygon(latlngs, style);

      poly.bindTooltip(makeTooltip(city, info), {{ sticky: true }});
      poly.bindPopup(makePopup(city, info, dateStr), {{ maxWidth: 320 }});

      poly.on('mouseover', function (e) {{
        e.target.setStyle(hoverStyle(style));
      }});
      poly.on('mouseout', function (e) {{
        e.target.setStyle(style);
      }});

      riskLayer.addLayer(poly);
    }});

    /* Update UI */
    var disp = document.getElementById('date-display');
    if (disp) disp.textContent = dateStr;

    var btnPrev = document.getElementById('btn-prev');
    var btnNext = document.getElementById('btn-next');
    if (btnPrev) btnPrev.disabled = (currentIdx === 0);
    if (btnNext) btnNext.disabled = (currentIdx === ALL_DATES.length - 1);
  }}

  /* ── Public: called by inline onclick handlers ──────────────────────────── */
  window.cityMapChangeDate = function (delta) {{
    var newIdx = currentIdx + delta;
    if (newIdx < 0 || newIdx >= ALL_DATES.length) return;
    currentIdx = newIdx;
    renderDate(ALL_DATES[currentIdx]);
  }};

  /* ── Bootstrap: wait for Leaflet map object to exist ───────────────────── */
  function init() {{
    if (typeof window['{map_var}'] === 'undefined') {{
      setTimeout(init, 100);
      return;
    }}
    if (ALL_DATES.length > 0) {{
      renderDate(ALL_DATES[currentIdx]);
    }} else {{
      var disp = document.getElementById('date-display');
      if (disp) disp.textContent = 'No data';
    }}
  }}

  init();
}})();
</script>
"""

    m.get_root().html.add_child(folium.Element(js))
    folium.LayerControl().add_to(m)
    m.save(output_path)


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def main(risk_factors: dict = None) -> None:
    """
    Generate the city risk zone map from a nested risk_factors dictionary.

    Args:
        risk_factors: Three-level nested dict:
          {
            "<city_lowercase>": {
              "<MM/YYYY>": {
                "<species_name>": <risk_float_0_to_10>,
                ...   ← other species with value 0 are ignored
              },
              ...
            },
            ...
          }

          City keys must be lowercase versions of the strings in
          track_cities.json.  Dates must be "MM/YYYY" strings.
          For each city+date, exactly one species should be non-zero.
          Cities / dates absent from the dict default to risk 0.

    Example
    ───────
        main({
            "detroit, mi": {
                "03/2024": {"Aedes aegypti": 7.5},
                "04/2024": {"Culex pipiens": 4.2},
            },
            "miami, fl": {
                "03/2024": {"Aedes albopictus": 6.1},
                "04/2024": {"Aedes albopictus": 8.3},
            },
        })
    """
    if risk_factors is None:
        risk_factors = {}

    banner = "  City Risk Zone Map Generator  "
    print("═" * (len(banner) + 4))
    print(f"  {banner}")
    print("═" * (len(banner) + 4))

    cities = load_cities(CITIES_JSON)
    print(f"\n▸ Loaded {len(cities)} cities from '{CITIES_JSON}'")

    # Parse nested dict
    parsed_risks, all_dates = parse_nested_risks(risk_factors)
    print(f"▸ Risk data: {len(risk_factors)} cities  |  {len(all_dates)} date(s): "
          f"{', '.join(all_dates) if all_dates else '(none)'}")

    non_zero_cities = sum(
        1 for city in cities
        if any(
            parsed_risks.get(city.lower(), {}).get(d, {}).get("risk", 0) > 0
            for d in all_dates
        )
    )
    print(f"▸ Cities with at least one non-zero date: {non_zero_cities}")

    cache        = load_cache(CACHE_FILE)
    cached_count = sum(1 for c in cities if c in cache)
    print(f"▸ Cache: {cached_count}/{len(cities)} cities already geocoded")
    print(f"▸ Geocoders: ArcGIS (primary) → Photon (fallback)\n")

    print("▸ Geocoding cities …\n")
    city_coords = geocode_all(cities, cache)
    save_cache(CACHE_FILE, cache)

    found = sum(1 for v in city_coords.values() if v)
    print(f"▸ Results: {found} geocoded  |  {len(city_coords) - found} not found")
    print(f"▸ Cache saved → {CACHE_FILE}\n")

    print("▸ Rendering Folium map …")
    build_map(city_coords, parsed_risks, all_dates, OUTPUT_HTML)
    print(f"  Map saved → {OUTPUT_HTML}")

    print(f"\n✓ Done!  Open '{OUTPUT_HTML}' in your browser.\n")


# ── Example usage ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    example_risks = {
        # city (lowercase)         date       species                  risk
        "detroit, mi": {
            "01/2024": {"Aedes aegypti":     8.5, "Culex pipiens":    0.0},
            "02/2024": {"Aedes aegypti":     6.0, "Culex pipiens":    0.0},
            "03/2024": {"Aedes aegypti":     0.0, "Culex pipiens":    3.2},
        },
        "miami, fl": {
            "01/2024": {"Aedes albopictus":  6.2, "Aedes aegypti":    0.0},
            "02/2024": {"Aedes albopictus":  7.8, "Aedes aegypti":    0.0},
            "03/2024": {"Aedes albopictus":  0.0, "Aedes aegypti":    5.1},
        },
        "new york, ny": {
            "01/2024": {"Culex pipiens":     9.1, "Aedes albopictus": 0.0},
            "02/2024": {"Culex pipiens":     0.0, "Aedes albopictus": 4.4},
            "03/2024": {"Culex pipiens":     7.3, "Aedes albopictus": 0.0},
        },
        "los angeles, ca": {
            "01/2024": {"Aedes aegypti":     7.4, "Culex quinquefasciatus": 0.0},
            "02/2024": {"Aedes aegypti":     0.0, "Culex quinquefasciatus": 5.9},
            "03/2024": {"Aedes aegypti":     8.2, "Culex quinquefasciatus": 0.0},
        },
        "chicago, il": {
            "01/2024": {"Culex pipiens":     5.8, "Aedes vexans":    0.0},
            "02/2024": {"Culex pipiens":     0.0, "Aedes vexans":    2.3},
            "03/2024": {"Culex pipiens":     4.1, "Aedes vexans":    0.0},
        },
        "houston, tx": {
            "01/2024": {"Aedes aegypti":     4.3, "Culex quinquefasciatus": 0.0},
            "02/2024": {"Aedes aegypti":     6.7, "Culex quinquefasciatus": 0.0},
            "03/2024": {"Aedes aegypti":     0.0, "Culex quinquefasciatus": 9.0},
        },
        "phoenix, az": {
            "01/2024": {"Aedes aegypti":     3.1, "Culex quinquefasciatus": 0.0},
            "03/2024": {"Aedes aegypti":     0.0, "Culex quinquefasciatus": 1.8},
        },
        "philadelphia, pa": {
            "01/2024": {"Culex pipiens":     6.7, "Aedes albopictus": 0.0},
            "02/2024": {"Culex pipiens":     0.0, "Aedes albopictus": 3.5},
        },
        "dallas/fort worth, tx": {
            "01/2024": {"Aedes aegypti":     2.9, "Culex quinquefasciatus": 0.0},
            "02/2024": {"Aedes aegypti":     0.0, "Culex quinquefasciatus": 7.1},
            "03/2024": {"Aedes aegypti":     5.5, "Culex quinquefasciatus": 0.0},
        },
        "atlanta, ga": {
            "01/2024": {"Aedes albopictus":  7.0, "Aedes aegypti":    0.0},
            "02/2024": {"Aedes albopictus":  0.0, "Aedes aegypti":    4.8},
            "03/2024": {"Aedes albopictus":  6.3, "Aedes aegypti":    0.0},
        },
    }

    main(risk_factors=example_risks)