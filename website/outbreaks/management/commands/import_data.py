from __future__ import annotations

import sys
import csv
import re
from collections import defaultdict
from datetime import date
from pathlib import Path

from django.core.management.base import BaseCommand, CommandError
from django.db import transaction

from outbreaks.models import AirTrafficAggregate, DetectionAggregate, PortTraffic
from outbreaks.utils import parse_month_date, parse_number, state_from_city_name


max_int = sys.maxsize

while True:
    try:
        csv.field_size_limit(max_int)
        break
    except OverflowError:
        max_int = int(max_int / 10)

COORD_RE = re.compile(r"(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)")


def safe_float(value):
    try:
        if value in (None, ""):
            return None
        return float(str(value).strip())
    except Exception:
        return None


def rough_wkt_centroid(wkt: str | None, max_pairs: int = 250):
    """Fast approximate centroid from the first coordinate pairs in WKT.

    This avoids loading GeoPandas/Shapely for huge county polygons. It is enough for
    map bubbles. If you need accurate polygon centroids, replace this with Shapely.
    """
    if not wkt:
        return None, None
    xs, ys = [], []
    for i, m in enumerate(COORD_RE.finditer(wkt)):
        if i >= max_pairs:
            break
        lon = safe_float(m.group(1)); lat = safe_float(m.group(2))
        if lon is not None and lat is not None:
            xs.append(lon); ys.append(lat)
    if not xs:
        return None, None
    return sum(ys) / len(ys), sum(xs) / len(xs)


class Command(BaseCommand):
    help = "Import fruit fly detections, international air traffic, and annual port traffic CSV files."

    def add_arguments(self, parser):
        parser.add_argument("--data-dir", default="data", help="Directory containing the CSV files.")
        parser.add_argument("--detections", default="fruit_fly_detections.csv")
        parser.add_argument("--air", default="international_segmentation.csv")
        parser.add_argument("--ports", nargs="*", default=["2020Ports.csv", "2021Ports.csv", "2022Ports.csv"])
        parser.add_argument("--clear", action="store_true", help="Delete existing imported rows first.")
        parser.add_argument("--batch-size", type=int, default=2500)
        parser.add_argument("--skip-geometry", action="store_true", help="Do not estimate centroids from detection WKT; faster import.")

    @transaction.atomic
    def handle(self, *args, **opts):
        data_dir = Path(opts["data_dir"]).resolve()
        if not data_dir.exists():
            raise CommandError(f"Data directory does not exist: {data_dir}")

        if opts["clear"]:
            self.stdout.write("Clearing old rows...")
            DetectionAggregate.objects.all().delete()
            AirTrafficAggregate.objects.all().delete()
            PortTraffic.objects.all().delete()

        self.import_detections(data_dir / opts["detections"], opts)
        self.import_air(data_dir / opts["air"], opts)
        for p in opts["ports"]:
            self.import_ports(data_dir / p)

        self.stdout.write(self.style.SUCCESS("Import complete."))

    def import_detections(self, path: Path, opts):
        if not path.exists():
            self.stdout.write(self.style.WARNING(f"Skipping missing detections file: {path}"))
            return
        self.stdout.write(f"Importing detections from {path.name}...")
        batch = []
        count = 0
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                d = parse_month_date(row.get("monthyear"))
                if not d:
                    continue
                lat = safe_float(row.get("geometry_lat") or row.get("lat") or row.get("LAT"))
                lon = safe_float(row.get("geometry_lon") or row.get("lon") or row.get("LON"))
                if (lat is None or lon is None) and not opts["skip_geometry"]:
                    lat, lon = rough_wkt_centroid(row.get("geometry_wkt"))
                batch.append(DetectionAggregate(
                    date=d,
                    year=d.year,
                    month=d.month,
                    common_name=(row.get("CommonName") or row.get("common_name") or "").strip(),
                    state_name=(row.get("STATE_NAME") or row.get("state") or "Unknown").strip(),
                    county_name=(row.get("NAME") or row.get("county") or "").strip(),
                    count=parse_number(row.get("Count_") or row.get("count") or 1),
                    lat=lat,
                    lon=lon,
                ))
                if len(batch) >= opts["batch_size"]:
                    DetectionAggregate.objects.bulk_create(batch)
                    count += len(batch)
                    batch.clear()
                    self.stdout.write(f"  detections: {count}")
        if batch:
            DetectionAggregate.objects.bulk_create(batch)
            count += len(batch)
        self.stdout.write(self.style.SUCCESS(f"  detections imported: {count}"))

    def import_air(self, path: Path, opts):
        if not path.exists():
            self.stdout.write(self.style.WARNING(f"Skipping missing air file: {path}"))
            return
        self.stdout.write(f"Importing inbound air traffic from {path.name}...")
        grouped = defaultdict(lambda: {"flights": 0.0, "passengers": 0.0, "freight": 0.0, "mail": 0.0, "payload": 0.0, "lat": None, "lon": None})
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                dest_country = (row.get("DEST_COUNTRY") or row.get("DEST_COUNTRY_NAME") or "").strip().upper()
                if dest_country not in {"US", "UNITED STATES"}:
                    continue
                d = parse_month_date(None, row.get("YEAR_WADS"), row.get("MONTH_WADS"))
                if not d:
                    continue
                city = row.get("DEST_CITY_NAME") or ""
                state_name = state_from_city_name(city)
                key = (
                    d,
                    state_name,
                    (row.get("DEST") or "").strip(),
                    city.strip(),
                    (row.get("ORIGIN_COUNTRY_NAME") or row.get("ORIGIN_COUNTRY") or "Unknown").strip(),
                    (row.get("ORIGIN_COUNTRY") or "").strip(),
                )
                g = grouped[key]
                g["flights"] += parse_number(row.get("DEPARTURES_PERFORMED") or row.get("DEPARTURES_SCHEDULED"))
                g["passengers"] += parse_number(row.get("PASSENGERS"))
                g["freight"] += parse_number(row.get("FREIGHT"))
                g["mail"] += parse_number(row.get("MAIL"))
                g["payload"] += parse_number(row.get("PAYLOAD"))
                g["lat"] = g["lat"] if g["lat"] is not None else safe_float(row.get("DEST_LAT"))
                g["lon"] = g["lon"] if g["lon"] is not None else safe_float(row.get("DEST_LON"))

        objs = []
        for (d, state_name, dest, city, country, country_code), g in grouped.items():
            objs.append(AirTrafficAggregate(
                date=d, year=d.year, month=d.month, state_name=state_name,
                dest_city_name=city, dest_airport=dest, dest_lat=g["lat"], dest_lon=g["lon"],
                origin_country=country or "Unknown", origin_country_code=country_code,
                flights=g["flights"], passengers=g["passengers"], freight=g["freight"], mail=g["mail"], payload=g["payload"],
            ))
        AirTrafficAggregate.objects.bulk_create(objs, batch_size=opts["batch_size"])
        self.stdout.write(self.style.SUCCESS(f"  inbound air rows imported: {len(objs)}"))

    def import_ports(self, path: Path):
        if not path.exists():
            self.stdout.write(self.style.WARNING(f"Skipping missing port file: {path}"))
            return
        year_match = re.search(r"(20\d{2})", path.name)
        year = int(year_match.group(1)) if year_match else 0
        self.stdout.write(f"Importing port traffic from {path.name}...")
        objs = []
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            rows = csv.reader(fh)
            for _ in range(3):
                next(rows, None)
            for row in rows:
                if len(row) < 11 or not row[0].strip():
                    continue
                objs.append(PortTraffic(
                    year=year,
                    port_name=row[0].strip(),
                    state=row[1].strip(),
                    foreign_inbound_loaded=parse_number(row[7]),
                    foreign_outbound_loaded=parse_number(row[8]),
                    total_foreign_loaded=parse_number(row[9]),
                    grand_total_loaded=parse_number(row[10]),
                ))
        PortTraffic.objects.bulk_create(objs)
        self.stdout.write(self.style.SUCCESS(f"  port rows imported: {len(objs)}"))
