from __future__ import annotations

from collections import defaultdict
from math import log1p

from django.db.models import Sum, Count
from django.http import JsonResponse
from django.shortcuts import render

from .models import AirTrafficAggregate, DetectionAggregate, PortTraffic
from .utils import (
    FRUIT_FLY_ORIGIN_COUNTRIES,
    haversine_miles,
    month_shift,
    normalize_metric,
    pearson,
    season_filter,
)

CENTROID_DISTANCE_LIMIT_MILES = 60.0


def dashboard(request):
    return render(request, "outbreaks/dashboard.html")


def api_options(request):
    states = list(DetectionAggregate.objects.order_by("state_name").values_list("state_name", flat=True).distinct())
    countries = list(AirTrafficAggregate.objects.order_by("origin_country").values_list("origin_country", flat=True).distinct()[:500])
    years = sorted(set(DetectionAggregate.objects.values_list("year", flat=True).distinct()) | set(AirTrafficAggregate.objects.values_list("year", flat=True).distinct()))
    return JsonResponse({"states": states, "countries": countries, "years": years})


def api_summary(request):
    det = DetectionAggregate.objects.aggregate(rows=Count("id"), total=Sum("count"))
    air = AirTrafficAggregate.objects.aggregate(rows=Count("id"), passengers=Sum("passengers"), freight=Sum("freight"), flights=Sum("flights"))
    ports = PortTraffic.objects.aggregate(rows=Count("id"), foreign=Sum("total_foreign_loaded"))
    return JsonResponse({"detections": det, "air_traffic": air, "ports": ports})


def _detection_series(state=None):
    qs = DetectionAggregate.objects.all()
    if state and state != "all":
        qs = qs.filter(state_name=state)
    rows = qs.values("year", "month").annotate(total=Sum("count")).order_by("year", "month")
    return {(__import__("datetime").date(r["year"], r["month"], 1)): float(r["total"] or 0) for r in rows}


def _county_centroids(state=None):
    qs = DetectionAggregate.objects.exclude(lat__isnull=True).exclude(lon__isnull=True)
    if state and state != "all":
        qs = qs.filter(state_name=state)
    rows = qs.values("state_name", "county_name", "lat", "lon").distinct()
    return [
        {
            "state_name": r["state_name"],
            "county_name": r["county_name"],
            "lat": float(r["lat"]),
            "lon": float(r["lon"]),
        }
        for r in rows
    ]


def _nearest_centroid(lat, lon, centroids):
    nearest = None
    nearest_distance = None
    for c in centroids:
        d = haversine_miles(float(lat), float(lon), c["lat"], c["lon"])
        if nearest_distance is None or d < nearest_distance:
            nearest_distance = d
            nearest = c
    return nearest, nearest_distance


def _nearby_air_rows(state, metric, country=None):
    centroids = _county_centroids(state)
    if not centroids:
        return []

    qs = AirTrafficAggregate.objects.exclude(dest_lat__isnull=True).exclude(dest_lon__isnull=True)
    if state and state != "all":
        qs = qs.filter(state_name=state)
    if country:
        qs = qs.filter(origin_country=country)
    rows = qs.values(
        "origin_country",
        "state_name",
        "dest_city_name",
        "dest_airport",
        "year",
        "month",
        "dest_lat",
        "dest_lon",
    ).annotate(v=Sum(metric))

    matched = []
    for r in rows:
        centroid, distance = _nearest_centroid(r["dest_lat"], r["dest_lon"], centroids)
        if centroid and distance is not None and distance <= CENTROID_DISTANCE_LIMIT_MILES:
            matched.append(
                {
                    "origin_country": r["origin_country"],
                    "state_name": r["state_name"],
                    "dest_city_name": r["dest_city_name"],
                    "dest_airport": r["dest_airport"],
                    "year": r["year"],
                    "month": r["month"],
                    "dest_lat": float(r["dest_lat"]),
                    "dest_lon": float(r["dest_lon"]),
                    "value": float(r["v"] or 0),
                    "nearest_county_name": centroid["county_name"],
                    "nearest_county_state": centroid["state_name"],
                    "nearest_county_distance_miles": distance,
                }
            )
    return matched


def api_countries(request):
    metric = normalize_metric(request.GET.get("metric"))
    state = request.GET.get("state") or "all"
    try:
        lag = int(request.GET.get("lag", 0))
    except ValueError:
        lag = 0

    detections = _detection_series(state)
    traffic_by_country = defaultdict(dict)
    totals = defaultdict(float)
    matched_airports = defaultdict(set)
    matched_counties = defaultdict(set)
    rows = _nearby_air_rows(state, metric)
    for r in rows:
        d = __import__("datetime").date(r["year"], r["month"], 1)
        country = r["origin_country"]
        val = float(r["value"] or 0)
        traffic_by_country[country][d] = traffic_by_country[country].get(d, 0.0) + val
        totals[country] += val
        matched_airports[country].add((r["dest_airport"], r["dest_city_name"], r["state_name"]))
        matched_counties[country].add((r["nearest_county_state"], r["nearest_county_name"]))

    detections_by_month = defaultdict(float)
    for d, val in detections.items():
        detections_by_month[d.month] += float(val or 0)
    peak_detection = max(detections_by_month.values(), default=0.0)
    peak_months = {
        m for m, total in detections_by_month.items()
        if peak_detection > 0 and total >= (0.9 * peak_detection)
    }

    def _country_peak_share(target_metric):
        by_country_month = defaultdict(lambda: defaultdict(float))
        metric_totals = defaultdict(float)
        for rr in _nearby_air_rows(state, target_metric):
            country_name = rr["origin_country"]
            month = int(rr["month"])
            val = float(rr["value"] or 0)
            by_country_month[country_name][month] += val
            metric_totals[country_name] += val
        share = {}
        for country_name, month_vals in by_country_month.items():
            total_val = metric_totals[country_name]
            peak_val = sum(v for m, v in month_vals.items() if m in peak_months)
            share[country_name] = (peak_val / total_val) if total_val > 0 else 0.0
        return share

    freight_peak_share = _country_peak_share("freight")
    flights_peak_share = _country_peak_share("flights")

    out = []
    for country, series in traffic_by_country.items():
        xs, ys = [], []
        for det_month, det_val in detections.items():
            traffic_month = month_shift(det_month, -lag)
            xs.append(series.get(traffic_month, 0.0))
            ys.append(det_val)
        corr = pearson(xs, ys)
        total_det = sum(ys)
        # Dampen traffic-volume influence so freight-heavy routes do not dominate risk ranking.
        base_risk = (max(corr or 0, 0) + 0.05) * (0.5 * log1p(totals[country])) * log1p(total_det)
        origin_multiplier = 2.5 if country in FRUIT_FLY_ORIGIN_COUNTRIES else 1.0
        country_peak_share = (freight_peak_share.get(country, 0.0) + flights_peak_share.get(country, 0.0)) / 2.0
        # Tiny max +3% bump when freight/flights are concentrated in peak detection months.
        peak_month_multiplier = 1.0 + (0.03 * country_peak_share)
        risk = base_risk * origin_multiplier * peak_month_multiplier
        out.append({
            "country": country,
            "correlation": corr,
            "traffic_total": totals[country],
            "detection_total": total_det,
            "risk_score": risk,
            "origin_multiplier": origin_multiplier,
            "peak_month_multiplier": peak_month_multiplier,
            "lag_months": lag,
            "matched_airports": len(matched_airports[country]),
            "matched_counties": len(matched_counties[country]),
        })
    out.sort(key=lambda r: (r["risk_score"], r["traffic_total"]), reverse=True)
    return JsonResponse(
        {
            "metric": metric,
            "state": state,
            "distance_limit_miles": CENTROID_DISTANCE_LIMIT_MILES,
            "countries": out[:75],
        }
    )


def api_timeseries(request):
    # Monthly comparison intentionally uses flight counts (not shipment volume).
    metric = "flights"
    country = request.GET.get("country")
    state = request.GET.get("state") or "all"
    detections = _detection_series(state)
    traffic = defaultdict(float)
    for r in _nearby_air_rows(state, metric, country=country):
        d = __import__("datetime").date(r["year"], r["month"], 1)
        traffic[d] += float(r["value"] or 0)
    months = sorted(set(detections) | set(traffic))
    return JsonResponse({
        "labels": [d.strftime("%Y-%m") for d in months],
        "detections": [detections.get(d, 0.0) for d in months],
        "traffic": [traffic.get(d, 0.0) for d in months],
        "metric": metric,
        "country": country,
        "state": state,
        "distance_limit_miles": CENTROID_DISTANCE_LIMIT_MILES,
    })


def api_monthly_risk(request):
    state = request.GET.get("state") or "all"
    year = request.GET.get("year")

    qs = DetectionAggregate.objects.all()
    if state != "all":
        qs = qs.filter(state_name=state)
    if year:
        qs = qs.filter(year=year)

    monthly_rows = qs.values("month").annotate(total=Sum("count"))
    totals_by_month = {int(r["month"]): float(r["total"] or 0) for r in monthly_rows}
    month_numbers = list(range(1, 13))
    totals = [totals_by_month.get(m, 0.0) for m in month_numbers]
    peak = max(totals) if totals else 0.0
    relative_risk = [(v / peak * 100.0) if peak > 0 else 0.0 for v in totals]
    labels = [
        "Jan", "Feb", "Mar", "Apr", "May", "Jun",
        "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    ]

    return JsonResponse(
        {
            "labels": labels,
            "relative_risk": relative_risk,
            "counts": totals,
            "state": state,
            "year": year or "",
        }
    )


def api_hotspots(request):
    metric = normalize_metric(request.GET.get("metric"))
    state = request.GET.get("state") or "all"
    country = request.GET.get("country")
    season = request.GET.get("season") or "all"
    year = request.GET.get("year")

    det_qs = DetectionAggregate.objects.exclude(lat__isnull=True).exclude(lon__isnull=True)
    port_qs = PortTraffic.objects.all()
    if state != "all":
        det_qs = det_qs.filter(state_name=state)
    if year:
        det_qs = det_qs.filter(year=year)
        port_qs = port_qs.filter(year=year)

    detections = []
    for r in det_qs.values("state_name", "county_name", "year", "month", "lat", "lon").annotate(total=Sum("count"))[:3000]:
        if season_filter(r["month"], season):
            detections.append({"type": "detection", "state": r["state_name"], "name": r["county_name"], "year": r["year"], "month": r["month"], "lat": r["lat"], "lon": r["lon"], "value": float(r["total"] or 0)})

    airports = []
    for r in _nearby_air_rows(state, metric, country=country):
        if year and int(r["year"]) != int(year):
            continue
        if season_filter(r["month"], season):
            airports.append(
                {
                    "type": "airport",
                    "state": r["state_name"],
                    "name": f"{r['dest_airport']} {r['dest_city_name']}",
                    "year": r["year"],
                    "month": r["month"],
                    "lat": r["dest_lat"],
                    "lon": r["dest_lon"],
                    "value": float(r["value"] or 0),
                    "nearest_county_name": r["nearest_county_name"],
                    "nearest_county_state": r["nearest_county_state"],
                    "nearest_county_distance_miles": r["nearest_county_distance_miles"],
                }
            )
    airports = airports[:3000]

    ports = list(port_qs.values("year", "port_name", "state").annotate(value=Sum("total_foreign_loaded")).order_by("-value")[:200])
    return JsonResponse(
        {
            "detections": detections,
            "airports": airports,
            "ports": ports,
            "metric": metric,
            "season": season,
            "distance_limit_miles": CENTROID_DISTANCE_LIMIT_MILES,
        }
    )
