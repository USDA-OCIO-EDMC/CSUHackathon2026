# Group21-Hackathon2026



## Project Overview

Risk-to-Country Analysis:
csurams.maps.arcgis.com/apps/mapviewer/index.html?webmap=3af11124622f4658962a2fdfb0960563

Our analysis combines multiple datasets related to:

- International inbound passenger travel
- Fruit fly detections in the United States
- Fruit production by country
- Country-to-fruit fly associations
- Monthly climate suitability indicators

Using these datasets, we created risk scores that help compare where risk from fruit fly introduction may be higher.

## Folder Structure

```text
Group21-Hackathon2026/
│
├── Results/
│   ├── visuals/
│   ├── clean_target_fly_country_associations.csv
│   ├── country_month_route_risk.csv
│   ├── fruit_production_by_country.csv
│   ├── international_inbound_segments.csv
│   ├── origin_country_risk_month_FINAL.csv
│   └── origin_country_risk_month_with_iso_min_temp.csv
│
└── TimelineDemo/
    ├── animate_origin_risk.py - Go month by month, analyzing risks from a country's fruit fly population
    └── animate_route_risk.py - Go month by month, analyzing risks from a country's fruit fly population spreading by passenger plane

