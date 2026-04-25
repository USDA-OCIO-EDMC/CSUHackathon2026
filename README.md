
# Team 23 — Fruit Fly Spread-Risk Analysis Pipeline
 
## Project Summary
 
Note: File paths and pip requirements will very likely need to be updated in order for the project to work, since the project was designed to work in a Databricks environment

This project investigates whether international flight traffic into U.S. states correlates with invasive fruit fly detections at the county level. Two notebooks form a pipeline: one **ingests** raw data from USDA PPQ ArcGIS Feature Servers, and the other **joins and analyzes** that data to produce combined datasets linking fruit fly detections with flight volume by origin country.
 
All intermediate and final outputs are cached as Parquet files so either notebook can be re-run without redundant downloads.
 
---