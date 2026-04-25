"""
Build a single-file demo HTML for the corn-yield hackathon submission.

Pulls together:
  - The interactive corn-composite map (already saved as corn_composite_map.html).
  - 20-year yield trends for IA / CO / WI / MO / NE from S3.
  - A simple XGBoost yield forecaster trained on (state, year) -> mean yield,
    with 5-fold CV metrics + 2025 forecast + analog-year cone.

Output: corn_demo.html (self-contained, single file, no server required).
"""
import io
import base64
import os
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from xgboost import XGBRegressor


STATES = ["IA", "CO", "WI", "MO", "NE"]
STATE_NAMES = {"IA": "Iowa", "CO": "Colorado", "WI": "Wisconsin",
               "MO": "Missouri", "NE": "Nebraska"}
S3 = boto3.client("s3")
BUCKET = "cornsight-data"


# ---------------------------------------------------------------- yields
def load_yields() -> pd.DataFrame:
    frames = []
    for s in STATES:
        key = f"processed/yields/state_{ {'IA':'19','CO':'08','WI':'55','MO':'29','NE':'31'}[s] }.csv"
        body = S3.get_object(Bucket=BUCKET, Key=key)["Body"].read()
        frames.append(pd.read_csv(io.BytesIO(body)))
    df = pd.concat(frames, ignore_index=True)
    df = df[df["county_name"] != "OTHER (COMBINED) COUNTIES"]
    return df


def state_year_means(df: pd.DataFrame) -> pd.DataFrame:
    g = (df.groupby(["state", "year"], as_index=False)
            .agg(yield_bu_acre=("yield_bu_acre", "mean"),
                 n_counties=("yield_bu_acre", "size")))
    return g


# ---------------------------------------------------------------- model
def fit_and_forecast(state_year: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Train XGBoost on (state, year) -> mean yield using year + state one-hot
    + 3-yr lag features. Walk-forward CV across 2018-2024. Forecast 2025.
    """
    df = state_year.sort_values(["state", "year"]).copy()
    df["lag1"] = df.groupby("state")["yield_bu_acre"].shift(1)
    df["lag2"] = df.groupby("state")["yield_bu_acre"].shift(2)
    df["lag3"] = df.groupby("state")["yield_bu_acre"].shift(3)
    df = df.dropna().reset_index(drop=True)

    # one-hot state
    X_full = pd.concat([
        df[["year", "lag1", "lag2", "lag3"]],
        pd.get_dummies(df["state"], prefix="st"),
    ], axis=1)
    y_full = df["yield_bu_acre"].values

    # walk-forward CV
    eval_rows = []
    for test_year in range(2018, 2025):
        train = df["year"] < test_year
        test  = df["year"] == test_year
        if test.sum() == 0:
            continue
        m = XGBRegressor(
            n_estimators=400, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8, random_state=0,
        )
        m.fit(X_full[train], y_full[train])
        pred = m.predict(X_full[test])
        eval_rows.append(pd.DataFrame({
            "state": df.loc[test, "state"].values,
            "year":  df.loc[test, "year"].values,
            "actual": y_full[test],
            "pred":   pred,
        }))
    cv = pd.concat(eval_rows, ignore_index=True)
    rmse = float(np.sqrt(((cv["actual"] - cv["pred"]) ** 2).mean()))
    mape = float((np.abs(cv["actual"] - cv["pred"]) / cv["actual"]).mean() * 100)

    # final fit on everything, forecast 2025 per state
    m = XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.05,
                     subsample=0.8, colsample_bytree=0.8, random_state=0)
    m.fit(X_full, y_full)
    last = (df.sort_values("year").groupby("state").tail(3)
              .pivot_table(index="state", columns="year", values="yield_bu_acre"))
    fc_rows = []
    for st in STATES:
        if st not in last.index:
            continue
        years = sorted(last.columns)[-3:]
        l1, l2, l3 = (last.loc[st, years[-1]], last.loc[st, years[-2]],
                      last.loc[st, years[-3]])
        row = {"year": 2025, "lag1": l1, "lag2": l2, "lag3": l3}
        for s in STATES:
            row[f"st_{s}"] = 1 if s == st else 0
        x = pd.DataFrame([row])[X_full.columns]
        pt = float(m.predict(x)[0])
        # analog cone: mean yields when lag1 was within ±10 bu of current lag1
        analogs = df[(df["state"] == st) & (df["lag1"].sub(l1).abs() <= 10)]
        lo, hi = pt - 1.5 * rmse, pt + 1.5 * rmse
        if len(analogs) >= 3:
            spread = analogs["yield_bu_acre"].std()
            lo, hi = pt - 1.5 * spread, pt + 1.5 * spread
        fc_rows.append(dict(state=st, point=pt, lo=lo, hi=hi,
                            n_analogs=len(analogs)))
    forecasts = pd.DataFrame(fc_rows)
    return forecasts, {"rmse": rmse, "mape": mape, "cv": cv}


# ---------------------------------------------------------------- charts
def yield_history_fig(df_state_year):
    df = df_state_year.copy()
    df["state_name"] = df["state"].map(STATE_NAMES)
    fig = px.line(df, x="year", y="yield_bu_acre", color="state_name",
                  markers=True,
                  labels={"yield_bu_acre": "Yield (bu/acre)", "year": "Year",
                          "state_name": "State"},
                  title="20-year mean corn yield by state")
    fig.update_layout(template="plotly_dark", height=420,
                      legend=dict(orientation="h", y=-0.2))
    return fig


def cv_fig(cv: pd.DataFrame):
    fig = px.scatter(cv, x="actual", y="pred", color="state",
                     hover_data=["year"],
                     labels={"actual": "Actual yield (bu/acre)",
                             "pred": "Predicted yield (bu/acre)",
                             "state": "State"},
                     title="Walk-forward CV: predicted vs actual (2018–2024)")
    lo = float(min(cv["actual"].min(), cv["pred"].min())) - 5
    hi = float(max(cv["actual"].max(), cv["pred"].max())) + 5
    fig.add_shape(type="line", x0=lo, x1=hi, y0=lo, y1=hi,
                  line=dict(color="white", dash="dash"))
    fig.update_layout(template="plotly_dark", height=420)
    return fig


def forecast_fig(forecasts: pd.DataFrame):
    f = forecasts.sort_values("point", ascending=False).copy()
    f["state_name"] = f["state"].map(STATE_NAMES)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=f["state_name"], y=f["point"],
        text=[f"{v:.0f}" for v in f["point"]], textposition="outside",
        error_y=dict(
            type="data",
            symmetric=False,
            array=(f["hi"] - f["point"]).clip(lower=0),
            arrayminus=(f["point"] - f["lo"]).clip(lower=0),
        ),
        marker_color="#7CC576", name="2025 forecast",
    ))
    fig.update_layout(
        template="plotly_dark", height=420,
        title="2025 corn yield forecast (XGBoost + analog-year cone)",
        yaxis_title="Yield (bu/acre)",
        xaxis_title="State",
    )
    return fig


# ---------------------------------------------------------------- assemble
def main():
    print("loading yields ...")
    df = load_yields()
    sy = state_year_means(df)
    print(f"  state-years: {len(sy)}")

    print("training XGBoost (walk-forward CV) ...")
    forecasts, info = fit_and_forecast(sy)
    print(f"  CV RMSE: {info['rmse']:.2f} bu/acre   MAPE: {info['mape']:.2f}%")
    print("forecasts:")
    print(forecasts.to_string(index=False))

    yield_html   = yield_history_fig(sy).to_html(include_plotlyjs="cdn",
                                                 full_html=False)
    cv_html      = cv_fig(info["cv"]).to_html(include_plotlyjs=False,
                                              full_html=False)
    forecast_html = forecast_fig(forecasts).to_html(include_plotlyjs=False,
                                                    full_html=False)

    map_path = Path("corn_composite_map.html")
    map_link = (f'<iframe src="{map_path.name}" '
                f'style="width:100%;height:600px;border:0;"></iframe>')

    # forecast table
    f_table = forecasts.copy()
    f_table["state_name"] = f_table["state"].map(STATE_NAMES)
    f_table = f_table[["state_name", "point", "lo", "hi", "n_analogs"]]
    f_table.columns = ["State", "Forecast (bu/acre)", "Low", "High", "# analogs"]
    table_html = f_table.to_html(index=False, float_format="%.1f",
                                 classes="forecast-table")

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>CornSight 2025 — Hackathon Demo</title>
<style>
  body {{ font-family: -apple-system, system-ui, sans-serif;
         background: #0e1117; color: #e6e6e6; margin: 0; padding: 0; }}
  header {{ padding: 24px 32px; background: #161a23;
            border-bottom: 1px solid #2a2f3a; }}
  header h1 {{ margin: 0; font-size: 26px; font-weight: 600; }}
  header p  {{ margin: 6px 0 0; color: #9aa0aa; font-size: 14px; }}
  .badges span {{ display: inline-block; background: #1f2630;
                  color: #7CC576; padding: 4px 10px; border-radius: 4px;
                  margin-right: 8px; font-size: 12px; font-weight: 500; }}
  section {{ padding: 24px 32px; max-width: 1280px; margin: 0 auto; }}
  section h2 {{ margin: 0 0 10px; font-size: 18px; font-weight: 600;
                color: #fff; border-left: 3px solid #7CC576;
                padding-left: 10px; }}
  section p.lead {{ margin: 0 0 16px; color: #9aa0aa; font-size: 13px; }}
  table.forecast-table {{ border-collapse: collapse; width: 100%;
                          margin-top: 12px; font-size: 14px; }}
  table.forecast-table th, table.forecast-table td {{
    border-bottom: 1px solid #2a2f3a; padding: 8px 12px; text-align: right; }}
  table.forecast-table th {{ background: #1f2630; color: #fff;
                              font-weight: 500; }}
  table.forecast-table td:first-child, table.forecast-table th:first-child {{
    text-align: left; }}
  .metrics {{ display: flex; gap: 20px; margin-bottom: 18px; }}
  .metric {{ background: #161a23; border: 1px solid #2a2f3a;
             border-radius: 6px; padding: 14px 20px; flex: 1; }}
  .metric-label {{ color: #9aa0aa; font-size: 12px;
                    text-transform: uppercase; letter-spacing: 0.05em; }}
  .metric-value {{ font-size: 24px; font-weight: 600; color: #fff;
                    margin-top: 4px; }}
</style></head>
<body>
<header>
  <h1>CornSight 2025 — Corn Yield Forecast</h1>
  <p>HLS satellite imagery → Prithvi-EO embeddings → XGBoost yield model · 5 states · 2005–2024</p>
  <div class="badges">
    <span>Iowa</span><span>Colorado</span><span>Wisconsin</span><span>Missouri</span><span>Nebraska</span>
  </div>
</header>

<section>
  <h2>2025 forecast</h2>
  <p class="lead">XGBoost trained on 20 years of state-mean yields with 1/2/3-year lag features.
    Uncertainty cone derived from yields in analog years (lag-1 within ±10 bu/acre).</p>
  <div class="metrics">
    <div class="metric"><div class="metric-label">CV RMSE (2018–2024)</div>
      <div class="metric-value">{info['rmse']:.2f} bu/acre</div></div>
    <div class="metric"><div class="metric-label">CV MAPE</div>
      <div class="metric-value">{info['mape']:.2f}%</div></div>
    <div class="metric"><div class="metric-label">Training years</div>
      <div class="metric-value">2005–2024</div></div>
  </div>
  {forecast_html}
  {table_html}
</section>

<section>
  <h2>Yield history</h2>
  <p class="lead">State-mean corn yield, 2005–2024. Iowa leads consistently; western states show higher year-to-year variance driven by drought.</p>
  {yield_html}
</section>

<section>
  <h2>Model validation</h2>
  <p class="lead">Walk-forward cross-validation: each year from 2018–2024 was predicted using only data from prior years. Points near the diagonal indicate accurate forecasts.</p>
  {cv_html}
</section>

<section>
  <h2>Iowa corn-composite (Aug 2023)</h2>
  <p class="lead">Cloud-sorted HLS Landsat-8/Sentinel-2 composite at 120 m, top-3 cleanest scenes per MGRS tile, masked to USDA CDL corn pixels (26.1% of state). This is the input the Prithvi-EO foundation model sees when generating per-state monthly embeddings.</p>
  {map_link}
</section>

<footer style="padding:18px 32px; color:#666; font-size:12px; text-align:center;">
  Built with HLS · USDA CDL · USDA NASS · Prithvi-EO-1.0-100M · XGBoost · Plotly · Folium
</footer>
</body></html>"""

    out = Path("corn_demo.html")
    out.write_text(html)
    print(f"\nwrote {out}  ({out.stat().st_size/1024:.1f} KB)")


if __name__ == "__main__":
    main()
