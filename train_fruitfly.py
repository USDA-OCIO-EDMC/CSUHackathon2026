
import os
import json
import pickle
import pandas as pd
import numpy as np
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import (
    LeaveOneOut, StratifiedKFold, cross_val_predict, KFold
)
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, roc_auc_score,
    confusion_matrix, accuracy_score,
    mean_absolute_error, r2_score,
)

# ----------------------------------------------------------
# Paths (SageMaker convention)
# ----------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR = os.environ.get("SM_CHANNEL_TRAINING") or BASE_DIR
if not os.path.isdir(INPUT_DIR):
    print(f"Training input directory not found: {INPUT_DIR}; using local project directory.")
    INPUT_DIR = BASE_DIR

MODEL_DIR = os.environ.get("SM_MODEL_DIR") or os.path.join(BASE_DIR, "training_output")
OUTPUT_DIR = os.environ.get("SM_OUTPUT_DATA_DIR") or os.path.join(BASE_DIR, "training_output")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Input directory contents: {os.listdir(INPUT_DIR)}")

# ----------------------------------------------------------
# 1. Load CSVs
# ----------------------------------------------------------
final     = pd.read_csv(os.path.join(INPUT_DIR, "Final.csv"))
temp_suit = pd.read_csv(os.path.join(INPUT_DIR, "temperature_suitability.csv"))
pest      = pd.read_csv(os.path.join(INPUT_DIR, "pest_status.csv"))
passenger = pd.read_csv(os.path.join(INPUT_DIR, "passenger_data.csv"))
trade     = pd.read_csv(os.path.join(INPUT_DIR, "trade_data.csv"))
us_port   = pd.read_csv(os.path.join(INPUT_DIR, "us_port.csv"))

origin_path = os.path.join(INPUT_DIR, "origin.csv")
origin = pd.read_csv(origin_path) if os.path.exists(origin_path) else None
arcgis_path = os.path.join(INPUT_DIR, "arcgis_enrichment.csv")
arcgis = pd.read_csv(arcgis_path) if os.path.exists(arcgis_path) else None
raw_detection_path = os.path.join(INPUT_DIR, "raw_detections.csv")
raw_detections = pd.read_csv(raw_detection_path) if os.path.exists(raw_detection_path) else None

print(f"Final.csv shape       : {final.shape}")
print(f"temperature_suit shape: {temp_suit.shape}")
if arcgis is not None:
    print(f"arcgis_enrichment shape: {arcgis.shape}")
if raw_detections is not None:
    print(f"raw_detections shape   : {raw_detections.shape}")

# ----------------------------------------------------------
# 2. Clean column names
# ----------------------------------------------------------
for d in [final, temp_suit, pest, passenger, trade, us_port]:
    d.columns = d.columns.str.strip().str.lower().str.replace(" ", "_")
if origin is not None:
    origin.columns = origin.columns.str.strip().str.lower().str.replace(" ", "_")
if arcgis is not None:
    arcgis.columns = arcgis.columns.str.strip().str.lower().str.replace(" ", "_")
if raw_detections is not None:
    raw_detections.columns = raw_detections.columns.str.strip().str.lower().str.replace(" ", "_")

# ----------------------------------------------------------
# 3. Standardise port names -> airport codes
# ----------------------------------------------------------
PORT_MAP = {
    "Atlanta": "ATL", "Dallas": "DFW", "Houston": "IAH",
    "JFK": "JFK", "LAX": "LAX", "Miami": "MIA",
    "Chicago": "ORD", "Seattle": "SEA",
}

def map_port(col):
    return col.astype(str).str.strip().map(PORT_MAP).fillna(col.str.strip())

for d in [passenger, us_port]:
    if "us_port" in d.columns:
        d["us_port"] = map_port(d["us_port"])

# ----------------------------------------------------------
# 4. Parse month number
# ----------------------------------------------------------
final["month_str"] = final["month"].astype(str).str.strip()
final["month_num"] = pd.to_datetime(
    final["month_str"], format="%Y-%m", errors="coerce"
).dt.month
final["year"] = pd.to_numeric(final["year"], errors="coerce")

# ----------------------------------------------------------
# 5. Merge temperature suitability
# ----------------------------------------------------------
temp_suit.rename(columns={"month": "month_int"}, inplace=True)
df = final.merge(
    temp_suit,
    left_on=["us_port", "month_num"],
    right_on=["us_port", "month_int"],
    how="left",
)

arcgis_feature_cols = []
if arcgis is not None and {"us_port", "month"}.issubset(arcgis.columns):
    arcgis["us_port"] = map_port(arcgis["us_port"])
    arcgis["month_num"] = pd.to_numeric(arcgis["month"], errors="coerce").fillna(1).astype(int)
    arcgis_feature_cols = [
        "arcgis_cpri",
        "arcgis_monthly_risk",
        "arcgis_detections_total",
        "arcgis_routes",
        "temp_weighted_risk",
        "species_diversity_idx",
        "cpri_temp_composite",
        "route_risk_composite",
        "risk_vs_annual",
        "log_routes",
        "log_detections",
    ]
    merge_cols = ["us_port", "month_num"] + [c for c in arcgis_feature_cols if c in arcgis.columns]
    df = df.merge(arcgis[merge_cols], on=["us_port", "month_num"], how="left")
    print(f"Merged ArcGIS enrichment features: {[c for c in arcgis_feature_cols if c in df.columns]}")

raw_detection_feature_cols = []
if raw_detections is not None:
    port_col = next((c for c in ["port", "us_port", "airport"] if c in raw_detections.columns), None)
    count_col = next((c for c in ["detectioncount", "detection_count", "count", "detections"] if c in raw_detections.columns), None)
    date_col = next((c for c in ["detectiondate", "detection_date", "month", "date"] if c in raw_detections.columns), None)
    species_col = next((c for c in ["species", "commonname", "common_name"] if c in raw_detections.columns), None)
    if port_col and count_col and date_col:
        raw = raw_detections.copy()
        raw["us_port"] = map_port(raw[port_col])
        raw["month_num"] = pd.to_datetime(raw[date_col], errors="coerce").dt.month
        raw[count_col] = pd.to_numeric(raw[count_col], errors="coerce").fillna(0)
        raw_agg = (
            raw.dropna(subset=["month_num"])
            .groupby(["us_port", "month_num"])[count_col]
            .sum()
            .reset_index()
            .rename(columns={count_col: "raw_detection_count"})
        )
        raw_agg["month_num"] = raw_agg["month_num"].astype(int)
        hist = (
            raw.groupby("us_port")[count_col]
            .agg(["sum", "mean", "max", "count"])
            .reset_index()
            .rename(
                columns={
                    "sum": "hist_total_detections",
                    "mean": "hist_mean_detections",
                    "max": "hist_max_detections",
                    "count": "hist_observation_count",
                }
            )
        )
        if species_col:
            species_diversity = (
                raw.groupby("us_port")[species_col]
                .nunique()
                .reset_index()
                .rename(columns={species_col: "hist_species_count"})
            )
            hist = hist.merge(species_diversity, on="us_port", how="left")
        df = df.merge(raw_agg, on=["us_port", "month_num"], how="left")
        df = df.merge(hist, on="us_port", how="left")
        raw_detection_feature_cols = [
            "raw_detection_count",
            "hist_total_detections",
            "hist_mean_detections",
            "hist_max_detections",
            "hist_observation_count",
            "hist_species_count",
        ]
        print(f"Merged raw detection history features: {[c for c in raw_detection_feature_cols if c in df.columns]}")

# ----------------------------------------------------------
# 6. Ensure numeric columns
# ----------------------------------------------------------
df["passengers"]    = pd.to_numeric(df["passengers"], errors="coerce").fillna(0)
df["fruit_imports"] = pd.to_numeric(df["fruit_imports"], errors="coerce").fillna(0)
df["detections"]    = pd.to_numeric(df["detections"], errors="coerce").fillna(0)
df["destination_temp_f"] = pd.to_numeric(
    df.get("destination_temp_f", pd.Series(dtype=float)), errors="coerce"
).fillna(60.0)
df["temp_suitability_score"] = pd.to_numeric(
    df.get("temp_suitability_score", pd.Series(dtype=float)), errors="coerce"
).fillna(0.5)

# ==========================================================
# 7. TEMPERATURE SUITABILITY MASK (Domain Knowledge)
#    Fruit flies CANNOT survive below ~40F.
#    Below 40F -> risk = 0, don't model it.
#    40-45F   -> marginal, reduced viability
#    Above 45F -> suitable for establishment
# ==========================================================
print("===== TEMPERATURE SUITABILITY MASK =====")

def temp_zone(temp_f):
    """Classify temperature into entomological risk zones."""
    if pd.isna(temp_f):
        return "UNKNOWN"
    elif temp_f < 40:
        return "DEAD_ZONE"
    elif temp_f < 45:
        return "MARGINAL"
    elif temp_f < 60:
        return "VIABLE"
    elif temp_f < 75:
        return "OPTIMAL"
    else:
        return "HOT_SUITABLE"

df["temp_zone"] = df["destination_temp_f"].apply(temp_zone)

# Binary mask: is this port-month even worth modeling?
df["temp_viable"] = (df["destination_temp_f"] >= 40).astype(int)

# Continuous suitability score (raster-style)
def suitability_score(temp_f):
    if pd.isna(temp_f) or temp_f < 40:
        return 0.0
    elif temp_f < 45:
        return 0.5 * (temp_f - 40) / 5.0
    elif temp_f < 75:
        return 0.5 + 0.5 * (temp_f - 45) / 30.0
    else:
        return 0.9

df["temp_risk_score"] = df["destination_temp_f"].apply(suitability_score)

zone_counts = df["temp_zone"].value_counts()
print(f"Temperature zones:{zone_counts}")
print(f"Viable rows (>=40F): {df['temp_viable'].sum()} / {len(df)}")

# ==========================================================
# 8. PEST STATUS ENCODING (Origin Country Risk)
# ==========================================================
PEST_SCORE_MAP = {"Absent": 0, "Emerging": 1, "Present": 2}
df["pest_score"] = df["pest_status"].map(PEST_SCORE_MAP).fillna(0).astype(int)

# ==========================================================
# 9. PATHWAY RISK SCORE
#    Risk = f(origin_pest_status, port_temperature, trade_volume, passenger_volume)
#    A pathway is: origin_country -> us_port in a given month
# ==========================================================
print("===== PATHWAY RISK FEATURES =====")

df["pathway_risk"]       = df["pest_score"] * df["temp_risk_score"]
df["trade_pathway_risk"] = df["fruit_imports"] * df["temp_risk_score"]
df["pax_pathway_risk"]   = df["passengers"] * df["pest_score"] * df["temp_risk_score"]
df["log_passengers"]     = np.log1p(df["passengers"])
df["log_imports"]        = np.log1p(df["fruit_imports"])
df["has_imports"]        = (df["fruit_imports"] > 0).astype(int)
df["import_density"]     = df["fruit_imports"] / (df["passengers"] + 1)
df["peak_season"]        = df["month_num"].isin([6, 7, 8, 9]).astype(int)
df["seasonal_temp_risk"] = df["peak_season"] * df["temp_risk_score"]

# ==========================================================
# 10. LABEL ENCODE CATEGORICALS
# ==========================================================
label_encoders = {}
for col in ["origin_country", "us_port", "detection_type"]:
    if col in df.columns:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col].astype(str))
        label_encoders[col] = le

ZONE_ORDER = {"DEAD_ZONE": 0, "MARGINAL": 1, "VIABLE": 2, "OPTIMAL": 3, "HOT_SUITABLE": 4, "UNKNOWN": 2}
df["temp_zone_enc"] = df["temp_zone"].map(ZONE_ORDER).fillna(2).astype(int)

# ==========================================================
# 11. DEFINE TARGETS
# ==========================================================
df["detection_count"] = df["detections"]

def risk_tier(det):
    if det == 0:
        return 0   # NO RISK
    elif det <= 5:
        return 1   # LOW
    elif det <= 10:
        return 2   # MEDIUM
    else:
        return 3   # HIGH

df["risk_tier"] = df["detections"].apply(risk_tier)
df["has_detection"] = (df["detections"] > 0).astype(int)

print(f"Dataset size: {df.shape}")
print(f"Risk tier distribution:")
print(df["risk_tier"].value_counts().sort_index().rename({0:"NO_RISK",1:"LOW",2:"MEDIUM",3:"HIGH"}))
print(f"Detection count stats:")
print(df["detection_count"].describe())

# ==========================================================
# 12. FEATURE SELECTION
# ==========================================================
feature_cols = [
    "passengers", "log_passengers",
    "fruit_imports", "log_imports", "has_imports",
    "import_density",
    "pest_score",
    "destination_temp_f",
    "temp_risk_score",
    "temp_viable",
    "temp_zone_enc",
    "pathway_risk",
    "trade_pathway_risk",
    "pax_pathway_risk",
    "month_num", "year",
    "peak_season",
    "seasonal_temp_risk",
    "origin_country_enc",
    "us_port_enc",
]
for optional_col in arcgis_feature_cols + raw_detection_feature_cols:
    if optional_col in df.columns:
        df[optional_col] = pd.to_numeric(df[optional_col], errors="coerce").fillna(0)
        feature_cols.append(optional_col)
feature_cols = [c for c in feature_cols if c in df.columns]

X = df[feature_cols].fillna(0)
print(f"Features ({len(feature_cols)}): {feature_cols}")
print(f"Samples: {len(X)}")

# ==========================================================
# 13. MODEL A: REGRESSION - Predict detection count
# ==========================================================
print("" + "=" * 60)
print("MODEL A: DETECTION COUNT REGRESSION")
print("=" * 60)

y_reg = df["detection_count"]

reg_model = XGBRegressor(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=1.0,
    reg_lambda=2.0,
    random_state=42,
)

print("Running LOO cross-validation for regression...")
loo = LeaveOneOut()
y_pred_reg_loo = cross_val_predict(reg_model, X, y_reg, cv=loo)

mae_loo = mean_absolute_error(y_reg, y_pred_reg_loo)
r2_loo  = r2_score(y_reg, y_pred_reg_loo)

print(f"LOO MAE : {mae_loo:.3f} detections")
print(f"LOO R2  : {r2_loo:.4f}")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
y_pred_reg_kf = cross_val_predict(reg_model, X, y_reg, cv=kf)
mae_kf = mean_absolute_error(y_reg, y_pred_reg_kf)
r2_kf  = r2_score(y_reg, y_pred_reg_kf)
print(f"5-Fold MAE: {mae_kf:.3f}")
print(f"5-Fold R2 : {r2_kf:.4f}")

# ==========================================================
# 14. MODEL B: RISK TIER CLASSIFICATION (4-class)
# ==========================================================
print("" + "=" * 60)
print("MODEL B: RISK TIER CLASSIFICATION")
print("=" * 60)

y_tier = df["risk_tier"]
print(f"Class distribution:{y_tier.value_counts().sort_index()}")

min_class_count = y_tier.value_counts().min()
n_splits = min(5, min_class_count) if min_class_count >= 2 else 2

tier_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=2,
    reg_alpha=1.0,
    reg_lambda=2.0,
    objective="multi:softprob",
    num_class=4,
    eval_metric="mlogloss",
    random_state=42,
)

print("Running LOO cross-validation for risk tiers...")
y_pred_tier_loo = cross_val_predict(tier_model, X, y_tier, cv=loo)

tier_acc_loo = accuracy_score(y_tier, y_pred_tier_loo)
print(f"LOO Accuracy: {tier_acc_loo:.4f}")
print("LOO Classification Report:")
tier_names = ["NO_RISK (0)", "LOW (1-5)", "MEDIUM (6-10)", "HIGH (11+)"]
existing_tiers = sorted(y_tier.unique())
existing_names = [tier_names[i] for i in existing_tiers]
print(classification_report(
    y_tier, y_pred_tier_loo,
    labels=existing_tiers,
    target_names=existing_names,
    zero_division=0,
))

if min_class_count >= 2:
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    y_pred_tier_skf = cross_val_predict(tier_model, X, y_tier, cv=skf)
    tier_acc_skf = accuracy_score(y_tier, y_pred_tier_skf)
    print(f"{n_splits}-Fold Accuracy: {tier_acc_skf:.4f}")
else:
    tier_acc_skf = tier_acc_loo

# ==========================================================
# 15. MODEL C: BINARY DETECTION (TEMP-MASKED)
#     Only model rows where temp >= 40F
#     Below 40F = automatic "no risk" prediction
# ==========================================================
print("" + "=" * 60)
print("MODEL C: BINARY DETECTION (TEMP-MASKED)")
print("=" * 60)

viable_mask = df["temp_viable"] == 1
df_viable = df[viable_mask].copy()
df_dead   = df[~viable_mask].copy()

print(f"Viable rows (>=40F): {len(df_viable)}")
print(f"Dead zone rows (<40F): {len(df_dead)}")
if len(df_dead) > 0:
    print(f"Dead zone detections (should be ~0): {df_dead['detections'].sum()}")

X_viable = df_viable[feature_cols].fillna(0)
y_viable = df_viable["has_detection"]

print(f"Viable zone target distribution:{y_viable.value_counts()}")

neg_count = int((y_viable == 0).sum())
pos_count = int((y_viable == 1).sum())
spw = neg_count / max(pos_count, 1)

binary_model = XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.08,
    subsample=0.8,
    colsample_bytree=0.7,
    min_child_weight=3,
    reg_alpha=1.0,
    reg_lambda=2.0,
    scale_pos_weight=spw,
    eval_metric="auc",
    random_state=42,
)

if len(X_viable) > 5 and y_viable.nunique() > 1:
    loo_v = LeaveOneOut()
    y_pred_bin_loo = cross_val_predict(binary_model, X_viable, y_viable, cv=loo_v)
    y_proba_bin_loo = cross_val_predict(binary_model, X_viable, y_viable, cv=loo_v, method="predict_proba")[:, 1]

    bin_acc_loo = accuracy_score(y_viable, y_pred_bin_loo)
    try:
        bin_auc_loo = roc_auc_score(y_viable, y_proba_bin_loo)
    except ValueError:
        bin_auc_loo = 0.0

    print(f"LOO Accuracy (viable zone): {bin_acc_loo:.4f}")
    print(f"LOO ROC-AUC (viable zone) : {bin_auc_loo:.4f}")
    print("Classification Report:")
    print(classification_report(
        y_viable, y_pred_bin_loo,
        target_names=["No Detection", "Detection"],
        zero_division=0,
    ))
else:
    bin_acc_loo = 0.0
    bin_auc_loo = 0.0
    print("Not enough viable-zone data for binary CV")

# ==========================================================
# 16. TRAIN FINAL MODELS ON ALL DATA (for deployment)
# ==========================================================
print("===== TRAINING FINAL MODELS ON ALL DATA =====")

reg_model.fit(X, y_reg)
print("Regression model trained.")

tier_model.fit(X, y_tier)
print("Risk tier model trained.")

if len(X_viable) > 5 and y_viable.nunique() > 1:
    binary_model.fit(X_viable, y_viable)
    print("Binary model (temp-masked) trained.")

# ==========================================================
# 17. FEATURE IMPORTANCE
# ==========================================================
print("===== FEATURE IMPORTANCE (Regression Model) =====")
importances = pd.Series(reg_model.feature_importances_, index=feature_cols)
for feat, imp in importances.nlargest(len(feature_cols)).items():
    bar = "X" * int(imp * 50)
    print(f"  {feat:25s}  {imp:.4f}  {bar}")

# ==========================================================
# 18. SAVE ALL ARTIFACTS
# ==========================================================
os.makedirs(MODEL_DIR, exist_ok=True)

reg_model.save_model(os.path.join(MODEL_DIR, "xgb_regression.json"))
tier_model.save_model(os.path.join(MODEL_DIR, "xgb_risk_tier.json"))

with open(os.path.join(MODEL_DIR, "model_regression.pkl"), "wb") as f:
    pickle.dump(reg_model, f)
with open(os.path.join(MODEL_DIR, "model_risk_tier.pkl"), "wb") as f:
    pickle.dump(tier_model, f)
with open(os.path.join(MODEL_DIR, "model_binary.pkl"), "wb") as f:
    pickle.dump(binary_model, f)

with open(os.path.join(MODEL_DIR, "feature_columns.json"), "w") as f:
    json.dump(feature_cols, f)

le_mappings = {}
for col, le in label_encoders.items():
    le_mappings[col] = dict(zip(
        le.classes_.tolist(),
        le.transform(le.classes_).tolist()
    ))
with open(os.path.join(MODEL_DIR, "label_encoders.json"), "w") as f:
    json.dump(le_mappings, f)

with open(os.path.join(MODEL_DIR, "port_mapping.json"), "w") as f:
    json.dump(PORT_MAP, f)

temp_thresholds = {
    "dead_zone_below_f": 40,
    "marginal_range_f": [40, 45],
    "viable_range_f": [45, 60],
    "optimal_range_f": [60, 75],
    "hot_suitable_above_f": 75,
    "note": "Fruit flies cannot survive below 40F. Below this threshold, risk is automatically zero regardless of other factors."
}
with open(os.path.join(MODEL_DIR, "temp_thresholds.json"), "w") as f:
    json.dump(temp_thresholds, f)

domain_knowledge = {
    "temp_dead_zone_f": 40,
    "temp_marginal_range_f": [40, 45],
    "temp_viable_range_f": [45, 60],
    "temp_optimal_range_f": [60, 75],
    "peak_months": [6, 7, 8, 9],
    "risk_class_names": ["NO_RISK (0)", "LOW (1-5)", "MEDIUM (6-10)", "HIGH (11+)"],
    "pest_score_map": PEST_SCORE_MAP,
    "port_map": PORT_MAP,
}
with open(os.path.join(MODEL_DIR, "domain_knowledge.json"), "w") as f:
    json.dump(domain_knowledge, f, indent=2)

metrics = {
    "regression": {
        "loo_mae": round(mae_loo, 4),
        "loo_r2": round(r2_loo, 4),
        "kfold_mae": round(mae_kf, 4),
        "kfold_r2": round(r2_kf, 4),
    },
    "risk_tier_classification": {
        "loo_accuracy": round(tier_acc_loo, 4),
        "kfold_accuracy": round(tier_acc_skf, 4),
        "n_classes": len(existing_tiers),
        "class_names": existing_names,
    },
    "binary_temp_masked": {
        "loo_accuracy": round(bin_acc_loo, 4),
        "loo_roc_auc": round(bin_auc_loo, 4),
        "viable_samples": int(len(X_viable)),
        "dead_zone_samples": int(len(df_dead)),
    },
    "dataset": {
        "n_samples": int(len(X)),
        "n_features": len(feature_cols),
        "positive_rate": round(float(df["has_detection"].mean()), 4),
        "detection_mean": round(float(df["detections"].mean()), 2),
        "detection_median": round(float(df["detections"].median()), 2),
    },
    "domain_knowledge": {
        "temp_threshold_f": 40,
        "peak_season_months": [6, 7, 8, 9],
        "risk_tiers": {"0": "NO_RISK", "1": "LOW (1-5)", "2": "MEDIUM (6-10)", "3": "HIGH (11+)"},
    }
}

with open(os.path.join(MODEL_DIR, "metrics.json"), "w") as f:
    json.dump(metrics, f, indent=2)

print(f"===== ALL ARTIFACTS SAVED TO {MODEL_DIR} =====")
print(f"Final metrics:{json.dumps(metrics, indent=2)}")
print("===== DONE =====")
