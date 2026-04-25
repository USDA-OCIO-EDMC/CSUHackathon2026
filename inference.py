import json
import os
import pickle

import numpy as np
import pandas as pd


PORT_MAP = {
    "Atlanta": "ATL",
    "Dallas": "DFW",
    "Houston": "IAH",
    "JFK": "JFK",
    "LAX": "LAX",
    "Miami": "MIA",
    "Chicago": "ORD",
    "Seattle": "SEA",
    "atlanta": "ATL",
    "dallas": "DFW",
    "houston": "IAH",
    "jfk": "JFK",
    "lax": "LAX",
    "miami": "MIA",
    "chicago": "ORD",
    "seattle": "SEA",
}

PEST_SCORE_MAP = {"absent": 0, "emerging": 1, "present": 2}
DEFAULT_RISK_CLASS_NAMES = {
    0: "NO_RISK (0)",
    1: "LOW (1-5)",
    2: "MEDIUM (6-10)",
    3: "HIGH (11+)",
}


def _load_json(path, default):
    if not os.path.exists(path):
        return default
    with open(path) as handle:
        return json.load(handle)


def _load_pickle(path):
    with open(path, "rb") as handle:
        return pickle.load(handle)


def _load_xgb_json(path, is_regressor=False):
    if is_regressor:
        from xgboost import XGBRegressor

        model = XGBRegressor()
    else:
        from xgboost import XGBClassifier

        model = XGBClassifier()
    model.load_model(path)
    return model


def model_fn(model_dir):
    bundle = {
        "feature_columns": _load_json(os.path.join(model_dir, "feature_columns.json"), []),
        "label_encoders": _load_json(os.path.join(model_dir, "label_encoders.json"), {}),
        "metrics": _load_json(os.path.join(model_dir, "metrics.json"), {}),
        "domain_knowledge": _load_json(os.path.join(model_dir, "domain_knowledge.json"), {}),
    }

    pickle_candidates = [
        ("model_bundle", "model_bundle.pkl"),
        ("models", "models.pkl"),
        ("regression_model", "model_regression.pkl"),
        ("regression_model", "regression_model.pkl"),
        ("regression_model", "xgb_regression.pkl"),
        ("regressor", "regressor.pkl"),
        ("risk_tier_model", "model_risk_tier.pkl"),
        ("risk_tier_model", "xgb_risk_tier.pkl"),
        ("risk_tier_model", "xgb_risk_class.pkl"),
        ("risk_tier_model", "risk_tier_model.pkl"),
        ("risk_classifier", "risk_classifier.pkl"),
        ("binary_model", "model_binary.pkl"),
        ("binary_model", "xgb_binary.pkl"),
        ("binary_model", "binary_model.pkl"),
        ("binary_classifier", "binary_classifier.pkl"),
        ("temp_masked_model", "temp_masked_model.pkl"),
        ("model", "model.pkl"),
    ]
    for key, filename in pickle_candidates:
        path = os.path.join(model_dir, filename)
        if not os.path.exists(path):
            continue
        if key == "model" and any(
            typed_key in bundle
            for typed_key in [
                "regression_model",
                "regressor",
                "risk_tier_model",
                "risk_classifier",
                "binary_model",
                "binary_classifier",
                "temp_masked_model",
            ]
        ):
            continue
        loaded = _load_pickle(path)
        if isinstance(loaded, dict):
            bundle.update(loaded)
        else:
            bundle[key] = loaded

    json_candidates = [
        ("model", "xgboost_model.json", False),
        ("regression_model", "xgb_regression.json", True),
        ("regression_model", "regression_model.json", True),
        ("regressor", "regressor.json", True),
        ("risk_tier_model", "xgb_risk_tier.json", False),
        ("risk_tier_model", "xgb_risk_class.json", False),
        ("risk_tier_model", "risk_tier_model.json", False),
        ("risk_classifier", "risk_classifier.json", False),
        ("binary_model", "xgb_binary.json", False),
        ("binary_model", "binary_model.json", False),
        ("binary_classifier", "binary_classifier.json", False),
        ("temp_masked_model", "temp_masked_model.json", False),
    ]
    for key, filename, is_regressor in json_candidates:
        path = os.path.join(model_dir, filename)
        if key == "model" and any(
            typed_key in bundle
            for typed_key in [
                "regression_model",
                "regressor",
                "risk_tier_model",
                "risk_classifier",
                "binary_model",
                "binary_classifier",
                "temp_masked_model",
            ]
        ):
            continue
        if os.path.exists(path) and key not in bundle:
            bundle[key] = _load_xgb_json(path, is_regressor=is_regressor)

    if "model" not in bundle and not any(
        key in bundle
        for key in [
            "regression_model",
            "regressor",
            "risk_tier_model",
            "risk_classifier",
            "binary_model",
            "binary_classifier",
            "temp_masked_model",
        ]
    ):
        raise FileNotFoundError("No supported model file found in model artifact.")

    return bundle


def input_fn(request_body, request_content_type):
    if request_content_type != "application/json":
        raise ValueError(f"Unsupported content type: {request_content_type}")

    payload = json.loads(request_body)
    if isinstance(payload, dict) and "instances" in payload:
        records = payload["instances"]
    elif isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
    elif isinstance(payload, list):
        records = payload
    else:
        records = [payload]

    return pd.DataFrame(records)


def _risk_tier_from_probability(probability):
    if probability >= 0.85:
        return "CRITICAL"
    if probability >= 0.70:
        return "HIGH"
    if probability >= 0.50:
        return "MEDIUM"
    return "LOW"


def _risk_tier_from_count(count):
    if count <= 0:
        return "NO_RISK (0)"
    if count <= 5:
        return "LOW (1-5)"
    if count <= 10:
        return "MEDIUM (6-10)"
    return "HIGH (11+)"


def _risk_class_name(class_id, metrics):
    risk_tiers = metrics.get("domain_knowledge", {}).get("risk_tiers", {})
    class_names = metrics.get("risk_tier_classification", {}).get("class_names", [])
    class_id = int(class_id)
    if str(class_id) in risk_tiers:
        return risk_tiers[str(class_id)]
    if 0 <= class_id < len(class_names):
        return class_names[class_id]
    return DEFAULT_RISK_CLASS_NAMES.get(class_id, f"CLASS_{class_id}")


def _encode_with_saved_mapping(series, mapping):
    return series.astype(str).map(lambda value: mapping.get(value, -1)).astype(int)


def _prepare_features(records, feature_columns, label_encoders, metrics):
    df = records.copy()

    for col in [
        "origin_country",
        "us_port",
        "pest_status",
        "detection_type",
        "month",
        "year",
        "passengers",
        "fruit_imports",
        "detections",
        "temp_suitability_score",
        "destination_temp_f",
    ]:
        if col not in df.columns:
            df[col] = None

    df["origin_country"] = df["origin_country"].astype(str).str.strip()
    df["us_port"] = df["us_port"].astype(str).str.strip().replace(PORT_MAP)
    df["pest_status"] = df["pest_status"].astype(str).str.strip()
    df["detection_type"] = df["detection_type"].astype(str).str.strip()

    month_numeric = pd.to_numeric(df["month"], errors="coerce")
    parsed_month = pd.to_datetime(df["month"], errors="coerce")
    df["month_num"] = month_numeric.where(month_numeric.between(1, 12), parsed_month.dt.month)
    df["month_num"] = df["month_num"].fillna(1).astype(int)
    df["quarter"] = ((df["month_num"] - 1) // 3 + 1).astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month_num"] / 12.0)
    df["month_cos"] = np.cos(2 * np.pi * df["month_num"] / 12.0)

    domain = metrics.get("domain_knowledge", {})
    threshold = float(domain.get("temp_threshold_f", domain.get("temp_dead_zone_f", 40)))
    peak_months = set(domain.get("peak_season_months", domain.get("peak_months", [6, 7, 8, 9])))

    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(2026)
    df["passengers"] = pd.to_numeric(df["passengers"], errors="coerce").fillna(0)
    df["fruit_imports"] = pd.to_numeric(df["fruit_imports"], errors="coerce").fillna(0)
    df["detections"] = pd.to_numeric(df["detections"], errors="coerce").fillna(0)
    df["temp_suitability_score"] = pd.to_numeric(
        df["temp_suitability_score"], errors="coerce"
    )
    df["destination_temp_f"] = pd.to_numeric(df["destination_temp_f"], errors="coerce").fillna(60)

    df["pest_score"] = (
        df["pest_status"].astype(str).str.lower().map(PEST_SCORE_MAP).fillna(0).astype(int)
    )
    df["temp_viable"] = (df["destination_temp_f"] >= threshold).astype(int)
    df["below_temp_threshold"] = (df["destination_temp_f"] < threshold).astype(int)
    df["temp_margin_f"] = df["destination_temp_f"] - threshold
    df["peak_season"] = df["month_num"].isin(peak_months).astype(int)
    df["has_detection"] = (df["detections"] > 0).astype(int)
    df["has_imports"] = (df["fruit_imports"] > 0).astype(int)

    temp = df["destination_temp_f"]
    df["temp_risk_score"] = np.select(
        [temp < threshold, temp < 45, temp <= 75],
        [0.0, 0.35, 1.0],
        default=0.9,
    )
    supplied_score = pd.to_numeric(df["temp_suitability_score"], errors="coerce")
    if supplied_score.notna().any():
        df["temp_risk_score"] = supplied_score.fillna(df["temp_risk_score"])
    df["temp_suitability_score"] = supplied_score.fillna(df["temp_risk_score"])
    df["temp_zone_enc"] = np.select(
        [temp < threshold, temp < 45, temp < 60, temp < 75],
        [0, 1, 2, 3],
        default=4,
    ).astype(int)

    df["log_passengers"] = np.log1p(df["passengers"])
    df["log_imports"] = np.log1p(df["fruit_imports"])
    df["log_detections"] = np.log1p(df["detections"])
    df["import_density"] = df["fruit_imports"] / (df["passengers"] + 1)
    df["import_x_temp"] = df["fruit_imports"] * df["temp_suitability_score"]
    df["passenger_x_pest"] = df["passengers"] * df["pest_score"]
    df["passenger_x_temp"] = df["passengers"] * df["temp_suitability_score"]
    df["imports_x_peak"] = df["fruit_imports"] * df["peak_season"]
    df["temp_x_peak"] = df["temp_suitability_score"] * df["peak_season"]
    df["import_per_passenger"] = df["fruit_imports"] / (df["passengers"] + 1)
    df["detection_rate"] = df["detections"] / (df["fruit_imports"] + 1)
    df["pathway_risk"] = df["pest_score"] * df["temp_risk_score"]
    df["trade_pathway_risk"] = df["fruit_imports"] * df["temp_risk_score"]
    df["pax_pathway_risk"] = df["passengers"] * df["pest_score"] * df["temp_risk_score"]
    df["seasonal_temp_risk"] = df["peak_season"] * df["temp_risk_score"]

    df = df.replace([np.inf, -np.inf], 0)

    for col, mapping in label_encoders.items():
        df[f"{col}_enc"] = _encode_with_saved_mapping(df[col], mapping)

    encoded = df.copy()
    for col in feature_columns:
        if col not in encoded.columns:
            encoded[col] = 0

    return encoded[feature_columns].apply(pd.to_numeric, errors="coerce").fillna(0), df


def _first_model(bundle, keys):
    for key in keys:
        if key in bundle:
            return bundle[key]
    return None


def _predict_probability(model, X):
    if model is None:
        return None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1] if proba.ndim == 2 and proba.shape[1] > 1 else proba.ravel()
    pred = model.predict(X)
    return np.asarray(pred, dtype=float).ravel()


def _predict_class(model, X):
    if model is None:
        return None, None
    pred = np.asarray(model.predict(X)).ravel()
    confidence = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if getattr(proba, "ndim", 1) == 2:
            confidence = np.max(proba, axis=1)
    return pred, confidence


def predict_fn(input_data, model_bundle):
    feature_columns = model_bundle.get("feature_columns", [])
    label_encoders = model_bundle.get("label_encoders", {})
    metrics = dict(model_bundle.get("metrics", {}))
    if model_bundle.get("domain_knowledge"):
        metrics["domain_knowledge"] = {
            **metrics.get("domain_knowledge", {}),
            **model_bundle["domain_knowledge"],
        }
    X, engineered = _prepare_features(input_data, feature_columns, label_encoders, metrics)

    regression_model = _first_model(
        model_bundle,
        ["regression_model", "regressor", "count_model", "detection_regressor"],
    )
    risk_model = _first_model(
        model_bundle,
        ["risk_tier_model", "risk_classifier", "tier_classifier", "classification_model"],
    )
    binary_model = _first_model(
        model_bundle,
        ["binary_model", "binary_classifier", "temp_masked_model", "model"],
    )

    predicted_counts = None if regression_model is None else np.asarray(regression_model.predict(X)).ravel()
    binary_probabilities = _predict_probability(binary_model, X)
    risk_classes, risk_confidence = _predict_class(risk_model, X)

    results = []
    for index in range(len(X)):
        temp_viable = bool(engineered.iloc[index]["temp_viable"])
        predicted_count = None if predicted_counts is None else max(0.0, float(predicted_counts[index]))
        probability = None if binary_probabilities is None else float(binary_probabilities[index])
        if probability is not None and not temp_viable:
            probability = 0.0

        risk_class_id = None if risk_classes is None else int(risk_classes[index])
        count_tier = _risk_tier_from_count(predicted_count or 0.0)
        class_tier = None if risk_class_id is None else _risk_class_name(risk_class_id, metrics)
        probability_tier = None if probability is None else _risk_tier_from_probability(probability)
        if not temp_viable:
            class_tier = "Below 40F filtered out"
            count_tier = "Below 40F filtered out"
            probability_tier = "Below 40F filtered out"

        result = {
            "index": int(index),
            "model_source": "sagemaker-endpoint",
            "temp_viable": temp_viable,
            "temperature_filter": "below_40F_filtered_out" if not temp_viable else "actionable_temperature",
            "risk_tier": class_tier or count_tier or probability_tier,
            "count_risk_tier": count_tier,
        }
        if predicted_count is not None:
            result["predicted_detections"] = round(predicted_count, 4)
        if probability is not None:
            result["future_detection_probability"] = round(probability, 6)
            result["prediction"] = int(probability >= 0.5)
            result["probability_risk_tier"] = probability_tier
        if risk_class_id is not None:
            result["risk_class_id"] = risk_class_id
            result["risk_class_name"] = class_tier
        if risk_confidence is not None:
            result["risk_class_confidence"] = round(float(risk_confidence[index]), 6)

        results.append(result)

    return {"predictions": results}


def output_fn(prediction, response_content_type):
    return json.dumps(prediction), "application/json"
