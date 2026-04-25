"""
=============================================================================
Amazon Bedrock Corn Yield Prediction Pipeline
=============================================================================
"""

import numpy as np
import pandas as pd
import boto3
import json
import warnings
warnings.filterwarnings("ignore")

# Configuration
STATES = ["Iowa", "Colorado", "Wisconsin", "Missouri", "Nebraska"]
FORECAST_DATES = ["Aug1", "Sep1", "Oct1", "Final"]
HIST_YEARS = list(range(2005, 2025))
TARGET_YEAR = 2025
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Initialize Bedrock
try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    BEDROCK_AVAILABLE = True
    print("✅ Bedrock initialized")
except Exception as e:
    print(f"⚠️ Bedrock unavailable: {e}")
    BEDROCK_AVAILABLE = False

# Generate sample data
BASE_YIELDS = {"Iowa": 170, "Colorado": 145, "Wisconsin": 155, "Missouri": 140, "Nebraska": 160}
TREND_PER_YEAR = 1.2

yield_records = []
for state in STATES:
    for i, year in enumerate(HIST_YEARS):
        yield_val = BASE_YIELDS[state] + TREND_PER_YEAR * i + np.random.normal(0, 12)
        yield_records.append({"year": year, "state": state, "yield_bu_acre": round(yield_val, 1)})

yield_df = pd.DataFrame(yield_records)

weather_records = []
for state in STATES:
    for year in HIST_YEARS + [TARGET_YEAR]:
        for fd in FORECAST_DATES:
            weather_records.append({
                "year": year, "state": state, "forecast_date": fd,
                "GDD": round(np.random.normal(1200, 150), 1),
                "Precip": round(np.random.normal(18, 4), 2),
                "PDSI": round(np.random.normal(0, 1.5), 2),
                "NDVI": round(np.random.normal(0.65, 0.08), 3)
            })

weather_df = pd.DataFrame(weather_records)

def predict_with_bedrock(state, forecast_date, weather_2025, historical_data):
    """Bedrock yield prediction"""
    if not BEDROCK_AVAILABLE:
        return fallback_prediction(state, forecast_date, historical_data)
    
    hist_yields = historical_data[historical_data['state'] == state]['yield_bu_acre'].tolist()
    
    prompt = f"""Predict 2025 corn yield for {state} at {forecast_date}.

Historical yields (Bu/Acre): {hist_yields[-5:]}
Average: {np.mean(hist_yields):.1f}

2025 conditions:
- GDD: {weather_2025['GDD']}
- Precipitation: {weather_2025['Precip']} inches
- PDSI: {weather_2025['PDSI']}
- NDVI: {weather_2025['NDVI']}

Return JSON: {{"point_estimate": X.X, "lower_bound": Y.Y, "upper_bound": Z.Z}}"""

    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 300, "temperature": 0.1}
            })
        )
        
        result = json.loads(response['body'].read())
        content = result['output']['message']['content'][0]['text']
        
        # Try to extract JSON from response
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        
        # If JSON parsing fails, extract numbers from text
        import re
        numbers = re.findall(r'\d+\.?\d*', content)
        if len(numbers) >= 3:
            return {
                "point_estimate": float(numbers[0]),
                "lower_bound": float(numbers[1]),
                "upper_bound": float(numbers[2])
            }
        
        # Final fallback
        raise ValueError("Could not parse response")
        
    except Exception as e:
        print(f"Bedrock error: {e}")
        return fallback_prediction(state, forecast_date, historical_data)

def fallback_prediction(state, forecast_date, historical_data):
    """Fallback when Bedrock unavailable"""
    recent_yields = historical_data[
        (historical_data['state'] == state) & (historical_data['year'] >= 2020)
    ]['yield_bu_acre'].values
    
    base_pred = np.mean(recent_yields) + 2.4
    uncertainty = {"Aug1": 0.15, "Sep1": 0.12, "Oct1": 0.08, "Final": 0.05}
    factor = uncertainty.get(forecast_date, 0.10)
    
    return {
        "point_estimate": round(base_pred, 1),
        "lower_bound": round(base_pred * (1 - factor), 1),
        "upper_bound": round(base_pred * (1 + factor), 1)
    }

# Generate predictions
print("Generating Bedrock predictions...")
cone_records = []

for state in STATES:
    for fd in FORECAST_DATES:
        weather_2025 = weather_df[
            (weather_df['state'] == state) & 
            (weather_df['year'] == TARGET_YEAR) & 
            (weather_df['forecast_date'] == fd)
        ].iloc[0]
        
        prediction = predict_with_bedrock(state, fd, weather_2025, yield_df)
        cone_width = prediction['upper_bound'] - prediction['lower_bound']
        
        cone_records.append({
            "state": state,
            "forecast_date": fd,
            "point_estimate": prediction['point_estimate'],
            "lower_10pct": prediction['lower_bound'],
            "upper_90pct": prediction['upper_bound'],
            "cone_width": round(cone_width, 1)
        })

cone_df = pd.DataFrame(cone_records)
cone_df["forecast_date"] = pd.Categorical(cone_df["forecast_date"], categories=FORECAST_DATES, ordered=True)
cone_df = cone_df.sort_values(["state", "forecast_date"]).reset_index(drop=True)

# Display results
print("\nBedrock Corn Yield Predictions 2025:")
print("=" * 60)
display_cols = ["state", "forecast_date", "point_estimate", "lower_10pct", "upper_90pct", "cone_width"]
print(cone_df[display_cols].to_string(index=False))

# Export
cone_df.to_csv("bedrock_corn_predictions_2025.csv", index=False)
print(f"\nResults saved to: bedrock_corn_predictions_2025.csv")

# Summary
summary = cone_df.groupby("state").agg(
    avg_yield=("point_estimate", "mean"),
    avg_uncertainty=("cone_width", "mean")
).round(1)

print(f"\nSummary by State:")
print(summary.to_string())
print("✅ Bedrock pipeline complete")"""
=============================================================================
Amazon Bedrock Corn Yield Prediction Pipeline
=============================================================================
"""

import numpy as np
import pandas as pd
import boto3
import json
import warnings
warnings.filterwarnings("ignore")

# Configuration
STATES = ["Iowa", "Colorado", "Wisconsin", "Missouri", "Nebraska"]
FORECAST_DATES = ["Aug1", "Sep1", "Oct1", "Final"]
HIST_YEARS = list(range(2005, 2025))
TARGET_YEAR = 2025
RANDOM_SEED = 42

np.random.seed(RANDOM_SEED)

# Initialize Bedrock
try:
    bedrock = boto3.client('bedrock-runtime', region_name='us-east-1')
    BEDROCK_AVAILABLE = True
    print("✅ Bedrock initialized")
except Exception as e:
    print(f"⚠️ Bedrock unavailable: {e}")
    BEDROCK_AVAILABLE = False

# Generate sample data
BASE_YIELDS = {"Iowa": 170, "Colorado": 145, "Wisconsin": 155, "Missouri": 140, "Nebraska": 160}
TREND_PER_YEAR = 1.2

yield_records = []
for state in STATES:
    for i, year in enumerate(HIST_YEARS):
        yield_val = BASE_YIELDS[state] + TREND_PER_YEAR * i + np.random.normal(0, 12)
        yield_records.append({"year": year, "state": state, "yield_bu_acre": round(yield_val, 1)})

yield_df = pd.DataFrame(yield_records)

weather_records = []
for state in STATES:
    for year in HIST_YEARS + [TARGET_YEAR]:
        for fd in FORECAST_DATES:
            weather_records.append({
                "year": year, "state": state, "forecast_date": fd,
                "GDD": round(np.random.normal(1200, 150), 1),
                "Precip": round(np.random.normal(18, 4), 2),
                "PDSI": round(np.random.normal(0, 1.5), 2),
                "NDVI": round(np.random.normal(0.65, 0.08), 3)
            })

weather_df = pd.DataFrame(weather_records)

def predict_with_bedrock(state, forecast_date, weather_2025, historical_data):
    """Bedrock yield prediction"""
    if not BEDROCK_AVAILABLE:
        return fallback_prediction(state, forecast_date, historical_data)
    
    hist_yields = historical_data[historical_data['state'] == state]['yield_bu_acre'].tolist()
    
    prompt = f"""Predict 2025 corn yield for {state} at {forecast_date}.

Historical yields (Bu/Acre): {hist_yields[-5:]}
Average: {np.mean(hist_yields):.1f}

2025 conditions:
- GDD: {weather_2025['GDD']}
- Precipitation: {weather_2025['Precip']} inches
- PDSI: {weather_2025['PDSI']}
- NDVI: {weather_2025['NDVI']}

Return JSON: {{"point_estimate": X.X, "lower_bound": Y.Y, "upper_bound": Z.Z}}"""

    try:
        response = bedrock.invoke_model(
            modelId='anthropic.claude-3-haiku-20240307-v1:0',
            body=json.dumps({
                "messages": [{"role": "user", "content": [{"text": prompt}]}],
                "inferenceConfig": {"maxTokens": 300, "temperature": 0.1}
            })
        )
        
        result = json.loads(response['body'].read())
        content = result['output']['message']['content'][0]['text']
        
        # Try to extract JSON from response
        try:
            start = content.find('{')
            end = content.rfind('}') + 1
            if start >= 0 and end > start:
                return json.loads(content[start:end])
        except:
            pass
        
        # If JSON parsing fails, extract numbers from text
        import re
        numbers = re.findall(r'\d+\.?\d*', content)
        if len(numbers) >= 3:
            return {
                "point_estimate": float(numbers[0]),
                "lower_bound": float(numbers[1]),
                "upper_bound": float(numbers[2])
            }
        
        # Final fallback
        raise ValueError("Could not parse response")
        
    except Exception as e:
        print(f"Bedrock error: {e}")
        return fallback_prediction(state, forecast_date, historical_data)

def fallback_prediction(state, forecast_date, historical_data):
    """Fallback when Bedrock unavailable"""
    recent_yields = historical_data[
        (historical_data['state'] == state) & (historical_data['year'] >= 2020)
    ]['yield_bu_acre'].values
    
    base_pred = np.mean(recent_yields) + 2.4
    uncertainty = {"Aug1": 0.15, "Sep1": 0.12, "Oct1": 0.08, "Final": 0.05}
    factor = uncertainty.get(forecast_date, 0.10)
    
    return {
        "point_estimate": round(base_pred, 1),
        "lower_bound": round(base_pred * (1 - factor), 1),
        "upper_bound": round(base_pred * (1 + factor), 1)
    }

# Generate predictions
print("Generating Bedrock predictions...")
cone_records = []

for state in STATES:
    for fd in FORECAST_DATES:
        weather_2025 = weather_df[
            (weather_df['state'] == state) & 
            (weather_df['year'] == TARGET_YEAR) & 
            (weather_df['forecast_date'] == fd)
        ].iloc[0]
        
        prediction = predict_with_bedrock(state, fd, weather_2025, yield_df)
        cone_width = prediction['upper_bound'] - prediction['lower_bound']
        
        cone_records.append({
            "state": state,
            "forecast_date": fd,
            "point_estimate": prediction['point_estimate'],
            "lower_10pct": prediction['lower_bound'],
            "upper_90pct": prediction['upper_bound'],
            "cone_width": round(cone_width, 1)
        })

cone_df = pd.DataFrame(cone_records)
cone_df["forecast_date"] = pd.Categorical(cone_df["forecast_date"], categories=FORECAST_DATES, ordered=True)
cone_df = cone_df.sort_values(["state", "forecast_date"]).reset_index(drop=True)

# Display results
print("\nBedrock Corn Yield Predictions 2025:")
print("=" * 60)
display_cols = ["state", "forecast_date", "point_estimate", "lower_10pct", "upper_90pct", "cone_width"]
print(cone_df[display_cols].to_string(index=False))

# Export
cone_df.to_csv("bedrock_corn_predictions_2025.csv", index=False)
print(f"\nResults saved to: bedrock_corn_predictions_2025.csv")

# Summary
summary = cone_df.groupby("state").agg(
    avg_yield=("point_estimate", "mean"),
    avg_uncertainty=("cone_width", "mean")
).round(1)

print(f"\nSummary by State:")
print(summary.to_string())
print("✅ Bedrock pipeline complete")