from fastapi import APIRouter, Request
from datetime import datetime
import pandas as pd
import numpy as np
from api.schemas import EventInput, PredictionResponse
from src.feature_engineering import FeatureEngineer

router = APIRouter()

@router.post("/predict", response_model=PredictionResponse)
async def predict(request: Request, event: EventInput):
    app = request.app
    
    # 1. Prepare input dataframe
    input_dict = event.dict()
    # Add dummy/default fields needed for feature engineering
    input_dict['start_datetime'] = pd.Timestamp.utcnow() # not strictly needed if we just mock the temporal feats
    
    df = pd.DataFrame([input_dict])
    
    # Mock some fields for FeatureEngineer if they are needed and missing
    # Actually FeatureEngineer uses hour, day_of_week, etc. directly.
    # Let's run a simplified feature extraction for inference.
    fe = FeatureEngineer()
    
    # Temporal
    df['hour'] = event.hour
    df['day_of_week'] = event.day_of_week
    df['is_weekend'] = 1 if event.day_of_week >= 5 else 0
    df['month'] = datetime.now().month
    if event.is_peak_hour is not None:
        df['is_peak_hour'] = event.is_peak_hour
    else:
        df['is_peak_hour'] = 1 if event.hour in [19, 20, 21, 22, 4, 5, 6] else 0
        
    # Categoricals target encode
    df = fe.encode_categoricals(df, is_train=False)
    
    # Distance / Hotspot
    df['junction'] = 'unknown' # Assume unknown unless provided
    df['displacement_km'] = 0.0 # Unknown at start
    df['is_junction_hotspot'] = 0
    
    # Historical density & saturation - since we don't have access to the whole active DB here easily,
    # we will use the median/mean from the lookup or use a default
    # A real system would query the active DB.
    df['historical_density_30d'] = 10.0 # Default fallback
    df['corridor_active_count'] = 1     # Default fallback
    df['station_cause_avg_duration'] = 60.0 # Default fallback
    
    # Get final columns
    with open('models/feature_columns.json', 'r') as f:
        import json
        feature_columns = json.load(f)
        
    # The models need specific features. 
    # Severity & Closure need all EXCEPT requires_road_closure
    features_sc = [c for c in feature_columns if c != 'requires_road_closure']
    X_sc = df[features_sc]
    
    # Run Closure model
    closure_prob = float(app.state.closure_model.predict_proba(X_sc)[0])
    
    # Run Severity model
    # Wait, the feature engineer doesn't know requires_road_closure at inference, so for the duration model it needs it!
    # But severity doesn't need it.
    severity_pred = app.state.severity_model.predict(X_sc)[0]
    
    # Compute confidence for severity (proxy from XGBoost probability)
    severity_probs = app.state.severity_model.model.predict_proba(X_sc)[0]
    severity_confidence = float(np.max(severity_probs))
    
    # Run Duration model
    # Duration needs all features including requires_road_closure
    df['requires_road_closure'] = 1 if closure_prob > 0.5 else 0
    X_dur = df[feature_columns]
    duration_pred = float(app.state.duration_model.predict(X_dur)[0])
    
    # Run Recommender
    rec = app.state.recommender.recommend(
        event_cause=event.event_cause,
        predicted_severity=severity_pred,
        corridor=event.corridor,
        hour=event.hour,
        requires_closure_prob=closure_prob,
        estimated_duration=duration_pred
    )
    
    # Run Explainer
    explanation = app.state.explainer.explain_severity(
        app.state.severity_model.model, X_sc, features_sc
    )
    text_exp = app.state.explainer.generate_explanation_text(explanation, severity_pred)
    # Add text to explanation list or just return the dict list
    # The schema wants List[dict], we can just append the text as a special dict
    explanation.append({'text': text_exp})
    
    return PredictionResponse(
        severity=severity_pred,
        severity_confidence=severity_confidence,
        closure_probability=closure_prob,
        estimated_duration_minutes=duration_pred,
        alert_level=rec['alert_level'],
        explanation=explanation,
        recommendation=rec,
        timestamp=datetime.utcnow().isoformat()
    )
