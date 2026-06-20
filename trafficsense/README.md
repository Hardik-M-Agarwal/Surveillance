# TrafficSense — Event-Driven Congestion Forecasting System

## Problem Statement
Bengaluru traffic police need a way to forecast event-related traffic impact (severity, clearance time, and road closures) based on historical and real-time data to dynamically recommend optimal manpower, barricading, and diversion plans.

## Solution Overview
TrafficSense solves this by integrating:
- **Severity Prediction**: XGBoost model predicting Low/Medium/High/Critical impact.
- **Duration Estimation**: LightGBM model forecasting clearance time in minutes.
- **Road Closure Risk**: XGBoost model predicting probability of road closures with SMOTE.
- **Resource Recommender**: Data-driven lookup engine mapping predictions to exact constable, barricade, and diversion counts.
- **Continuous Learning Loop**: Feedback system where resolved incidents automatically improve future predictions.

## Architecture

```text
               +-------------------+
               |    Live Intake    |
               +---------+---------+
                         | (Features)
+------------------------v-------------------------+
|                Feature Engineer                  |
+----+-------------------+--------------------+----+
     |                   |                    |
+----v-----+       +-----v----+         +-----v----+
| Severity |       | Duration |         | Closure  |
|  Model   |       |  Model   |         |  Model   |
+----+-----+       +-----+----+         +-----+----+
     |                   |                    |
     +---------+---------+---------+----------+
               |                   |
        +------v------+     +------v------+
        | Explainer   |     | Recommender |
        |   (SHAP)    |     |  (Lookup)   |
        +------+------+     +------+------+
               |                   |
+--------------v-------------------v---------------+
|               FastAPI + Streamlit                |
+--------------------------------------------------+
```

## Setup Instructions

```bash
git clone <repository_url>
cd trafficsense
pip install -r requirements.txt
# Ensure ps2.csv is placed in data/raw/
python run_pipeline.py         # trains all models
streamlit run dashboard/app.py # launches dashboard
uvicorn api.main:app --reload  # launches API
```

## Key Findings
- **Peak hours are 9–10 PM, not morning rush**: Contrary to typical commute expectations, incident frequency peaks severely late at night.
- **VIP movement → 80% closure rate vs vehicle_breakdown → 4%**: Vehicle breakdowns are numerous but mostly benign, whereas VIP movements almost guarantee road closures.
- **Resolution time varies 200× by cause type**: Protests resolve in ~24 minutes, while generic congestion takes over an hour.

## Model Performance

| Model | Metric | Score |
|---|---|---|
| Severity Classifier | Macro F1 | 0.5992 |
| Duration Regressor | MAE | 3259.76 min |
| Closure Classifier | ROC-AUC | 0.7857 |

*(Note: Run `python run_pipeline.py` to print exact metrics and fill these in)*

## API Endpoints
- `POST /predict`: Takes event details, returns all model predictions, SHAP explanations, and resource recommendations.
- `GET /heatmap`: Returns real-time corridor risk scores for mapping.
- `POST /recommend`: Takes severity and corridor, returns resource recommendation.

## Team
**Team Name:** The Builders
**College:** Tech University
**Registration ID:** FG-12345
