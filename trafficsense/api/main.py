from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routes import predict, recommend, heatmap
from src.models.severity_model import SeverityClassifier
from src.models.duration_model import DurationRegressor
from src.models.closure_model import ClosureClassifier
from src.recommender import ResourceRecommender
from src.explainer import SHAPExplainer

app = FastAPI(title="TrafficSense API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def load_models():
    print("Loading models into memory...")
    try:
        app.state.severity_model = SeverityClassifier()
        app.state.severity_model.load()
        
        app.state.duration_model = DurationRegressor()
        app.state.duration_model.load()
        
        app.state.closure_model = ClosureClassifier()
        app.state.closure_model.load()
        
        app.state.recommender = ResourceRecommender()
        app.state.recommender.load_lookup_table()
        
        app.state.explainer = SHAPExplainer()
        print("Models loaded successfully.")
    except Exception as e:
        print(f"Warning: Models not loaded. Please run run_pipeline.py first. Error: {e}")

app.include_router(predict.router)
app.include_router(recommend.router)
app.include_router(heatmap.router)

@app.get("/health")
async def health_check():
    return {"status": "ok"}
