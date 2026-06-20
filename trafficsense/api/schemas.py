from pydantic import BaseModel
from typing import Optional, List

class EventInput(BaseModel):
    event_type: str              # "planned" or "unplanned"
    event_cause: str             # one of the 17 causes
    latitude: float
    longitude: float
    corridor: str
    police_station: str
    hour: int                    # 0-23
    day_of_week: int             # 0-6
    veh_type: Optional[str] = "unknown"
    priority: str = "High"
    is_peak_hour: Optional[int] = None   # auto-computed if None

class PredictionResponse(BaseModel):
    severity: str
    severity_confidence: float
    closure_probability: float
    estimated_duration_minutes: float
    alert_level: str
    explanation: List[dict]
    recommendation: dict
    timestamp: str

class RecommendInput(BaseModel):
    event_cause: str
    severity: str
    corridor: str
    hour: int
    closure_probability: float
    estimated_duration: Optional[float] = None
