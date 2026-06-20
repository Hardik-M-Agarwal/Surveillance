from fastapi import APIRouter, Request
from api.schemas import RecommendInput

router = APIRouter()

@router.post("/recommend")
async def recommend(request: Request, rec_input: RecommendInput):
    app = request.app
    rec = app.state.recommender.recommend(
        event_cause=rec_input.event_cause,
        predicted_severity=rec_input.severity,
        corridor=rec_input.corridor,
        hour=rec_input.hour,
        requires_closure_prob=rec_input.closure_probability,
        estimated_duration=rec_input.estimated_duration
    )
    return rec
