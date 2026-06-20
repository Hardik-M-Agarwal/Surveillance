from fastapi import APIRouter
from pathlib import Path
import pandas as pd
import numpy as np

router = APIRouter()

CORRIDOR_COORDS = {
    "Mysore Road":         (12.9358, 77.5264),
    "Bellary Road 1":      (13.0358, 77.5800),
    "Tumkur Road":         (13.0154, 77.5107),
    "Hosur Road":          (12.8893, 77.6387),
    "ORR North 1":         (13.0604, 77.6218),
    "Old Madras Road":     (13.0032, 77.6540),
    "Magadi Road":         (12.9683, 77.5025),
    "Bellary Road 2":      (13.0558, 77.5900),
    "ORR East 1":          (12.9450, 77.6800),
    "Bannerghatta Road":   (12.8735, 77.5985),
    "Kanakapura Road":     (12.9068, 77.5736),
    "Sarjapur Road":       (12.9239, 77.6384),
    "MG Road":             (12.9738, 77.6119),
    "Sankey Road":         (13.0022, 77.5815),
    "Outer Ring Road West":(12.9734, 77.5126),
    "Residency Road":      (12.9732, 77.6044),
}

_risk_cache: dict | None = None


def _compute_risk_scores() -> dict[str, float]:
    global _risk_cache
    if _risk_cache is not None:
        return _risk_cache

    csv_path = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "cleaned.csv"
    if not csv_path.exists():
        return {name: 50.0 for name in CORRIDOR_COORDS}

    try:
        df = pd.read_csv(csv_path, low_memory=False,
                         usecols=lambda c: c in ["corridor","severity","requires_road_closure"])
        df["requires_road_closure"] = (
            df["requires_road_closure"].astype(str).str.lower()
            .map({"true": 1, "1": 1, "false": 0, "0": 0}).fillna(0)
        )
        df["sev_score"] = df["severity"].map(
            {"Critical": 4, "High": 3, "Medium": 2, "Low": 1}
        ).fillna(2)

        stats = (
            df[df["corridor"].isin(CORRIDOR_COORDS)]
            .groupby("corridor")
            .agg(total=("sev_score","count"),
                 closure=("requires_road_closure","mean"),
                 high_crit=("severity", lambda x: ((x=="High")|(x=="Critical")).sum()))
            .reset_index()
        )
        if stats.empty:
            return {name: 50.0 for name in CORRIDOR_COORDS}

        max_total = stats["total"].max() or 1
        stats["risk"] = (
            0.40 * (stats["total"] / max_total)
            + 0.35 * stats["closure"]
            + 0.25 * (stats["high_crit"] / stats["total"].clip(lower=1))
        ) * 100
        stats["risk"] = stats["risk"].clip(0, 100)

        scores = dict(zip(stats["corridor"], stats["risk"].round(1)))
        for name in CORRIDOR_COORDS:
            if name not in scores:
                scores[name] = 40.0

        _risk_cache = scores
        return scores
    except Exception:
        return {name: 50.0 for name in CORRIDOR_COORDS}


@router.get("/heatmap")
async def get_heatmap():
    risk_scores = _compute_risk_scores()

    corridors = []
    for name, coords in CORRIDOR_COORDS.items():
        corridors.append({
            "name":       name,
            "latitude":   coords[0],
            "longitude":  coords[1],
            "risk_score": risk_scores.get(name, 50.0),
        })

    return {
        "corridors": corridors,
        "hotspots": [
            {"name": "Mekri Circle",         "latitude": 13.0134, "longitude": 77.5821, "count": 64},
            {"name": "Ayyappa Temple Junc",   "latitude": 13.0298, "longitude": 77.5459, "count": 49},
            {"name": "Satellite Bus Stand",   "latitude": 12.9540, "longitude": 77.5381, "count": 43},
            {"name": "Yeshwanthpura Circle",  "latitude": 13.0260, "longitude": 77.5501, "count": 38},
            {"name": "Yelahanka Circle",      "latitude": 13.1008, "longitude": 77.5963, "count": 34},
            {"name": "Silk Board Junc",       "latitude": 12.9177, "longitude": 77.6238, "count": 33},
        ],
    }
