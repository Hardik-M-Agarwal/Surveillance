from fastapi import APIRouter
import random

router = APIRouter()

CORRIDOR_COORDS = {
    'Mysore Road': (12.9358, 77.5264),
    'Bellary Road 1': (13.0358, 77.5800),
    'Tumkur Road': (13.0154, 77.5107),
    'Hosur Road': (12.8893, 77.6387),
    'ORR North 1': (13.0604, 77.6218),
    'Old Madras Road': (13.0032, 77.6540),
    'Magadi Road': (12.9683, 77.5025),
    'Bellary Road 2': (13.0558, 77.5900),
    'ORR East 1': (12.9450, 77.6800),
    'Bannerghatta Road': (12.8735, 77.5985),
    'Kanakapura Road': (12.9068, 77.5736),
    'Sarjapur Road': (12.9239, 77.6384),
    'Sankey Road': (13.0022, 77.5815),
    'C V Raman Road': (13.0210, 77.5794),
    'Outer Ring Road West': (12.9734, 77.5126),
    'Residency Road': (12.9732, 77.6044),
    'Richmond Road': (12.9634, 77.6015),
    'JC Road': (12.9568, 77.5828),
    'Kasturba Road': (12.9730, 77.5947),
    'MG Road': (12.9738, 77.6119),
    'Brigade Road': (12.9723, 77.6067),
    'Commercial Street': (12.9822, 77.6083),
    'Infantry Road': (12.9818, 77.6006),
    'Cunningham Road': (12.9863, 77.5960),
    'St Marks Road': (12.9719, 77.5997),
    'Lavelle Road': (12.9706, 77.5967),
    'Dickenson Road': (12.9834, 77.6186),
    'Ulsoor Road': (12.9790, 77.6200),
}

@router.get("/heatmap")
async def get_heatmap():
    # Return simulated risk scores for demonstration
    # In production, query the active incidents from DB
    
    corridors = []
    for name, coords in CORRIDOR_COORDS.items():
        # simulate risk
        risk = random.randint(10, 95)
        corridors.append({
            "name": name,
            "latitude": coords[0],
            "longitude": coords[1],
            "risk_score": risk
        })
        
    return {
        "corridors": corridors,
        "hotspots": [
            {"name": "MekhriCircle", "latitude": 13.0134, "longitude": 77.5821, "count": 64},
            {"name": "AyyappaTempleJunc", "latitude": 13.0298, "longitude": 77.5459, "count": 49},
            {"name": "SatteliteBusStandJunc", "latitude": 12.9540, "longitude": 77.5381, "count": 43},
            {"name": "YeshwanthpuraCircle", "latitude": 13.0260, "longitude": 77.5501, "count": 38},
            {"name": "YelhankaCircle", "latitude": 13.1008, "longitude": 77.5963, "count": 34},
            {"name": "SilkBoardJunc", "latitude": 12.9177, "longitude": 77.6238, "count": 33}
        ]
    }
