import os
import requests
import streamlit as st

BENGALURU = (12.9716, 77.5946)

CORRIDOR_COORDS = {
    "Mysore Road":          (12.9358, 77.5264),
    "Bellary Road 1":       (13.0358, 77.5800),
    "Tumkur Road":          (13.0154, 77.5107),
    "Hosur Road":           (12.8893, 77.6387),
    "ORR North 1":          (13.0604, 77.6218),
    "Old Madras Road":      (13.0032, 77.6540),
    "Magadi Road":          (12.9683, 77.5025),
    "Bellary Road 2":       (13.0558, 77.5900),
    "ORR East 1":           (12.9450, 77.6800),
    "Bannerghatta Road":    (12.8735, 77.5985),
    "Non-corridor":         BENGALURU,
}


def _get_api_key() -> str:
    try:
        return st.secrets.get("OPENWEATHER_API_KEY", "") or os.environ.get("OPENWEATHER_API_KEY", "")
    except Exception:
        return os.environ.get("OPENWEATHER_API_KEY", "")


@st.cache_data(ttl=600)
def fetch_weather(lat: float = BENGALURU[0], lon: float = BENGALURU[1]) -> dict | None:
    api_key = _get_api_key()
    if not api_key:
        return None
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={api_key}&units=metric"
        )
        r = requests.get(url, timeout=6)
        if r.status_code == 200:
            d = r.json()
            return {
                "condition":   d["weather"][0]["main"],
                "description": d["weather"][0]["description"],
                "temp":        d["main"]["temp"],
                "humidity":    d["main"]["humidity"],
                "wind_speed":  d["wind"]["speed"],
                "visibility":  d.get("visibility", 10000) / 1000,
            }
    except Exception:
        pass
    return None


def weather_for_corridor(corridor: str) -> dict | None:
    coords = CORRIDOR_COORDS.get(corridor, BENGALURU)
    return fetch_weather(lat=coords[0], lon=coords[1])


def duration_multiplier(weather: dict | None) -> tuple[float, str | None]:
    """Return (multiplier, warning_text) based on weather conditions."""
    if not weather:
        return 1.0, None
    cond = weather.get("condition", "").lower()
    vis  = weather.get("visibility", 10)
    if "thunderstorm" in cond:
        return 1.55, "⛈ Thunderstorm — clearance time +55%; risk of secondary incidents."
    if "snow" in cond:
        return 1.60, "❄️ Snow/ice — extreme delays expected; +60% clearance time."
    if "fog" in cond or vis < 2:
        return 1.45, "🌫 Dense fog — visibility <2 km; risk of secondary incidents high."
    if "rain" in cond or "drizzle" in cond:
        if weather.get("humidity", 0) > 85:
            return 1.35, "🌧 Heavy rain — clearance time +35%."
        return 1.20, "🌦 Rainy conditions — clearance time +20%."
    if "haze" in cond or vis < 5:
        return 1.10, "🌫 Hazy — reduced visibility; slight delay expected."
    return 1.0, None
