"""AI narrative generation: Gemini (multi-model cascade) → Groq → rule-based fallback."""
import os
import streamlit as st

# ── Model priority lists ───────────────────────────────────────────────────────
GEMINI_MODELS = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.0-flash",
    "gemini-2.0-flash-lite",
    "gemini-1.5-flash",
]

GROQ_MODELS = [
    "llama-3.3-70b-versatile",
    "llama-3.1-70b-versatile",
    "mixtral-8x7b-32768",
]

# ── Alternate route knowledge base (Bengaluru corridors) ──────────────────────
ALTERNATE_ROUTES = {
    "Mysore Road": [
        "Kanakapura Road via NICE Road interchange",
        "Magadi Road → Chord Road → destination",
        "Outer Ring Road (ORR) South via Banashankari",
    ],
    "Bellary Road 1": [
        "Hebbal flyover → Thanisandra Road",
        "ORR North → Nagavara junction",
        "Outer Ring Road via Kogilu cross",
    ],
    "Bellary Road 2": [
        "Yelahanka → NH 44 bypass",
        "Bagalur Road via Jakkur",
        "ORR North → Hebbal connector",
    ],
    "Tumkur Road": [
        "Peenya Industrial Area inner roads",
        "Yeshwanthapura → Chord Road",
        "ORR North via Jalahalli cross",
    ],
    "Hosur Road": [
        "Sarjapur Road → Outer Ring Road East",
        "Bannerghatta Road via JP Nagar",
        "NICE Road → Electronic City Phase 2",
    ],
    "ORR North 1": [
        "Bellary Road → Hebbal flyover",
        "Thanisandra Main Road as parallel route",
        "Nagawara → Kammanahalli connector",
    ],
    "Old Madras Road": [
        "Whitefield Road via Hoodi junction",
        "Outer Ring Road East → KR Puram bridge",
        "ITPL Road → Varthur Road",
    ],
    "Magadi Road": [
        "Mysore Road via NICE Road",
        "Chord Road → Rajajinagar",
        "Tumkur Road via Yeshwanthapura",
    ],
    "ORR East 1": [
        "Sarjapur Road as parallel corridor",
        "Varthur Road → Whitefield",
        "KR Puram → Old Madras Road connector",
    ],
    "Bannerghatta Road": [
        "Kanakapura Road via JP Nagar",
        "NICE Road South interchange",
        "Uttarahalli Road → Mysore Road",
    ],
    "Non-corridor": [
        "Use nearest arterial road bypass",
        "Coordinate with local traffic unit for diversion",
    ],
}

# ── Staging & deployment knowledge ────────────────────────────────────────────
STAGING_POINTS = {
    "Mysore Road":       "Stage at Nayandahalli junction and Kengeri satellite town entry",
    "Bellary Road 1":    "Stage at Hebbal flyover base and Mekhri Circle",
    "Bellary Road 2":    "Stage at Yelahanka New Town junction and Air Force Station gate",
    "Tumkur Road":       "Stage at Yeshwanthapura junction and Peenya 2nd Stage",
    "Hosur Road":        "Stage at Silk Board junction and Electronic City toll",
    "ORR North 1":       "Stage at Hebbal ORR junction and Nagavara signal",
    "Old Madras Road":   "Stage at KR Puram bridge and Hoodi junction",
    "Magadi Road":       "Stage at Goraguntepalya junction and Vijayanagar circle",
    "ORR East 1":        "Stage at KR Puram ORR and Marathahalli bridge",
    "Bannerghatta Road": "Stage at JP Nagar 7th Phase and Gottigere junction",
    "Non-corridor":      "Stage at nearest junction; confirm with control room",
}


def _key(name: str) -> str:
    try:
        return st.secrets.get(name, "") or os.environ.get(name, "")
    except Exception:
        return os.environ.get(name, "")


def _gemini(prompt: str) -> str | None:
    api_key = _key("GEMINI_API_KEY")
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
    except Exception:
        return None

    for model_name in GEMINI_MODELS:
        try:
            model  = genai.GenerativeModel(model_name)
            result = model.generate_content(prompt).text.strip()
            if result:
                return result
        except Exception:
            continue
    return None


def _groq(prompt: str) -> str | None:
    api_key = _key("GROQ_API_KEY")
    if not api_key:
        return None
    try:
        from groq import Groq
        client = Groq(api_key=api_key)
    except Exception:
        return None

    for model_name in GROQ_MODELS:
        try:
            resp   = client.chat.completions.create(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=700,
                temperature=0.3,
            )
            result = resp.choices[0].message.content.strip()
            if result:
                return result
        except Exception:
            continue
    return None


def _llm(prompt: str) -> str | None:
    return _gemini(prompt) or _groq(prompt)


# ── Prompt builder ─────────────────────────────────────────────────────────────

def _build_prompt(pred: dict, event: dict, weather: dict | None) -> str:
    rec        = pred.get("recommendation", {})
    cause      = event.get("event_cause", "").replace("_", " ").title()
    corridor   = event.get("corridor", "Non-corridor")
    hour       = event.get("hour", 0)
    e_type     = event.get("event_type", "unplanned").title()
    severity   = pred.get("severity", "Unknown")
    confidence = pred.get("severity_confidence", 0) * 100
    duration   = pred.get("estimated_duration_minutes", 0)
    closure_p  = pred.get("closure_probability", 0) * 100
    alert      = pred.get("alert_level", "GREEN")
    constables = rec.get("constable_count", 0)
    barricades = rec.get("barricade_count", 0)
    diversion  = rec.get("diversion_needed", False)
    station    = rec.get("suggested_police_station", "nearest station")

    alt_routes   = ALTERNATE_ROUTES.get(corridor, ALTERNATE_ROUTES["Non-corridor"])
    staging      = STAGING_POINTS.get(corridor, STAGING_POINTS["Non-corridor"])
    alt_str      = "\n".join(f"  • {r}" for r in alt_routes)
    weather_str  = ""
    weather_warn = ""
    if weather:
        vis   = weather.get("visibility", 10)
        cond  = weather.get("condition", "Clear")
        temp  = weather.get("temp", 25)
        wind  = weather.get("wind_speed", 0)
        weather_str = f"{cond}, {temp:.0f}°C, visibility {vis:.1f} km, wind {wind:.1f} m/s"
        if vis < 3 or "fog" in cond.lower() or "mist" in cond.lower():
            weather_warn = "CRITICAL: Low visibility conditions — mandate additional lighting and reflective gear for deployed personnel."
        elif "rain" in cond.lower() or "drizzle" in cond.lower():
            weather_warn = "NOTE: Wet road conditions — factor in extended braking distances and higher secondary incident risk."

    time_context = ""
    if 7 <= hour <= 10:
        time_context = "This is morning peak hour — expect high commuter and office-going traffic compounding the incident impact."
    elif 17 <= hour <= 21:
        time_context = "This is evening peak hour — congestion will build rapidly; act within the first 10 minutes to prevent gridlock."
    elif 0 <= hour <= 5:
        time_context = "Late night / early morning — lower traffic volume but higher speeds; risk of secondary accidents is elevated."
    else:
        time_context = "Off-peak hour — traffic volume is moderate; standard deployment should suffice if actioned promptly."

    return f"""You are an experienced Bengaluru Traffic Police operations commander generating a real-time actionable field brief.

INCIDENT DATA:
- Cause: {cause}
- Corridor: {corridor}
- Time: {hour:02d}:00 hrs ({e_type} event)
- Alert Level: {alert}
- Predicted Severity: {severity} (ML confidence: {confidence:.0f}%)
- Estimated Clearance Time: {duration:.0f} minutes
- Road Closure Probability: {closure_p:.0f}%
- Resources Required: {constables} constables, {barricades} barricades
- Nearest Station: {station}
- Diversion Required: {"YES" if diversion else "NO"}
- Live Weather: {weather_str if weather_str else "Not available"}
- Time Context: {time_context}
{f"- Weather Advisory: {weather_warn}" if weather_warn else ""}

Known alternate routes for {corridor}:
{alt_str}

Staging recommendation: {staging}

Generate a detailed, structured field operations brief that a traffic police officer can immediately act on. Cover the following in order:

1. SITUATION SUMMARY — What is happening, how serious it is, and why it matters right now given the time and weather.

2. IMMEDIATE ACTIONS (first 5 minutes) — Exactly what the first responding officer should do the moment they arrive. Be specific: where to stand, what to signal, who to call.

3. DEPLOYMENT PLAN — How to position the {constables} constables and {barricades} barricades. Specify which points to cover first and why.

4. TRAFFIC DIVERSION — {"Since diversion is required, specify which of the alternate routes to activate, in what order, and how to signal motorists toward them." if diversion else "Diversion is not currently required, but specify the trigger condition (queue length or closure probability threshold) at which diversion should be activated and which route to open first."}

5. ESCALATION TRIGGERS — What specific on-ground signs should prompt the officer to request additional resources or escalate to a senior officer.

6. ESTIMATED RESOLUTION — Realistic timeline given the severity, weather, and time of day.

Use plain, direct language. Avoid jargon. Write as if briefing a constable over radio who must act immediately. No bullet points within sections — write each section as 2–3 clear sentences. Use the section headers exactly as listed above."""


# ── Public API ─────────────────────────────────────────────────────────────────

def generate_situation_report(pred: dict, event: dict, weather: dict | None = None) -> str:
    prompt = _build_prompt(pred, event, weather)
    result = _llm(prompt)
    return result if result else _rule_based_report(pred, event)


def generate_scenario_brief(scenario: dict) -> str | None:
    prompt = f"""You are a senior Bengaluru traffic police planning officer.
Write a concise pre-event DEPLOYMENT BRIEF (5–6 sentences) for:

- Event type: {scenario.get('event_type','').replace('_',' ').title()}
- Location: {scenario.get('corridor','')} corridor
- Date/Time: {scenario.get('date','')} starting {scenario.get('start_hour',0):02d}:00 hrs
- Expected duration: {scenario.get('duration_hours',0)} hours
- Peak risk window: {scenario.get('peak_hour',0):02d}:00 – {(scenario.get('peak_hour',0)+1)%24:02d}:00 hrs
- Max resources needed: {scenario.get('max_constables',0)} constables, {scenario.get('max_barricades',0)} barricades
- Diversion needed: {'Yes' if scenario.get('diversion_needed') else 'No'}
- Historical context: {scenario.get('similar_events','N/A')}

Include: staging time, barricade placement, diversion corridor suggestions, escalation triggers.
Professional, concise, police planning style."""

    return _llm(prompt)


def _rule_based_report(pred: dict, event: dict) -> str:
    severity   = pred.get("severity", "Unknown")
    duration   = pred.get("estimated_duration_minutes", 0)
    closure_p  = pred.get("closure_probability", 0)
    confidence = pred.get("severity_confidence", 0)
    rec        = pred.get("recommendation", {})
    cause      = event.get("event_cause", "").replace("_", " ").title()
    corridor   = event.get("corridor", "Non-corridor")
    hour       = event.get("hour", 0)
    diversion  = rec.get("diversion_needed", False)
    alt_routes = ALTERNATE_ROUTES.get(corridor, ALTERNATE_ROUTES["Non-corridor"])
    staging    = STAGING_POINTS.get(corridor, STAGING_POINTS["Non-corridor"])

    diversion_text = ""
    if diversion:
        diversion_text = (
            f" Diversion is required — activate {alt_routes[0]} as primary alternate"
            f" and {alt_routes[1]} as secondary if primary saturates."
        )
    else:
        diversion_text = (
            f" Diversion not yet required; monitor queue length and activate"
            f" {alt_routes[0]} if closure probability crosses 60%."
        )

    closure_note = (
        "Road closure is likely — deploy barricades at corridor entry points immediately and halt inbound traffic."
        if closure_p > 0.5
        else "Road closure unlikely at this stage; maintain single-lane flow and monitor."
    )

    return (
        f"**SITUATION SUMMARY:** {cause} reported on {corridor} at {hour:02d}:00 hrs. "
        f"ML predicts **{severity}** severity with {confidence*100:.0f}% confidence; estimated clearance **{duration:.0f} minutes**. "
        f"{closure_note}\n\n"
        f"**IMMEDIATE ACTIONS:** {staging}. "
        f"Deploy {rec.get('constable_count', 0)} constables to manage flow and {rec.get('barricade_count', 0)} barricades at entry points. "
        f"Contact {rec.get('suggested_police_station', 'nearest station')} for backup and notify control room.\n\n"
        f"**TRAFFIC DIVERSION:**{diversion_text}\n\n"
        f"**ESCALATION:** Request additional units if clearance exceeds {duration + 15:.0f} minutes or a secondary incident occurs."
    )