import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import requests
import pandas as pd
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium

from utils.theme import (apply_theme, sidebar_brand, alert_html, kpi_card,
                         rec_card, section_header, C, PLOTLY, SEV_COLOR,
                         AXIS_STYLE, BASE_MARGIN, page_header)
from utils.weather import weather_for_corridor, duration_multiplier
from utils.theme import weather_widget
from utils.ai_narrative import generate_situation_report

st.set_page_config(page_title="Live Prediction · TrafficSense", page_icon="⚡", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_brand()

# ── Session state ─────────────────────────────────────────────────────────────
for key, default in {
    "pred_data":    None,
    "pred_ctx":     None,
    "pred_weather": None,
    "pred_report":  None,
    "pred_error":   None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = default

# ── Constants ─────────────────────────────────────────────────────────────────
CORRIDOR_COORDS = {
    "Non-corridor":      (12.9716, 77.5946),
    "Mysore Road":       (12.9358, 77.5264),
    "Bellary Road 1":    (13.0358, 77.5800),
    "Tumkur Road":       (13.0154, 77.5107),
    "Hosur Road":        (12.8893, 77.6387),
    "ORR North 1":       (13.0604, 77.6218),
    "Old Madras Road":   (13.0032, 77.6540),
    "Magadi Road":       (12.9683, 77.5025),
    "Bellary Road 2":    (13.0558, 77.5900),
    "ORR East 1":        (12.9450, 77.6800),
    "Bannerghatta Road": (12.8735, 77.5985),
}
EVENT_CAUSES = [
    "vehicle_breakdown","accident","congestion","tree_fall","water_logging",
    "pot_holes","construction","road_conditions","public_event","procession",
    "vip_movement","protest","debris","fog_low_visibility","others",
]
CAUSE_ICONS = {
    "vehicle_breakdown":"🚗","accident":"💥","congestion":"🚦","tree_fall":"🌲",
    "water_logging":"🌊","pot_holes":"🕳️","construction":"🚧","road_conditions":"🛣️",
    "public_event":"🎉","procession":"🚶","vip_movement":"👔","protest":"✊",
    "debris":"🗑️","fog_low_visibility":"🌫️","others":"❓",
}
VEH_TYPES = ["unknown","bmtc_bus","heavy_vehicle","lcv","others","private_bus",
             "private_car","truck","ksrtc_bus","taxi","auto"]

# ── Page header ───────────────────────────────────────────────────────────────
page_header("⚡", "Live Incident Prediction",
            "Officer intake → ML severity · duration · closure · resource deployment")

# Weather banner — full width, always visible above the form
ctx_corridor = (st.session_state.pred_ctx or {}).get("corridor", "Non-corridor")
_w = st.session_state.pred_weather or weather_for_corridor(ctx_corridor)
if _w:
    weather_widget(_w)
else:
    st.info("Add OPENWEATHER_API_KEY to .streamlit/secrets.toml for live weather.")

col_form, col_result = st.columns([1, 1.6], gap="large")

# ── Left column: Intake form ──────────────────────────────────────────────────
with col_form:

    # Card wrapper — white, clean
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                padding:20px 20px 8px;box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:12px;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:16px;
                    padding-bottom:12px;border-bottom:1px solid #f3f4f6;">
            <span style="font-size:15px;">🗒️</span>
            <span style="font-size:12px;font-weight:700;color:#374151;
                         text-transform:uppercase;letter-spacing:.08em;">Officer Intake Form</span>
        </div>
    """, unsafe_allow_html=True)

    with st.form("intake_form", clear_on_submit=False):
        cause_display = [f"{CAUSE_ICONS.get(c,'•')} {c.replace('_',' ').title()}"
                         for c in EVENT_CAUSES]
        cause_idx   = st.selectbox("Event Cause", range(len(EVENT_CAUSES)),
                                   format_func=lambda i: cause_display[i])
        event_cause = EVENT_CAUSES[cause_idx]

        corridors = list(CORRIDOR_COORDS.keys())
        corridor  = st.selectbox("Corridor", corridors)

        col_et, col_pr = st.columns(2)
        with col_et:
            event_type = st.radio("Event Type", ["unplanned","planned"], horizontal=True)
        with col_pr:
            priority = st.radio("Priority", ["High","Low"], horizontal=True)

        col_h, col_d = st.columns(2)
        with col_h:
            hour = st.slider("Hour of Day", 0, 23, 19, format="%d:00",
                             help="Peak hours: 19–22 and 04–06")
        with col_d:
            day_of_week = st.selectbox(
                "Day of Week", range(7),
                format_func=lambda x: ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"][x]
            )

        veh_type   = st.selectbox("Vehicle Type", VEH_TYPES)
        police_stn = st.selectbox("Police Station", [
            "Yelahanka","Madiwala","K R Puram","Electronic City",
            "Whitefield","Yeshwanthapura","High Grounds","Hebbal",
            "Byatarayanapura","R.T. Nagar","Jayanagara",
        ])

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    # Clear button
    if st.session_state.pred_data is not None:
        if st.button("Clear Results", use_container_width=True):
            for key in ["pred_data","pred_ctx","pred_weather","pred_report","pred_error"]:
                st.session_state[key] = None
            st.rerun()

    # ── Submit handler ────────────────────────────────────────────────────────
    if submitted:
        lat, lon = CORRIDOR_COORDS.get(corridor, (12.9716, 77.5946))
        payload  = {
            "event_type": event_type, "event_cause": event_cause,
            "latitude": lat, "longitude": lon,
            "corridor": corridor, "police_station": police_stn,
            "hour": hour, "day_of_week": day_of_week,
            "veh_type": veh_type, "priority": priority,
        }
        event_ctx = {
            "event_cause": event_cause, "corridor": corridor,
            "hour": hour, "event_type": event_type,
        }

        with st.spinner("Running ML pipeline…"):
            try:
                res  = requests.post("http://127.0.0.1:8000/predict", json=payload, timeout=10)
                data = res.json() if res.status_code == 200 else None
                err  = None
            except Exception as e:
                data, err = None, str(e)

        if data is None:
            st.session_state.pred_data  = None
            st.session_state.pred_error = err or "API returned an error"
        else:
            with st.spinner("Fetching weather & generating AI report…"):
                weather_data = weather_for_corridor(corridor)
                report       = generate_situation_report(data, event_ctx, weather_data)
            st.session_state.pred_data    = data
            st.session_state.pred_ctx     = event_ctx
            st.session_state.pred_weather = weather_data
            st.session_state.pred_report  = report
            st.session_state.pred_error   = None

        st.rerun()



# ── Right column: Results ─────────────────────────────────────────────────────
with col_result:

    # Error state
    if st.session_state.pred_error is not None:
        st.error("API server not reachable. Start it with:")
        st.code("python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000")

    # Empty state
    elif st.session_state.pred_data is None:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                    padding:64px 32px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05);">
            <div style="font-size:36px;margin-bottom:14px;opacity:.35;">⚡</div>
            <div style="color:#111827;font-size:16px;font-weight:600;margin-bottom:8px;">
                Ready for Prediction
            </div>
            <div style="color:#6b7280;font-size:13px;line-height:1.75;max-width:340px;margin:0 auto;">
                Fill the intake form and click <strong style="color:#1a56db;">Run Prediction</strong>
                to get AI-powered severity, duration, closure probability
                and optimal resource deployment recommendations.
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Results state
    else:
        data         = st.session_state.pred_data
        event_ctx    = st.session_state.pred_ctx
        weather_data = st.session_state.pred_weather
        report       = st.session_state.pred_report

        alert_level = data["alert_level"]
        severity    = data["severity"]
        duration    = data["estimated_duration_minutes"]
        closure_p   = data["closure_probability"]
        rec         = data["recommendation"]

        lat = CORRIDOR_COORDS.get(event_ctx["corridor"], (12.9716, 77.5946))[0]
        lon = CORRIDOR_COORDS.get(event_ctx["corridor"], (12.9716, 77.5946))[1]

        # Alert banner
        st.markdown(alert_html(alert_level), unsafe_allow_html=True)

        # Weather-adjusted duration
        mult, warn = duration_multiplier(weather_data)
        adj_duration = duration * mult
        if warn:
            st.warning(warn)

        # KPI cards
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(kpi_card("🎯", "Severity", severity,
                                 accent=SEV_COLOR.get(severity, C["primary"])),
                        unsafe_allow_html=True)
        with m2:
            dur_label = f"{adj_duration:.0f} min" + (" *" if mult > 1 else "")
            st.markdown(kpi_card("⏱️", "Est. Clearance", dur_label, accent=C["warning"]),
                        unsafe_allow_html=True)
        with m3:
            cl_color = C["error"] if closure_p > 0.6 else C["warning"] if closure_p > 0.4 else C["success"]
            st.markdown(kpi_card("🚧", "Closure Risk", f"{closure_p*100:.0f}%", accent=cl_color),
                        unsafe_allow_html=True)

        if mult > 1:
            st.caption(f"* Duration adjusted +{(mult-1)*100:.0f}% for weather conditions")

        # Resource deployment
        section_header("Resource Deployment", "👮")
        r1, r2, r3, r4 = st.columns(4)
        conf_colors = {"high": C["success"], "medium": C["warning"], "low": C["error"]}
        with r1:
            st.markdown(rec_card("👮", "Constables", str(rec["constable_count"]), C["primary"]),
                        unsafe_allow_html=True)
        with r2:
            st.markdown(rec_card("🚧", "Barricades", str(rec["barricade_count"]), C["warning"]),
                        unsafe_allow_html=True)
        with r3:
            st.markdown(rec_card("🔀", "Diversion",
                                 "YES" if rec["diversion_needed"] else "NO",
                                 C["error"] if rec["diversion_needed"] else C["success"]),
                        unsafe_allow_html=True)
        with r4:
            st.markdown(rec_card("📍", "Station", rec["suggested_police_station"],
                                 conf_colors.get(rec["confidence"], C["muted"])),
                        unsafe_allow_html=True)

        # AI Situation Report
        section_header("AI Situation Report", "🤖")
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;
                    border-left:3px solid {C['primary']};border-radius:10px;
                    padding:16px 20px;font-size:14px;color:#374151;line-height:1.8;">
            {report}
        </div>
        """, unsafe_allow_html=True)

        # ── Tactical Deployment Map ───────────────────────────────────────────
        section_header("Tactical Deployment Map", "🗺️")

        # Colour by alert level
        alert_map_colors = {
            "CRITICAL": "#dc2626", "RED": "#ea580c",
            "AMBER":    "#d97706", "GREEN": "#15803d"
        }
        incident_color = alert_map_colors.get(alert_level, "#1a56db")
        diversion_needed = rec.get("diversion_needed", False)
        corridor_name    = event_ctx["corridor"]
        cause_label      = event_ctx["event_cause"].replace("_", " ").title()

        # Staging / barricade / alternate-route data (mirrors ai_narrative.py)
        STAGING_OFFSETS = {
            "Mysore Road":       [(-0.008, -0.012), (-0.014,  0.006)],
            "Bellary Road 1":    [(-0.009,  0.005), ( 0.005,  0.010)],
            "Bellary Road 2":    [(-0.010,  0.003), ( 0.008, -0.005)],
            "Tumkur Road":       [(-0.007, -0.008), (-0.012,  0.004)],
            "Hosur Road":        [( 0.008, -0.006), ( 0.015,  0.005)],
            "ORR North 1":       [(-0.006,  0.010), ( 0.004,  0.015)],
            "Old Madras Road":   [( 0.007,  0.008), (-0.005,  0.012)],
            "Magadi Road":       [(-0.009, -0.010), (-0.006,  0.008)],
            "ORR East 1":        [( 0.010,  0.007), (-0.008,  0.012)],
            "Bannerghatta Road": [( 0.009, -0.008), ( 0.014,  0.003)],
            "Non-corridor":      [(-0.005, -0.005), ( 0.005,  0.005)],
        }
        BARRICADE_OFFSETS = {
            "Mysore Road":       [(-0.003, -0.004), ( 0.003, -0.004), (-0.005,  0.002)],
            "Bellary Road 1":    [(-0.004,  0.002), ( 0.003,  0.004), (-0.002, -0.005)],
            "Bellary Road 2":    [(-0.004,  0.001), ( 0.004, -0.002), ( 0.001,  0.005)],
            "Tumkur Road":       [(-0.003, -0.003), ( 0.003, -0.005), (-0.005,  0.003)],
            "Hosur Road":        [( 0.004, -0.003), (-0.003, -0.004), ( 0.005,  0.002)],
            "ORR North 1":       [(-0.003,  0.005), ( 0.004,  0.006), (-0.002, -0.004)],
            "Old Madras Road":   [( 0.003,  0.004), (-0.004,  0.005), ( 0.005, -0.003)],
            "Magadi Road":       [(-0.004, -0.004), (-0.002,  0.004), ( 0.003, -0.003)],
            "ORR East 1":        [( 0.005,  0.003), (-0.004,  0.006), ( 0.003, -0.004)],
            "Bannerghatta Road": [( 0.004, -0.004), (-0.003, -0.005), ( 0.005,  0.002)],
            "Non-corridor":      [(-0.002, -0.003), ( 0.003,  0.002)],
        }
        ALT_ROUTE_COORDS = {
            "Mysore Road":       [(12.9220, 77.5240, 12.9420, 77.5590), (12.9500, 77.5100, 12.9600, 77.5400)],
            "Bellary Road 1":    [(13.0450, 77.6100, 13.0300, 77.5850), (13.0500, 77.6300, 13.0350, 77.6000)],
            "Bellary Road 2":    [(13.0680, 77.5700, 13.0450, 77.5900), (13.0600, 77.5500, 13.0350, 77.5800)],
            "Tumkur Road":       [(13.0200, 77.5200, 13.0100, 77.5600), (13.0250, 77.5350, 13.0050, 77.5650)],
            "Hosur Road":        [(12.8800, 77.6500, 12.9100, 77.6300), (12.8700, 77.6100, 12.9000, 77.6000)],
            "ORR North 1":       [(13.0700, 77.6100, 13.0500, 77.6400), (13.0550, 77.5900, 13.0400, 77.6200)],
            "Old Madras Road":   [(13.0100, 77.6700, 13.0000, 77.6400), (12.9950, 77.6800, 12.9850, 77.6500)],
            "Magadi Road":       [(12.9600, 77.4900, 12.9700, 77.5200), (12.9750, 77.5000, 12.9800, 77.5300)],
            "ORR East 1":        [(12.9550, 77.6900, 12.9400, 77.6600), (12.9300, 77.7000, 12.9200, 77.6700)],
            "Bannerghatta Road": [(12.8650, 77.5800, 12.8900, 77.6000), (12.8800, 77.5700, 12.9000, 77.5900)],
            "Non-corridor":      [],
        }
        STAGING_LABELS = {
            "Mysore Road":       ["Nayandahalli Jn.", "Kengeri Entry"],
            "Bellary Road 1":    ["Hebbal Flyover", "Mekhri Circle"],
            "Bellary Road 2":    ["Yelahanka New Town", "Air Force Gate"],
            "Tumkur Road":       ["Yeshwanthapura Jn.", "Peenya 2nd Stage"],
            "Hosur Road":        ["Silk Board Jn.", "EC Toll"],
            "ORR North 1":       ["Hebbal ORR Jn.", "Nagavara Signal"],
            "Old Madras Road":   ["KR Puram Bridge", "Hoodi Jn."],
            "Magadi Road":       ["Goraguntepalya Jn.", "Vijayanagar Circle"],
            "ORR East 1":        ["KR Puram ORR", "Marathahalli Bridge"],
            "Bannerghatta Road": ["JP Nagar 7th Phase", "Gottigere Jn."],
            "Non-corridor":      ["Staging Point A", "Staging Point B"],
        }
        ALT_ROUTE_NAMES = {
            "Mysore Road":       ["Kanakapura Rd via NICE", "Magadi Rd → Chord Rd"],
            "Bellary Road 1":    ["Thanisandra Rd", "ORR North → Nagavara"],
            "Bellary Road 2":    ["Yelahanka NH44 Bypass", "Bagalur Rd via Jakkur"],
            "Tumkur Road":       ["Peenya Inner Roads", "Chord Rd via Yeshwanthapura"],
            "Hosur Road":        ["Sarjapur Rd → ORR East", "NICE Rd → EC Phase 2"],
            "ORR North 1":       ["Bellary Rd → Hebbal", "Thanisandra Main Rd"],
            "Old Madras Road":   ["Whitefield Rd via Hoodi", "ITPL Rd → Varthur"],
            "Magadi Road":       ["Mysore Rd via NICE", "Chord Rd → Rajajinagar"],
            "ORR East 1":        ["Sarjapur Rd Parallel", "Varthur Rd → Whitefield"],
            "Bannerghatta Road": ["Kanakapura Rd via JP Nagar", "NICE Rd South"],
            "Non-corridor":      [],
        }

        stg_offsets  = STAGING_OFFSETS.get(corridor_name,  STAGING_OFFSETS["Non-corridor"])
        bar_offsets  = BARRICADE_OFFSETS.get(corridor_name, BARRICADE_OFFSETS["Non-corridor"])
        alt_coords   = ALT_ROUTE_COORDS.get(corridor_name,  [])
        stg_labels   = STAGING_LABELS.get(corridor_name,   STAGING_LABELS["Non-corridor"])
        alt_names    = ALT_ROUTE_NAMES.get(corridor_name,   [])

        # Build barricade subset based on count recommended
        bar_count    = min(rec.get("barricade_count", 2), len(bar_offsets))
        bar_offsets  = bar_offsets[:bar_count] if bar_count > 0 else bar_offsets[:2]

        # Build map
        tac_map = folium.Map(location=[lat, lon], zoom_start=14, tiles="CartoDB positron")

        # Incident marker — pulsing red circle + emoji
        folium.CircleMarker(
            location=[lat, lon], radius=22,
            color=incident_color, fill=True,
            fill_color=incident_color, fill_opacity=0.15, weight=3,
            tooltip=f"🚨 INCIDENT: {cause_label}",
            popup=folium.Popup(
                f"<b style='color:{incident_color}'>⚠ {cause_label}</b><br>"
                f"Corridor: {corridor_name}<br>"
                f"Severity: <b>{severity}</b><br>"
                f"Clearance: {adj_duration:.0f} min<br>"
                f"Closure risk: {closure_p*100:.0f}%",
                max_width=220,
            ),
        ).add_to(tac_map)
        folium.Marker(
            location=[lat, lon],
            icon=folium.DivIcon(
                html=f'''<div style="font-size:26px;margin-left:-13px;margin-top:-26px;
                                     filter:drop-shadow(0 2px 4px rgba(0,0,0,.3));">🚨</div>''',
                icon_size=(26, 26),
            ),
        ).add_to(tac_map)

        # Staging / constable deployment points — blue badge
        for i, (dlat, dlon) in enumerate(stg_offsets):
            slat, slon = lat + dlat, lon + dlon
            label = stg_labels[i] if i < len(stg_labels) else f"Staging {i+1}"
            folium.Marker(
                location=[slat, slon],
                icon=folium.DivIcon(
                    html=f'''<div style="background:#1a56db;color:#fff;font-size:11px;
                                         font-weight:700;padding:4px 8px;border-radius:6px;
                                         white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.25);
                                         border:2px solid #fff;">👮 {label}</div>''',
                    icon_size=(140, 30),
                    icon_anchor=(70, 15),
                ),
                tooltip=f"👮 Staging Point: {label}",
                popup=folium.Popup(
                    f"<b>Staging Point</b><br>{label}<br>"
                    f"Deploy constables here to manage flow",
                    max_width=180,
                ),
            ).add_to(tac_map)

        # Barricade points — orange badge
        for i, (dlat, dlon) in enumerate(bar_offsets):
            blat, blon = lat + dlat, lon + dlon
            folium.Marker(
                location=[blat, blon],
                icon=folium.DivIcon(
                    html=f'''<div style="background:#d97706;color:#fff;font-size:11px;
                                         font-weight:700;padding:4px 8px;border-radius:6px;
                                         white-space:nowrap;box-shadow:0 2px 6px rgba(0,0,0,.25);
                                         border:2px solid #fff;">🚧 Barricade {i+1}</div>''',
                    icon_size=(120, 30),
                    icon_anchor=(60, 15),
                ),
                tooltip=f"🚧 Barricade Point {i+1}",
                popup=folium.Popup(
                    f"<b>Barricade Position {i+1}</b><br>Place barricade here to block/divert traffic",
                    max_width=180,
                ),
            ).add_to(tac_map)

        # Alternate route polylines — green dashed lines (only if diversion needed)
        if diversion_needed:
            route_colors = ["#15803d", "#0891b2"]
            for ri, coords in enumerate(alt_coords[:2]):
                if len(coords) == 4:
                    r_color = route_colors[ri % len(route_colors)]
                    r_name  = alt_names[ri] if ri < len(alt_names) else f"Alt Route {ri+1}"
                    folium.PolyLine(
                        locations=[[coords[0], coords[1]], [coords[2], coords[3]]],
                        color=r_color, weight=4, opacity=0.8, dash_array="8 4",
                        tooltip=f"🔀 Alt Route {ri+1}: {r_name}",
                        popup=folium.Popup(
                            f"<b style='color:{r_color}'>Alternate Route {ri+1}</b><br>{r_name}<br>"
                            f"<i>Activate this diversion to reduce congestion on {corridor_name}</i>",
                            max_width=220,
                        ),
                    ).add_to(tac_map)
                    # Route label at midpoint
                    mid_lat = (coords[0] + coords[2]) / 2
                    mid_lon = (coords[1] + coords[3]) / 2
                    folium.Marker(
                        location=[mid_lat, mid_lon],
                        icon=folium.DivIcon(
                            html=f'''<div style="background:{r_color};color:#fff;font-size:10px;
                                                 font-weight:700;padding:3px 7px;border-radius:5px;
                                                 white-space:nowrap;box-shadow:0 1px 4px rgba(0,0,0,.2);
                                                 border:1px solid #fff;">🔀 {r_name}</div>''',
                            icon_size=(160, 24),
                            icon_anchor=(80, 12),
                        ),
                    ).add_to(tac_map)

        # Map legend
        legend_items = [
            f'<span style="color:{incident_color};font-size:16px;">🚨</span> Incident Point',
            '<span style="color:#1a56db;font-size:14px;">👮</span> Constable Staging',
            '<span style="color:#d97706;font-size:14px;">🚧</span> Barricade Position',
        ]
        if diversion_needed:
            legend_items.append('<span style="color:#15803d;">──</span> Alternate Route')

        legend_html = "".join(
            f'<div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">{item}</div>'
            for item in legend_items
        )
        tac_map.get_root().html.add_child(folium.Element(f'''
            <div style="position:fixed;bottom:24px;left:24px;z-index:9999;
                        background:white;border:1px solid #e5e7eb;border-radius:8px;
                        padding:10px 14px;font-family:Inter,sans-serif;font-size:12px;
                        color:#374151;box-shadow:0 2px 8px rgba(0,0,0,.12);">
                <div style="font-weight:700;margin-bottom:6px;color:#111827;">Map Legend</div>
                {legend_html}
            </div>
        '''))

        st_folium(tac_map, height=400, use_container_width=True)

        # Quick legend note below map
        if diversion_needed:
            st.markdown(
                f'<p style="font-size:12px;color:#6b7280;margin-top:6px;">' +
                f'🔀 Diversion routes shown. Activate <b>{alt_names[0] if alt_names else "primary alternate"}</b> first.' +
                f'</p>',
                unsafe_allow_html=True,
            )