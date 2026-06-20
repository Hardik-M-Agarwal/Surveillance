"""
Event Scenario Planner — Pre-plan police deployment for upcoming events.
Full light-theme redesign: planning form on page, rich results layout.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import folium
from streamlit_folium import st_folium
from datetime import datetime, timedelta, date

from utils.theme import (apply_theme, sidebar_brand, page_header, section_header,
                         kpi_card, rec_card, C, PLOTLY, AXIS_STYLE, BASE_MARGIN)
from utils.ai_narrative import generate_scenario_brief

st.set_page_config(page_title="Scenario Planner · TrafficSense",
                   page_icon="🗓️", layout="wide")
apply_theme()

# ── Extra page CSS ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
div[data-testid="column"] { padding: 0 5px !important; }
div[data-testid="stDateInput"] input { background:#ffffff !important; color:#111827 !important;
        border:1px solid #d1d5db !important; border-radius:8px !important; font-size:14px !important; }
/* Date picker calendar popup */
div[data-baseweb="calendar"] { background:#ffffff !important; border:1px solid #e5e7eb !important; }
div[data-baseweb="calendar"] button { color:#111827 !important; background:#ffffff !important; }
div[data-baseweb="calendar"] button:hover { background:#eff6ff !important; }
div[data-baseweb="calendar"] [aria-selected="true"] button {
    background:#1a56db !important; color:#ffffff !important; border-radius:50% !important; }
div[data-baseweb="calendar"] div[role="columnheader"] { color:#6b7280 !important; font-weight:600 !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    sidebar_brand()

# ── Constants ──────────────────────────────────────────────────────────────────
CORRIDOR_COORDS = {
    "Any":                  (12.9716, 77.5946),
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
}

EVENT_CAUSE_OPTIONS = {
    "🎪 Public Event / Rally":     "public_event",
    "✊ Protest / Agitation":      "protest",
    "🚶 Procession / March":       "procession",
    "👔 VIP Movement":             "vip_movement",
    "🚧 Road Construction":        "construction",
    "🚗 Vehicle Breakdown (mass)": "vehicle_breakdown",
    "💥 Accident":                 "accident",
    "🚦 General Congestion":       "congestion",
}

CAUSE_BASE = {
    "vip_movement":      dict(constables=9,  barricades=18, closure_prob=0.80),
    "protest":           dict(constables=7,  barricades=14, closure_prob=0.45),
    "procession":        dict(constables=6,  barricades=12, closure_prob=0.35),
    "public_event":      dict(constables=5,  barricades=10, closure_prob=0.30),
    "accident":          dict(constables=5,  barricades=10, closure_prob=0.15),
    "vehicle_breakdown": dict(constables=3,  barricades=6,  closure_prob=0.04),
    "congestion":        dict(constables=4,  barricades=8,  closure_prob=0.10),
    "construction":      dict(constables=4,  barricades=8,  closure_prob=0.12),
}

RISK_PALETTE = {"Low":"#15803d","Moderate":"#d97706","High":"#ea580c","Critical":"#b91c1c"}
RISK_BG      = {"Low":"#f0fdf4","Moderate":"#fffbeb","High":"#fff7ed","Critical":"#fef2f2"}
RISK_BORDER  = {"Low":"#bbf7d0","Moderate":"#fde68a","High":"#fed7aa","Critical":"#fecaca"}

def _rlabel(s):
    if s >= 75: return "Critical"
    if s >= 50: return "High"
    if s >= 25: return "Moderate"
    return "Low"

# ── Data ───────────────────────────────────────────────────────────────────────
CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"

@st.cache_data(show_spinner=False)
def load_historical():
    if not CLEANED_CSV.exists():
        return None
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["start_datetime"]        = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]                  = df["start_datetime"].dt.hour
    df["day_of_week"]           = df["start_datetime"].dt.dayofweek
    df["duration_minutes"]      = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower()
        .map({"true":1,"1":1,"false":0,"0":0}).fillna(0).astype(int)
    )
    df["sev_score"] = df["severity"].map(
        {"Critical":4,"High":3,"Medium":2,"Low":1}).fillna(2)
    return df


def compute_hourly_profile(df, cause, corridor, day_of_week):
    CAUSE_GROUPS = {
        "procession":        ["procession","public_event","vip_movement"],
        "vip_movement":      ["vip_movement","procession"],
        "protest":           ["protest"],
        "public_event":      ["public_event","procession"],
        "accident":          ["accident"],
        "vehicle_breakdown": ["vehicle_breakdown"],
        "congestion":        ["congestion"],
        "construction":      ["construction","road_conditions","debris"],
    }
    cause_group = CAUSE_GROUPS.get(cause, [cause])
    mask = df["event_cause"].isin(cause_group)
    if corridor != "Any":
        corr_mask = mask & (df["corridor"] == corridor)
        if corr_mask.sum() >= 30:
            mask = corr_mask
    if day_of_week >= 0:
        dow_mask = mask & (df["day_of_week"] == day_of_week)
        if dow_mask.sum() >= 20:
            mask = dow_mask
    sub = df[mask].copy()
    if len(sub) < 5:
        sub = df[df["event_cause"].isin(cause_group)]
    if len(sub) < 5:
        sub = df.copy()

    base = CAUSE_BASE.get(cause, dict(constables=4, barricades=8, closure_prob=0.20))
    hourly = []
    for h in range(24):
        hour_sub = sub[sub["hour"] == h] if "hour" in sub.columns else sub
        n_incidents   = len(hour_sub)
        global_n      = max(len(df[df["hour"] == h]), 1)
        density_ratio = n_incidents / max(global_n, 1)
        if len(hour_sub) > 0:
            avg_sev = hour_sub["sev_score"].mean()
            closure = hour_sub["requires_road_closure"].mean()
        else:
            avg_sev = 2.0
            closure = base["closure_prob"]
        peak_bonus = 1.3 if h in [4,5,6,19,20,21,22] else 1.0
        risk = min(100, (
            0.30*(avg_sev/4) + 0.35*closure +
            0.20*density_ratio + 0.15*(peak_bonus-1)
        )*100*peak_bonus*(1+base["closure_prob"]))
        constables = max(1, int(base["constables"]*(0.5+0.5*(risk/100))*peak_bonus))
        barricades  = constables * 2
        diversion   = closure > 0.3 or base["closure_prob"] > 0.3
        med_dur     = hour_sub["duration_minutes"].median() if len(hour_sub) > 0 else 45.0
        if np.isnan(med_dur): med_dur = 45.0
        hourly.append({
            "hour": h, "risk_score": round(risk,1),
            "avg_severity": round(avg_sev,2), "closure_prob": round(closure,2),
            "constables": constables, "barricades": barricades,
            "diversion": diversion, "median_duration": round(med_dur,0),
            "n_historical": n_incidents,
        })
    return pd.DataFrame(hourly)


# ── Session state ─────────────────────────────────────────────────────────────
for _k, _v in {
    "sp_results":   None,   # computed results dict
    "sp_hourly":    None,   # hourly DataFrame
    "sp_event":     None,   # event DataFrame (window only)
    "sp_similar":   None,   # similar historical DataFrame
    "sp_brief":     None,   # AI brief text
    "sp_inputs":    None,   # form inputs snapshot
}.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

# ── Page header ────────────────────────────────────────────────────────────────
page_header("🗓️", "Event Scenario Planner",
            "Pre-plan police deployment for any upcoming event · powered by historical patterns")

df_hist = load_historical()

# ── PLANNING FORM ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
            padding:20px 24px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
    <div style="font-size:12px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:.08em;margin-bottom:16px;display:flex;align-items:center;gap:8px;">
        <span>🎯</span> Plan Your Event Deployment
    </div>
""", unsafe_allow_html=True)

f1, f2, f3, f4, f5 = st.columns([1.4, 1.2, 0.8, 0.8, 0.9])

with f1:
    cause_label = st.selectbox("Event Type", list(EVENT_CAUSE_OPTIONS.keys()))
    event_cause = EVENT_CAUSE_OPTIONS[cause_label]

with f2:
    corridor = st.selectbox("Affected Corridor", list(CORRIDOR_COORDS.keys()))

with f3:
    event_date  = st.date_input("Event Date", value=date.today() + timedelta(days=1))
    day_of_week = event_date.weekday()

with f4:
    start_hour   = st.number_input("Start Hour", 0, 23, 9, format="%02d",
                                   help="e.g. 9 for 09:00")
    duration_hrs = st.number_input("Duration (hrs)", 1, 12, 3)

with f5:
    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
    # Summary pill
    end_h = (start_hour + duration_hrs) % 24
    dow_names = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    st.markdown(f"""
    <div style="background:#f0f4f8;border:1px solid #e2e8f0;border-radius:8px;
                padding:10px 14px;font-size:12px;color:#374151;line-height:1.8;margin-top:2px;">
        <b>{event_date.strftime('%d %b %Y')}</b> · {dow_names[day_of_week]}<br>
        🕐 {start_hour:02d}:00 – {end_h:02d}:00<br>
        📍 {corridor}
    </div>
    """, unsafe_allow_html=True)
    st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)
    run_btn = st.button("Generate Deployment Plan", use_container_width=True)

st.markdown("</div>", unsafe_allow_html=True)

# ── ON BUTTON CLICK: compute and store in session_state ───────────────────────
if run_btn:
    if df_hist is None:
        st.error("Historical data not found — run `python run_pipeline.py` first.")
        st.stop()

    with st.spinner("Analysing historical patterns…"):
        hourly_df_c  = compute_hourly_profile(df_hist, event_cause, corridor, day_of_week)
        event_hours_c= [(start_hour + i) % 24 for i in range(int(duration_hrs))]
        event_df_c   = hourly_df_c[hourly_df_c["hour"].isin(event_hours_c)].copy()

        peak_idx_c       = event_df_c["risk_score"].idxmax()
        peak_hour_c      = int(event_df_c.loc[peak_idx_c, "hour"])
        max_risk_c       = event_df_c["risk_score"].max()
        max_constables_c = int(event_df_c["constables"].max())
        max_barricades_c = int(event_df_c["barricades"].max())
        total_need_c     = int(event_df_c["constables"].sum())
        need_diversion_c = bool(event_df_c["diversion"].any())
        closure_p_c      = event_df_c["closure_prob"].mean()

        hist_filter_c = df_hist["event_cause"] == event_cause
        if corridor != "Any":
            hist_filter_c &= df_hist["corridor"] == corridor
        similar_c  = df_hist[hist_filter_c]
        n_similar_c = len(similar_c)
        stage_time_c = max(int(start_hour) - 1, 0)

        # AI brief
        scenario_ctx_c = {
            "event_type":       event_cause,
            "corridor":         corridor,
            "date":             str(event_date),
            "start_hour":       int(start_hour),
            "duration_hours":   int(duration_hrs),
            "peak_hour":        peak_hour_c,
            "max_constables":   max_constables_c,
            "max_barricades":   max_barricades_c,
            "diversion_needed": need_diversion_c,
            "similar_events":   f"{n_similar_c:,} historical incidents",
        }
        brief_c = generate_scenario_brief(scenario_ctx_c)

    # Store everything
    st.session_state.sp_results = {
        "peak_hour":       peak_hour_c,
        "max_risk":        max_risk_c,
        "max_constables":  max_constables_c,
        "max_barricades":  max_barricades_c,
        "total_need":      total_need_c,
        "need_diversion":  need_diversion_c,
        "closure_p":       closure_p_c,
        "n_similar":       n_similar_c,
        "stage_time":      stage_time_c,
        "event_hours":     event_hours_c,
        "risk_label_str":  _rlabel(max_risk_c),
        "end_h":           (int(start_hour) + int(duration_hrs)) % 24,
        "cause_label":     cause_label,
        "event_cause":     event_cause,
        "corridor":        corridor,
        "event_date":      str(event_date),
        "start_hour":      int(start_hour),
        "duration_hrs":    int(duration_hrs),
    }
    st.session_state.sp_hourly  = hourly_df_c
    st.session_state.sp_event   = event_df_c
    st.session_state.sp_similar = similar_c
    st.session_state.sp_brief   = brief_c
    st.rerun()

# ── EMPTY STATE (no results yet) ──────────────────────────────────────────────
if st.session_state.sp_results is None:
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                padding:56px 40px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05);">
        <div style="font-size:40px;margin-bottom:14px;opacity:.4;">🗓️</div>
        <div style="color:#111827;font-size:16px;font-weight:600;margin-bottom:8px;">
            Ready to Plan
        </div>
        <div style="color:#6b7280;font-size:13px;line-height:1.9;max-width:500px;
                    margin:0 auto;">
            Fill in the event details above and click
            <strong style="color:#1a56db;">Generate Deployment Plan</strong> to get:<br>
            Hour-by-hour risk timeline &nbsp;·&nbsp; Constable &amp; barricade schedule
            &nbsp;·&nbsp; Diversion requirements &nbsp;·&nbsp;
            AI pre-deployment brief &nbsp;·&nbsp; Historical context
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── UNPACK from session_state ─────────────────────────────────────────────────
_r            = st.session_state.sp_results
hourly_df     = st.session_state.sp_hourly
event_df      = st.session_state.sp_event
similar       = st.session_state.sp_similar
brief         = st.session_state.sp_brief

peak_hour      = _r["peak_hour"]
max_risk       = _r["max_risk"]
max_constables = _r["max_constables"]
max_barricades = _r["max_barricades"]
total_need     = _r["total_need"]
need_diversion = _r["need_diversion"]
closure_p      = _r["closure_p"]
n_similar      = _r["n_similar"]
stage_time     = _r["stage_time"]
event_hours    = _r["event_hours"]
risk_label_str = _r["risk_label_str"]
end_h          = _r["end_h"]
cause_label    = _r["cause_label"]
event_cause    = _r["event_cause"]
corridor       = _r["corridor"]
event_date     = date.fromisoformat(_r["event_date"])
start_hour     = _r["start_hour"]
duration_hrs   = _r["duration_hrs"]

# ── KPI STRIP ──────────────────────────────────────────────────────────────────
risk_color = RISK_PALETTE[risk_label_str]
risk_bg    = RISK_BG[risk_label_str]

# Alert banner
st.markdown(f"""
<div style="background:{risk_bg};border:1px solid {RISK_BORDER[risk_label_str]};
            border-left:4px solid {risk_color};border-radius:10px;
            padding:14px 20px;margin-bottom:16px;
            display:flex;align-items:center;gap:16px;">
    <div style="flex:1;">
        <div style="font-size:11px;font-weight:700;color:{risk_color};
                    text-transform:uppercase;letter-spacing:.08em;margin-bottom:2px;">
            Predicted Event Risk Level
        </div>
        <div style="font-size:20px;font-weight:800;color:{risk_color};">
            {risk_label_str} — {max_risk:.0f}/100
        </div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:12px;color:#6b7280;">Peak risk at</div>
        <div style="font-size:22px;font-weight:800;color:#111827;">{peak_hour:02d}:00</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:12px;color:#6b7280;">Stage by</div>
        <div style="font-size:22px;font-weight:800;color:#1a56db;">{stage_time:02d}:00</div>
    </div>
    <div style="text-align:right;">
        <div style="font-size:12px;color:#6b7280;">Diversion</div>
        <div style="font-size:22px;font-weight:800;
                    color:{'#b91c1c' if need_diversion else '#15803d'};">
            {'YES' if need_diversion else 'NO'}
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

k1,k2,k3,k4,k5 = st.columns(5)
for col, icon, lbl, val, acc in [
    (k1,"👮","Peak Constables",      str(max_constables),  C["primary"]),
    (k2,"🚧","Peak Barricades",      str(max_barricades),  C["warning"]),
    (k3,"📊","Officer-Hours Total",  str(total_need),      C["cyan"]),
    (k4,"🔒","Avg Closure Risk",     f"{closure_p*100:.0f}%", C["error"]),
    (k5,"📜","Similar Past Events",  f"{n_similar:,}",     C["purple"]),
]:
    col.markdown(kpi_card(icon, lbl, val, accent=acc), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 1: Risk timeline chart (left) + Map (right)
# ══════════════════════════════════════════════════════════════════════════════
col_chart, col_map = st.columns([1.6, 1], gap="large")

with col_chart:
    section_header("Hour-by-Hour Risk & Resource Timeline", "📈")

    hours_all   = hourly_df["hour"].tolist()
    risk_all    = hourly_df["risk_score"].tolist()
    const_all   = hourly_df["constables"].tolist()
    in_event    = [h in event_hours for h in hours_all]

    fig = go.Figure()

    # Event window shading
    fig.add_vrect(
        x0=start_hour - 0.4, x1=(start_hour + duration_hrs - 1 + 0.4),
        fillcolor="rgba(26,86,219,0.06)",
        line_color="rgba(26,86,219,0.2)", line_width=1,
        annotation_text=f"Event Window  {start_hour:02d}:00–{end_h:02d}:00",
        annotation_position="top left",
        annotation_font=dict(color="#1a56db", size=11),
    )

    # Risk area
    fig.add_trace(go.Scatter(
        x=hours_all, y=risk_all,
        name="Risk Score",
        mode="lines+markers",
        line=dict(color=risk_color, width=2.5),
        marker=dict(
            color=[risk_color if ie else "#d1d5db" for ie in in_event],
            size=[9 if ie else 5 for ie in in_event],
            line=dict(color="#fff", width=1.5),
        ),
        fill="tozeroy",
        fillcolor=f"rgba({int(risk_color[1:3],16)},{int(risk_color[3:5],16)},{int(risk_color[5:7],16)},0.06)",
        hovertemplate="<b>%{x:02d}:00</b><br>Risk: %{y:.0f}/100<extra></extra>",
    ))

    # Constables line (secondary Y)
    fig.add_trace(go.Scatter(
        x=hours_all, y=const_all,
        name="Constables Needed",
        mode="lines+markers", yaxis="y2",
        line=dict(color="#1a56db", width=2, dash="dot"),
        marker=dict(size=5, color="#1a56db"),
        hovertemplate="<b>%{x:02d}:00</b><br>Constables: %{y}<extra></extra>",
    ))

    # Peak marker
    fig.add_trace(go.Scatter(
        x=[peak_hour], y=[max_risk],
        name="Peak Risk",
        mode="markers+text",
        marker=dict(color=risk_color, size=14, symbol="star",
                    line=dict(color="#fff", width=2)),
        text=[f"Peak {max_risk:.0f}"],
        textposition="top center",
        textfont=dict(size=11, color=risk_color),
        hovertemplate=f"Peak at {peak_hour:02d}:00 — Risk {max_risk:.0f}/100<extra></extra>",
    ))

    # Threshold line
    fig.add_hline(y=50, line=dict(color="#ea580c", dash="dot", width=1),
                  annotation_text="High Risk", annotation_position="top right",
                  annotation_font=dict(color="#ea580c", size=10))

    fig.update_layout(
        **PLOTLY, height=340,
        margin=dict(l=55, r=55, t=44, b=60),
        title=dict(text="Risk score & constable requirement across all 24 hours",
                   font=dict(size=13, color="#111827"), x=0, xanchor="left"),
        yaxis=dict(title="Risk Score (0–100)", range=[0,110],
                   gridcolor="#e5e7eb", linecolor="#d1d5db",
                   tickfont=dict(color="#374151"),
                   title_font=dict(color="#374151")),
        yaxis2=dict(title="Constables", overlaying="y", side="right",
                    range=[0, max_constables*2.2],
                    gridcolor="rgba(0,0,0,0)", linecolor="#d1d5db",
                    tickfont=dict(color="#1a56db"),
                    title_font=dict(color="#1a56db")),
        legend=dict(orientation="h", y=-0.22,
                    font=dict(size=11, color="#374151"),
                    bgcolor="rgba(0,0,0,0)"),
    )
    fig.update_xaxes(
        gridcolor="#e5e7eb", linecolor="#d1d5db",
        tickmode="array", tickvals=list(range(0,24,2)),
        ticktext=[f"{h:02d}:00" for h in range(0,24,2)],
        tickfont=dict(color="#374151"), title="Hour of Day",
        title_font=dict(color="#374151"),
    )
    st.plotly_chart(fig, config={"displayModeBar":False}, use_container_width=True)

with col_map:
    section_header("Deployment Map", "📍")
    lat, lon = CORRIDOR_COORDS.get(corridor, (12.9716, 77.5946))
    m = folium.Map(location=[lat, lon], zoom_start=13, tiles="CartoDB positron")

    # Incident zone
    folium.CircleMarker(
        location=[lat, lon], radius=22,
        color=risk_color, fill=True,
        fill_color=risk_color, fill_opacity=0.15, weight=3,
        tooltip=f"{corridor} — {risk_label_str} Risk ({max_risk:.0f}/100)",
        popup=folium.Popup(
            f"<div style='font-family:Inter,sans-serif;min-width:180px;'>"
            f"<b style='color:{risk_color};font-size:14px;'>{corridor}</b><hr>"
            f"Event: {cause_label}<br>"
            f"Date: {event_date.strftime('%d %b %Y')}<br>"
            f"Time: {start_hour:02d}:00 – {end_h:02d}:00<br>"
            f"Peak Risk: {max_risk:.0f}/100 at {peak_hour:02d}:00<br>"
            f"Max constables: {max_constables}<br>"
            f"Diversion: {'Yes' if need_diversion else 'No'}"
            f"</div>", max_width=220,
        ),
    ).add_to(m)

    folium.Marker(location=[lat, lon], icon=folium.DivIcon(
        html=f'<div style="font-size:24px;margin:-12px 0 0 -12px;">🚨</div>',
        icon_size=(24,24),
    )).add_to(m)

    # Staging points
    staging_offsets = [(0.005, 0.008), (-0.006, 0.010), (0.009, -0.005)]
    staging_labels  = ["Staging Point A", "Staging Point B", "Staging Point C"]
    for i, (dlat, dlon) in enumerate(staging_offsets):
        folium.Marker(
            location=[lat+dlat, lon+dlon],
            icon=folium.DivIcon(
                html=(f'<div style="background:#eff6ff;border:1.5px solid #bfdbfe;'
                      f'color:#1a56db;font-size:10px;font-weight:700;'
                      f'padding:3px 8px;border-radius:6px;white-space:nowrap;'
                      f'box-shadow:0 1px 4px rgba(0,0,0,.1);">'
                      f'👮 {staging_labels[i]}</div>'),
                icon_size=(130,22), icon_anchor=(0,11),
            ),
            tooltip=staging_labels[i],
        ).add_to(m)

    # Deployment zone ring
    folium.CircleMarker(
        location=[lat, lon], radius=max_risk/3.5,
        color="#1a56db", fill=False,
        weight=1.5, dash_array="6 4", opacity=0.5,
        tooltip="Deployment zone radius",
    ).add_to(m)

    # Legend
    m.get_root().html.add_child(folium.Element(f"""
    <div style="position:fixed;bottom:20px;left:20px;z-index:9999;
                background:#fff;border:1px solid #e5e7eb;border-radius:8px;
                padding:10px 14px;font-family:Inter,sans-serif;font-size:11px;">
        <b style="color:#111827;">Event: {cause_label.split(' ',1)[1]}</b><br>
        <span style="color:{risk_color};">■</span> {risk_label_str} Risk Zone<br>
        <span style="color:#1a56db;">– –</span> Deployment zone<br>
        <span style="color:#1a56db;">👮</span> Staging points
    </div>"""))

    st_folium(m, height=400, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# ROW 2: Deployment schedule table (left) + Resource summary (right)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
col_tbl, col_res = st.columns([1.6, 1], gap="large")

with col_tbl:
    section_header("Hour-by-Hour Deployment Schedule", "📋")

    # Render as styled cards instead of a dataframe
    # Header row
    st.markdown("""
    <div style="display:grid;grid-template-columns:60px 90px 70px 90px 90px 90px 80px;
                gap:4px;padding:8px 12px;background:#f3f4f6;border-radius:8px;
                margin-bottom:6px;font-size:10px;font-weight:700;color:#6b7280;
                text-transform:uppercase;letter-spacing:.06em;">
        <div>Hour</div><div>Alert</div><div>Risk</div>
        <div>Constables</div><div>Barricades</div>
        <div>Closure %</div><div>Diversion</div>
    </div>
    """, unsafe_allow_html=True)

    for _, row in event_df.iterrows():
        h      = int(row["hour"])
        risk   = row["risk_score"]
        rl     = _rlabel(risk)
        rc     = RISK_PALETTE[rl]
        rb     = RISK_BG[rl]
        alert  = ("🔴 CRITICAL" if risk>75 else "🟠 HIGH" if risk>50
                  else "🟡 CAUTION" if risk>25 else "🟢 LOW")
        div_yn = "✅ YES" if row["diversion"] else "❌ NO"
        is_peak = h == peak_hour

        st.markdown(f"""
        <div style="display:grid;grid-template-columns:60px 90px 70px 90px 90px 90px 80px;
                    gap:4px;padding:9px 12px;
                    background:{'#eff6ff' if is_peak else '#ffffff'};
                    border:1px solid {'#bfdbfe' if is_peak else '#e5e7eb'};
                    border-left:4px solid {rc};
                    border-radius:8px;margin-bottom:4px;font-size:12px;
                    box-shadow:0 1px 3px rgba(0,0,0,.03);">
            <div style="font-weight:700;color:#111827;">
                {h:02d}:00{'⭐' if is_peak else ''}</div>
            <div style="color:{rc};font-weight:600;font-size:11px;">{alert}</div>
            <div style="font-weight:800;color:{rc};">{risk:.0f}</div>
            <div style="font-weight:700;color:#1a56db;">👮 {int(row['constables'])}</div>
            <div style="font-weight:700;color:#d97706;">🚧 {int(row['barricades'])}</div>
            <div style="color:#374151;">{row['closure_prob']*100:.0f}%</div>
            <div>{div_yn}</div>
        </div>
        """, unsafe_allow_html=True)

with col_res:
    section_header("Resource Summary", "👮")

    # Resource cards
    r1, r2 = st.columns(2)
    with r1:
        st.markdown(rec_card("👮","Peak Constables", str(max_constables), C["primary"]),
                    unsafe_allow_html=True)
    with r2:
        st.markdown(rec_card("🚧","Peak Barricades", str(max_barricades), C["warning"]),
                    unsafe_allow_html=True)
    r3, r4 = st.columns(2)
    with r3:
        div_col = C["error"] if need_diversion else C["success"]
        st.markdown(rec_card("🔀","Diversion","YES" if need_diversion else "NO", div_col),
                    unsafe_allow_html=True)
    with r4:
        st.markdown(rec_card("⏱️","Total Officer-Hours", str(total_need), C["cyan"]),
                    unsafe_allow_html=True)

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)

    # Quick action checklist
    section_header("Pre-Event Checklist", "✅")
    checklist = [
        (True,  f"Stage units by <b>{stage_time:02d}:00</b> (1 hr before start)"),
        (True,  f"Position <b>{max_constables} constables</b> at ingress points"),
        (True,  f"Place <b>{max_barricades} barricades</b> at corridor entries"),
        (need_diversion, "Activate diversion route at event start"),
        (closure_p > 0.3, f"Pre-inform public of likely closure ({closure_p*100:.0f}% probability)"),
        (True,  f"Contact nearest station 30 min before peak ({peak_hour:02d}:00)"),
        (max_risk > 50,   "Request additional units on standby"),
    ]
    for active, text in checklist:
        if active:
            badge = '<span style="display:inline-flex;align-items:center;justify-content:center;' \
                     'width:18px;height:18px;border-radius:4px;background:#1a56db;' \
                     'flex-shrink:0;margin-top:1px;">' \
                     '<svg width="10" height="8" viewBox="0 0 10 8" fill="none">' \
                     '<path d="M1 4L3.5 6.5L9 1" stroke="white" stroke-width="1.8" ' \
                     'stroke-linecap="round" stroke-linejoin="round"/></svg></span>'
            txt_color = "#111827"
        else:
            badge = '<span style="display:inline-block;width:18px;height:18px;' \
                     'border-radius:4px;border:1.5px solid #d1d5db;' \
                     'flex-shrink:0;margin-top:1px;"></span>'
            txt_color = "#9ca3af"
        st.markdown(
            f'<div style="display:flex;align-items:flex-start;gap:10px;' \
            f'padding:8px 0;border-bottom:1px solid #f3f4f6;font-size:13px;color:{txt_color};">' \
            f'{badge}<span style="line-height:1.5;">{text}</span></div>',
            unsafe_allow_html=True,
        )

# ══════════════════════════════════════════════════════════════════════════════
# ROW 3: AI Brief (left) + Historical context (right)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
col_brief, col_hist = st.columns([1, 1], gap="large")

with col_brief:
    section_header("AI Pre-Deployment Brief", "🤖")

    if brief:
        import re as _re
        # Convert markdown **bold** → <b>bold</b>, strip leftover *
        brief_html = _re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", brief)
        brief_html = brief_html.replace("*", "")
        # Convert newlines to <br>
        brief_html = brief_html.replace("\n", "<br>")
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;
                    border-left:3px solid {C['primary']};border-radius:10px;
                    padding:18px 22px;font-size:13px;color:#374151;line-height:1.95;">
            {brief_html}
        </div>
        """, unsafe_allow_html=True)
    else:
        # Rule-based fallback — rendered as clean HTML
        div_line = (
            "Activate diversion routes at event start and station a constable at each alternate entry."
            if need_diversion else
            "Diversion not required at this stage; activate only if closure probability exceeds 50%."
        )
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;
                    border-left:3px solid {C['primary']};border-radius:10px;
                    padding:16px 20px;font-size:13px;color:#374151;line-height:1.9;">
            <div style="font-size:12px;font-weight:700;color:#6b7280;
                        text-transform:uppercase;letter-spacing:.07em;margin-bottom:10px;">
                Deployment Brief · {event_date.strftime('%d %b %Y')}
            </div>
            A <b>{cause_label}</b> is expected on <b>{corridor}</b> from
            <b>{start_hour:02d}:00 to {end_h:02d}:00 hrs</b>.
            Based on <b>{n_similar:,} similar historical incidents</b>, peak traffic impact
            is forecast at <b>{peak_hour:02d}:00 hrs</b> with a risk score of
            <b style="color:{risk_color};">{max_risk:.0f}/100 ({risk_label_str})</b>.<br><br>
            <b>Staging:</b> Deploy all units to their positions by <b>{stage_time:02d}:00 hrs</b>,
            one hour before the event starts.
            Position <b>{max_constables} constables</b> and <b>{max_barricades} barricades</b>
            at key ingress points on {corridor}.<br><br>
            <b>Diversion:</b> {div_line}<br><br>
            <b>Escalation trigger:</b> If crowd exceeds estimate or closure probability rises
            above 70%, immediately request 3 additional constables and notify the
            {corridor.split()[0]} area supervisor.
        </div>
        """, unsafe_allow_html=True)

with col_hist:
    section_header(f"Historical Context — {n_similar:,} Similar Incidents", "📜")

    if n_similar == 0:
        st.info("No similar historical incidents found for this combination.")
    else:
        # Summary stats from similar events
        _dur_vals = similar["duration_minutes"].dropna()
        avg_dur  = _dur_vals.mean() if len(_dur_vals) > 0 else 0.0
        avg_dur  = avg_dur if not (avg_dur != avg_dur) else 0.0  # NaN guard
        avg_sev  = similar["sev_score"].mean() if "sev_score" in similar.columns else 2.0
        cl_rate  = similar["requires_road_closure"].mean() * 100
        sev_dist = similar["severity"].value_counts()

        s1, s2, s3 = st.columns(3)
        for col, lbl, val, acc in [
            (s1, "Avg Duration",   f"{avg_dur:.0f} min" if avg_dur > 0 else "—",  C["warning"]),
            (s2, "Closure Rate",   f"{cl_rate:.0f}%",     C["error"]),
            (s3, "Avg Severity",   f"{avg_sev:.1f}/4",    C["primary"]),
        ]:
            col.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
                        padding:10px 12px;text-align:center;border-top:2px solid {acc};">
                <div style="font-size:18px;font-weight:800;color:{acc};">{val}</div>
                <div style="font-size:10px;color:#6b7280;font-weight:600;
                            text-transform:uppercase;margin-top:3px;">{lbl}</div>
            </div>
            """, unsafe_allow_html=True)

        # Severity distribution mini-bar
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        sev_order  = ["Critical","High","Medium","Low"]
        sev_colors = {"Critical":"#b91c1c","High":"#ea580c","Medium":"#d97706","Low":"#15803d"}
        total_sev  = len(similar.dropna(subset=["severity"]))
        if total_sev > 0:
            st.markdown('<div style="font-size:11px;font-weight:700;color:#6b7280;'
                        'text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">'
                        'Severity Distribution</div>', unsafe_allow_html=True)
            for sev in sev_order:
                cnt = int(sev_dist.get(sev, 0))
                pct = cnt / total_sev * 100
                if cnt == 0: continue
                sc  = sev_colors[sev]
                st.markdown(f"""
                <div style="display:flex;align-items:center;gap:8px;margin-bottom:5px;">
                    <div style="width:70px;font-size:11px;font-weight:600;color:#374151;">
                        {sev}</div>
                    <div style="flex:1;background:#f3f4f6;border-radius:99px;height:8px;">
                        <div style="background:{sc};width:{pct}%;height:8px;
                                    border-radius:99px;"></div>
                    </div>
                    <div style="width:40px;text-align:right;font-size:11px;
                                font-weight:700;color:{sc};">{cnt}</div>
                </div>
                """, unsafe_allow_html=True)

        # Recent incidents table
        st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
        st.markdown('<div style="font-size:11px;font-weight:700;color:#6b7280;'
                    'text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">'
                    'Recent Similar Incidents</div>', unsafe_allow_html=True)

        hist_show = (
            similar[["start_datetime","corridor","severity","duration_minutes","requires_road_closure"]]
            .dropna(subset=["duration_minutes"])
            .sort_values("duration_minutes", ascending=False)
            .head(6)
        )
        hist_show["start_datetime"]       = pd.to_datetime(
            hist_show["start_datetime"]).dt.strftime("%d %b %Y")
        hist_show["duration_minutes"]     = hist_show["duration_minutes"].apply(
            lambda x: f"{x:.0f} min")
        hist_show["requires_road_closure"]= hist_show["requires_road_closure"].map(
            {1:"Yes",0:"No",True:"Yes",False:"No"})
        hist_show.columns = ["Date","Corridor","Severity","Duration","Closed"]

        for _, hrow in hist_show.iterrows():
            sc = sev_colors.get(hrow["Severity"], "#6b7280")
            closed_color = "#b91c1c" if hrow["Closed"]=="Yes" else "#15803d"
            st.markdown(f"""
            <div style="display:flex;justify-content:space-between;align-items:center;
                        padding:7px 10px;background:#ffffff;border:1px solid #e5e7eb;
                        border-radius:7px;margin-bottom:4px;font-size:11px;">
                <div>
                    <span style="font-weight:600;color:#111827;">{hrow['Corridor']}</span>
                    <span style="color:#9ca3af;margin-left:6px;">{hrow['Date']}</span>
                </div>
                <div style="display:flex;gap:8px;align-items:center;flex-shrink:0;">
                    <span style="color:{sc};font-weight:700;">{hrow['Severity']}</span>
                    <span style="color:#374151;">{hrow['Duration']}</span>
                    <span style="color:{closed_color};font-weight:600;">
                        {'🔒' if hrow['Closed']=='Yes' else '✓'} {hrow['Closed']}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)