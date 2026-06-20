"""
Predictive Risk Forecast — full page redesign
Filters on page, corridor grid, tactical brief, meaningful charts
"""
import sys
from pathlib import Path
from datetime import datetime

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import folium
from folium.plugins import HeatMap
from streamlit_folium import st_folium
import plotly.graph_objects as go

from utils.theme import (apply_theme, sidebar_brand, page_header,
                         section_header, kpi_card, C, PLOTLY,
                         AXIS_STYLE, BASE_MARGIN, weather_widget)
from utils.weather import fetch_weather, duration_multiplier

st.set_page_config(page_title="Risk Forecast · TrafficSense", page_icon="🔮", layout="wide")
apply_theme()

# ── Extra CSS for this page ────────────────────────────────────────────────────
st.markdown("""
<style>
/* tighter column gaps */
div[data-testid="column"] { padding: 0 4px !important; }
/* filter row card */
.filter-row { background:#fff;border:1px solid #e5e7eb;border-radius:12px;
              padding:16px 20px;margin-bottom:20px;
              box-shadow:0 1px 4px rgba(0,0,0,.05); }
</style>
""", unsafe_allow_html=True)

BENGALURU   = (12.9716, 77.5946)
CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"

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
    "Kanakapura Road":      (12.9068, 77.5736),
    "Sarjapur Road":        (12.9239, 77.6384),
    "MG Road":              (12.9738, 77.6119),
    "Sankey Road":          (13.0022, 77.5815),
    "Outer Ring Road West": (12.9734, 77.5126),
}

HOTSPOTS = [
    ("Mekri Circle",         13.0134, 77.5821, 64),
    ("Ayyappa Temple Junc",  13.0298, 77.5459, 49),
    ("Satellite Bus Stand",  12.9540, 77.5381, 43),
    ("Yeshwanthpura Circle", 13.0260, 77.5501, 38),
    ("Yelahanka Circle",     13.1008, 77.5963, 34),
    ("Silk Board Junc",      12.9177, 77.6238, 33),
    ("Hebbal Junc",          13.0355, 77.5968, 30),
    ("KR Puram Junc",        13.0078, 77.6950, 28),
    ("Tin Factory Junc",     12.9990, 77.6610, 26),
    ("Goraguntepalya Junc",  13.0198, 77.5199, 24),
]

EVENT_CAUSES = [
    "vehicle_breakdown","accident","congestion","tree_fall","water_logging",
    "pot_holes","construction","road_conditions","public_event","procession",
    "vip_movement","protest","debris","fog_low_visibility","others",
]

QUICK_WINDOWS = {
    "Early Morning  04–06": (4,  6),
    "Morning Peak   07–10": (7,  10),
    "Midday         11–14": (11, 14),
    "Afternoon      14–17": (14, 17),
    "Evening Peak   17–21": (17, 21),
    "Night          21–24": (21, 23),
    "Late Night     00–04": (0,  4),
    "Custom":               None,
}

RISK_PALETTE = {"Low":"#15803d","Moderate":"#d97706","High":"#ea580c","Critical":"#b91c1c"}
RISK_BG      = {"Low":"#f0fdf4","Moderate":"#fffbeb","High":"#fff7ed","Critical":"#fef2f2"}
RISK_BORDER  = {"Low":"#bbf7d0","Moderate":"#fde68a","High":"#fed7aa","Critical":"#fecaca"}

def _rlabel(s):
    if s >= 75: return "Critical"
    if s >= 50: return "High"
    if s >= 30: return "Moderate"
    return "Low"

def _rcolor(s): return RISK_PALETTE[_rlabel(s)]

# ── Sidebar (brand only) ───────────────────────────────────────────────────────
with st.sidebar:
    sidebar_brand()
    show_hotspots    = st.toggle("Historical Hotspots",    value=True)
    show_deploy      = st.toggle("Deployment Zones",       value=True)
    show_heatmap_lyr = st.toggle("Incident Density Layer", value=True)

# ── Page header ────────────────────────────────────────────────────────────────
now      = datetime.now()
cur_hour = now.hour
cur_dow  = now.weekday()

page_header("🔮", "Predictive Risk Forecast",
            "Set your shift window below — the map and rankings update instantly")

# ── FILTER ROW (on page, not sidebar) ─────────────────────────────────────────
st.markdown('<div class="filter-row">', unsafe_allow_html=True)

fc1, fc2, fc3, fc4, fc5 = st.columns([1.4, 0.7, 0.7, 1.2, 1.2])

with fc1:
    quick_sel = st.selectbox(
        "⏱ Time Window", list(QUICK_WINDOWS.keys()), index=4,
        help="Pick a preset or choose Custom to enter your own hours",
    )

with fc2:
    if quick_sel == "Custom":
        start_h = st.number_input("From (hr)", 0, 23, cur_hour, step=1, format="%02d")
    else:
        start_h = QUICK_WINDOWS[quick_sel][0]
        st.metric("From", f"{start_h:02d}:00")

with fc3:
    if quick_sel == "Custom":
        end_h = st.number_input("To (hr)", 0, 23, (cur_hour + 2) % 24, step=1, format="%02d")
    else:
        end_h = QUICK_WINDOWS[quick_sel][1]
        st.metric("To", f"{end_h:02d}:00")

with fc4:
    day_options = ["Today (auto)", "Monday", "Tuesday", "Wednesday",
                   "Thursday", "Friday", "Saturday", "Sunday",
                   "Weekdays", "Weekends"]
    # auto-index today
    auto_idx = 0
    day_sel  = st.selectbox("📅 Day", day_options, index=auto_idx)
    day_name = day_sel if day_sel != "Today (auto)" else now.strftime("%A")

with fc5:
    all_causes = st.checkbox("All event causes", value=True)
    if not all_causes:
        cause_sel = st.multiselect("Causes", EVENT_CAUSES, default=EVENT_CAUSES,
                                   label_visibility="collapsed")
    else:
        cause_sel = EVENT_CAUSES

st.markdown('</div>', unsafe_allow_html=True)

# ── Resolve filters ────────────────────────────────────────────────────────────
DOW_MAP = {"Monday":0,"Tuesday":1,"Wednesday":2,"Thursday":3,
           "Friday":4,"Saturday":5,"Sunday":6}
if day_sel == "Today (auto)":
    day_filter = "weekend" if cur_dow >= 5 else "weekday"
elif day_sel == "Weekdays": day_filter = "weekday"
elif day_sel == "Weekends": day_filter = "weekend"
else:                       day_filter = str(DOW_MAP[day_sel])

if end_h > start_h:
    window_hours = tuple(range(start_h, end_h + 1))
else:
    window_hours = tuple(list(range(start_h, 24)) + list(range(0, end_h + 1)))

# ── Data ───────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not CLEANED_CSV.exists():
        return None
    df = pd.read_csv(CLEANED_CSV, low_memory=False,
                     usecols=lambda c: c in [
                         "corridor","severity","requires_road_closure",
                         "duration_minutes","event_cause","start_datetime",
                         "latitude","longitude",
                     ])
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]           = df["start_datetime"].dt.hour
    df["day_of_week"]    = df["start_datetime"].dt.dayofweek
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower()
        .map({"true":1,"1":1,"false":0,"0":0}).fillna(0)
    )
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["sev_score"]        = df["severity"].map(
        {"Critical":4,"High":3,"Medium":2,"Low":1}).fillna(1)
    return df


@st.cache_data(show_spinner=False)
def compute_risk(df, window_hours, day_filter, causes):
    if day_filter == "weekday":   dff = df[df["day_of_week"] < 5].copy()
    elif day_filter == "weekend": dff = df[df["day_of_week"] >= 5].copy()
    else:                         dff = df[df["day_of_week"] == int(day_filter)].copy()
    dff = dff[dff["hour"].isin(window_hours)]
    if causes: dff = dff[dff["event_cause"].isin(causes)]
    n_hours = max(len(set(window_hours)), 1)
    rows = []
    for corridor in CORRIDOR_COORDS:
        cdata = dff[dff["corridor"] == corridor]
        if len(cdata) == 0:
            rows.append(dict(corridor=corridor, risk_score=0.0, incident_rate=0.0,
                             closure_rate=0.0, high_crit_rate=0.0, avg_duration=0.0,
                             incident_count=0, most_common_cause="—",
                             peak_hour=start_h, no_data=True,
                             recommended_constables=2, recommended_barricades=1,
                             risk_label="Low", total=0, high_crit=0))
            continue
        incident_rate  = len(cdata) / n_hours
        closure_rate   = float(cdata["requires_road_closure"].mean())
        high_crit_rate = float(cdata["severity"].isin(["High","Critical"]).mean())
        avg_duration   = float(cdata["duration_minutes"].dropna().mean() or 0)
        most_common    = cdata["event_cause"].value_counts().index[0]
        peak_hr        = int(cdata.groupby("hour").size().idxmax())
        rows.append(dict(
            corridor=corridor, incident_rate=incident_rate, closure_rate=closure_rate,
            high_crit_rate=high_crit_rate, avg_duration=avg_duration,
            incident_count=len(cdata), most_common_cause=most_common,
            peak_hour=peak_hr, no_data=False, total=len(cdata),
            high_crit=int(cdata["severity"].isin(["High","Critical"]).sum()),
        ))
    rdf = pd.DataFrame(rows)
    mx_r = rdf["incident_rate"].max() or 1.0
    mx_d = rdf["avg_duration"].max()  or 1.0
    rdf["risk_score"] = (
        0.35 * (rdf["incident_rate"] / mx_r) +
        0.30 * rdf["closure_rate"] +
        0.20 * rdf["high_crit_rate"] +
        0.15 * (rdf["avg_duration"] / mx_d)
    ).clip(0, 1) * 100
    rdf["risk_score"]             = rdf["risk_score"].round(1)
    rdf["recommended_constables"] = (2 + (rdf["risk_score"] / 100) * 10).astype(int)
    rdf["recommended_barricades"] = (1 + (rdf["risk_score"] / 100) * 7).astype(int)
    rdf["risk_label"]             = rdf["risk_score"].apply(_rlabel)
    return rdf.sort_values("risk_score", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner=False)
def compute_hourly_trend(df, top_corridors, trend_hours, day_filter):
    if day_filter == "weekday":   dff = df[df["day_of_week"] < 5]
    elif day_filter == "weekend": dff = df[df["day_of_week"] >= 5]
    else:                         dff = df[df["day_of_week"] == int(day_filter)]
    rows = []
    for h in trend_hours:
        hour_df = dff[dff["hour"] == h]
        for corridor in top_corridors:
            cdata   = hour_df[hour_df["corridor"] == corridor]
            n_total = max(len(dff[dff["corridor"] == corridor]), 1)
            if len(cdata) == 0:
                score = 0.0
            else:
                closure = float(cdata["requires_road_closure"].mean())
                hc      = float(cdata["severity"].isin(["High","Critical"]).mean())
                score   = min(
                    (0.45*(len(cdata)/n_total*10) + 0.35*closure + 0.20*hc)*100, 100)
            rows.append({"hour":h, "corridor":corridor, "score":round(score,1)})
    return pd.DataFrame(rows)


# Load
df      = load_data()
data_ok = df is not None
if not data_ok:
    st.warning("Processed data not found — run `python run_pipeline.py` first.")

risk_df = (
    compute_risk(df, window_hours, day_filter, tuple(cause_sel))
    if data_ok else
    pd.DataFrame([{"corridor":c,"risk_score":0,"risk_label":"Low",
                   "recommended_constables":2,"recommended_barricades":1,
                   "peak_hour":start_h,"most_common_cause":"—","no_data":True,
                   "total":0,"high_crit":0,"closure_rate":0,"incident_count":0}
                  for c in CORRIDOR_COORDS])
)

weather = fetch_weather(*BENGALURU)

# ── Weather strip ──────────────────────────────────────────────────────────────
if weather:
    weather_widget(weather)

# ── KPI row ────────────────────────────────────────────────────────────────────
high_crit_count  = int((risk_df["risk_score"] >= 50).sum())
top_row          = risk_df.iloc[0] if len(risk_df) else {}
top_corridor     = top_row.get("corridor","—")
top_score        = top_row.get("risk_score", 0)
total_constables = int(risk_df["recommended_constables"].sum())
total_incidents  = int(risk_df["total"].sum())

k1,k2,k3,k4 = st.columns(4)
for col, icon, lbl, val, accent in [
    (k1,"⚠️","High / Critical Corridors",
     str(high_crit_count),
     C["error"] if high_crit_count >= 3 else C["warning"] if high_crit_count else C["success"]),
    (k2,"🛣️","Highest Risk Corridor",    top_corridor,            C["error"]),
    (k3,"📊","Max Risk Score",            f"{top_score:.0f} / 100",C["warning"]),
    (k4,"👮","Total Constables Needed",   str(total_constables),   C["primary"]),
]:
    col.markdown(kpi_card(icon, lbl, val, accent=accent), unsafe_allow_html=True)

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# CORRIDOR GRID — replaces the long right-panel scroll
# ═══════════════════════════════════════════════════════════════════════════════
section_header("Corridor Risk Overview", "🛣️")

# Split into corridors with data vs without
has_data = risk_df[~risk_df["no_data"].fillna(False)]
no_data  = risk_df[risk_df["no_data"].fillna(False)]

# Render corridors with data in a 5-column grid
cols_per_row = 5
rows         = [has_data.iloc[i:i+cols_per_row]
                for i in range(0, len(has_data), cols_per_row)]

for row_df in rows:
    cols = st.columns(cols_per_row)
    for ci, (_, r) in enumerate(row_df.iterrows()):
        score  = r["risk_score"]
        label  = r["risk_label"]
        color  = RISK_PALETTE[label]
        bg     = RISK_BG[label]
        border = RISK_BORDER[label]
        cause  = str(r.get("most_common_cause","—")).replace("_"," ").title()
        peak_h = int(r.get("peak_hour", start_h))
        nc, nb = r["recommended_constables"], r["recommended_barricades"]

        cols[ci].markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:12px 13px;border-top:3px solid {color};
                    box-shadow:0 1px 3px rgba(0,0,0,.04);height:100%;">
            <div style="font-size:12px;font-weight:700;color:#111827;
                        margin-bottom:4px;line-height:1.3;">{r['corridor']}</div>
            <div style="display:flex;align-items:baseline;gap:4px;margin-bottom:6px;">
                <span style="font-size:22px;font-weight:800;color:{color};
                             line-height:1;">{score:.0f}</span>
                <span style="font-size:10px;color:{color};font-weight:600;">/100</span>
                <span style="font-size:10px;background:{color}20;color:{color};
                             font-weight:700;padding:1px 6px;border-radius:99px;
                             margin-left:2px;">{label}</span>
            </div>
            <div style="background:rgba(0,0,0,.08);border-radius:99px;
                        height:3px;margin-bottom:8px;">
                <div style="background:{color};width:{score}%;height:3px;
                            border-radius:99px;"></div>
            </div>
            <div style="font-size:10px;color:#6b7280;line-height:1.6;">
                Peak {peak_h:02d}:00 &middot; {cause}<br>
                👮 {nc} &nbsp;🚧 {nb}
            </div>
        </div>
        """, unsafe_allow_html=True)
    # fill empty cells in last row
    for ci in range(len(row_df), cols_per_row):
        cols[ci].markdown("")

if not no_data.empty:
    st.markdown(
        f'<p style="font-size:11px;color:#9ca3af;margin-top:6px;">'
        f'{len(no_data)} corridor(s) have no historical data for this window.</p>',
        unsafe_allow_html=True,
    )

st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# MAP + BRIEF side by side
# ═══════════════════════════════════════════════════════════════════════════════
col_map, col_brief = st.columns([1.6, 1], gap="large")

with col_map:
    section_header("Tactical Risk Map", "🗺️")
    m = folium.Map(location=[12.9716, 77.5946], zoom_start=11, tiles="CartoDB positron")

    if show_heatmap_lyr and data_ok and "latitude" in df.columns:
        heat_data = (df[df["hour"].isin(window_hours)][["latitude","longitude"]]
                     .dropna().values.tolist())
        if heat_data:
            HeatMap(heat_data, radius=14, blur=18, max_zoom=13,
                    gradient={0.0:"#dbeafe",0.4:"#93c5fd",
                               0.7:"#f59e0b",1.0:"#b91c1c"}).add_to(m)

    for _, row in risk_df.iterrows():
        coords = CORRIDOR_COORDS.get(row["corridor"])
        if not coords: continue
        score  = row["risk_score"]
        color  = _rcolor(score)
        label  = _rlabel(score)
        cause  = str(row.get("most_common_cause","—")).replace("_"," ").title()
        peak_h = int(row.get("peak_hour", start_h))
        nc, nb = row["recommended_constables"], row["recommended_barricades"]

        popup_html = f"""
        <div style='font-family:Inter,sans-serif;min-width:200px;padding:4px;'>
            <div style='font-size:13px;font-weight:700;color:#111827;margin-bottom:6px;'>
                {row['corridor']}</div>
            <div style='display:flex;justify-content:space-between;margin-bottom:6px;'>
                <span style='color:#6b7280;font-size:12px;'>Risk Score</span>
                <span style='color:{color};font-size:16px;font-weight:800;'>{score:.0f}/100</span>
            </div>
            <div style='background:#f3f4f6;border-radius:4px;height:4px;margin-bottom:8px;'>
                <div style='background:{color};width:{score}%;height:4px;border-radius:4px;'>
                </div></div>
            <div style='font-size:11px;color:#374151;line-height:1.8;'>
                Peak: <b>{peak_h:02d}:00</b><br>
                Cause: <b>{cause}</b><br>
                Closure risk: <b>{row['closure_rate']*100:.0f}%</b><br>
                <hr style='border-color:#e5e7eb;margin:5px 0;'>
                <span style='color:#1a56db;font-weight:700;'>
                    Deploy: {nc} constables · {nb} barricades</span>
            </div>
        </div>"""

        folium.CircleMarker(
            location=list(coords),
            radius=max(10, min(30, score/3.5)),
            color=color, fill=True, fill_color=color, fill_opacity=0.2,
            weight=2.5,
            popup=folium.Popup(popup_html, max_width=240),
            tooltip=f"{row['corridor']} · {label} · {score:.0f}",
        ).add_to(m)

        # Score label
        folium.Marker(location=list(coords), icon=folium.DivIcon(
            html=(f'<div style="font-family:Inter,sans-serif;font-size:10px;font-weight:700;'
                  f'color:{color};margin-top:20px;margin-left:-16px;'
                  f'text-shadow:0 1px 3px #fff;">{score:.0f}</div>'),
            icon_size=(36,18), icon_anchor=(18,0),
        )).add_to(m)

        if show_deploy and score > 40:
            folium.CircleMarker(
                location=list(coords), radius=score/3.2,
                color="#1a56db", fill=False,
                weight=1.5, dash_array="6 4", opacity=0.5,
            ).add_to(m)
            folium.Marker(
                location=[coords[0]+0.004, coords[1]+0.005],
                icon=folium.DivIcon(
                    html=(f'<div style="background:#eff6ff;border:1px solid #bfdbfe;'
                          f'color:#1a56db;font-size:10px;font-weight:700;'
                          f'padding:2px 7px;border-radius:5px;white-space:nowrap;">👮 {nc}</div>'),
                    icon_size=(70,20), icon_anchor=(0,10),
                ),
            ).add_to(m)

    if show_hotspots:
        for name, lat, lon, count in HOTSPOTS:
            folium.Marker(
                location=[lat, lon],
                icon=folium.DivIcon(
                    html=(f'<div style="background:#fff;border:1.5px solid #fca5a5;'
                          f'color:#b91c1c;font-size:10px;font-weight:700;'
                          f'padding:2px 7px;border-radius:5px;white-space:nowrap;">⚑ {name}</div>'),
                    icon_size=(130,20), icon_anchor=(0,10),
                ),
                tooltip=f"{name}: {count} incidents",
            ).add_to(m)

    m.get_root().html.add_child(folium.Element("""
    <div style="position:fixed;bottom:24px;left:24px;z-index:9999;background:#fff;
                border:1px solid #e5e7eb;border-radius:10px;padding:10px 14px;
                font-family:Inter,sans-serif;font-size:11px;color:#374151;
                box-shadow:0 2px 8px rgba(0,0,0,.1);">
        <div style="font-weight:700;color:#111827;margin-bottom:6px;">Risk Level</div>
        <div style="display:flex;flex-direction:column;gap:4px;">
            <div><span style="color:#15803d;font-size:14px;">●</span>&ensp;Low (&lt;30)</div>
            <div><span style="color:#d97706;font-size:14px;">●</span>&ensp;Moderate (30–50)</div>
            <div><span style="color:#ea580c;font-size:14px;">●</span>&ensp;High (50–75)</div>
            <div><span style="color:#b91c1c;font-size:14px;">●</span>&ensp;Critical (&gt;75)</div>
            <div style="border-top:1px solid #f3f4f6;padding-top:4px;margin-top:2px;">
                <span style="color:#1a56db;">– –</span>&ensp;Deploy zone
            </div>
        </div>
    </div>"""))

    st_folium(m, height=500, use_container_width=True)

# ── SHIFT DEPLOYMENT BRIEF ─────────────────────────────────────────────────────
with col_brief:
    section_header("Shift Deployment Brief", "📋")

    priority = risk_df[risk_df["risk_score"] >= 50]
    monitor  = risk_df[(risk_df["risk_score"] >= 30) & (risk_df["risk_score"] < 50)
                       & (~risk_df["no_data"].fillna(False))]
    low      = risk_df[(risk_df["risk_score"] < 30)
                       & (~risk_df["no_data"].fillna(False))]

    # Window + day header card
    mult_val, w_warn = duration_multiplier(weather) if weather else (1.0, None)
    weather_note = f'<div style="margin-top:8px;padding:6px 10px;background:#fffbeb;border:1px solid #fcd34d;border-radius:6px;font-size:11px;color:#92400e;">⚠️ {w_warn}</div>' if w_warn else ""

    st.markdown(f"""
    <div style="background:#1a56db;border-radius:10px;padding:14px 16px;
                margin-bottom:14px;color:#fff;">
        <div style="font-size:11px;font-weight:600;opacity:.75;letter-spacing:.06em;
                    text-transform:uppercase;margin-bottom:4px;">Forecast Window</div>
        <div style="font-size:20px;font-weight:800;letter-spacing:-.3px;">
            {start_h:02d}:00 – {end_h:02d}:00</div>
        <div style="font-size:13px;opacity:.85;margin-top:2px;">{day_name}</div>
        {f'<div style="margin-top:8px;background:rgba(255,255,255,.15);border-radius:6px;padding:5px 10px;font-size:11px;">⚠️ {w_warn}</div>' if w_warn else ""}
    </div>
    """, unsafe_allow_html=True)

    # Priority section
    if not priority.empty:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin:0 0 8px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#b91c1c;flex-shrink:0;"></div>
            <span style="font-size:11px;font-weight:800;color:#b91c1c;
                         text-transform:uppercase;letter-spacing:.08em;">
                Priority Deployment
            </span>
        </div>
        """, unsafe_allow_html=True)
        for _, r in priority.head(6).iterrows():
            cause = str(r.get("most_common_cause","")).replace("_"," ").title()
            nc, nb = r["recommended_constables"], r["recommended_barricades"]
            score  = r["risk_score"]
            color  = _rcolor(score)
            st.markdown(f"""
            <div style="background:#fff;border:1px solid #fecaca;border-left:4px solid {color};
                        border-radius:8px;padding:10px 12px;margin-bottom:6px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:13px;font-weight:700;color:#111827;">
                        {r['corridor']}</span>
                    <span style="font-size:11px;font-weight:700;color:{color};
                                 background:{RISK_BG[r['risk_label']]};
                                 padding:2px 8px;border-radius:99px;">
                        {r['risk_label']} · {score:.0f}
                    </span>
                </div>
                <div style="display:flex;gap:12px;margin-top:6px;">
                    <div style="text-align:center;background:#eff6ff;border-radius:6px;
                                padding:4px 10px;flex:1;">
                        <div style="font-size:16px;font-weight:800;color:#1a56db;">{nc}</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;
                                    text-transform:uppercase;">Constables</div>
                    </div>
                    <div style="text-align:center;background:#fffbeb;border-radius:6px;
                                padding:4px 10px;flex:1;">
                        <div style="font-size:16px;font-weight:800;color:#d97706;">{nb}</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;
                                    text-transform:uppercase;">Barricades</div>
                    </div>
                    <div style="text-align:center;background:#f8fafc;border-radius:6px;
                                padding:4px 10px;flex:1;">
                        <div style="font-size:12px;font-weight:700;color:#374151;">
                            {int(r['peak_hour']):02d}:00</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;
                                    text-transform:uppercase;">Peak Hr</div>
                    </div>
                </div>
                <div style="font-size:10px;color:#9ca3af;margin-top:5px;">
                    Main cause: {cause}
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Monitor section
    if not monitor.empty:
        st.markdown("""
        <div style="display:flex;align-items:center;gap:8px;margin:10px 0 8px;">
            <div style="width:10px;height:10px;border-radius:50%;
                        background:#d97706;flex-shrink:0;"></div>
            <span style="font-size:11px;font-weight:800;color:#d97706;
                         text-transform:uppercase;letter-spacing:.08em;">
                Monitor Closely
            </span>
        </div>
        """, unsafe_allow_html=True)
        for _, r in monitor.head(4).iterrows():
            nc, nb = r["recommended_constables"], r["recommended_barricades"]
            st.markdown(f"""
            <div style="background:#fffbeb;border:1px solid #fde68a;
                        border-radius:8px;padding:9px 12px;margin-bottom:5px;
                        display:flex;justify-content:space-between;align-items:center;">
                <div>
                    <div style="font-size:12px;font-weight:600;color:#111827;">
                        {r['corridor']}</div>
                    <div style="font-size:10px;color:#92400e;margin-top:2px;">
                        Peak {int(r['peak_hour']):02d}:00
                    </div>
                </div>
                <div style="text-align:right;flex-shrink:0;margin-left:8px;">
                    <div style="font-size:12px;font-weight:700;color:#d97706;">
                        👮 {nc} &nbsp; 🚧 {nb}</div>
                    <div style="font-size:10px;color:#9ca3af;">
                        Risk {r['risk_score']:.0f}/100</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

    # Standard patrol
    if not low.empty:
        names = ", ".join(low["corridor"].tolist()[:6])
        more  = f" +{len(low)-6} more" if len(low) > 6 else ""
        st.markdown(f"""
        <div style="background:#f0fdf4;border:1px solid #bbf7d0;
                    border-radius:8px;padding:10px 12px;margin-top:8px;">
            <div style="display:flex;align-items:center;gap:6px;margin-bottom:4px;">
                <div style="width:8px;height:8px;border-radius:50%;
                            background:#15803d;"></div>
                <span style="font-size:11px;font-weight:800;color:#15803d;
                             text-transform:uppercase;letter-spacing:.06em;">
                    Standard Patrol — {len(low)} corridor(s)
                </span>
            </div>
            <div style="font-size:11px;color:#374151;line-height:1.6;">
                {names}{more}
            </div>
            <div style="font-size:10px;color:#15803d;margin-top:4px;font-weight:600;">
                Routine deployment · 2 constables each
            </div>
        </div>
        """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
# HOUR-BY-HOUR TREND — only top corridors with actual risk
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
section_header("When Does Risk Peak? — Hour-by-Hour Trend", "📈")

# Only show corridors that have data and meaningful risk
top5 = tuple(
    risk_df[~risk_df["no_data"].fillna(False)]
    .head(5)["corridor"].tolist()
)
trend_hrs = tuple((start_h + i) % 24 for i in range(6))

if data_ok and top5:
    trend_df  = compute_hourly_trend(df, top5, trend_hrs, day_filter)
    hr_labels = [f"{h:02d}:00" for h in trend_hrs]
    t_colors  = ["#1a56db","#b91c1c","#ea580c","#d97706","#15803d"]

    fig = go.Figure()

    # Threshold band
    fig.add_hrect(y0=50, y1=105, fillcolor="rgba(185,28,28,.04)",
                  line_width=0, layer="below")
    fig.add_hline(y=50, line=dict(color="#ea580c", dash="dot", width=1.5),
                  annotation_text="⚠ High Risk",
                  annotation_position="top left",
                  annotation_font=dict(color="#ea580c", size=11))

    for i, corridor in enumerate(top5):
        cdata  = trend_df[trend_df["corridor"] == corridor]
        scores = [
            float(cdata[cdata["hour"] == h]["score"].values[0])
            if len(cdata[cdata["hour"] == h]) > 0 else 0
            for h in trend_hrs
        ]
        # Area fill for the top corridor only — use rgba, not 8-digit hex
        fill      = "tozeroy" if i == 0 else "none"
        hex_color = t_colors[i].lstrip("#")
        r_val     = int(hex_color[0:2], 16)
        g_val     = int(hex_color[2:4], 16)
        b_val     = int(hex_color[4:6], 16)
        fill_rgba = f"rgba({r_val},{g_val},{b_val},0.07)"

        fig.add_trace(go.Scatter(
            x=hr_labels, y=scores,
            name=corridor,
            mode="lines+markers",
            fill=fill,
            fillcolor=fill_rgba,
            line=dict(color=t_colors[i], width=2.5),
            marker=dict(size=8, line=dict(color="#fff", width=1.5)),
            hovertemplate=f"<b>{corridor}</b><br>%{{x}}: Risk %{{y:.0f}}/100<extra></extra>",
        ))

    max_score = max(trend_df["score"].max() if not trend_df.empty else 0, 45)
    fig.update_layout(
        **PLOTLY, height=340,
        margin=dict(l=60, r=30, t=48, b=80),
        title=dict(
            text=f"Risk by hour · top corridors · {start_h:02d}:00 window",
            font=dict(size=13, color="#111827", weight=700),
            x=0, xanchor="left",
        ),
        xaxis_title="Hour of Day",
        yaxis_title="Risk Score (0–100)",
        yaxis_range=[0, min(max_score * 1.3, 105)],
        legend=dict(
            orientation="h", y=-0.32,
            font=dict(size=11, color="#374151"),
            bgcolor="rgba(0,0,0,0)",
        ),
    )
    fig.update_xaxes(**AXIS_STYLE)
    fig.update_yaxes(**AXIS_STYLE)
    st.plotly_chart(fig, config={"displayModeBar": False}, use_container_width=True)

    # Plain-English insight below the chart
    if not trend_df.empty:
        peak_row = trend_df.loc[trend_df["score"].idxmax()]
        st.markdown(
            f'<div style="background:#eff6ff;border:1px solid #bfdbfe;border-radius:8px;'
            f'padding:10px 14px;font-size:13px;color:#1a56db;margin-top:-8px;">'
            f'💡 <b>Peak risk hour:</b> {int(peak_row["hour"]):02d}:00 on '
            f'<b>{peak_row["corridor"]}</b> — pre-deploy constables by '
            f'{max(int(peak_row["hour"])-1, 0):02d}:30 to get ahead of congestion.'
            f'</div>',
            unsafe_allow_html=True,
        )
else:
    st.info("No trend data for the selected window.")

# ═══════════════════════════════════════════════════════════════════════════════
# CAUSE BREAKDOWN — two columns: chart + insight cards
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
section_header("What's Causing Incidents This Window", "🔍")

if data_ok:
    if day_filter == "weekday":   cause_base = df[df["day_of_week"] < 5]
    elif day_filter == "weekend": cause_base = df[df["day_of_week"] >= 5]
    else:                         cause_base = df[df["day_of_week"] == int(day_filter)]
    cause_base = cause_base[cause_base["hour"].isin(window_hours)]

    if not cause_base.empty:
        grp = cause_base.groupby("event_cause").agg(
            count        =("event_cause",           "count"),
            closure_rate =("requires_road_closure", "mean"),
            avg_duration =("duration_minutes",       "mean"),
            high_crit    =("severity", lambda x: x.isin(["High","Critical"]).mean()),
        ).reset_index()
        grp["sev"]         = 0.5*grp["closure_rate"] + 0.5*grp["high_crit"]
        grp["cause_label"] = grp["event_cause"].str.replace("_"," ").str.title()
        grp                = grp.nlargest(6, "count").reset_index(drop=True)

        cc1, cc2 = st.columns([1.6, 1], gap="large")

        with cc1:
            bar_colors = [
                "#b91c1c" if s>=0.5 else "#ea580c" if s>=0.3 else
                "#d97706" if s>=0.15 else "#15803d"
                for s in grp["sev"]
            ]
            # Compute avg_duration safely (may have NaN)
            grp["avg_duration_disp"] = grp["avg_duration"].fillna(0)

            fig2 = go.Figure(go.Bar(
                x=grp["count"], y=grp["cause_label"],
                orientation="h",
                marker_color=bar_colors, marker_line_width=0,
                text=[f"{r['count']} incidents" for _, r in grp.iterrows()],
                textposition="outside",
                textfont=dict(size=11, color="#111827"),
                hovertemplate=(
                    "<b>%{y}</b><br>Incidents: %{x}<br>"
                    "Closure rate: %{customdata[0]:.0%}<br>"
                    "Avg duration: %{customdata[1]:.0f} min<extra></extra>"
                ),
                customdata=grp[["closure_rate","avg_duration_disp"]].values,
            ))
            fig2.update_layout(
                **PLOTLY, height=300,
                margin=dict(l=160, r=130, t=40, b=30),
                title=dict(
                    text="Incident count by cause · bar color = severity",
                    font=dict(size=13, color="#111827", weight=700),
                    x=0, xanchor="left",
                ),
                xaxis_title="Incident count",
            )
            fig2.update_xaxes(**AXIS_STYLE)
            fig2.update_yaxes(**AXIS_STYLE, automargin=True)
            st.plotly_chart(fig2, config={"displayModeBar":False}, use_container_width=True)

        with cc2:
            st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)
            # Top 3 causes as insight cards
            for _, r in grp.head(3).iterrows():
                sev_score = r["sev"]
                c = ("#b91c1c" if sev_score>=0.5 else "#ea580c" if sev_score>=0.3
                     else "#d97706" if sev_score>=0.15 else "#15803d")
                bg = ("#fef2f2" if sev_score>=0.5 else "#fff7ed" if sev_score>=0.3
                      else "#fffbeb" if sev_score>=0.15 else "#f0fdf4")
                closure_pct = r["closure_rate"] * 100
                avg_dur     = r["avg_duration_disp"]
                st.markdown(f"""
                <div style="background:{bg};border:1px solid {c}30;
                            border-left:4px solid {c};border-radius:8px;
                            padding:10px 12px;margin-bottom:8px;">
                    <div style="font-size:12px;font-weight:700;color:#111827;
                                margin-bottom:6px;">{r['cause_label']}</div>
                    <div style="display:flex;gap:8px;">
                        <div style="flex:1;text-align:center;background:rgba(255,255,255,.7);
                                    border-radius:5px;padding:4px;">
                            <div style="font-size:14px;font-weight:800;color:{c};">
                                {r['count']}</div>
                            <div style="font-size:9px;color:#6b7280;font-weight:600;">
                                INCIDENTS</div>
                        </div>
                        <div style="flex:1;text-align:center;background:rgba(255,255,255,.7);
                                    border-radius:5px;padding:4px;">
                            <div style="font-size:14px;font-weight:800;color:{c};">
                                {closure_pct:.0f}%</div>
                            <div style="font-size:9px;color:#6b7280;font-weight:600;">
                                CLOSURES</div>
                        </div>
                        <div style="flex:1;text-align:center;background:rgba(255,255,255,.7);
                                    border-radius:5px;padding:4px;">
                            <div style="font-size:14px;font-weight:800;color:{c};">
                                {avg_dur:.0f}m</div>
                            <div style="font-size:9px;color:#6b7280;font-weight:600;">
                                AVG DUR</div>
                        </div>
                    </div>
                </div>
                """, unsafe_allow_html=True)
    else:
        st.info("No cause data available for the selected window.")