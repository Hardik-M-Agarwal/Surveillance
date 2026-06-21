import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

try:
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
except Exception:
    pass

import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import plotly.graph_objects as go

from utils.theme import (apply_theme, sidebar_brand, C, PLOTLY, BASE_MARGIN)
from utils.weather import fetch_weather, duration_multiplier

st.set_page_config(
    page_title="TrafficSense · Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

with st.sidebar:
    sidebar_brand()

# ── Axis helper ────────────────────────────────────────────────────────────────
def ax(title="", **kw):
    d = dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
             tickfont=dict(color="#111827", size=11),
             title_font=dict(color="#111827", size=12))
    if title: d["title"] = title
    d.update(kw)
    return d

# ── Constants ──────────────────────────────────────────────────────────────────
BENGALURU       = (12.9716, 77.5946)
CLEANED_CSV     = ROOT / "data" / "processed" / "cleaned.csv"
FEEDBACK_CSV    = ROOT / "data" / "processed" / "feedback_log.csv"
MIN_FEEDBACK    = 50
DAYS            = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
DAYS_SHORT      = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
RISK_PALETTE    = {"Low":"#15803d","Moderate":"#d97706","High":"#ea580c","Critical":"#b91c1c"}
RISK_BG         = {"Low":"#f0fdf4","Moderate":"#fffbeb","High":"#fff7ed","Critical":"#fef2f2"}
RISK_BORDER     = {"Low":"#bbf7d0","Moderate":"#fde68a","High":"#fed7aa","Critical":"#fecaca"}

CORRIDOR_COORDS = {
    "Mysore Road":       (12.9358,77.5264), "Bellary Road 1":  (13.0358,77.5800),
    "Tumkur Road":       (13.0154,77.5107), "Hosur Road":      (12.8893,77.6387),
    "ORR North 1":       (13.0604,77.6218), "Old Madras Road": (13.0032,77.6540),
    "Magadi Road":       (12.9683,77.5025), "Bellary Road 2":  (13.0558,77.5900),
    "ORR East 1":        (12.9450,77.6800), "Bannerghatta Road":(12.8735,77.5985),
}

def _rlabel(s):
    if s >= 70: return "Critical"
    if s >= 45: return "High"
    if s >= 25: return "Moderate"
    return "Low"

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_data():
    if not CLEANED_CSV.exists():
        return None
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]           = df["start_datetime"].dt.hour
    df["day_of_week"]    = df["start_datetime"].dt.dayofweek
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower()
        .map({"true":1,"1":1,"false":0,"0":0}).fillna(0)
    )
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["sev_score"]        = df["severity"].map(
        {"Critical":4,"High":3,"Medium":2,"Low":1}).fillna(2)
    return df


def load_feedback():
    if FEEDBACK_CSV.exists():
        try:
            fb = pd.read_csv(FEEDBACK_CSV)
            fb["timestamp"] = pd.to_datetime(fb["timestamp"], errors="coerce")
            return fb
        except Exception:
            pass
    return pd.DataFrame()


def compute_corridor_risk(df, window_hours, day_filter):
    """Reuse the same risk logic as Predictive Risk Forecast."""
    if day_filter == "weekday":   dff = df[df["day_of_week"] < 5]
    elif day_filter == "weekend": dff = df[df["day_of_week"] >= 5]
    else:                         dff = df[df["day_of_week"] == int(day_filter)]
    dff = dff[dff["hour"].isin(window_hours)]

    rows = []
    for corridor in CORRIDOR_COORDS:
        cdata = dff[dff["corridor"] == corridor]
        if len(cdata) == 0:
            rows.append(dict(corridor=corridor, risk_score=0, risk_label="Low",
                             no_data=True, incident_count=0,
                             recommended_constables=2, recommended_barricades=1,
                             most_common_cause="—", closure_rate=0))
            continue
        n_h            = max(len(set(window_hours)), 1)
        incident_rate  = len(cdata) / n_h
        closure_rate   = float(cdata["requires_road_closure"].mean())
        high_crit_rate = float(cdata["severity"].isin(["High","Critical"]).mean())
        avg_duration   = float(cdata["duration_minutes"].dropna().mean() or 0)
        most_common    = cdata["event_cause"].value_counts().index[0]
        rows.append(dict(
            corridor=corridor, incident_rate=incident_rate,
            closure_rate=closure_rate, high_crit_rate=high_crit_rate,
            avg_duration=avg_duration, incident_count=len(cdata),
            most_common_cause=most_common, no_data=False,
        ))

    rdf      = pd.DataFrame(rows)
    mx_r     = rdf["incident_rate"].max() or 1
    mx_d     = rdf.get("avg_duration", pd.Series([1])).max() or 1
    rdf["risk_score"] = (
        0.35*(rdf["incident_rate"]/mx_r) +
        0.30* rdf["closure_rate"] +
        0.20* rdf["high_crit_rate"] +
        0.15*(rdf.get("avg_duration", 0)/mx_d)
    ).clip(0,1) * 100
    rdf["risk_score"]             = rdf["risk_score"].fillna(0).round(1)
    rdf["risk_label"]             = rdf["risk_score"].apply(_rlabel)
    rdf["recommended_constables"] = (2+(rdf["risk_score"]/100)*10).fillna(2).astype(int)
    rdf["recommended_barricades"] = (1+(rdf["risk_score"]/100)*7).fillna(1).astype(int)
    return rdf.sort_values("risk_score", ascending=False).reset_index(drop=True)


# ── Runtime context ────────────────────────────────────────────────────────────
now       = datetime.now()
cur_hour  = now.hour
cur_dow   = now.weekday()
day_name  = DAYS[cur_dow]
day_filter= "weekend" if cur_dow >= 5 else "weekday"

if 6 <= cur_hour < 14:   shift_name, shift_icon = "Morning Shift  06:00–14:00",  "🌅"
elif 14 <= cur_hour < 22: shift_name, shift_icon = "Afternoon Shift  14:00–22:00","☀️"
else:                     shift_name, shift_icon = "Night Shift  22:00–06:00",    "🌙"

window_hours = tuple((cur_hour + i) % 24 for i in range(3))

df       = load_data()
feedback = load_feedback()
weather  = fetch_weather(*BENGALURU)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 1 — LIVE STATUS BAR
# ══════════════════════════════════════════════════════════════════════════════
weather_str  = ""
weather_warn = ""
if weather:
    cond = weather.get("condition","")
    temp = weather.get("temp", 0)
    vis  = weather.get("visibility", 10)
    _, warn = duration_multiplier(weather)
    weather_str  = f"{cond} · {temp:.0f}°C · Vis {vis:.1f} km"
    weather_warn = warn or ""

data_status  = "✅ Online" if df is not None else "⚠️ No Data"
fb_count     = len(feedback)
fb_status    = f"{fb_count}/{MIN_FEEDBACK} this cycle"

# Build status bar as list of HTML parts to avoid f-string quoting issues
_sb_parts = [
    '<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;'
    'padding:10px 20px;margin-bottom:18px;display:flex;align-items:center;gap:0;'
    'box-shadow:0 1px 3px rgba(0,0,0,.04);">',

    # System Online
    '<div style="display:flex;align-items:center;gap:8px;padding-right:20px;'
    'border-right:1px solid #f3f4f6;margin-right:20px;">'
    '<span style="width:8px;height:8px;border-radius:50%;background:#15803d;'
    'display:inline-block;flex-shrink:0;"></span>'
    '<span style="font-size:12px;font-weight:700;color:#15803d;">System Online</span>'
    '</div>',

    # Shift
    '<div style="display:flex;align-items:center;gap:6px;padding-right:20px;'
    'border-right:1px solid #f3f4f6;margin-right:20px;">'
    '<span style="font-size:14px;">' + shift_icon + '</span>'
    '<span style="font-size:12px;font-weight:600;color:#111827;">' + shift_name + '</span>'
    '<span style="font-size:12px;color:#9ca3af;margin-left:4px;">' + day_name + '</span>'
    '</div>',

    # Time
    '<div style="display:flex;align-items:center;gap:6px;padding-right:20px;'
    'border-right:1px solid #f3f4f6;margin-right:20px;">'
    '<span style="font-size:13px;">' + chr(128336) + '</span>'
    '<span style="font-size:12px;font-weight:600;color:#111827;">'
    + now.strftime("%H:%M") + '</span>'
    '<span style="font-size:12px;color:#9ca3af;">'
    ' Forecast: ' + str(cur_hour).zfill(2) + ':00–'
    + str((cur_hour+3)%24).zfill(2) + ':00</span>'
    '</div>',
]

# Weather block (only if available)
if weather_str:
    _warn_badge = (
        '<span style="background:#fffbeb;border:1px solid #fcd34d;color:#92400e;'
        'font-size:10px;font-weight:600;padding:1px 7px;border-radius:4px;margin-left:4px;">'
        + chr(9888) + ' ' + weather_warn[:40] + '</span>'
        if weather_warn else ""
    )
    _sb_parts.append(
        '<div style="display:flex;align-items:center;gap:6px;padding-right:20px;'
        'border-right:1px solid #f3f4f6;margin-right:20px;">'
        '<span style="font-size:13px;">' + chr(127780) + '</span>'
        '<span style="font-size:12px;color:#374151;">' + weather_str + '</span>'
        + _warn_badge +
        '</div>'
    )

# Right side
_sb_parts.append(
    '<div style="margin-left:auto;display:flex;align-items:center;gap:16px;">'
    '<span style="font-size:11px;color:#6b7280;">Data: '
    '<b style="color:#111827;">' + data_status + '</b></span>'
    '<span style="font-size:11px;color:#6b7280;">Feedback: '
    '<b style="color:#1a56db;">' + fb_status + '</b></span>'
    '</div>'
)
_sb_parts.append('</div>')

st.markdown("".join(_sb_parts), unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 2 — TODAY'S INTELLIGENCE BRIEF
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
            letter-spacing:.09em;margin-bottom:12px;">
    📋 Today's Intelligence Brief
</div>""", unsafe_allow_html=True)

if df is not None:
    # Filter to today's day-of-week + current hour window historically
    today_mask  = (df["day_of_week"] == cur_dow) & (df["hour"].isin(window_hours))
    today_df    = df[today_mask]
    n_today     = len(today_df)
    top_cause   = (today_df["event_cause"].value_counts().index[0].replace("_"," ").title()
                   if n_today > 0 else "—")
    top_corr    = (today_df[today_df["corridor"]!="Non-corridor"]["corridor"]
                   .value_counts().index[0] if n_today > 0 else "—")
    cl_rate     = today_df["requires_road_closure"].mean()*100 if n_today > 0 else 0
    hc_rate     = (today_df["severity"].isin(["High","Critical"]).mean()*100
                   if n_today > 0 and "severity" in today_df.columns else 0)
    avg_dur     = today_df["duration_minutes"].dropna().mean() if n_today > 0 else 0
    avg_dur     = avg_dur if not np.isnan(avg_dur) else 0

    # Risk level for this window
    window_risk = "High" if hc_rate > 40 else "Moderate" if hc_rate > 20 else "Low"
    wr_color    = RISK_PALETTE[window_risk]
    wr_bg       = RISK_BG[window_risk]
    wr_border   = RISK_BORDER[window_risk]

    b1,b2,b3,b4,b5 = st.columns(5)
    for col, icon, lbl, val, acc in [
        (b1,"📊","Hist. Incidents\nThis Window",  f"{n_today:,}",        C["primary"]),
        (b2,"🎯","High+Critical Rate",            f"{hc_rate:.0f}%",
         C["error"] if hc_rate>40 else C["warning"] if hc_rate>20 else C["success"]),
        (b3,"🔒","Closure Rate",                  f"{cl_rate:.0f}%",
         C["error"] if cl_rate>30 else C["warning"] if cl_rate>15 else C["success"]),
        (b4,"🔝","Top Cause Today",               top_cause,             C["primary"]),
        (b5,"🛣️","Highest Risk Corridor",         top_corr,              C["error"]),
    ]:
        col.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                    padding:13px 14px;border-top:3px solid {acc};
                    box-shadow:0 1px 3px rgba(0,0,0,.04);margin-bottom:16px;">
            <div style="font-size:9px;font-weight:700;color:#6b7280;
                        text-transform:uppercase;letter-spacing:.08em;margin-bottom:5px;
                        white-space:pre-line;">{lbl}</div>
            <div style="font-size:17px;font-weight:700;color:#111827;line-height:1.2;">
                {val}</div>
        </div>""", unsafe_allow_html=True)

    # Narrative brief card
    if avg_dur > 0:
        dur_note = f"Avg clearance time is <b>{avg_dur:.0f} min</b>."
    else:
        dur_note = ""
    if weather_warn:
        wx_note = f' <span style="color:#b45309;">Weather advisory: {weather_warn}</span>'
    else:
        wx_note = ""

    st.markdown(f"""
    <div style="background:{wr_bg};border:1px solid {wr_border};
                border-left:4px solid {wr_color};border-radius:10px;
                padding:13px 18px;margin-bottom:20px;font-size:13px;color:#374151;
                line-height:1.8;">
        <b style="color:{wr_color};">📋 {day_name} {cur_hour:02d}:00–{(cur_hour+3)%24:02d}:00
        · {window_risk} Risk Window</b><br>
        Historically, this time slot on {day_name}s sees
        <b>{n_today:,} incidents</b> in the training data,
        with <b>{hc_rate:.0f}%</b> being High or Critical severity
        and a <b>{cl_rate:.0f}%</b> road closure rate.
        Most common cause: <b>{top_cause}</b>.
        Corridor needing most attention: <b>{top_corr}</b>.
        {dur_note}{wx_note}
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Run `python run_pipeline.py` to enable the intelligence brief.")

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 3 — CORRIDOR ALERT GRID
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
            letter-spacing:.09em;margin-bottom:12px;">
    🗺️ Corridor Risk — Next 3 Hours
</div>""", unsafe_allow_html=True)

if df is not None:
    risk_df = compute_corridor_risk(df, window_hours, day_filter)

    # Show corridors with data first, sorted by risk
    has_data = risk_df[~risk_df["no_data"].fillna(False)].head(10)
    cols     = st.columns(5)

    for i, (_, row) in enumerate(has_data.iterrows()):
        col    = cols[i % 5]
        score  = row["risk_score"]
        label  = row["risk_label"]
        color  = RISK_PALETTE[label]
        bg     = RISK_BG[label]
        border = RISK_BORDER[label]
        cause  = str(row.get("most_common_cause","—")).replace("_"," ").title()
        nc     = row["recommended_constables"]

        col.markdown(f"""
        <div style="background:{bg};border:1px solid {border};border-radius:10px;
                    padding:12px 13px;border-top:3px solid {color};
                    box-shadow:0 1px 3px rgba(0,0,0,.04);margin-bottom:12px;">
            <div style="display:flex;justify-content:space-between;align-items:flex-start;
                        margin-bottom:6px;">
                <div style="font-size:12px;font-weight:700;color:#111827;
                            line-height:1.3;flex:1;margin-right:4px;">
                    {row['corridor']}</div>
                <div style="background:{color};color:#fff;font-size:9px;font-weight:700;
                            padding:2px 7px;border-radius:99px;white-space:nowrap;
                            flex-shrink:0;">{label}</div>
            </div>
            <div style="font-size:22px;font-weight:800;color:{color};
                        line-height:1;margin-bottom:6px;">{score:.0f}
                <span style="font-size:11px;font-weight:500;color:{color};">/100</span>
            </div>
            <div style="background:rgba(0,0,0,.07);border-radius:99px;
                        height:3px;margin-bottom:7px;">
                <div style="background:{color};width:{score:.0f}%;
                            height:3px;border-radius:99px;"></div>
            </div>
            <div style="font-size:10px;color:#6b7280;line-height:1.5;">
                {cause}<br>
                <span style="color:#1a56db;font-weight:600;">👮 {nc} constables</span>
            </div>
        </div>""", unsafe_allow_html=True)

    # Second row if more than 5
    if len(has_data) > 5:
        cols2 = st.columns(5)
        for i, (_, row) in enumerate(has_data.iloc[5:].iterrows()):
            col    = cols2[i % 5]
            score  = row["risk_score"]
            label  = row["risk_label"]
            color  = RISK_PALETTE[label]
            bg     = RISK_BG[label]
            border = RISK_BORDER[label]
            cause  = str(row.get("most_common_cause","—")).replace("_"," ").title()
            nc     = row["recommended_constables"]
            col.markdown(f"""
            <div style="background:{bg};border:1px solid {border};border-radius:10px;
                        padding:12px 13px;border-top:3px solid {color};
                        box-shadow:0 1px 3px rgba(0,0,0,.04);margin-bottom:12px;">
                <div style="display:flex;justify-content:space-between;align-items:flex-start;
                            margin-bottom:6px;">
                    <div style="font-size:12px;font-weight:700;color:#111827;
                                line-height:1.3;flex:1;margin-right:4px;">
                        {row['corridor']}</div>
                    <div style="background:{color};color:#fff;font-size:9px;font-weight:700;
                                padding:2px 7px;border-radius:99px;white-space:nowrap;
                                flex-shrink:0;">{label}</div>
                </div>
                <div style="font-size:22px;font-weight:800;color:{color};
                            line-height:1;margin-bottom:6px;">{score:.0f}
                    <span style="font-size:11px;font-weight:500;color:{color};">/100</span>
                </div>
                <div style="background:rgba(0,0,0,.07);border-radius:99px;
                            height:3px;margin-bottom:7px;">
                    <div style="background:{color};width:{score:.0f}%;
                                height:3px;border-radius:99px;"></div>
                </div>
                <div style="font-size:10px;color:#6b7280;line-height:1.5;">
                    {cause}<br>
                    <span style="color:#1a56db;font-weight:600;">👮 {nc} constables</span>
                </div>
            </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 4 — UPCOMING EVENT WARNING + RECENT ACTIVITY (side by side)
# ══════════════════════════════════════════════════════════════════════════════
col_warn, col_feed = st.columns([1, 1], gap="large")

with col_warn:
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:.09em;margin-bottom:12px;">
        ⚡ Upcoming Risk Patterns
    </div>""", unsafe_allow_html=True)

    if df is not None:
        # Look at the next 4 hours and find which cause+corridor combos are historically
        # most dangerous in that window on this day type
        next_hours = [(cur_hour + i) % 24 for i in range(1, 5)]
        nxt_mask   = (df["day_of_week"] == cur_dow) & (df["hour"].isin(next_hours))
        nxt_df     = df[nxt_mask]

        if len(nxt_df) > 0:
            # Top 3 cause+corridor combos by closure rate
            combo = (nxt_df[nxt_df["corridor"]!="Non-corridor"]
                     .groupby(["corridor","event_cause"])
                     .agg(count=("event_cause","count"),
                          closure_rate=("requires_road_closure","mean"),
                          hc_rate=("severity",
                                   lambda x: x.isin(["High","Critical"]).mean()))
                     .reset_index())
            combo = combo[combo["count"] >= 3].copy()
            combo["danger_score"] = (0.4*combo["closure_rate"] +
                                     0.4*combo["hc_rate"] +
                                     0.2*(combo["count"]/combo["count"].max()))
            combo = combo.nlargest(4, "danger_score")

            for _, row in combo.iterrows():
                cause_clean = row["event_cause"].replace("_"," ").title()
                cl_pct      = row["closure_rate"]*100
                hc_pct      = row["hc_rate"]*100
                danger      = row["danger_score"]
                d_color     = "#b91c1c" if danger>0.5 else "#ea580c" if danger>0.3 else "#d97706"
                d_bg        = "#fef2f2" if danger>0.5 else "#fff7ed" if danger>0.3 else "#fffbeb"
                d_border    = "#fecaca" if danger>0.5 else "#fed7aa" if danger>0.3 else "#fde68a"
                st.markdown(f"""
                <div style="background:{d_bg};border:1px solid {d_border};
                            border-radius:9px;padding:10px 13px;margin-bottom:8px;
                            display:flex;align-items:center;gap:12px;">
                    <div style="flex:1;">
                        <div style="font-size:12px;font-weight:700;color:#111827;">
                            {row['corridor']}</div>
                        <div style="font-size:11px;color:#6b7280;margin-top:2px;">
                            {cause_clean} · {row['count']} historical incidents</div>
                    </div>
                    <div style="text-align:right;flex-shrink:0;">
                        <div style="font-size:11px;font-weight:700;color:{d_color};">
                            {cl_pct:.0f}% closure</div>
                        <div style="font-size:10px;color:#9ca3af;">{hc_pct:.0f}% High+Critical</div>
                    </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:9px;
                        padding:14px 16px;font-size:13px;color:#15803d;">
                ✅ No historically high-risk patterns found for the next 4 hours.
            </div>""", unsafe_allow_html=True)
    else:
        st.info("No data available.")

with col_feed:
    st.markdown("""
    <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                letter-spacing:.09em;margin-bottom:12px;">
        🗂️ Recent Activity Feed
    </div>""", unsafe_allow_html=True)

    if len(feedback) > 0:
        recent = feedback.sort_values("timestamp", ascending=False).head(6)
        for _, row in recent.iterrows():
            ts      = pd.to_datetime(row.get("timestamp",""), errors="coerce")
            ts_str  = ts.strftime("%d %b %H:%M") if pd.notna(ts) else "—"
            iid     = str(row.get("incident_id","—"))
            corr    = str(row.get("corridor","—"))
            a_sev   = str(row.get("actual_severity","—"))
            a_dur   = row.get("actual_duration", None)
            dur_str = f"{a_dur:.0f} min" if pd.notna(a_dur) else "—"
            p_sev   = str(row.get("predicted_severity","—"))
            match   = "✅" if a_sev == p_sev else "❌"
            sev_c   = {"Critical":"#b91c1c","High":"#ea580c",
                       "Medium":"#d97706","Low":"#15803d"}.get(a_sev,"#6b7280")

            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:8px;
                        padding:9px 13px;margin-bottom:6px;
                        display:flex;align-items:center;gap:10px;
                        box-shadow:0 1px 3px rgba(0,0,0,.03);">
                <div style="flex:1;min-width:0;">
                    <div style="display:flex;align-items:center;gap:6px;">
                        <span style="font-size:12px;font-weight:700;color:#111827;">
                            {iid}</span>
                        <span style="font-size:10px;color:#9ca3af;">{ts_str}</span>
                    </div>
                    <div style="font-size:11px;color:#6b7280;margin-top:2px;">
                        {corr} · {dur_str}</div>
                </div>
                <div style="text-align:right;flex-shrink:0;">
                    <div style="font-size:11px;font-weight:700;color:{sev_c};">{a_sev}</div>
                    <div style="font-size:10px;color:#9ca3af;">{match} Pred: {p_sev}</div>
                </div>
            </div>""", unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;border-radius:9px;
                    padding:24px 16px;text-align:center;">
            <div style="font-size:24px;opacity:.3;margin-bottom:8px;">📋</div>
            <div style="font-size:13px;color:#6b7280;">No feedback logged yet.</div>
            <div style="font-size:12px;color:#9ca3af;margin-top:4px;">
                Use Post-Event Log to record incident outcomes.</div>
        </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECTION 5 — MODULE STATUS TILES
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
            letter-spacing:.09em;margin:8px 0 12px;">
    🧭 System Modules
</div>""", unsafe_allow_html=True)

# Compute live context for each tile
n_high_corr  = int((risk_df["risk_score"] >= 45).sum()) if df is not None else 0
top_risk_corr= risk_df.iloc[0]["corridor"] if df is not None else "—"
fb_pct       = min(int(fb_count / MIN_FEEDBACK * 100), 100)
sev_acc      = 0
if len(feedback) >= 3:
    sev_acc = int((feedback["predicted_severity"]==feedback["actual_severity"]).mean()*100)

MODULES = [
    (
        "⚡", "Live Incident Prediction",
        "Submit an active incident for ML-powered severity, clearance time, closure "
        "probability, resource deployment and AI situation report.",
        "#1a56db","#eff6ff","#bfdbfe",
        f"{n_high_corr} corridors elevated · use for active incidents",
    ),
    (
        "🔮", "Predictive Risk Forecast",
        "Set your shift window and get corridor-by-corridor risk predictions, "
        "deployment brief, and hour-by-hour trend.",
        "#0891b2","#f0f9ff","#bae6fd",
        f"Current window: {cur_hour:02d}:00–{(cur_hour+3)%24:02d}:00 · {n_high_corr} at risk",
    ),
    (
        "📊", "Operations Intelligence",
        "Corridor scorecards, incident cost analysis, temporal patterns and "
        "monthly review for shift briefings and DCP reports.",
        "#6d28d9","#f5f3ff","#ddd6fe",
        "Top risk: " + (top_risk_corr if df is not None else "No data"),
    ),
    (
        "🗓️", "Event Scenario Planner",
        "Pre-plan resources for any upcoming event with hour-by-hour deployment "
        "forecast, checklist, and AI brief.",
        "#b45309","#fffbeb","#fde68a",
        f"Plan before an event — uses {len(df):,} historical incidents" if df is not None else "No data",
    ),
    (
        "📋", "Post-Event Feedback Log",
        "Log actual outcomes after each incident to track ML accuracy "
        "and drive the continuous learning pipeline.",
        "#15803d","#f0fdf4","#bbf7d0",
        f"{fb_count}/{MIN_FEEDBACK} logged · {sev_acc}% severity accuracy" if fb_count > 0
        else f"0/{MIN_FEEDBACK} logged · Start logging to improve predictions",
    ),
]

cols = st.columns(5, gap="small")
for col, (icon, title, desc, accent, bg, border, status) in zip(cols, MODULES):
    col.markdown(f"""
    <div style="background:{bg};border:1px solid {border};border-radius:12px;
                padding:16px 14px 14px;min-height:210px;
                box-shadow:0 1px 3px rgba(0,0,0,.04);">
        <div style="width:36px;height:36px;border-radius:9px;display:flex;
                    align-items:center;justify-content:center;margin-bottom:10px;
                    background:#ffffff;border:1px solid {border};">
            <span style="font-size:17px;line-height:1;">{icon}</span>
        </div>
        <div style="font-size:13px;font-weight:700;color:#111827;
                    margin-bottom:6px;line-height:1.3;">{title}</div>
        <div style="font-size:11px;color:#4b5563;line-height:1.6;
                    margin-bottom:10px;">{desc}</div>
        <div style="background:#ffffff;border:1px solid {border};border-radius:6px;
                    padding:5px 9px;font-size:10px;color:{accent};font-weight:600;
                    line-height:1.4;">{status}</div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="margin-top:28px;padding-top:14px;border-top:1px solid #e5e7eb;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:8px;">
    <span style="color:#9ca3af;font-size:12px;">
        TrafficSense &nbsp;·&nbsp; Bengaluru Traffic Police AI Command Center
    </span>
    <span style="color:#9ca3af;font-size:12px;">
        XGBoost · LightGBM · SHAP · Gemini AI · Groq · OpenWeatherMap
    </span>
</div>
""", unsafe_allow_html=True)