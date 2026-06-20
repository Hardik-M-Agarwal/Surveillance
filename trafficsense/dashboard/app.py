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
import plotly.graph_objects as go
from utils.theme import (apply_theme, sidebar_brand, C, PLOTLY,
                         AXIS_STYLE, BASE_MARGIN)

st.set_page_config(
    page_title="TrafficSense · Command Center",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

with st.sidebar:
    sidebar_brand()

# ── Data ───────────────────────────────────────────────────────────────────────
CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"

@st.cache_data(show_spinner=False)
def load_cleaned():
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
    return df

df = load_cleaned()

# ── Hero ───────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="text-align:center;padding:40px 0 32px;
            background:radial-gradient(ellipse at 50% -20%,#1d4ed820 0%,transparent 70%);
            margin:-1.5rem -2rem 2rem;border-bottom:1px solid #162440;">
    <div style="display:inline-flex;align-items:center;gap:8px;
                background:#0d1b30;border:1px solid #1e3452;
                border-radius:99px;padding:5px 16px;margin-bottom:18px;">
        <span style="width:7px;height:7px;border-radius:50%;background:#10b981;
                     display:inline-block;animation:dot-pulse 2s ease-in-out infinite;"></span>
        <span style="font-size:11px;font-weight:600;color:#10b981;letter-spacing:.08em;">
            BENGALURU TRAFFIC POLICE · AI COMMAND CENTER
        </span>
    </div>
    <h1 style="font-size:42px;font-weight:900;color:#f0f6ff;margin:0;letter-spacing:-1px;
               background:linear-gradient(135deg,#f0f6ff 30%,#93c5fd 100%);
               -webkit-background-clip:text;-webkit-text-fill-color:transparent;
               background-clip:text;">
        TrafficSense
    </h1>
    <p style="color:#8da4c0;font-size:16px;margin:10px 0 0;max-width:560px;
              margin-left:auto;margin-right:auto;line-height:1.6;">
        Event-driven congestion forecasting &amp; real-time resource deployment
        powered by machine learning
    </p>
</div>
""", unsafe_allow_html=True)

# ── KPIs ───────────────────────────────────────────────────────────────────────
if df is not None:
    total       = len(df)
    n_critical  = (df["severity"] == "Critical").sum() if "severity" in df.columns else 0
    n_high      = (df["severity"] == "High").sum()     if "severity" in df.columns else 0
    closure_pct = df["requires_road_closure"].mean() * 100
    med_dur     = df["duration_minutes"].dropna().median() if "duration_minutes" in df.columns else 0
    peak_hour   = int(df["hour"].value_counts().index[0]) if "hour" in df.columns else 0

    c1, c2, c3, c4, c5 = st.columns(5)
    metrics = [
        (c1, "📁", "Training Records",  f"{total:,}",           C["primary"]),
        (c2, "⚠️", "High + Critical",   f"{n_critical+n_high:,}",C["error"]),
        (c3, "🚧", "Closure Rate",      f"{closure_pct:.1f}%",   C["warning"]),
        (c4, "⏱️", "Median Clearance",  f"{med_dur:.0f} min",    C["cyan"]),
        (c5, "🕐", "Peak Incident Hour", f"{peak_hour:02d}:00",   C["purple"]),
    ]
    for col, icon, label, val, color in metrics:
        with col:
            st.markdown(f"""
            <div style="background:linear-gradient(135deg,#0a1628,#0d1e35);
                        border:1px solid #162440;border-left:3px solid {color};
                        border-radius:14px;padding:18px 20px;text-align:center;
                        box-shadow:0 4px 20px rgba(0,0,0,.4);">
                <div style="font-size:18px;">{icon}</div>
                <div style="color:#8da4c0;font-size:10px;font-weight:700;
                            text-transform:uppercase;letter-spacing:.09em;margin-top:6px;">
                    {label}</div>
                <div style="color:#f0f6ff;font-size:26px;font-weight:800;
                            margin-top:6px;line-height:1;">{val}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Charts ─────────────────────────────────────────────────────────────
    col_a, col_b, col_c = st.columns([1.5, 1.2, 1.3], gap="medium")

    with col_a:
        st.markdown("""<div style="background:linear-gradient(135deg,#0a1628,#0d1e35);
                    border:1px solid #162440;border-radius:14px;padding:20px;
                    box-shadow:0 4px 20px rgba(0,0,0,.3);">
                    <div style="font-size:12px;font-weight:700;color:#8da4c0;
                    text-transform:uppercase;letter-spacing:.09em;margin-bottom:2px;">
                    Incident Frequency by Hour</div>""",
                    unsafe_allow_html=True)
        hc = df.groupby("hour").size().reset_index(name="count")
        pk = [4,5,6,19,20,21,22]
        fig = go.Figure(go.Bar(
            x=hc["hour"], y=hc["count"],
            marker_color=[C["error"] if h in pk else C["primary"] for h in hc["hour"]],
            marker_line_width=0,
            hovertemplate="<b>%{x}:00</b> — %{y} incidents<extra></extra>",
        ))
        fig.update_layout(**PLOTLY, height=200, margin=BASE_MARGIN)
        fig.update_xaxes(**AXIS_STYLE, tickmode="array", tickvals=list(range(0,24,4)),
                         ticktext=[f"{h:02d}" for h in range(0,24,4)])
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, config={"displayModeBar":False})
        st.markdown("<div style='font-size:11px;color:#4a6080;text-align:center;margin-top:-8px;'>Red = peak hours (19-22, 04-06)</div></div>",
                    unsafe_allow_html=True)

    with col_b:
        st.markdown("""<div style="background:linear-gradient(135deg,#0a1628,#0d1e35);
                    border:1px solid #162440;border-radius:14px;padding:20px;
                    box-shadow:0 4px 20px rgba(0,0,0,.3);">
                    <div style="font-size:12px;font-weight:700;color:#8da4c0;
                    text-transform:uppercase;letter-spacing:.09em;margin-bottom:2px;">
                    Severity Split</div>""",
                    unsafe_allow_html=True)
        if "severity" in df.columns:
            sev_order  = ["Critical","High","Medium","Low"]
            sev_counts = df["severity"].value_counts()
            sev_vals   = [sev_counts.get(s,0) for s in sev_order]
            fig2 = go.Figure(go.Pie(
                labels=sev_order, values=sev_vals,
                marker_colors=["#dc2626","#ef4444","#f59e0b","#10b981"],
                hole=0.6, textinfo="percent", textfont_size=11,
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            ))
            fig2.add_annotation(
                text=f"<b>{sum(sev_vals):,}</b>",
                x=0.5, y=0.5, font=dict(size=16, color="#f0f6ff"), showarrow=False,
            )
            fig2.update_layout(**PLOTLY, height=200, margin=dict(l=0,r=0,t=8,b=0),
                               showlegend=True,
                               legend=dict(bgcolor="rgba(0,0,0,0)", orientation="v",
                                           font=dict(size=11, color=C["muted"])))
            st.plotly_chart(fig2, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_c:
        st.markdown("""<div style="background:linear-gradient(135deg,#0a1628,#0d1e35);
                    border:1px solid #162440;border-radius:14px;padding:20px;
                    box-shadow:0 4px 20px rgba(0,0,0,.3);">
                    <div style="font-size:12px;font-weight:700;color:#8da4c0;
                    text-transform:uppercase;letter-spacing:.09em;margin-bottom:2px;">
                    Top Corridors by Incidents</div>""",
                    unsafe_allow_html=True)
        if "corridor" in df.columns:
            top5 = df[df["corridor"] != "Non-corridor"]["corridor"].value_counts().head(5)
            fig3 = go.Figure(go.Bar(
                y=top5.index[::-1], x=top5.values[::-1], orientation="h",
                marker_color=C["cyan"], marker_line_width=0,
                hovertemplate="%{y}: %{x:,} incidents<extra></extra>",
            ))
            fig3.update_layout(**PLOTLY, height=200, margin=BASE_MARGIN)
            fig3.update_xaxes(**AXIS_STYLE)
            fig3.update_yaxes(**AXIS_STYLE, tickfont=dict(size=10))
            st.plotly_chart(fig3, config={"displayModeBar":False})
        st.markdown("</div>", unsafe_allow_html=True)

else:
    st.markdown("""
    <div style="background:linear-gradient(135deg,#0a1628,#0d1e35);
                border:1px solid #1e3452;border-radius:14px;
                padding:40px;text-align:center;margin:20px 0;">
        <div style="font-size:40px;margin-bottom:12px;">⚙️</div>
        <div style="color:#f0f6ff;font-size:16px;font-weight:700;">Setup Required</div>
        <div style="color:#8da4c0;font-size:14px;margin-top:6px;">
            Run <code style="background:#162440;color:#93c5fd;padding:2px 8px;
            border-radius:5px;">python run_pipeline.py</code> to train models and generate data.
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Module cards ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin:8px 0 16px;">
    <div style="font-size:11px;font-weight:700;color:#8da4c0;
                text-transform:uppercase;letter-spacing:.1em;margin-bottom:14px;">
        System Modules
    </div>
</div>
""", unsafe_allow_html=True)

modules = [
    ("⚡", "Live Prediction",
     "Submit an active incident → ML severity, duration, closure probability & resource deployment with AI narrative.",
     C["primary"], "#1d4ed820"),
    ("🗺️", "Corridor Risk Map",
     "Real-time heatmap of 28 Bengaluru corridors. Data-driven risk scores, density layers, junction hotspots.",
     C["cyan"], "#0369a120"),
    ("📊", "Data Insights",
     "Interactive analytics on 8,173+ historical incidents — temporal patterns, cause breakdown, corridor comparisons.",
     C["purple"], "#7c3aed20"),
    ("🗓️", "Scenario Planner",
     "Pre-plan resources for upcoming events. Hour-by-hour deployment forecast + AI deployment brief.",
     C["warning"], "#d9770620"),
    ("📋", "Post-Event Log",
     "Log incident outcomes, track prediction accuracy, monitor model drift for continuous learning.",
     C["success"], "#05966920"),
]

cols = st.columns(5, gap="small")
for col, (icon, title, desc, accent, bg) in zip(cols, modules):
    with col:
        r,g,b = int(accent[1:3],16), int(accent[3:5],16), int(accent[5:7],16)
        st.markdown(f"""
        <div style="background:linear-gradient(160deg,#0a1628 0%,#0d1e35 100%);
                    border:1px solid #162440;border-radius:14px;padding:20px 16px;
                    min-height:180px;position:relative;overflow:hidden;
                    transition:all .25s ease;cursor:default;
                    box-shadow:0 4px 16px rgba(0,0,0,.3);">
            <div style="position:absolute;top:0;right:0;width:100px;height:100px;
                        background:radial-gradient(circle at top right,rgba({r},{g},{b},.1),transparent 65%);
                        pointer-events:none;border-radius:0 14px 0 0;"></div>
            <div style="width:40px;height:40px;border-radius:11px;display:flex;
                        align-items:center;justify-content:center;margin-bottom:12px;
                        background:rgba({r},{g},{b},.12);border:1px solid rgba({r},{g},{b},.25);">
                <span style="font-size:20px;line-height:1;">{icon}</span>
            </div>
            <div style="color:#f0f6ff;font-size:14px;font-weight:700;
                        margin-bottom:8px;line-height:1.2;">{title}</div>
            <div style="color:#8da4c0;font-size:12px;line-height:1.55;">{desc}</div>
            <div style="position:absolute;bottom:16px;left:16px;">
                <span style="font-size:10px;font-weight:600;color:{accent};
                             letter-spacing:.06em;text-transform:uppercase;">Open →</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px;padding-top:20px;border-top:1px solid #162440;
            display:flex;align-items:center;justify-content:space-between;
            flex-wrap:wrap;gap:8px;">
    <span style="color:#4a6080;font-size:12px;">
        TrafficSense &nbsp;·&nbsp; Built for Bengaluru Traffic Police
    </span>
    <span style="color:#4a6080;font-size:12px;">
        XGBoost · LightGBM · SHAP · Gemini AI · OpenWeatherMap
    </span>
</div>
""", unsafe_allow_html=True)
