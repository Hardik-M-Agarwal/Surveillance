"""
Operations Intelligence — Corridor Scorecard, Incident Cost Analysis,
Temporal Patterns, Monthly Review
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from utils.theme import (apply_theme, sidebar_brand, page_header, section_header,
                         C, PLOTLY, SEV_COLOR, AXIS_STYLE, AXIS_NO_GRID, BASE_MARGIN)

st.set_page_config(page_title="Operations Intelligence · TrafficSense",
                   page_icon="📊", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_brand()

CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"

@st.cache_data(show_spinner=False)
def load_data():
    if not CLEANED_CSV.exists():
        return None
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]           = df["start_datetime"].dt.hour
    df["day_of_week"]    = df["start_datetime"].dt.dayofweek
    df["month"]          = df["start_datetime"].dt.month
    df["year_month"]     = df["start_datetime"].dt.to_period("M").astype(str)
    df["start_date"]     = df["start_datetime"].dt.date
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower()
        .map({"true":1,"1":1,"false":0,"0":0}).fillna(0).astype(int)
    )
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    df["sev_score"]        = df["severity"].map(
        {"Critical":4,"High":3,"Medium":2,"Low":1}).fillna(2)
    return df

df = load_data()

page_header("📊", "Operations Intelligence",
            "Corridor scorecards · Incident cost analysis · Monthly review for command briefings")

if df is None:
    st.warning("Processed data not found — run `python run_pipeline.py` to generate.")
    st.stop()

with st.sidebar:
    st.markdown("""<div style="font-size:11px;font-weight:700;color:#6b7280;
    text-transform:uppercase;letter-spacing:.08em;margin:8px 0 6px 2px;">Filters</div>""",
    unsafe_allow_html=True)
    corridors_avail = sorted(df["corridor"].dropna().unique().tolist())
    sel_corridors   = st.multiselect("Corridor", corridors_avail,
                                     default=[c for c in corridors_avail if c != "Non-corridor"])
    causes_avail = sorted(df["event_cause"].dropna().unique().tolist())
    sel_causes   = st.multiselect("Event Cause", causes_avail, default=causes_avail)
    sel_sev      = st.multiselect("Severity", ["Critical","High","Medium","Low"],
                                  default=["Critical","High","Medium","Low"])
    if df["start_datetime"].notna().any():
        min_d = df["start_datetime"].min().date()
        max_d = df["start_datetime"].max().date()
        date_range = st.date_input("Date Range", value=(min_d, max_d),
                                   min_value=min_d, max_value=max_d)
    else:
        date_range = None

mask = pd.Series(True, index=df.index)
if sel_corridors: mask &= df["corridor"].isin(sel_corridors)
if sel_causes:    mask &= df["event_cause"].isin(sel_causes)
if sel_sev and "severity" in df.columns: mask &= df["severity"].isin(sel_sev)
if date_range and len(date_range) == 2 and df["start_datetime"].notna().any():
    mask &= df["start_datetime"].between(
        pd.Timestamp(date_range[0], tz="UTC"),
        pd.Timestamp(date_range[1], tz="UTC")
    )
dff = df[mask].copy()
n   = len(dff)

med_dur  = dff["duration_minutes"].dropna().median()
cr       = dff["requires_road_closure"].mean()*100
high_pct = (dff["severity"].isin(["Critical","High"]).sum()/max(n,1))*100 if "severity" in dff.columns else 0
top_cause= dff["event_cause"].value_counts().index[0].replace("_"," ").title() if n>0 else "—"

k1,k2,k3,k4,k5 = st.columns(5)
for col, icon, lbl, val, acc in [
    (k1,"📋","Total Incidents",    f"{n:,}",                                           C["primary"]),
    (k2,"⏱️","Median Clearance",  f"{med_dur:.0f} min" if not np.isnan(med_dur) else "—", C["warning"]),
    (k3,"🔒","Closure Rate",      f"{cr:.1f}%",                                        C["error"]),
    (k4,"⚠️","High / Critical %", f"{high_pct:.1f}%",                                  C["error"]),
    (k5,"🔝","Top Cause",         top_cause,                                            C["primary"]),
]:
    col.markdown(f"""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                padding:14px 16px;border-top:3px solid {acc};
                box-shadow:0 1px 3px rgba(0,0,0,.04);margin-bottom:16px;">
        <div style="font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:6px;">{icon}&ensp;{lbl}</div>
        <div style="font-size:20px;font-weight:700;color:#111827;line-height:1.2;">{val}</div>
    </div>""", unsafe_allow_html=True)

DAYS       = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
SEV_COLORS = {"Critical":"#b91c1c","High":"#ea580c","Medium":"#d97706","Low":"#15803d"}

tab1, tab2, tab3, tab4 = st.tabs([
    "🛣️ Corridor Scorecard",
    "💰 Incident Cost Analysis",
    "📅 Temporal Patterns",
    "📈 Monthly Review",
])

# ── Helper: clean axis layout dict (no tickfont/title_font — AXIS_STYLE has them) ──
def ax(title="", **kw):
    d = dict(
        gridcolor="#e5e7eb",
        linecolor="#d1d5db",
        tickfont=dict(color="#111827", size=12, family="Inter, sans-serif"),
        title_font=dict(color="#111827", size=13, family="Inter, sans-serif"),
        title_standoff=10,
    )
    if title:
        d["title"] = title
        d["title_font"] = dict(color="#111827", size=13, family="Inter, sans-serif")
    d.update(kw)
    return d

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1 — CORRIDOR SCORECARD
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    section_header("Corridor Performance Scorecard", "🛣️")
    st.markdown('<p style="color:#6b7280;font-size:13px;margin:-8px 0 14px;">Use this in shift briefings and weekly reviews.</p>', unsafe_allow_html=True)

    dff_corr = dff[dff["corridor"] != "Non-corridor"]
    if not dff_corr.empty:
        scorecard = dff_corr.groupby("corridor").agg(
            total        =("sev_score",             "count"),
            avg_sev      =("sev_score",             "mean"),
            closure_rate =("requires_road_closure", "mean"),
            avg_duration =("duration_minutes",       "mean"),
            high_crit    =("severity", lambda x: x.isin(["High","Critical"]).mean()),
        ).reset_index()
        scorecard["most_common_cause"] = (
            dff_corr.groupby("corridor")["event_cause"]
            .agg(lambda x: x.value_counts().index[0])
            .str.replace("_"," ").str.title()
        )
        mx_dur = scorecard["avg_duration"].max()
        scorecard["risk_score"] = (
            0.30*(scorecard["avg_sev"]/4) +
            0.35* scorecard["closure_rate"] +
            0.20* scorecard["high_crit"] +
            0.15*(scorecard["avg_duration"].fillna(0) / max(mx_dur, 1))
        ) * 100
        scorecard = scorecard.sort_values("risk_score", ascending=False).reset_index(drop=True)

        t1,t2,t3 = st.columns(3)
        for ci, (col, row) in enumerate(zip([t1,t2,t3], scorecard.head(3).itertuples())):
            rank_label = ["🔴 Priority 1","🟠 Priority 2","🟡 Priority 3"][ci]
            score = row.risk_score
            color = "#b91c1c" if score>70 else "#ea580c" if score>50 else "#d97706"
            col.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                        border-top:4px solid {color};padding:14px 16px;
                        box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:16px;">
                <div style="font-size:10px;font-weight:700;color:{color};
                            text-transform:uppercase;letter-spacing:.07em;margin-bottom:6px;">{rank_label}</div>
                <div style="font-size:15px;font-weight:700;color:#111827;margin-bottom:10px;">{row.corridor}</div>
                <div style="display:flex;gap:8px;margin-bottom:8px;">
                    <div style="flex:1;text-align:center;background:#f8fafc;border-radius:6px;padding:6px 4px;">
                        <div style="font-size:16px;font-weight:800;color:{color};">{score:.0f}</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;">RISK</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f8fafc;border-radius:6px;padding:6px 4px;">
                        <div style="font-size:16px;font-weight:800;color:#111827;">{row.total:,}</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;">INCIDENTS</div>
                    </div>
                    <div style="flex:1;text-align:center;background:#f8fafc;border-radius:6px;padding:6px 4px;">
                        <div style="font-size:16px;font-weight:800;color:#b91c1c;">{row.closure_rate*100:.0f}%</div>
                        <div style="font-size:9px;color:#6b7280;font-weight:600;">CLOSURE</div>
                    </div>
                </div>
                <div style="font-size:11px;color:#6b7280;">Top cause: <b style="color:#374151;">{row.most_common_cause}</b></div>
            </div>""", unsafe_allow_html=True)

        section_header("Full Corridor Rankings", "📋")
        st.markdown("""<div style="display:grid;grid-template-columns:160px 60px 80px 80px 80px 90px 130px;
            gap:4px;padding:8px 12px;background:#f3f4f6;border-radius:8px;margin-bottom:6px;
            font-size:10px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.06em;">
            <div>Corridor</div><div>Risk</div><div>Incidents</div><div>Avg Sev</div>
            <div>Closure</div><div>Avg Duration</div><div>Top Cause</div></div>""",
            unsafe_allow_html=True)

        for _, row in scorecard.iterrows():
            sc = row["risk_score"]
            rc = "#b91c1c" if sc>70 else "#ea580c" if sc>50 else "#d97706" if sc>30 else "#15803d"
            ad = row["avg_duration"]
            ad_str = f"{ad:.0f} min" if not np.isnan(ad) else "—"
            st.markdown(f"""
            <div style="display:grid;grid-template-columns:160px 60px 80px 80px 80px 90px 130px;
                        gap:4px;padding:9px 12px;background:#ffffff;
                        border:1px solid #e5e7eb;border-left:4px solid {rc};
                        border-radius:8px;margin-bottom:4px;font-size:12px;">
                <div style="font-weight:600;color:#111827;">{row['corridor']}</div>
                <div style="font-weight:800;color:{rc};">{sc:.0f}</div>
                <div style="color:#374151;">{int(row['total']):,}</div>
                <div style="color:#374151;">{row['avg_sev']:.1f}/4</div>
                <div style="color:{'#b91c1c' if row['closure_rate']>0.3 else '#374151'};">{row['closure_rate']*100:.0f}%</div>
                <div style="color:#374151;">{ad_str}</div>
                <div style="color:#6b7280;font-size:11px;">{row['most_common_cause']}</div>
            </div>""", unsafe_allow_html=True)

        ch1, ch2 = st.columns(2)
        with ch1:
            section_header("Incident Count by Corridor", "📊")
            fig = go.Figure(go.Bar(
                x=scorecard["total"], y=scorecard["corridor"], orientation="h",
                marker_color=C["primary"],
                text=scorecard["total"].apply(lambda v: f"{v:,}"), textposition="outside",
                hovertemplate="<b>%{y}</b> — %{x:,}<extra></extra>",
            ))
            fig.update_layout(**PLOTLY, height=380, margin=dict(l=12,r=60,t=36,b=12),
                              title=dict(text="Total incidents per corridor",
                                         font=dict(size=13,color="#111827"),x=0,xanchor="left"))
            fig.update_xaxes(**ax("Incidents"))
            fig.update_yaxes(**ax())
            st.plotly_chart(fig, config={"displayModeBar":False}, use_container_width=True)

        with ch2:
            section_header("Closure Rate by Corridor", "🔒")
            fig2 = go.Figure(go.Bar(
                x=scorecard["closure_rate"]*100, y=scorecard["corridor"], orientation="h",
                marker_color=scorecard["closure_rate"].apply(
                    lambda r: "#b91c1c" if r>0.3 else "#ea580c" if r>0.15 else "#d97706" if r>0.05 else "#15803d"),
                text=scorecard["closure_rate"].apply(lambda r: f"{r*100:.0f}%"), textposition="outside",
                hovertemplate="<b>%{y}</b> — %{x:.1f}%<extra></extra>",
            ))
            fig2.update_layout(**PLOTLY, height=380, margin=dict(l=12,r=60,t=36,b=12),
                               title=dict(text="% incidents causing road closure",
                                          font=dict(size=13,color="#111827"),x=0,xanchor="left"))
            fig2.update_xaxes(**ax("Closure Rate (%)", range=[0,100]))
            fig2.update_yaxes(**ax())
            st.plotly_chart(fig2, config={"displayModeBar":False}, use_container_width=True)

        if "severity" in dff_corr.columns:
            section_header("Severity Matrix — Top 10 Corridors", "🎯")
            corr_sev = dff_corr.groupby(["corridor","severity"]).size().reset_index(name="count")
            top10    = dff_corr["corridor"].value_counts().head(10).index
            corr_sev = corr_sev[corr_sev["corridor"].isin(top10)]
            fig3 = px.bar(corr_sev, x="corridor", y="count", color="severity",
                          color_discrete_map=SEV_COLOR, barmode="stack",
                          category_orders={"severity":["Critical","High","Medium","Low"]})
            fig3.update_layout(**PLOTLY, height=320, margin=dict(l=12,r=12,t=36,b=60),
                               title=dict(text="Incident severity breakdown by corridor",
                                          font=dict(size=13,color="#111827"),x=0,xanchor="left"),
                               xaxis_title="", yaxis_title="Incidents",
                               legend=dict(orientation="h",y=-0.3,
                                           font=dict(size=11,color="#374151"),bgcolor="rgba(0,0,0,0)"))
            fig3.update_xaxes(**ax(tickangle=-30))
            fig3.update_yaxes(**ax())
            st.plotly_chart(fig3, config={"displayModeBar":False}, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 2 — INCIDENT COST ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    section_header("Which Incidents Cost the Most?", "💰")
    st.markdown('<p style="color:#6b7280;font-size:13px;margin:-8px 0 14px;">Ranked by operational cost — clearance time × closure rate × frequency.</p>', unsafe_allow_html=True)

    cause_cost = dff.groupby("event_cause").agg(
        count          =("event_cause",           "count"),
        avg_duration   =("duration_minutes",       "mean"),
        closure_rate   =("requires_road_closure",  "mean"),
        high_crit_rate =("severity", lambda x: x.isin(["High","Critical"]).mean()),
    ).reset_index()
    cause_cost = cause_cost[cause_cost["count"] >= 5].copy()
    cause_cost["avg_duration"]     = cause_cost["avg_duration"].fillna(0)
    cause_cost["operational_cost"] = (
        cause_cost["count"] * (cause_cost["avg_duration"].clip(upper=300)/60) *
        (1 + cause_cost["closure_rate"])
    )
    cause_cost["cause_label"] = cause_cost["event_cause"].str.replace("_"," ").str.title()
    cause_cost = cause_cost.sort_values("operational_cost", ascending=False)

    section_header("Priority Matrix: Frequency vs Duration", "🎯")
    st.markdown('<p style="color:#6b7280;font-size:13px;margin:-8px 0 12px;">Bubble size = closure rate. Top-right = highest priority.</p>', unsafe_allow_html=True)

    med_count = cause_cost["count"].median()
    med_dur_c = cause_cost["avg_duration"].median()

    fig_scatter = go.Figure(go.Scatter(
        x=cause_cost["count"], y=cause_cost["avg_duration"],
        mode="markers+text", text=cause_cost["cause_label"],
        textposition="top center", textfont=dict(size=10, color="#374151"),
        marker=dict(
            size=cause_cost["closure_rate"]*120+12,
            color=cause_cost["closure_rate"].apply(
                lambda r: "#b91c1c" if r>0.4 else "#ea580c" if r>0.2 else "#d97706" if r>0.1 else "#15803d"),
            line=dict(color="#ffffff", width=1.5), opacity=0.8,
        ),
        hovertemplate="<b>%{text}</b><br>Incidents: %{x:,}<br>Avg duration: %{y:.0f} min<extra></extra>",
    ))
    fig_scatter.add_vline(x=med_count, line=dict(color="#d1d5db", dash="dash", width=1))
    fig_scatter.add_hline(y=med_dur_c,  line=dict(color="#d1d5db", dash="dash", width=1))
    fig_scatter.update_layout(
        **PLOTLY, height=420, margin=dict(l=60,r=20,t=44,b=60),
        title=dict(text="Incident Priority Matrix — bubble size = closure rate",
                   font=dict(size=13,color="#111827"),x=0,xanchor="left"),
        xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
                   title="Incident Frequency (count)",
                   tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
        yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
                   title="Average Duration (minutes)",
                   tickfont=dict(color="#374151"), title_font=dict(color="#374151")),
    )
    st.plotly_chart(fig_scatter, config={"displayModeBar":False}, use_container_width=True)

    c1, c2 = st.columns(2)
    with c1:
        section_header("Average Clearance Time by Cause", "⏱️")
        dur_df = cause_cost.sort_values("avg_duration", ascending=True).tail(12)
        fig_dur = go.Figure(go.Bar(
            x=dur_df["avg_duration"], y=dur_df["cause_label"], orientation="h",
            marker_color=dur_df["avg_duration"].apply(
                lambda d: "#b91c1c" if d>120 else "#ea580c" if d>60 else "#d97706" if d>30 else "#15803d"),
            text=dur_df["avg_duration"].apply(lambda d: f"{d:.0f} min"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Avg: %{x:.0f} min<extra></extra>",
        ))
        fig_dur.update_layout(**PLOTLY, height=380, margin=dict(l=150,r=80,t=36,b=12),
                              title=dict(text="Which causes take longest to clear?",
                                         font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_dur.update_xaxes(**ax("Avg Duration (min)"))
        fig_dur.update_yaxes(**ax())
        st.plotly_chart(fig_dur, config={"displayModeBar":False}, use_container_width=True)

    with c2:
        section_header("Road Closure Rate by Cause", "🚧")
        cl_df = cause_cost.sort_values("closure_rate", ascending=True)
        fig_cl = go.Figure(go.Bar(
            x=cl_df["closure_rate"]*100, y=cl_df["cause_label"], orientation="h",
            marker_color=cl_df["closure_rate"].apply(
                lambda r: "#b91c1c" if r>0.5 else "#ea580c" if r>0.25 else "#d97706" if r>0.1 else "#15803d"),
            text=cl_df["closure_rate"].apply(lambda r: f"{r*100:.0f}%"), textposition="outside",
            hovertemplate="<b>%{y}</b><br>Closure: %{x:.1f}%<extra></extra>",
        ))
        fig_cl.update_layout(**PLOTLY, height=380, margin=dict(l=150,r=70,t=36,b=12),
                             title=dict(text="Which causes most often close the road?",
                                        font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_cl.update_xaxes(**ax("Closure Rate (%)", range=[0,100]))
        fig_cl.update_yaxes(**ax())
        st.plotly_chart(fig_cl, config={"displayModeBar":False}, use_container_width=True)

    if "event_type" in dff.columns:
        section_header("Event Type vs Cause — Planned vs Unplanned", "🔀")
        top_causes_et = dff["event_cause"].value_counts().head(10).index
        et_df = (dff[dff["event_cause"].isin(top_causes_et)]
                 .groupby(["event_cause","event_type"]).size().reset_index(name="count"))
        et_df["cause_label"] = et_df["event_cause"].str.replace("_"," ").str.title()
        fig_et = px.bar(et_df, x="count", y="cause_label", color="event_type",
                        orientation="h", barmode="stack",
                        color_discrete_sequence=["#1a56db","#d97706"])
        fig_et.update_layout(**PLOTLY, height=340, margin=dict(l=150,r=20,t=36,b=12),
                             title=dict(text="Top 10 causes: planned vs unplanned split",
                                        font=dict(size=13,color="#111827"),x=0,xanchor="left"),
                             legend=dict(orientation="h",y=-0.2,
                                         font=dict(size=11,color="#374151"),bgcolor="rgba(0,0,0,0)"))
        fig_et.update_xaxes(**ax("Incidents"))
        fig_et.update_yaxes(**ax())
        st.plotly_chart(fig_et, config={"displayModeBar":False}, use_container_width=True)

    section_header("Top Causes Requiring Road Closure", "🔒")
    closure_data = dff[dff["requires_road_closure"]==1]
    if not closure_data.empty:
        cl_causes = closure_data["event_cause"].value_counts().head(10).reset_index()
        cl_causes.columns = ["cause","count"]
        cl_causes["cause_label"] = cl_causes["cause"].str.replace("_"," ").str.title()
        fig_clc = go.Figure(go.Bar(
            x=cl_causes["count"], y=cl_causes["cause_label"], orientation="h",
            marker_color="#b91c1c",
            text=cl_causes["count"].apply(lambda v: f"{v:,}"), textposition="outside",
            hovertemplate="<b>%{y}</b> — %{x:,} closures<extra></extra>",
        ))
        fig_clc.update_layout(**PLOTLY, height=320, margin=dict(l=150,r=80,t=36,b=12),
                              title=dict(text="Which causes most frequently forced road closures?",
                                         font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_clc.update_xaxes(**ax("Closure Incidents"))
        fig_clc.update_yaxes(**ax())
        st.plotly_chart(fig_clc, config={"displayModeBar":False}, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 3 — TEMPORAL PATTERNS
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    c1, c2 = st.columns(2)
    with c1:
        section_header("Incidents by Hour of Day", "🕐")
        hour_data = dff.groupby("hour").size().reset_index(name="count")
        peak_hrs  = [4,5,6,19,20,21,22]
        hour_data["color"] = hour_data["hour"].apply(
            lambda h: "#b91c1c" if h in peak_hrs else C["primary"])
        fig_h = go.Figure(go.Bar(
            x=hour_data["hour"], y=hour_data["count"],
            marker_color=hour_data["color"],
            hovertemplate="<b>%{x}:00</b> — %{y:,}<extra></extra>",
        ))
        fig_h.add_annotation(text="🔴 Peak hours (04–06 and 19–22)",
                              x=0, y=1.08, xref="paper", yref="paper",
                              showarrow=False, font=dict(size=11,color="#6b7280"), xanchor="left")
        fig_h.update_layout(**PLOTLY, height=300, margin=dict(l=50,r=20,t=48,b=50),
                            title=dict(text="Hourly incident frequency",
                                       font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_h.update_xaxes(**ax("Hour of Day", tickmode="array",
                                tickvals=list(range(0,24,2)),
                                ticktext=[f"{h:02d}:00" for h in range(0,24,2)]))
        fig_h.update_yaxes(**ax("Incidents"))
        st.plotly_chart(fig_h, config={"displayModeBar":False}, use_container_width=True)

    with c2:
        section_header("Incidents by Day of Week", "📅")
        day_data = dff.groupby("day_of_week").size().reset_index(name="count")
        day_data["day_name"] = day_data["day_of_week"].map(dict(enumerate(DAYS)))
        day_data["color"]    = day_data["day_of_week"].apply(
            lambda d: C["success"] if d>=5 else C["primary"])
        fig_d = go.Figure(go.Bar(
            x=day_data["day_name"], y=day_data["count"],
            marker_color=day_data["color"],
            hovertemplate="<b>%{x}</b> — %{y:,}<extra></extra>",
        ))
        fig_d.update_layout(**PLOTLY, height=300, margin=dict(l=50,r=20,t=44,b=50),
                            title=dict(text="Weekly distribution · green = weekend",
                                       font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_d.update_xaxes(**ax("Day of Week"))
        fig_d.update_yaxes(**ax("Incidents"))
        st.plotly_chart(fig_d, config={"displayModeBar":False}, use_container_width=True)

    section_header("Incident Density Heatmap — Hour × Day", "🔥")
    pivot = dff.pivot_table(index="day_of_week", columns="hour",
                             values="event_cause", aggfunc="count", fill_value=0)
    pivot.index = [DAYS[int(i)] for i in pivot.index]
    fig_heat = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{int(h):02d}:00" for h in pivot.columns], y=pivot.index,
        colorscale=[[0,"#f0f9ff"],[0.3,"#93c5fd"],[0.7,"#ea580c"],[1.0,"#b91c1c"]],
        hovertemplate="<b>%{y} at %{x}</b><br>Incidents: %{z}<extra></extra>",
        colorbar=dict(title=dict(text="Incidents",font=dict(color="#374151",size=11)),
                      tickfont=dict(color="#374151")),
    ))
    fig_heat.update_layout(**PLOTLY, height=320, margin=dict(l=50,r=20,t=44,b=60),
                           title=dict(text="When are incidents most concentrated?",
                                      font=dict(size=13,color="#111827"),x=0,xanchor="left"))
    fig_heat.update_xaxes(**ax("Hour of Day"))
    fig_heat.update_yaxes(**ax())
    st.plotly_chart(fig_heat, config={"displayModeBar":False}, use_container_width=True)

    if "event_type" in dff.columns:
        section_header("Planned vs Unplanned by Hour", "📊")
        hourly_type = dff.groupby(["hour","event_type"]).size().reset_index(name="count")
        fig_ht = px.bar(hourly_type, x="hour", y="count", color="event_type",
                        barmode="stack", color_discrete_sequence=["#1a56db","#d97706"])
        fig_ht.update_layout(**PLOTLY, height=300, margin=dict(l=50,r=20,t=44,b=50),
                             title=dict(text="Planned vs unplanned incidents by hour",
                                        font=dict(size=13,color="#111827"),x=0,xanchor="left"),
                             legend=dict(orientation="h",y=-0.25,
                                         font=dict(size=11,color="#374151"),bgcolor="rgba(0,0,0,0)"))
        fig_ht.update_xaxes(**ax("Hour of Day", tickmode="array",
                                  tickvals=list(range(0,24,2)),
                                  ticktext=[f"{h:02d}" for h in range(0,24,2)]))
        fig_ht.update_yaxes(**ax("Incidents"))
        st.plotly_chart(fig_ht, config={"displayModeBar":False}, use_container_width=True)

    if "priority" in dff.columns:
        section_header("Event Priority Distribution", "🎯")
        p1, p2 = st.columns(2)
        with p1:
            pri_counts = dff["priority"].value_counts().reset_index()
            pri_counts.columns = ["priority","count"]
            pri_colors = {"High":"#b91c1c","Medium":"#d97706","Low":"#15803d"}
            fig_pri = go.Figure(go.Bar(
                x=pri_counts["priority"], y=pri_counts["count"],
                marker_color=pri_counts["priority"].map(pri_colors).fillna("#6b7280"),
                text=pri_counts["count"].apply(lambda v: f"{v:,}"), textposition="outside",
                hovertemplate="<b>%{x}</b> — %{y:,}<extra></extra>",
            ))
            fig_pri.update_layout(**PLOTLY, height=280, margin=dict(l=50,r=20,t=44,b=40),
                                  title=dict(text="Incidents by priority level",
                                             font=dict(size=13,color="#111827"),x=0,xanchor="left"))
            fig_pri.update_xaxes(**ax())
            fig_pri.update_yaxes(**ax("Incidents"))
            st.plotly_chart(fig_pri, config={"displayModeBar":False}, use_container_width=True)

        with p2:
            if "event_cause" in dff.columns:
                crosstab = pd.crosstab(dff["event_cause"], dff["priority"])
                crosstab.index = crosstab.index.str.replace("_"," ").str.title()
                top_ct = crosstab.sum(axis=1).nlargest(8).index
                crosstab = crosstab.loc[crosstab.index.isin(top_ct)]
                st.markdown('<div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;letter-spacing:.07em;margin-bottom:8px;">Cause × Priority Cross-Tab (Top 8)</div>', unsafe_allow_html=True)
                st.dataframe(crosstab, use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# TAB 4 — MONTHLY REVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    section_header("Month-over-Month Trends", "📈")
    st.markdown('<p style="color:#6b7280;font-size:13px;margin:-8px 0 14px;">For senior officer monthly reviews and DCP briefings.</p>', unsafe_allow_html=True)

    monthly = dff.groupby("year_month").agg(
        incidents    =("event_cause",           "count"),
        closure_rate =("requires_road_closure", "mean"),
        avg_duration =("duration_minutes",       "mean"),
    ).reset_index().sort_values("year_month")
    monthly["avg_duration"] = monthly["avg_duration"].fillna(0)

    if len(monthly) >= 2:
        last  = monthly.iloc[-1]
        prev  = monthly.iloc[-2]
        inc_ch = ((last["incidents"]-prev["incidents"])/max(prev["incidents"],1))*100
        dur_ch = last["avg_duration"] - prev["avg_duration"]
        cl_ch  = (last["closure_rate"]-prev["closure_rate"])*100
        inc_c  = "#b91c1c" if inc_ch>10 else "#15803d" if inc_ch<-5 else "#d97706"
        dur_c  = "#b91c1c" if dur_ch>10 else "#15803d" if dur_ch<-5 else "#d97706"
        inc_w  = f"↑ {inc_ch:.0f}% more" if inc_ch>0 else f"↓ {abs(inc_ch):.0f}% fewer"
        dur_w  = f"↑ +{dur_ch:.0f} min longer" if dur_ch>0 else f"↓ {abs(dur_ch):.0f} min shorter"
        cl_w   = f"↑ +{cl_ch:.1f}%" if cl_ch>0 else f"↓ {abs(cl_ch):.1f}%"
        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                    padding:18px 22px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);">
            <div style="font-size:12px;font-weight:700;color:#6b7280;text-transform:uppercase;
                        letter-spacing:.07em;margin-bottom:10px;">
                📋 Monthly Summary — {last['year_month']} vs {prev['year_month']}
            </div>
            <div style="font-size:14px;color:#374151;line-height:1.9;">
                This period saw <b style="color:{inc_c};">{last['incidents']:,} incidents</b>
                — <span style="color:{inc_c};font-weight:700;">{inc_w}</span> than the previous month.
                Average clearance time was <b style="color:{dur_c};">{last['avg_duration']:.0f} minutes</b>
                (<span style="color:{dur_c};font-weight:700;">{dur_w}</span>).
                Road closure rate was <b>{last['closure_rate']*100:.1f}%</b>
                (<span style="font-weight:700;">{cl_w} vs previous</span>).
            </div>
        </div>""", unsafe_allow_html=True)

    m1, m2 = st.columns(2)
    with m1:
        fig_inc = go.Figure(go.Scatter(
            x=monthly["year_month"], y=monthly["incidents"],
            mode="lines+markers", line=dict(color=C["primary"], width=2.5),
            marker=dict(size=7, line=dict(color="#fff",width=1.5)),
            fill="tozeroy", fillcolor="rgba(26,86,219,0.07)",
            hovertemplate="<b>%{x}</b><br>Incidents: %{y:,}<extra></extra>",
        ))
        fig_inc.update_layout(**PLOTLY, height=280, margin=dict(l=55,r=20,t=44,b=60),
                              title=dict(text="Monthly incident volume",
                                         font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_inc.update_xaxes(**ax(tickangle=-30))
        fig_inc.update_yaxes(**ax("Incidents"))
        st.plotly_chart(fig_inc, config={"displayModeBar":False}, use_container_width=True)

    with m2:
        fig_cl2 = go.Figure(go.Scatter(
            x=monthly["year_month"], y=monthly["closure_rate"]*100,
            mode="lines+markers", line=dict(color="#b91c1c", width=2.5),
            marker=dict(size=7, line=dict(color="#fff",width=1.5)),
            hovertemplate="<b>%{x}</b><br>Closure rate: %{y:.1f}%<extra></extra>",
        ))
        fig_cl2.update_layout(**PLOTLY, height=280, margin=dict(l=55,r=20,t=44,b=60),
                              title=dict(text="Monthly road closure rate (%)",
                                         font=dict(size=13,color="#111827"),x=0,xanchor="left"))
        fig_cl2.update_xaxes(**ax(tickangle=-30))
        fig_cl2.update_yaxes(**ax("Closure Rate (%)"))
        st.plotly_chart(fig_cl2, config={"displayModeBar":False}, use_container_width=True)

    fig_dur2 = go.Figure(go.Scatter(
        x=monthly["year_month"], y=monthly["avg_duration"],
        mode="lines+markers", line=dict(color="#d97706", width=2.5),
        marker=dict(size=7, line=dict(color="#fff",width=1.5)),
        fill="tozeroy", fillcolor="rgba(217,119,6,0.07)",
        hovertemplate="<b>%{x}</b><br>Avg clearance: %{y:.0f} min<extra></extra>",
    ))
    fig_dur2.update_layout(**PLOTLY, height=260, margin=dict(l=55,r=20,t=44,b=60),
                           title=dict(text="Monthly average clearance time — lower is better",
                                      font=dict(size=13,color="#111827"),x=0,xanchor="left"))
    fig_dur2.update_xaxes(**ax(tickangle=-30))
    fig_dur2.update_yaxes(**ax("Avg Duration (min)"))
    st.plotly_chart(fig_dur2, config={"displayModeBar":False}, use_container_width=True)

    if "start_date" in dff.columns:
        section_header("Daily Incident Volume Over Time", "📅")
        daily    = dff.groupby("start_date").size().reset_index(name="count")
        daily["start_date"] = pd.to_datetime(daily["start_date"])
        daily_7ma = daily["count"].rolling(7, min_periods=1).mean()
        fig_daily = go.Figure()
        fig_daily.add_trace(go.Scatter(
            x=daily["start_date"], y=daily["count"], name="Daily",
            mode="lines", line=dict(color="#bfdbfe", width=1),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>%{y}<extra></extra>",
        ))
        fig_daily.add_trace(go.Scatter(
            x=daily["start_date"], y=daily_7ma, name="7-day avg",
            mode="lines", line=dict(color="#1a56db", width=2.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>7-day avg: %{y:.1f}<extra></extra>",
        ))
        fig_daily.update_layout(**PLOTLY, height=300, margin=dict(l=55,r=20,t=44,b=60),
                                title=dict(text="Daily incidents with 7-day moving average",
                                           font=dict(size=13,color="#111827"),x=0,xanchor="left"),
                                legend=dict(orientation="h",y=-0.25,
                                            font=dict(size=11,color="#374151"),bgcolor="rgba(0,0,0,0)"))
        fig_daily.update_xaxes(**ax())
        fig_daily.update_yaxes(**ax("Incidents"))
        st.plotly_chart(fig_daily, config={"displayModeBar":False}, use_container_width=True)

    section_header("Key Findings for Command Briefing", "🔍")
    findings = []
    if "hour" in dff.columns:
        ph = int(dff["hour"].value_counts().index[0])
        findings.append(("🕐","Peak Incident Hour",f"{ph:02d}:00 hrs — ensure elevated staffing from {max(ph-1,0):02d}:30"))
    if "event_cause" in dff.columns and "requires_road_closure" in dff.columns:
        top_cl = (dff.groupby("event_cause")["requires_road_closure"].mean()
                  .sort_values(ascending=False).index[0].replace("_"," ").title())
        findings.append(("🚧","Highest Closure Cause", top_cl))
    if "corridor" in dff.columns:
        top_c = dff[dff["corridor"]!="Non-corridor"]["corridor"].value_counts().index[0]
        findings.append(("🛣️","Most Affected Corridor", top_c))
    if "duration_minutes" in dff.columns:
        p90 = dff["duration_minutes"].quantile(0.90)
        findings.append(("⏱️","90th Percentile Duration",f"{p90:.0f} min — plan resource availability for this window"))
    if "event_type" in dff.columns:
        unp = (dff["event_type"]=="unplanned").mean()*100
        findings.append(("⚡","Unplanned Event Rate",f"{unp:.0f}% of all incidents — focus reactive capacity"))

    cols = st.columns(2)
    for i, (icon, label, text) in enumerate(findings):
        with cols[i%2]:
            st.markdown(f"""
            <div style="background:#ffffff;border:1px solid #e5e7eb;
                        border-left:4px solid {C['primary']};border-radius:10px;
                        padding:14px 16px;margin-bottom:10px;
                        box-shadow:0 1px 3px rgba(0,0,0,.04);">
                <div style="font-size:11px;font-weight:700;color:#6b7280;
                            text-transform:uppercase;letter-spacing:.07em;margin-bottom:4px;">
                    {icon} {label}</div>
                <div style="font-size:14px;font-weight:600;color:#111827;">{text}</div>
            </div>""", unsafe_allow_html=True)