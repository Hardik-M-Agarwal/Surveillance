import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from utils.theme import apply_theme, sidebar_brand, page_header, section_header, C, PLOTLY, SEV_COLOR, AXIS_STYLE, AXIS_NO_GRID, LEGEND_STYLE, BASE_MARGIN

st.set_page_config(page_title="Data Insights · TrafficSense", page_icon="📊", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_brand()

# ── Load data ─────────────────────────────────────────────────────────────────
CLEANED_CSV = ROOT / "data" / "processed" / "cleaned.csv"

@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame | None:
    if not CLEANED_CSV.exists():
        return None
    df = pd.read_csv(CLEANED_CSV, low_memory=False)
    df["start_datetime"] = pd.to_datetime(df["start_datetime"], utc=True, errors="coerce")
    df["hour"]        = df["start_datetime"].dt.hour
    df["day_of_week"] = df["start_datetime"].dt.dayofweek
    df["month"]       = df["start_datetime"].dt.month
    df["requires_road_closure"] = (
        df["requires_road_closure"].astype(str).str.lower()
        .map({"true": 1, "1": 1, "false": 0, "0": 0}).fillna(0).astype(int)
    )
    df["duration_minutes"] = pd.to_numeric(df["duration_minutes"], errors="coerce")
    return df

df = load_data()

page_header("📊", "Data Insights", f"Interactive analysis of {'8,173' if df is not None else '?'} historical traffic incidents in Bengaluru")

if df is None:
    st.warning("⚠️ Processed data not found. Run `python run_pipeline.py` to generate it.")
    st.stop()

# ── Sidebar filters ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("**Filters**")

    corridors_avail = sorted(df["corridor"].dropna().unique().tolist())
    sel_corridors   = st.multiselect("Corridor", corridors_avail,
                                     default=["Mysore Road", "Bellary Road 1", "Tumkur Road",
                                              "Non-corridor", "Hosur Road"])

    causes_avail = sorted(df["event_cause"].dropna().unique().tolist())
    sel_causes   = st.multiselect("Event Cause", causes_avail, default=causes_avail[:8])

    severities   = ["Critical", "High", "Medium", "Low"]
    sel_sev      = st.multiselect("Severity", severities, default=severities)

    # Date range
    if df["start_datetime"].notna().any():
        min_date = df["start_datetime"].min().date()
        max_date = df["start_datetime"].max().date()
        date_range = st.date_input("Date Range", value=(min_date, max_date),
                                   min_value=min_date, max_value=max_date)
    else:
        date_range = None

    apply_btn = st.button("Apply Filters", use_container_width=True)

# ── Apply filters ─────────────────────────────────────────────────────────────
mask = pd.Series(True, index=df.index)
if sel_corridors:
    mask &= df["corridor"].isin(sel_corridors)
if sel_causes:
    mask &= df["event_cause"].isin(sel_causes)
if sel_sev and "severity" in df.columns:
    mask &= df["severity"].isin(sel_sev)
if date_range and len(date_range) == 2 and df["start_datetime"].notna().any():
    s, e = pd.Timestamp(date_range[0], tz="UTC"), pd.Timestamp(date_range[1], tz="UTC")
    mask &= df["start_datetime"].between(s, e)

dff = df[mask].copy()
n   = len(dff)

# ── Summary KPIs ──────────────────────────────────────────────────────────────
k1, k2, k3, k4, k5 = st.columns(5)
with k1: st.metric("Incidents", f"{n:,}")
with k2:
    med_dur = dff["duration_minutes"].dropna().median()
    st.metric("Median Duration", f"{med_dur:.0f} min" if not np.isnan(med_dur) else "—")
with k3:
    cr = dff["requires_road_closure"].mean() * 100
    st.metric("Closure Rate", f"{cr:.1f}%")
with k4:
    high_sev = ((dff["severity"].isin(["Critical","High"])).sum() / max(n,1)) * 100 if "severity" in dff.columns else 0
    st.metric("High/Critical %", f"{high_sev:.1f}%")
with k5:
    top_cause = dff["event_cause"].value_counts().index[0] if n > 0 else "—"
    st.metric("Top Cause", top_cause.replace("_"," ").title())

st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ───────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(["🕐 Temporal Patterns", "⚠️ Event Causes", "🛣️ Corridors", "📦 Severity & Duration"])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1: TEMPORAL PATTERNS
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2 = st.columns(2)

    with c1:
        section_header("Incidents by Hour", "🕐")
        hour_data  = dff.groupby("hour").size().reset_index(name="count")
        peak_hours = [4, 5, 6, 19, 20, 21, 22]
        hour_data["is_peak"] = hour_data["hour"].isin(peak_hours)
        hour_data["color"]   = hour_data["is_peak"].map({True: C["error"], False: C["primary"]})

        fig = go.Figure(go.Bar(
            x=hour_data["hour"], y=hour_data["count"],
            marker_color=hour_data["color"],
            hovertemplate="<b>%{x}:00</b> — %{y} incidents<extra></extra>",
        ))
        fig.add_annotation(text="● Peak hours highlighted in red", x=0.5, y=1.08,
                           xref="paper", yref="paper", showarrow=False,
                           font=dict(size=11, color=C["muted"]))
        fig.update_layout(**PLOTLY, title="Hourly Incident Frequency", height=300, margin=BASE_MARGIN)
        fig.update_xaxes(**AXIS_STYLE, tickmode="array", tickvals=list(range(0,24,2)),
                         ticktext=[f"{h:02d}" for h in range(0,24,2)])
        fig.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig, config={"displayModeBar": False})

    with c2:
        section_header("Day of Week Pattern", "📅")
        days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
        day_data = dff.groupby("day_of_week").size().reset_index(name="count")
        day_data["day_name"] = day_data["day_of_week"].map(dict(enumerate(days)))
        day_data["color"]    = day_data["day_of_week"].apply(
            lambda d: C["success"] if d >= 5 else C["primary"])

        fig2 = go.Figure(go.Bar(
            x=day_data["day_name"], y=day_data["count"],
            marker_color=day_data["color"],
            hovertemplate="<b>%{x}</b> — %{y} incidents<extra></extra>",
        ))
        fig2.update_layout(**PLOTLY, title="Weekly Incident Distribution", height=300, margin=BASE_MARGIN)
        fig2.update_xaxes(**AXIS_STYLE)
        fig2.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig2, config={"displayModeBar": False})

    section_header("Hour vs Day Heatmap (Incident Density)", "🔥")
    pivot = dff.pivot_table(index="day_of_week", columns="hour", values="event_cause",
                             aggfunc="count", fill_value=0)
    pivot.index = [days[int(i)] for i in pivot.index]

    fig3 = go.Figure(go.Heatmap(
        z=pivot.values, x=[f"{int(h):02d}:00" for h in pivot.columns],
        y=pivot.index, colorscale="Blues",
        hovertemplate="<b>%{y} %{x}</b><br>Incidents: %{z}<extra></extra>",
        colorbar=dict(title=dict(text="Count", font=dict(color=C["muted"])),
                      tickfont=dict(color=C["muted"])),
    ))
    fig3.update_layout(**PLOTLY, title="Incident Density: Day of Week × Hour of Day",
                       height=300, margin=BASE_MARGIN, xaxis_title="Hour")
    fig3.update_xaxes(**AXIS_NO_GRID)
    fig3.update_yaxes(**AXIS_NO_GRID)
    st.plotly_chart(fig3, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2: EVENT CAUSES
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    c1, c2 = st.columns(2)

    with c1:
        section_header("Road Closure Rate by Cause", "🚧")
        cause_closure = dff.groupby("event_cause").agg(
            incidents=("requires_road_closure", "count"),
            closure_rate=("requires_road_closure", "mean"),
        ).reset_index()
        cause_closure = cause_closure[cause_closure["incidents"] >= 10].sort_values("closure_rate")
        cause_closure["cause_label"] = cause_closure["event_cause"].str.replace("_", " ").str.title()

        fig4 = go.Figure(go.Bar(
            x=cause_closure["closure_rate"] * 100,
            y=cause_closure["cause_label"],
            orientation="h",
            marker_color=cause_closure["closure_rate"].apply(
                lambda r: C["critical"] if r > 0.6 else C["error"] if r > 0.3 else C["warning"] if r > 0.1 else C["primary"]
            ),
            hovertemplate="<b>%{y}</b><br>Closure rate: %{x:.1f}%<extra></extra>",
            text=cause_closure["closure_rate"].apply(lambda r: f"{r*100:.0f}%"),
            textposition="outside",
            textfont=dict(color=C["text"], size=11),
        ))
        fig4.update_layout(**PLOTLY, title="Road Closure Rate by Event Cause",
                           height=380, margin=BASE_MARGIN, xaxis_title="Closure Rate (%)")
        fig4.update_xaxes(**AXIS_STYLE, range=[0, 100])
        fig4.update_yaxes(**AXIS_STYLE, tickfont=dict(size=11))
        st.plotly_chart(fig4, config={"displayModeBar": False})

    with c2:
        section_header("Incident Volume by Cause", "📊")
        cause_counts = dff["event_cause"].value_counts().reset_index()
        cause_counts.columns = ["cause", "count"]
        cause_counts["cause_label"] = cause_counts["cause"].str.replace("_", " ").str.title()
        cause_counts = cause_counts.head(12)

        fig5 = go.Figure(go.Bar(
            x=cause_counts["count"], y=cause_counts["cause_label"],
            orientation="h", marker_color=C["cyan"],
            hovertemplate="<b>%{y}</b> — %{x:,} incidents<extra></extra>",
        ))
        fig5.update_layout(**PLOTLY, title="Incident Volume (Top 12 Causes)",
                           height=380, margin=BASE_MARGIN, xaxis_title="Incidents")
        fig5.update_xaxes(**AXIS_STYLE)
        fig5.update_yaxes(**AXIS_STYLE, tickfont=dict(size=11))
        st.plotly_chart(fig5, config={"displayModeBar": False})

    if "severity" in dff.columns:
        section_header("Severity Distribution by Cause", "🎯")
        cause_sev = dff.groupby(["event_cause", "severity"]).size().reset_index(name="count")
        cause_sev["cause_label"] = cause_sev["event_cause"].str.replace("_", " ").str.title()
        top_causes = dff["event_cause"].value_counts().head(10).index
        cause_sev  = cause_sev[cause_sev["event_cause"].isin(top_causes)]

        fig6 = px.bar(cause_sev, x="cause_label", y="count", color="severity",
                      color_discrete_map=SEV_COLOR, barmode="stack",
                      labels={"cause_label": "Event Cause", "count": "Incidents", "severity": "Severity"},
                      category_orders={"severity": ["Critical", "High", "Medium", "Low"]})
        fig6.update_layout(**PLOTLY, title="Severity Stack by Top 10 Event Causes",
                           height=320, margin=BASE_MARGIN)
        fig6.update_xaxes(**AXIS_STYLE, tickangle=-30)
        fig6.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig6, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3: CORRIDORS
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    c1, c2 = st.columns(2)

    with c1:
        section_header("Incident Count by Corridor", "🛣️")
        corr_counts = (dff[dff["corridor"] != "Non-corridor"]["corridor"]
                       .value_counts().reset_index())
        corr_counts.columns = ["corridor", "count"]

        fig7 = go.Figure(go.Bar(
            x=corr_counts["count"], y=corr_counts["corridor"],
            orientation="h", marker_color=C["primary"],
            hovertemplate="<b>%{y}</b> — %{x:,} incidents<extra></extra>",
        ))
        fig7.update_layout(**PLOTLY, title="Incidents per Corridor",
                           height=380, margin=BASE_MARGIN, xaxis_title="Incidents")
        fig7.update_xaxes(**AXIS_STYLE)
        fig7.update_yaxes(**AXIS_STYLE, tickfont=dict(size=11))
        st.plotly_chart(fig7, config={"displayModeBar": False})

    with c2:
        section_header("Closure Rate by Corridor", "🚧")
        corr_cl = (dff[dff["corridor"] != "Non-corridor"]
                   .groupby("corridor")["requires_road_closure"].agg(["mean", "count"])
                   .reset_index())
        corr_cl = corr_cl[corr_cl["count"] >= 20].sort_values("mean")
        corr_cl.columns = ["corridor", "closure_rate", "count"]

        fig8 = go.Figure(go.Bar(
            x=corr_cl["closure_rate"] * 100, y=corr_cl["corridor"],
            orientation="h",
            marker_color=corr_cl["closure_rate"].apply(
                lambda r: C["error"] if r > 0.2 else C["warning"] if r > 0.1 else C["success"]),
            hovertemplate="<b>%{y}</b><br>Closure rate: %{x:.1f}%<extra></extra>",
        ))
        fig8.update_layout(**PLOTLY, title="Road Closure Rate per Corridor",
                           height=380, margin=BASE_MARGIN, xaxis_title="Closure Rate (%)")
        fig8.update_xaxes(**AXIS_STYLE)
        fig8.update_yaxes(**AXIS_STYLE, tickfont=dict(size=11))
        st.plotly_chart(fig8, config={"displayModeBar": False})

    if "severity" in dff.columns:
        section_header("Corridor Risk Matrix", "📊")
        corr_sev = (dff[dff["corridor"] != "Non-corridor"]
                    .groupby(["corridor", "severity"]).size().reset_index(name="count"))
        corr_sev = corr_sev[corr_sev["corridor"].isin(
            dff["corridor"].value_counts().head(10).index)]

        fig9 = px.bar(corr_sev, x="corridor", y="count", color="severity",
                      color_discrete_map=SEV_COLOR, barmode="stack",
                      labels={"corridor": "Corridor", "count": "Incidents"},
                      category_orders={"severity": ["Critical", "High", "Medium", "Low"]})
        fig9.update_layout(**PLOTLY, title="Severity Matrix: Top 10 Corridors",
                           height=340, margin=BASE_MARGIN)
        fig9.update_xaxes(**AXIS_STYLE, tickangle=-30)
        fig9.update_yaxes(**AXIS_STYLE)
        st.plotly_chart(fig9, config={"displayModeBar": False})

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4: SEVERITY & DURATION
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    c1, c2 = st.columns(2)

    with c1:
        section_header("Severity Distribution", "🎯")
        if "severity" in dff.columns:
            sev_counts = dff["severity"].value_counts()
            sev_order  = ["Critical", "High", "Medium", "Low"]
            vals       = [sev_counts.get(s, 0) for s in sev_order]
            colors_pie = [SEV_COLOR.get(s, C["muted"]) for s in sev_order]

            fig10 = go.Figure(go.Pie(
                labels=sev_order, values=vals,
                marker_colors=colors_pie, hole=0.55,
                textinfo="label+percent", textfont_size=12,
                hovertemplate="%{label}: %{value:,} (%{percent})<extra></extra>",
            ))
            fig10.add_annotation(text=f"<b>{sum(vals):,}</b><br>incidents",
                                 x=0.5, y=0.5, font_size=14, showarrow=False,
                                 font=dict(color=C["text"]))
            fig10.update_layout(**PLOTLY, title="Severity Breakdown", height=340, margin=BASE_MARGIN,
                                showlegend=True,
                                legend=dict(bgcolor="rgba(0,0,0,0)", orientation="h", y=-0.1, font=dict(size=12)))
            st.plotly_chart(fig10, config={"displayModeBar": False})

    with c2:
        section_header("Duration Distribution by Cause", "⏱️")
        dur_data = dff[dff["duration_minutes"].between(0, 480)].copy()
        if not dur_data.empty:
            top_causes_dur = dur_data["event_cause"].value_counts().head(8).index
            dur_data = dur_data[dur_data["event_cause"].isin(top_causes_dur)]
            dur_data["cause_label"] = dur_data["event_cause"].str.replace("_", " ").str.title()

            fig11 = go.Figure()
            for cause in dur_data["cause_label"].unique():
                vals = dur_data[dur_data["cause_label"] == cause]["duration_minutes"]
                fig11.add_trace(go.Box(
                    y=vals, name=cause, boxpoints=False,
                    marker_color=C["primary"],
                    hovertemplate=f"<b>{cause}</b><br>Duration: %{{y:.0f}} min<extra></extra>",
                ))
            fig11.update_layout(**PLOTLY, title="Duration by Event Cause (0–480 min)",
                                height=340, margin=BASE_MARGIN,
                                yaxis_title="Duration (min)", showlegend=False)
            fig11.update_xaxes(**AXIS_STYLE, tickangle=-30)
            fig11.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig11, config={"displayModeBar": False})

    section_header("Key Statistical Findings", "🔍")
    key_findings = []
    if "hour" in dff.columns:
        peak_h = int(dff["hour"].value_counts().index[0])
        key_findings.append(f"🕐 **Peak incident hour:** {peak_h:02d}:00 — counterintuitively not morning rush hour")

    if "requires_road_closure" in dff.columns and "event_cause" in dff.columns:
        top_closure_cause = (dff.groupby("event_cause")["requires_road_closure"].mean()
                             .sort_values(ascending=False).index[0].replace("_", " ").title())
        key_findings.append(f"🚧 **Highest closure cause:** {top_closure_cause}")

    if "corridor" in dff.columns:
        top_corr = dff[dff["corridor"] != "Non-corridor"]["corridor"].value_counts().index[0]
        key_findings.append(f"🛣️ **Most affected corridor:** {top_corr}")

    if "duration_minutes" in dff.columns:
        p90_dur = dff["duration_minutes"].quantile(0.90)
        key_findings.append(f"⏱️ **90th percentile duration:** {p90_dur:.0f} min — plan for worst case")

    if key_findings:
        cols = st.columns(2)
        for i, finding in enumerate(key_findings):
            with cols[i % 2]:
                st.markdown(f"""
                <div style="background:#141928;border:1px solid #1e2a45;border-left:3px solid {C['primary']};
                            border-radius:10px;padding:14px 16px;margin-bottom:12px;font-size:14px;
                            color:#cbd5e1;line-height:1.5;">
                    {finding}
                </div>
                """, unsafe_allow_html=True)
