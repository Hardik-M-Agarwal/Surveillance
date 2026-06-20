import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils.theme import apply_theme, sidebar_brand, page_header, section_header, C, PLOTLY, AXIS_STYLE, AXIS_NO_GRID, BASE_MARGIN

st.set_page_config(page_title="Post-Event Log · TrafficSense", page_icon="📋", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_brand()

page_header("📋", "Post-Event Feedback Log",
            "Log actual incident outcomes to track model accuracy and enable continuous learning")

LOG_FILE = ROOT / "data" / "processed" / "feedback_log.csv"

# ── Load existing log ─────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_log(path: Path) -> pd.DataFrame:
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame(columns=[
        "timestamp", "incident_id", "event_cause", "corridor",
        "predicted_severity", "actual_severity",
        "predicted_duration", "actual_duration",
        "predicted_constables", "actual_constables",
        "predicted_closure", "actual_closure",
        "notes",
    ])

log_df = load_log(LOG_FILE)

col_form, col_analytics = st.columns([1, 1.5], gap="large")

# ── Intake Form ────────────────────────────────────────────────────────────────
with col_form:
    st.markdown("""<div style="background:#141928;border:1px solid #1e2a45;border-radius:14px;padding:24px;">""",
                unsafe_allow_html=True)
    st.markdown("#### 📝 Log Incident Resolution")

    with st.form("feedback_form", clear_on_submit=True):
        incident_id = st.text_input("Incident ID", placeholder="e.g. INC-2024-00847")

        col_a, col_b = st.columns(2)
        with col_a:
            event_cause = st.selectbox("Event Cause", [
                "vehicle_breakdown","accident","congestion","tree_fall","water_logging",
                "pot_holes","construction","road_conditions","public_event","procession",
                "vip_movement","protest","debris","fog_low_visibility","others",
            ])
        with col_b:
            corridor = st.selectbox("Corridor", [
                "Non-corridor","Mysore Road","Bellary Road 1","Tumkur Road","Hosur Road",
                "ORR North 1","Old Madras Road","Magadi Road","Bellary Road 2",
                "ORR East 1","Bannerghatta Road",
            ])

        st.markdown("##### Predicted vs Actual")
        col_c, col_d = st.columns(2)
        with col_c:
            pred_severity = st.selectbox("Predicted Severity", ["Low","Medium","High","Critical"])
            pred_duration = st.number_input("Predicted Duration (min)", min_value=0, value=45)
            pred_const    = st.number_input("Predicted Constables", min_value=0, value=3)
        with col_d:
            actual_severity = st.selectbox("Actual Severity", ["Low","Medium","High","Critical"])
            actual_duration = st.number_input("Actual Duration (min)", min_value=0, value=45)
            actual_const    = st.number_input("Actual Constables Deployed", min_value=0, value=3)

        col_e, col_f = st.columns(2)
        with col_e:
            pred_closure   = st.checkbox("Predicted: Road Closed")
        with col_f:
            actual_closure = st.checkbox("Actual: Road Closed")

        notes   = st.text_area("Officer Notes", placeholder="Any observations or anomalies…", height=80)
        sub_btn = st.form_submit_button("✅  Log Resolution", use_container_width=True)

        if sub_btn:
            if not incident_id.strip():
                st.error("Please provide an Incident ID.")
            else:
                new_row = {
                    "timestamp":           datetime.now().isoformat(),
                    "incident_id":         incident_id.strip(),
                    "event_cause":         event_cause,
                    "corridor":            corridor,
                    "predicted_severity":  pred_severity,
                    "actual_severity":     actual_severity,
                    "predicted_duration":  pred_duration,
                    "actual_duration":     actual_duration,
                    "predicted_constables":pred_const,
                    "actual_constables":   actual_const,
                    "predicted_closure":   pred_closure,
                    "actual_closure":      actual_closure,
                    "notes":               notes,
                }
                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([new_row]).to_csv(
                    LOG_FILE, mode="a",
                    header=not LOG_FILE.exists(),
                    index=False,
                )
                st.cache_data.clear()
                st.success(f"✅ Incident **{incident_id}** logged successfully!")
                log_df = load_log(LOG_FILE)

    st.markdown("</div>", unsafe_allow_html=True)

    # Learning cycle info
    st.markdown("""
    <div style="background:#141928;border:1px solid #1e2a45;border-radius:10px;
                padding:16px;margin-top:16px;">
        <div style="color:#e2e8f0;font-size:14px;font-weight:600;margin-bottom:8px;">
            🔄 Continuous Learning Pipeline
        </div>
        <div style="color:#94a3b8;font-size:13px;line-height:1.6;">
            Feedback logs are used to detect model drift and trigger retraining.<br><br>
            <b style="color:#e2e8f0;">Retraining schedule:</b> Weekly (Sunday 02:00 AM)<br>
            <b style="color:#e2e8f0;">Trigger threshold:</b> &gt;15% MAE degradation<br>
            <b style="color:#e2e8f0;">Min feedback required:</b> 50 incidents per cycle
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Analytics ──────────────────────────────────────────────────────────────────
with col_analytics:
    if len(log_df) == 0:
        st.markdown("""
        <div style="background:#141928;border:1px solid #1e2a45;border-radius:14px;
                    padding:60px 40px;text-align:center;">
            <div style="font-size:48px;margin-bottom:16px;">📋</div>
            <div style="color:#e2e8f0;font-size:16px;font-weight:600;">No Feedback Logged Yet</div>
            <div style="color:#94a3b8;font-size:14px;margin-top:8px;line-height:1.6;">
                Use the form on the left to log your first incident resolution.<br>
                Analytics and model performance tracking will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        log_df["timestamp"]        = pd.to_datetime(log_df["timestamp"], errors="coerce")
        log_df["predicted_duration"] = pd.to_numeric(log_df["predicted_duration"], errors="coerce")
        log_df["actual_duration"]    = pd.to_numeric(log_df["actual_duration"], errors="coerce")
        log_df["predicted_constables"] = pd.to_numeric(log_df["predicted_constables"], errors="coerce")
        log_df["actual_constables"]    = pd.to_numeric(log_df["actual_constables"], errors="coerce")

        # KPIs
        n_logs    = len(log_df)
        sev_match = (log_df["predicted_severity"] == log_df["actual_severity"]).mean() * 100
        dur_err   = (log_df["actual_duration"] - log_df["predicted_duration"]).abs().median()
        closure_acc = (log_df["predicted_closure"] == log_df["actual_closure"]).mean() * 100 \
                      if "predicted_closure" in log_df else 0

        k1, k2, k3, k4 = st.columns(4)
        with k1: st.metric("Feedback Logs", n_logs)
        with k2: st.metric("Severity Match", f"{sev_match:.0f}%")
        with k3: st.metric("Median Duration Error", f"{dur_err:.0f} min")
        with k4: st.metric("Closure Accuracy", f"{closure_acc:.0f}%")

        # Prediction vs Actual Duration scatter
        section_header("Duration: Predicted vs Actual", "⏱️")
        dur_df = log_df[["predicted_duration","actual_duration"]].dropna()
        if len(dur_df) > 0:
            perfect = [min(dur_df.min().min(), 0), max(dur_df.max().max(), 10)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=dur_df["predicted_duration"], y=dur_df["actual_duration"],
                mode="markers", name="Incidents",
                marker=dict(color=C["primary"], size=8, opacity=0.8),
                hovertemplate="Predicted: %{x:.0f} min<br>Actual: %{y:.0f} min<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=perfect, y=perfect, mode="lines", name="Perfect Prediction",
                line=dict(color=C["success"], dash="dash", width=1.5),
            ))
            fig.update_layout(**PLOTLY, title="Predicted vs Actual Duration (each dot = one incident)",
                              height=280, margin=BASE_MARGIN,
                              xaxis_title="Predicted (min)", yaxis_title="Actual (min)")
            fig.update_xaxes(**AXIS_STYLE)
            fig.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig, config={"displayModeBar": False})

        # Constable deployment accuracy
        section_header("Constable Deployment: Predicted vs Actual", "👮")
        con_df = log_df[["predicted_constables","actual_constables"]].dropna()
        if len(con_df) > 0:
            diff = con_df["actual_constables"] - con_df["predicted_constables"]
            colors_bar = [C["error"] if d > 0 else C["success"] for d in diff]
            fig2 = go.Figure(go.Bar(
                x=list(range(len(diff))), y=diff.values,
                marker_color=colors_bar,
                hovertemplate="Incident #%{x}<br>Under/Over-predict: %{y:+.0f}<extra></extra>",
            ))
            fig2.add_hline(y=0, line_color=C["muted"], line_dash="dot")
            fig2.update_layout(**PLOTLY, title="Δ Constables (Actual − Predicted): + = under-predicted",
                               height=220, margin=BASE_MARGIN,
                               xaxis_title="Incident #", yaxis_title="Δ Constables")
            fig2.update_xaxes(**AXIS_STYLE)
            fig2.update_yaxes(**AXIS_STYLE)
            st.plotly_chart(fig2, config={"displayModeBar": False})

        # Severity confusion
        section_header("Severity Prediction Accuracy", "🎯")
        sev_cols = ["predicted_severity", "actual_severity"]
        if all(c in log_df.columns for c in sev_cols):
            sev_df = log_df[sev_cols].dropna()
            if len(sev_df) > 0:
                sev_order = ["Critical", "High", "Medium", "Low"]
                conf = sev_df.groupby(sev_cols).size().reset_index(name="count")
                pivot = conf.pivot(index="actual_severity", columns="predicted_severity", values="count").fillna(0)
                pivot = pivot.reindex(index=sev_order, columns=sev_order, fill_value=0)
                fig3 = go.Figure(go.Heatmap(
                    z=pivot.values, x=sev_order, y=sev_order,
                    colorscale="Blues", showscale=True,
                    hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                    texttemplate="%{z:.0f}", textfont=dict(size=13, color="white"),
                ))
                fig3.update_layout(**PLOTLY, title="Severity Confusion Matrix",
                                   height=220, margin=BASE_MARGIN,
                                   xaxis_title="Predicted", yaxis_title="Actual")
                fig3.update_xaxes(**AXIS_NO_GRID)
                fig3.update_yaxes(**AXIS_NO_GRID)
                st.plotly_chart(fig3, config={"displayModeBar": False})

        # Recent log table
        section_header("Recent Feedback", "🗂️")
        display_cols = ["timestamp","incident_id","corridor","event_cause",
                        "predicted_severity","actual_severity","predicted_duration","actual_duration"]
        disp = log_df[[c for c in display_cols if c in log_df.columns]].tail(15).iloc[::-1]
        disp["timestamp"] = disp["timestamp"].dt.strftime("%Y-%m-%d %H:%M") if hasattr(disp["timestamp"], "dt") else disp["timestamp"]
        st.dataframe(disp, use_container_width=True, hide_index=True)
