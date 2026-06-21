"""
Post-Event Feedback Log — Log incident outcomes, track model accuracy,
drive continuous learning. Full light-theme redesign.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "dashboard"))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime

from utils.theme import (apply_theme, sidebar_brand, page_header,
                         section_header, kpi_card, C, PLOTLY, BASE_MARGIN)

st.set_page_config(page_title="Post-Event Log · TrafficSense",
                   page_icon="📋", layout="wide")
apply_theme()

with st.sidebar:
    sidebar_brand()

LOG_FILE = ROOT / "data" / "processed" / "feedback_log.csv"
MIN_FEEDBACK   = 5    # incidents needed to trigger retraining cycle
RETRAIN_THRESH = 15.0  # % MAE degradation threshold

COLS = [
    "timestamp","incident_id","event_cause","corridor",
    "predicted_severity","actual_severity",
    "predicted_duration","actual_duration",
    "predicted_constables","actual_constables",
    "predicted_closure","actual_closure","notes",
]

EVENT_CAUSES = [
    "vehicle_breakdown","accident","congestion","tree_fall","water_logging",
    "pot_holes","construction","road_conditions","public_event","procession",
    "vip_movement","protest","debris","fog_low_visibility","others",
]
CORRIDORS = [
    "Non-corridor","Mysore Road","Bellary Road 1","Tumkur Road","Hosur Road",
    "ORR North 1","Old Madras Road","Magadi Road","Bellary Road 2",
    "ORR East 1","Bannerghatta Road",
]
SEV_LEVELS = ["Low","Medium","High","Critical"]
SEV_COLORS = {"Critical":"#b91c1c","High":"#ea580c","Medium":"#d97706","Low":"#15803d"}

# ── Session state ──────────────────────────────────────────────────────────────
if "log_df" not in st.session_state:
    st.session_state.log_df = None
if "log_success" not in st.session_state:
    st.session_state.log_success = None

def load_log() -> pd.DataFrame:
    if LOG_FILE.exists():
        df = pd.read_csv(LOG_FILE)
        df["timestamp"]            = pd.to_datetime(df["timestamp"], errors="coerce")
        df["predicted_duration"]   = pd.to_numeric(df["predicted_duration"],   errors="coerce")
        df["actual_duration"]      = pd.to_numeric(df["actual_duration"],       errors="coerce")
        df["predicted_constables"] = pd.to_numeric(df["predicted_constables"], errors="coerce")
        df["actual_constables"]    = pd.to_numeric(df["actual_constables"],     errors="coerce")
        return df
    return pd.DataFrame(columns=COLS)

def refresh_log():
    st.session_state.log_df = load_log()

if st.session_state.log_df is None:
    refresh_log()

log_df = st.session_state.log_df

# ── Page header ────────────────────────────────────────────────────────────────
page_header("📋", "Post-Event Feedback Log",
            "Log actual outcomes after each incident · improve ML predictions over time")

# ── Why this matters banner ────────────────────────────────────────────────────
n_logged   = len(log_df)
n_needed   = max(MIN_FEEDBACK - n_logged, 0)
pct_done   = min(n_logged / MIN_FEEDBACK * 100, 100)
cycle_color= "#15803d" if pct_done >= 100 else "#1a56db" if pct_done >= 60 else "#d97706"

_threshold_badge = (
    '<div style="font-size:11px;color:#15803d;font-weight:700;margin-top:5px;">'
    + chr(9989) + ' Threshold reached ' + chr(8212) + ' retraining queued</div>'
    if pct_done >= 100 else ""
)

def _build_banner(n_needed, cycle_color, n_logged, MIN_FEEDBACK, pct_done,
                  threshold_badge, RETRAIN_THRESH):
    lines = [
        '<div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;'
        'padding:16px 22px;margin-bottom:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);'
        'display:flex;align-items:center;gap:24px;">',

        '<div style="flex:1;">',
        '<div style="font-size:13px;font-weight:700;color:#111827;margin-bottom:3px;">',
        chr(128260) + ' Continuous Learning Pipeline</div>',
        '<div style="font-size:12px;color:#6b7280;line-height:1.7;">',
        "Every log you submit improves the ML model predictions for your corridor and shift. ",
        '<b style="color:#111827;">' + str(n_needed) + " more incidents</b>",
        " needed to trigger the next retraining cycle.</div>",
        '</div>',

        '<div style="flex:2;">',
        '<div style="display:flex;justify-content:space-between;',
        'font-size:11px;font-weight:600;color:#6b7280;margin-bottom:5px;">',
        '<span>Feedback this cycle</span>',
        '<span style="color:' + cycle_color + ';">' + str(n_logged) + ' / ' + str(MIN_FEEDBACK) + '</span>',
        '</div>',
        '<div style="background:#f3f4f6;border-radius:99px;height:10px;">',
        '<div style="background:' + cycle_color + ';width:' + str(int(pct_done)) + '%;height:10px;border-radius:99px;"></div>',
        '</div>',
        threshold_badge,
        '</div>',

        '<div style="text-align:center;flex-shrink:0;background:#f8fafc;border-radius:10px;padding:10px 18px;">',
        '<div style="font-size:11px;color:#6b7280;font-weight:600;text-transform:uppercase;letter-spacing:.06em;margin-bottom:2px;">Retrain Trigger</div>',
        '<div style="font-size:13px;font-weight:700;color:' + ('#15803d' if pct_done >= 100 else '#111827') + ';">' + ('✅ Ready' if pct_done >= 100 else str(MIN_FEEDBACK - n_logged) + ' more incidents') + '</div>',
        '<div style="font-size:10px;color:#9ca3af;margin-top:1px;">Trigger: ' + str(MIN_FEEDBACK) + ' incidents logged</div>',
        '</div>',

        '</div>',
    ]
    return "".join(lines)

st.markdown(
    _build_banner(n_needed, cycle_color, n_logged, MIN_FEEDBACK,
                  pct_done, _threshold_badge, RETRAIN_THRESH),
    unsafe_allow_html=True,
)

# ── Success message ────────────────────────────────────────────────────────────
if st.session_state.log_success:
    st.success(st.session_state.log_success)
    st.session_state.log_success = None

# ── Layout ─────────────────────────────────────────────────────────────────────
col_form, col_analytics = st.columns([1, 1.6], gap="large")

# ══════════════════════════════════════════════════════════════════════════════
# LEFT — LOGGING FORM
# ══════════════════════════════════════════════════════════════════════════════
with col_form:
    st.markdown("""
    <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                padding:20px;box-shadow:0 1px 4px rgba(0,0,0,.05);margin-bottom:12px;">
        <div style="font-size:12px;font-weight:700;color:#6b7280;text-transform:uppercase;
                    letter-spacing:.08em;margin-bottom:16px;">📝 Log Incident Resolution</div>
    """, unsafe_allow_html=True)

    with st.form("feedback_form", clear_on_submit=True):
        incident_id = st.text_input("Incident ID", placeholder="e.g. INC-2024-00847")

        col_a, col_b = st.columns(2)
        with col_a:
            event_cause = st.selectbox("Event Cause", EVENT_CAUSES,
                                       format_func=lambda x: x.replace("_"," ").title())
        with col_b:
            corridor = st.selectbox("Corridor", CORRIDORS)

        # Divider
        st.markdown("""
        <div style="height:1px;background:#f3f4f6;margin:8px 0 12px;"></div>
        <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                    letter-spacing:.07em;margin-bottom:10px;">Predicted vs Actual Outcomes</div>
        """, unsafe_allow_html=True)

        col_c, col_d = st.columns(2)
        with col_c:
            st.markdown('<div style="font-size:11px;font-weight:600;color:#1a56db;margin-bottom:6px;">🤖 ML Predicted</div>', unsafe_allow_html=True)
            pred_severity = st.selectbox("Predicted Severity", SEV_LEVELS, key="ps")
            pred_duration = st.number_input("Predicted Duration (min)", min_value=0, value=45, key="pd")
            pred_const    = st.number_input("Predicted Constables",     min_value=0, value=3,  key="pc")
            pred_closure  = st.checkbox("Predicted: Road Closed", key="pcl")
        with col_d:
            st.markdown('<div style="font-size:11px;font-weight:600;color:#15803d;margin-bottom:6px;">✅ Actual Outcome</div>', unsafe_allow_html=True)
            actual_severity = st.selectbox("Actual Severity",          SEV_LEVELS, key="as_")
            actual_duration = st.number_input("Actual Duration (min)", min_value=0, value=45, key="ad")
            actual_const    = st.number_input("Actual Constables",     min_value=0, value=3,  key="ac")
            actual_closure  = st.checkbox("Actual: Road Closed", key="acl")

        notes   = st.text_area("Officer Notes", placeholder="Any anomalies, unusual conditions, or observations…", height=70)
        sub_btn = st.form_submit_button("Submit Feedback", use_container_width=True)

        if sub_btn:
            if not incident_id.strip():
                st.error("Please provide an Incident ID.")
            elif n_logged > 0 and incident_id.strip() in log_df["incident_id"].astype(str).values:
                st.warning(f"Incident {incident_id} already logged. Check for duplicates.")
            else:
                new_row = {
                    "timestamp":            datetime.now().isoformat(),
                    "incident_id":          incident_id.strip(),
                    "event_cause":          event_cause,
                    "corridor":             corridor,
                    "predicted_severity":   pred_severity,
                    "actual_severity":      actual_severity,
                    "predicted_duration":   pred_duration,
                    "actual_duration":      actual_duration,
                    "predicted_constables": pred_const,
                    "actual_constables":    actual_const,
                    "predicted_closure":    pred_closure,
                    "actual_closure":       actual_closure,
                    "notes":                notes,
                }
                LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame([new_row]).to_csv(
                    LOG_FILE, mode="a",
                    header=not LOG_FILE.exists(),
                    index=False,
                )
                st.session_state.log_success = f"✅ Incident {incident_id.strip()} logged successfully!"
                refresh_log()
                log_df = st.session_state.log_df
                st.rerun()

    st.markdown("</div>", unsafe_allow_html=True)

    # Download log
    if n_logged > 0:
        csv_bytes = log_df.to_csv(index=False).encode()
        st.download_button(
            "⬇ Download Full Log (CSV)",
            data=csv_bytes,
            file_name=f"feedback_log_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Model accuracy insight card (if enough data)
    if n_logged >= 5:
        sev_acc  = (log_df["predicted_severity"]==log_df["actual_severity"]).mean()*100
        dur_mae  = (log_df["actual_duration"]-log_df["predicted_duration"]).abs().mean()
        const_mae= (log_df["actual_constables"]-log_df["predicted_constables"]).abs().mean()

        acc_color = "#15803d" if sev_acc>=80 else "#d97706" if sev_acc>=60 else "#b91c1c"
        dur_color = "#15803d" if dur_mae<=15 else "#d97706" if dur_mae<=30 else "#b91c1c"

        st.markdown(f"""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:10px;
                    padding:14px 16px;margin-top:10px;box-shadow:0 1px 3px rgba(0,0,0,.04);">
            <div style="font-size:11px;font-weight:700;color:#6b7280;text-transform:uppercase;
                        letter-spacing:.07em;margin-bottom:10px;">📊 Current Model Health</div>
            <div style="display:flex;flex-direction:column;gap:8px;">
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:12px;color:#374151;">Severity accuracy</span>
                    <span style="font-size:13px;font-weight:700;color:{acc_color};">{sev_acc:.0f}%</span>
                </div>
                <div style="background:#f3f4f6;border-radius:99px;height:5px;">
                    <div style="background:{acc_color};width:{sev_acc:.0f}%;height:5px;border-radius:99px;"></div>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;margin-top:4px;">
                    <span style="font-size:12px;color:#374151;">Avg duration error</span>
                    <span style="font-size:13px;font-weight:700;color:{dur_color};">{dur_mae:.0f} min</span>
                </div>
                <div style="display:flex;justify-content:space-between;align-items:center;">
                    <span style="font-size:12px;color:#374151;">Avg constable error</span>
                    <span style="font-size:13px;font-weight:700;color:#374151;">{const_mae:.1f}</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# RIGHT — ANALYTICS
# ══════════════════════════════════════════════════════════════════════════════
with col_analytics:

    if n_logged == 0:
        st.markdown("""
        <div style="background:#ffffff;border:1px solid #e5e7eb;border-radius:12px;
                    padding:64px 40px;text-align:center;box-shadow:0 1px 4px rgba(0,0,0,.05);">
            <div style="font-size:40px;margin-bottom:14px;opacity:.35;">📋</div>
            <div style="color:#111827;font-size:16px;font-weight:600;margin-bottom:8px;">
                No Feedback Logged Yet
            </div>
            <div style="color:#6b7280;font-size:13px;line-height:1.8;max-width:380px;margin:0 auto;">
                Use the form on the left to log your first incident resolution.
                Model accuracy tracking, prediction drift analysis, and
                corridor-level insights will appear here.
            </div>
        </div>
        """, unsafe_allow_html=True)

    else:
        # ── KPI strip ──────────────────────────────────────────────────────────
        sev_acc     = (log_df["predicted_severity"]==log_df["actual_severity"]).mean()*100
        dur_mae     = (log_df["actual_duration"]-log_df["predicted_duration"]).abs().median()
        cl_acc      = (log_df["predicted_closure"]==log_df["actual_closure"]).mean()*100 \
                      if "predicted_closure" in log_df else 0
        const_bias  = (log_df["actual_constables"]-log_df["predicted_constables"]).mean()

        k1,k2,k3,k4 = st.columns(4)
        for col, icon, lbl, val, acc in [
            (k1,"📝","Total Logs",          f"{n_logged}",             C["primary"]),
            (k2,"🎯","Severity Accuracy",   f"{sev_acc:.0f}%",
             C["success"] if sev_acc>=80 else C["warning"] if sev_acc>=60 else C["error"]),
            (k3,"⏱️","Median Duration Err", f"{dur_mae:.0f} min",
             C["success"] if dur_mae<=15 else C["warning"] if dur_mae<=30 else C["error"]),
            (k4,"🔒","Closure Accuracy",    f"{cl_acc:.0f}%",
             C["success"] if cl_acc>=80 else C["warning"] if cl_acc>=60 else C["error"]),
        ]:
            col.markdown(kpi_card(icon, lbl, val, accent=acc), unsafe_allow_html=True)

        st.markdown("<div style='height:6px'></div>", unsafe_allow_html=True)

        # Constable bias note
        bias_color = "#b91c1c" if abs(const_bias)>2 else "#d97706" if abs(const_bias)>1 else "#15803d"
        bias_text  = (f"Model consistently <b>under-predicts</b> constables by {abs(const_bias):.1f} on average"
                      if const_bias > 0 else
                      f"Model consistently <b>over-predicts</b> constables by {abs(const_bias):.1f} on average"
                      if const_bias < 0 else
                      "Constable predictions are well-calibrated")
        st.markdown(f"""
        <div style="background:#f8fafc;border:1px solid #e5e7eb;border-left:3px solid {bias_color};
                    border-radius:8px;padding:10px 14px;margin-bottom:16px;
                    font-size:13px;color:#374151;">
            👮 {bias_text} — avg bias: <b style="color:{bias_color};">{const_bias:+.1f}</b>
        </div>
        """, unsafe_allow_html=True)

        # ── Chart 1: Duration scatter ──────────────────────────────────────────
        section_header("Predicted vs Actual Duration", "⏱️")
        dur_df = log_df[["predicted_duration","actual_duration","corridor"]].dropna()
        if len(dur_df) > 0:
            perfect = [0, max(dur_df[["predicted_duration","actual_duration"]].max().max()*1.1, 10)]
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=perfect, y=perfect, mode="lines", name="Perfect prediction",
                line=dict(color="#15803d", dash="dash", width=1.5),
                hoverinfo="skip",
            ))
            fig.add_trace(go.Scatter(
                x=dur_df["predicted_duration"], y=dur_df["actual_duration"],
                mode="markers", name="Incidents",
                marker=dict(color=C["primary"], size=9, opacity=0.75,
                            line=dict(color="#fff", width=1.5)),
                text=dur_df["corridor"],
                hovertemplate="<b>%{text}</b><br>Predicted: %{x:.0f} min<br>Actual: %{y:.0f} min<extra></extra>",
            ))
            fig.update_layout(
                **PLOTLY, height=280, margin=dict(l=55,r=20,t=36,b=50),
                title=dict(text="Each dot = one incident · dots above the line = under-predicted",
                           font=dict(size=12,color="#6b7280"),x=0,xanchor="left"),
                xaxis=dict(gridcolor="#e5e7eb",linecolor="#d1d5db",title="Predicted (min)",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
                yaxis=dict(gridcolor="#e5e7eb",linecolor="#d1d5db",title="Actual (min)",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
                legend=dict(orientation="h",y=-0.2,font=dict(size=11,color="#374151"),
                            bgcolor="rgba(0,0,0,0)"),
            )
            st.plotly_chart(fig, config={"displayModeBar":False}, use_container_width=True)

        # ── Chart 2: Constable delta ───────────────────────────────────────────
        section_header("Constable Deployment Gap", "👮")
        con_df = log_df[["predicted_constables","actual_constables","incident_id"]].dropna()
        if len(con_df) > 0:
            diff = con_df["actual_constables"] - con_df["predicted_constables"]
            fig2 = go.Figure(go.Bar(
                x=list(range(len(diff))), y=diff.values,
                marker_color=["#b91c1c" if d>0 else "#15803d" for d in diff],
                text=[f"{d:+.0f}" for d in diff], textposition="outside",
                hovertemplate="Incident #%{x}<br>Gap: %{y:+.0f} constables<extra></extra>",
            ))
            fig2.add_hline(y=0, line=dict(color="#9ca3af", dash="dot", width=1.5))
            fig2.update_layout(
                **PLOTLY, height=220, margin=dict(l=50,r=20,t=36,b=50),
                title=dict(text="Red = under-predicted (more needed), Green = over-predicted",
                           font=dict(size=12,color="#6b7280"),x=0,xanchor="left"),
                xaxis=dict(gridcolor="#e5e7eb",linecolor="#d1d5db",title="Incident #",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
                yaxis=dict(gridcolor="#e5e7eb",linecolor="#d1d5db",title="Δ Constables",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
            )
            st.plotly_chart(fig2, config={"displayModeBar":False}, use_container_width=True)

        # ── Chart 3: Severity confusion matrix ────────────────────────────────
        section_header("Severity Prediction Accuracy", "🎯")
        sev_df = log_df[["predicted_severity","actual_severity"]].dropna()
        if len(sev_df) > 0:
            sev_order = ["Low","Medium","High","Critical"]
            conf  = sev_df.groupby(["actual_severity","predicted_severity"]).size().reset_index(name="count")
            pivot = conf.pivot(index="actual_severity",columns="predicted_severity",values="count").fillna(0)
            pivot = pivot.reindex(index=sev_order, columns=sev_order, fill_value=0)
            fig3  = go.Figure(go.Heatmap(
                z=pivot.values, x=sev_order, y=sev_order,
                colorscale=[[0,"#f0f9ff"],[0.5,"#93c5fd"],[1.0,"#1a56db"]],
                showscale=False,
                hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>",
                texttemplate="%{z:.0f}",
                textfont=dict(size=14, color="#111827"),
            ))
            fig3.update_layout(
                **PLOTLY, height=220, margin=dict(l=70,r=20,t=44,b=60),
                title=dict(text="Diagonal = correct predictions",
                           font=dict(size=12,color="#6b7280"),x=0,xanchor="left"),
                xaxis=dict(gridcolor="rgba(0,0,0,0)",linecolor="#d1d5db",
                           title="Predicted Severity",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
                yaxis=dict(gridcolor="rgba(0,0,0,0)",linecolor="#d1d5db",
                           title="Actual Severity",
                           tickfont=dict(color="#111827"),title_font=dict(color="#111827",size=12)),
            )
            st.plotly_chart(fig3, config={"displayModeBar":False}, use_container_width=True)

        # ── Corridor accuracy breakdown ────────────────────────────────────────
        if n_logged >= 5 and "corridor" in log_df.columns:
            section_header("Model Accuracy by Corridor", "🛣️")
            corr_acc = log_df.groupby("corridor").agg(
                logs         =("incident_id",         "count"),
                sev_acc      =("predicted_severity",
                               lambda x: (x==log_df.loc[x.index,"actual_severity"]).mean()*100),
                dur_mae      =("predicted_duration",
                               lambda x: (log_df.loc[x.index,"actual_duration"]-x).abs().mean()),
            ).reset_index()
            corr_acc = corr_acc[corr_acc["logs"] >= 2].sort_values("sev_acc", ascending=False)

            if not corr_acc.empty:
                st.markdown("""
                <div style="display:grid;grid-template-columns:140px 50px 100px 90px;
                            gap:4px;padding:7px 12px;background:#f3f4f6;border-radius:7px;
                            margin-bottom:5px;font-size:10px;font-weight:700;color:#6b7280;
                            text-transform:uppercase;letter-spacing:.06em;">
                    <div>Corridor</div><div>Logs</div><div>Sev Accuracy</div><div>Dur Error</div>
                </div>""", unsafe_allow_html=True)

                for _, row in corr_acc.iterrows():
                    sa  = row["sev_acc"]
                    sc  = "#15803d" if sa>=80 else "#d97706" if sa>=60 else "#b91c1c"
                    dm  = row["dur_mae"]
                    dc  = "#15803d" if dm<=15 else "#d97706" if dm<=30 else "#b91c1c"
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:140px 50px 100px 90px;
                                gap:4px;padding:8px 12px;background:#ffffff;
                                border:1px solid #e5e7eb;border-radius:7px;margin-bottom:4px;
                                font-size:12px;">
                        <div style="font-weight:600;color:#111827;">{row['corridor']}</div>
                        <div style="color:#6b7280;">{int(row['logs'])}</div>
                        <div style="font-weight:700;color:{sc};">{sa:.0f}%</div>
                        <div style="font-weight:700;color:{dc};">
                            {"—" if np.isnan(dm) else f"{dm:.0f} min"}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Cause accuracy breakdown ───────────────────────────────────────────
        if n_logged >= 5 and "event_cause" in log_df.columns:
            section_header("Model Accuracy by Event Cause", "🔍")
            cause_acc = log_df.groupby("event_cause").agg(
                logs    =("incident_id", "count"),
                sev_acc =("predicted_severity",
                          lambda x: (x==log_df.loc[x.index,"actual_severity"]).mean()*100),
                dur_mae =("predicted_duration",
                          lambda x: (log_df.loc[x.index,"actual_duration"]-x).abs().mean()),
            ).reset_index()
            cause_acc = cause_acc[cause_acc["logs"]>=2].sort_values("sev_acc", ascending=False)
            cause_acc["cause_label"] = cause_acc["event_cause"].str.replace("_"," ").str.title()

            if not cause_acc.empty:
                # Highlight worst performing cause
                worst = cause_acc.iloc[-1]
                wc    = "#b91c1c"
                st.markdown(f"""
                <div style="background:#fef2f2;border:1px solid #fecaca;border-radius:8px;
                            padding:10px 14px;margin-bottom:10px;font-size:12px;color:#374151;">
                    ⚠️ Lowest accuracy cause:
                    <b style="color:{wc};">{worst['cause_label']}</b>
                    — {worst['sev_acc']:.0f}% severity match, avg {worst['dur_mae']:.0f} min duration error.
                    Consider adjusting manual estimates for this cause type.
                </div>
                """, unsafe_allow_html=True)

                st.markdown("""
                <div style="display:grid;grid-template-columns:150px 50px 100px 90px;
                            gap:4px;padding:7px 12px;background:#f3f4f6;border-radius:7px;
                            margin-bottom:5px;font-size:10px;font-weight:700;color:#6b7280;
                            text-transform:uppercase;letter-spacing:.06em;">
                    <div>Event Cause</div><div>Logs</div><div>Sev Accuracy</div><div>Dur Error</div>
                </div>""", unsafe_allow_html=True)

                for _, row in cause_acc.iterrows():
                    sa = row["sev_acc"]
                    sc = "#15803d" if sa>=80 else "#d97706" if sa>=60 else "#b91c1c"
                    dm = row["dur_mae"]
                    dc = "#15803d" if dm<=15 else "#d97706" if dm<=30 else "#b91c1c"
                    st.markdown(f"""
                    <div style="display:grid;grid-template-columns:150px 50px 100px 90px;
                                gap:4px;padding:8px 12px;background:#ffffff;
                                border:1px solid #e5e7eb;border-radius:7px;margin-bottom:4px;
                                font-size:12px;">
                        <div style="font-weight:600;color:#111827;">{row['cause_label']}</div>
                        <div style="color:#6b7280;">{int(row['logs'])}</div>
                        <div style="font-weight:700;color:{sc};">{sa:.0f}%</div>
                        <div style="font-weight:700;color:{dc};">
                            {"—" if np.isnan(dm) else f"{dm:.0f} min"}</div>
                    </div>
                    """, unsafe_allow_html=True)

        # ── Learning Curve + Retrain Trigger ──────────────────────────────────
        if n_logged >= MIN_FEEDBACK:
            section_header("Model Learning — Cumulative Accuracy Trend", "📈")

            # Compute rolling accuracy as more incidents are logged
            sorted_log = log_df.sort_values("timestamp").reset_index(drop=True)
            cumulative_acc = []
            for i in range(1, len(sorted_log) + 1):
                window  = sorted_log.iloc[:i]
                acc     = (window["predicted_severity"] == window["actual_severity"]).mean() * 100
                dur_err = (window["actual_duration"] - window["predicted_duration"]).abs().mean()
                cumulative_acc.append({"n": i, "sev_acc": acc, "dur_mae": dur_err})
            cum_df = pd.DataFrame(cumulative_acc)

            fig_lc = go.Figure()
            fig_lc.add_trace(go.Scatter(
                x=cum_df["n"], y=cum_df["sev_acc"],
                name="Severity Accuracy",
                mode="lines+markers",
                line=dict(color=C["success"], width=2.5),
                marker=dict(size=7, color=C["success"],
                            line=dict(color="#fff", width=1.5)),
                hovertemplate="After %{x} logs: Severity accuracy = %{y:.1f}%<extra></extra>",
            ))
            fig_lc.add_hline(y=80, line=dict(color=C["success"], dash="dot", width=1),
                             annotation_text="80% target",
                             annotation_position="top right",
                             annotation_font=dict(color=C["success"], size=10))
            fig_lc.update_layout(
                **PLOTLY, height=230,
                margin=dict(l=55, r=30, t=44, b=50),
                title=dict(
                    text="Severity prediction accuracy as more real outcomes are logged",
                    font=dict(size=12, color="#6b7280"), x=0, xanchor="left",
                ),
                xaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
                           title="Number of incidents logged",
                           tickfont=dict(color="#111827"),
                           title_font=dict(color="#111827", size=12)),
                yaxis=dict(gridcolor="#e5e7eb", linecolor="#d1d5db",
                           title="Severity Accuracy (%)", range=[0, 105],
                           tickfont=dict(color="#111827"),
                           title_font=dict(color="#111827", size=12)),
            )
            st.plotly_chart(fig_lc, config={"displayModeBar": False},
                            use_container_width=True)

            # Retrain-ready call-to-action
            final_acc = cum_df["sev_acc"].iloc[-1]
            st.markdown(f"""
            <div style="background:#f0fdf4;border:1px solid #bbf7d0;border-radius:10px;
                        padding:14px 18px;margin-top:4px;">
                <div style="font-size:13px;font-weight:700;color:#15803d;margin-bottom:6px;">
                    ✅ Retraining Threshold Reached — {n_logged} incidents logged
                </div>
                <div style="font-size:12px;color:#374151;line-height:1.8;">
                    Current field-validated severity accuracy: <b>{final_acc:.0f}%</b>.
                    The feedback dataset is ready to be merged with the historical training
                    corpus to retrain the XGBoost severity and LightGBM duration/closure models.
                    Run <code>python run_pipeline.py --retrain-with-feedback</code> to trigger
                    the full retraining cycle.
                </div>
            </div>
            """, unsafe_allow_html=True)

        # ── Recent log table ───────────────────────────────────────────────────
        section_header("Recent Feedback Entries", "🗂️")
        disp_cols = {
            "timestamp":           "Time",
            "incident_id":         "Incident ID",
            "corridor":            "Corridor",
            "event_cause":         "Cause",
            "predicted_severity":  "Pred Sev",
            "actual_severity":     "Actual Sev",
            "predicted_duration":  "Pred Dur",
            "actual_duration":     "Actual Dur",
        }
        disp = log_df[[c for c in disp_cols if c in log_df.columns]].tail(12).iloc[::-1].copy()
        disp.columns = [disp_cols[c] for c in disp.columns]
        if "Time" in disp.columns:
            disp["Time"] = pd.to_datetime(disp["Time"], errors="coerce").dt.strftime("%d %b %H:%M")
        if "Cause" in disp.columns:
            disp["Cause"] = disp["Cause"].str.replace("_"," ").str.title()
        if "Pred Dur" in disp.columns:
            disp["Pred Dur"] = disp["Pred Dur"].apply(lambda v: f"{v:.0f} min" if pd.notna(v) else "—")
        if "Actual Dur" in disp.columns:
            disp["Actual Dur"] = disp["Actual Dur"].apply(lambda v: f"{v:.0f} min" if pd.notna(v) else "—")
        st.dataframe(disp, use_container_width=True, hide_index=True)