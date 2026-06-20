import streamlit as st

st.set_page_config(
    page_title="TrafficSense",
    page_icon="🚦",
    layout="wide"
)

st.title("TrafficSense: Event-Driven Congestion Forecasting")
st.markdown("""
Welcome to **TrafficSense**, an intelligent event-driven congestion forecasting and resource recommendation engine for Bengaluru city traffic police.

Please select a page from the sidebar:
- **Live Prediction**: Officer intake form and live prediction dashboard
- **Heatmap**: Live Bengaluru corridor risk map
- **EDA Insights**: Key historical findings
- **Post-Event Log**: Feedback loop for continuous learning
""")
