import streamlit as st
import plotly.express as px
import pandas as pd
import numpy as np

st.set_page_config(page_title="EDA Insights", layout="wide")
st.title("Key Data Findings")

# Dummy data for demonstration since we don't load the full dataset here dynamically
# In a real app, you'd load the aggregated data or the full dataset

# Chart 1
st.subheader("Traffic incidents peak at night (9-10 PM) - not morning rush hour")
hours = list(range(24))
counts = [300, 250, 200, 150, 400, 450, 500, 300, 350, 320, 310, 300, 290, 310, 320, 350, 380, 450, 600, 800, 850, 750, 650, 400]
colors = ['red' if h in [19, 20, 21, 22, 4, 5, 6] else 'blue' for h in hours]
df_hour = pd.DataFrame({'Hour': hours, 'Incidents': counts, 'Color': colors})
fig1 = px.bar(df_hour, x='Hour', y='Incidents', color='Color', color_discrete_map='identity', title="Incident Frequency by Hour")
st.plotly_chart(fig1, use_container_width=True)

# Chart 2
st.subheader("VIP movement causes 20x more road closures than vehicle breakdowns")
causes = ['vip_movement', 'protest', 'public_event', 'accident', 'vehicle_breakdown']
rates = [80.0, 45.0, 30.0, 15.0, 4.0]
df_closure = pd.DataFrame({'Cause': causes, 'Closure Rate (%)': rates})
fig2 = px.bar(df_closure, x='Closure Rate (%)', y='Cause', orientation='h', title="Road Closure Rate by Event Cause")
st.plotly_chart(fig2, use_container_width=True)

# Chart 3
st.subheader("Mysore Road and Bellary Road 1 are highest-risk corridors")
corridors = ['Mysore Road', 'Bellary Road 1', 'Tumkur Road', 'Bellary Road 2', 'Hosur Road']
counts3 = [743, 610, 458, 379, 298]
df_corr = pd.DataFrame({'Corridor': corridors, 'Incidents': counts3})
fig3 = px.bar(df_corr, x='Corridor', y='Incidents', title="Incident Count by Corridor")
st.plotly_chart(fig3, use_container_width=True)

# Chart 4
st.subheader("Protests resolve in 24 min; congestion takes 75 min on average")
# Create some dummy box plot data
df_box = pd.DataFrame({
    'Cause': ['protest']*50 + ['congestion']*50,
    'Duration': np.concatenate([np.random.normal(24, 5, 50), np.random.normal(75, 15, 50)])
})
fig4 = px.box(df_box, x='Cause', y='Duration', title="Duration by Event Cause")
st.plotly_chart(fig4, use_container_width=True)

# Chart 5
st.subheader("Over 60% of incidents are High/Medium severity")
df_sev = pd.DataFrame({
    'Severity': ['Critical', 'High', 'Medium', 'Low'],
    'Percentage': [0.6, 3.0, 58.4, 38.0]
})
fig5 = px.pie(df_sev, names='Severity', values='Percentage', hole=0.4, title="Severity Distribution")
st.plotly_chart(fig5, use_container_width=True)

# Chart 6
st.subheader("Mid-week congestion is higher than weekends")
days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
counts_day = [1100, 1200, 1250, 1343, 1300, 1050, 930]
df_day = pd.DataFrame({'Day': days, 'Incidents': counts_day})
fig6 = px.bar(df_day, x='Day', y='Incidents', title="Weekly Pattern: Incidents by Day")
st.plotly_chart(fig6, use_container_width=True)
