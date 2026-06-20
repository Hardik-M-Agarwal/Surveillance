import streamlit as st
import requests
import folium
from streamlit_folium import st_folium

st.set_page_config(page_title="Corridor Risk Heatmap", layout="wide")
st.title("Bengaluru Corridor Risk Map")

try:
    res = requests.get("http://127.0.0.1:8000/heatmap")
    if res.status_code == 200:
        data = res.json()
        
        m = folium.Map(location=[12.9716, 77.5946], zoom_start=11)
        
        for c in data['corridors']:
            risk = c['risk_score']
            if risk < 30: color = 'green'
            elif risk < 60: color = 'orange'
            else: color = 'red'
            
            # Since we only have centroids, we draw a circle for the corridor
            folium.CircleMarker(
                location=[c['latitude'], c['longitude']],
                radius=15,
                popup=f"{c['name']} Risk: {risk}%",
                color=color,
                fill=True,
                fill_opacity=0.7
            ).add_to(m)
            
        for hs in data['hotspots']:
            folium.Marker(
                location=[hs['latitude'], hs['longitude']],
                popup=f"Junction: {hs['name']} ({hs['count']} incidents)",
                icon=folium.Icon(color='darkred', icon='info-sign')
            ).add_to(m)
            
        st_folium(m, width=1000, height=600)
        
except requests.exceptions.ConnectionError:
    st.error("API is not running. Start the FastAPI server first: `uvicorn api.main:app --reload`")
