import streamlit as st
import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Live Prediction", layout="wide")
st.title("Live Incident Prediction & Recommendation")

col1, col2 = st.columns([1, 1.5])

with col1:
    st.header("Officer Intake Form")
    with st.form("intake_form"):
        event_cause = st.selectbox("Event Cause", ['vehicle_breakdown', 'others', 'pot_holes', 'construction', 'water_logging', 'accident', 'tree_fall', 'road_conditions', 'congestion', 'public_event', 'procession', 'vip_movement', 'protest', 'debris', 'fog_low_visibility'])
        corridor = st.selectbox("Corridor", ['Non-corridor', 'Mysore Road', 'Bellary Road 1', 'Tumkur Road', 'Hosur Road', 'ORR North 1', 'Old Madras Road', 'Magadi Road', 'Bellary Road 2', 'ORR East 1', 'Bannerghatta Road'])
        event_type = st.radio("Event Type", ["unplanned", "planned"])
        hour = st.slider("Hour of Day", 0, 23, 12)
        day_of_week = st.selectbox("Day of Week", [0, 1, 2, 3, 4, 5, 6], format_func=lambda x: ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'][x])
        veh_type = st.selectbox("Vehicle Type", ["unknown", "bmtc_bus", "heavy_vehicle", "lcv", "others", "private_bus", "private_car", "truck", "ksrtc_bus", "taxi", "auto"])
        police_station = st.selectbox("Police Station", ["Yelahanka", "Madiwala", "K R Puram", "Electronic City", "Whitefield", "Yeshwanthapura", "High Grounds", "Hebbal"])
        
        submitted = st.form_submit_button("Predict Impact")

with col2:
    if submitted:
        payload = {
            "event_type": event_type,
            "event_cause": event_cause,
            "latitude": 12.9716, # default mock
            "longitude": 77.5946, # default mock
            "corridor": corridor,
            "police_station": police_station,
            "hour": hour,
            "day_of_week": day_of_week,
            "veh_type": veh_type,
            "priority": "High"
        }
        
        with st.spinner("Predicting..."):
            try:
                res = requests.post("http://127.0.0.gov:8000/predict", json=payload)
                # If API is not running, we'll gracefully handle it
                if res.status_code == 200:
                    data = res.json()
                    
                    alert = data['alert_level']
                    color = {"CRITICAL": "red", "RED": "orange", "AMBER": "yellow", "GREEN": "green"}.get(alert, "grey")
                    st.markdown(f"<h2 style='text-align: center; color: white; background-color: {color}; padding: 10px; border-radius: 5px;'>{alert} ALERT</h2>", unsafe_allow_html=True)
                    
                    c1, c2, c3 = st.columns(3)
                    c1.metric("Severity", data['severity'])
                    c2.metric("Est. Duration", f"{data['estimated_duration_minutes']:.0f} min")
                    c3.metric("Closure Prob", f"{data['closure_probability']*100:.1f}%")
                    
                    st.subheader("Resource Recommendation")
                    rec = data['recommendation']
                    st.write(f"**Constables**: {rec['constable_count']}")
                    st.write(f"**Barricades**: {rec['barricade_count']}")
                    st.write(f"**Diversion Needed**: {'Yes' if rec['diversion_needed'] else 'No'}")
                    st.write(f"**Assigned Station**: {rec['suggested_police_station']}")
                    
                    st.subheader("Explanation")
                    # data['explanation'] contains top features and the text summary at the end
                    exp_text = data['explanation'][-1]['text']
                    st.info(exp_text)
                    
                    # Plot horizontal bar
                    feats = data['explanation'][:-1]
                    if len(feats) > 0:
                        df_plot = pd.DataFrame(feats)
                        fig, ax = plt.subplots(figsize=(6,3))
                        sns.barplot(data=df_plot, x='impact', y='feature', ax=ax, palette='coolwarm')
                        plt.title("Top SHAP Features")
                        st.pyplot(fig)
                        
            except requests.exceptions.ConnectionError:
                st.error("API is not running. Start the FastAPI server first: `uvicorn api.main:app --reload`")
    else:
        st.info("Submit the form to see predictions.")
