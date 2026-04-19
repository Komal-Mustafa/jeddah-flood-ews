import streamlit as st
import pandas as pd
import numpy as np
import torch
import pickle
import requests
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import sys
sys.path.append('data/src')
from model import FloodCNNLSTM

st.set_page_config(page_title="JFFEWS", page_icon="🌊", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;700;900&family=Exo+2:wght@300;400;600&display=swap');
html, body, [data-testid="stAppViewContainer"] { background:#020B18 !important; color:#E0F4FF !important; }
[data-testid="stSidebar"] { background:#010810 !important; border-right:1px solid #0A3A5C !important; }
h1,h2,h3 { font-family:'Orbitron',monospace !important; letter-spacing:2px; }
p,div,span,label { font-family:'Exo 2',sans-serif !important; }
.stButton>button { background:linear-gradient(135deg,#0A4F8C,#0D7FCA) !important; color:white !important;
    border:1px solid #1AA3FF !important; border-radius:4px !important; font-family:'Orbitron',monospace !important;
    font-size:13px !important; letter-spacing:2px !important; padding:12px 30px !important; text-transform:uppercase !important; }
.stButton>button:hover { background:linear-gradient(135deg,#0D7FCA,#1AA3FF) !important; box-shadow:0 0 20px #1AA3FF55 !important; }
[data-testid="metric-container"] { background:#041628 !important; border:1px solid #0A3A5C !important; border-radius:8px !important; padding:15px !important; }
[data-testid="stMetricValue"] { font-family:'Orbitron',monospace !important; color:#1AA3FF !important; font-size:1.6em !important; }
[data-testid="stMetricLabel"] { color:#7EC8E3 !important; font-size:0.8em !important; text-transform:uppercase !important; letter-spacing:1px !important; }
.section-header { font-family:'Orbitron',monospace; font-size:1.1em; color:#1AA3FF; letter-spacing:3px;
    text-transform:uppercase; border-bottom:1px solid #0A3A5C; padding-bottom:8px; margin:30px 0 15px 0; }
.info-card { background:#041628; border:1px solid #0A3A5C; border-left:3px solid #1AA3FF; border-radius:8px; padding:20px; margin:10px 0; }
.threshold-card { background:#041628; border-radius:8px; padding:15px; text-align:center; border:1px solid #0A3A5C; }
div[data-testid="stForm"] { background:#041628 !important; border:1px solid #0A3A5C !important; border-radius:12px !important; padding:20px !important; }
</style>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:20px 0;'>
        <div style='font-family:Orbitron;font-size:1.4em;color:#1AA3FF;letter-spacing:3px;'>JFFEWS</div>
        <div style='font-family:Exo 2;font-size:0.7em;color:#4A8FA8;letter-spacing:2px;margin-top:5px;'>
        JEDDAH FLASH FLOOD<br>EARLY WARNING SYSTEM</div>
    </div>
    <hr style='border-color:#0A3A5C;'>
    """, unsafe_allow_html=True)
    page = st.radio("", ["Live Prediction", "Flood Risk Map", "Historical Analysis", "Alert Settings"])
    st.markdown("<hr style='border-color:#0A3A5C;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style='font-family:Exo 2;font-size:0.75em;color:#2A6A8A;line-height:1.8;'>
    Model: CNN-LSTM<br>Data: ERA5 1985-2024<br>Accuracy: 99.96%<br>
    Source: Open-Meteo API<br>Built by: Komal</div>
    """, unsafe_allow_html=True)


@st.cache_resource
def load_model():
    model = FloodCNNLSTM(n_features=21, seq_length=24)
    model.load_state_dict(torch.load('models/flood_model.pth', map_location='cpu'))
    model.eval()
    return model


@st.cache_resource
def load_scaler():
    with open('models/scaler.pkl', 'rb') as f:
        return pickle.load(f)


def fetch_forecast():
    url = 'https://api.open-meteo.com/v1/forecast'
    params = {
        'latitude': 21.5433, 'longitude': 39.1728,
        'hourly': ['precipitation', 'rain', 'temperature_2m', 'relativehumidity_2m',
                   'windspeed_10m', 'surface_pressure', 'cape',
                   'soil_moisture_0_1cm', 'soil_moisture_1_3cm', 'dewpoint_2m'],
        'forecast_days': 3, 'timezone': 'Asia/Riyadh'
    }
    r = requests.get(url, params=params, timeout=30)
    df = pd.DataFrame(r.json()['hourly'])
    df['time'] = pd.to_datetime(df['time'])
    return df


def engineer_features(df):
    df['precip_1hr'] = df['precipitation'].rolling(1, min_periods=1).sum()
    df['precip_3hr'] = df['precipitation'].rolling(3, min_periods=1).sum()
    df['precip_6hr'] = df['precipitation'].rolling(6, min_periods=1).sum()
    df['precip_12hr'] = df['precipitation'].rolling(12, min_periods=1).sum()
    df['precip_24hr'] = df['precipitation'].rolling(24, min_periods=1).sum()
    df['precip_rate'] = df['precipitation'].diff().fillna(0).clip(lower=0)
    df['precip_max3hr'] = df['precipitation'].rolling(3, min_periods=1).max()
    df['hour'] = df['time'].dt.hour
    df['month'] = df['time'].dt.month
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['cape_category'] = pd.cut(df['cape'], bins=[-1, 500, 1500, 3000, 99999], labels=[0, 1, 2, 3]).astype(float)
    df['soil_moisture_avg'] = df[['soil_moisture_0_1cm', 'soil_moisture_1_3cm']].mean(axis=1)
    df['dry_hours_before'] = 0
    df['high_risk_month'] = df['month'].isin([11, 12, 1, 2]).astype(int)
    return df


FEATURE_COLS = [
    'precip_1hr', 'precip_3hr', 'precip_6hr', 'precip_12hr', 'precip_24hr',
    'precip_rate', 'precip_max3hr', 'cape', 'cape_category', 'soil_moisture_avg',
    'relativehumidity_2m', 'surface_pressure', 'windspeed_10m', 'temperature_2m',
    'dewpoint_2m', 'hour_sin', 'hour_cos', 'month_sin', 'month_cos',
    'dry_hours_before', 'high_risk_month'
]

DARK = dict(
    paper_bgcolor='#020B18', plot_bgcolor='#041628',
    font=dict(color='#7EC8E3', family='Exo 2'),
    xaxis=dict(gridcolor='#0A3A5C', linecolor='#0A3A5C'),
    yaxis=dict(gridcolor='#0A3A5C', linecolor='#0A3A5C')
)


def get_alert(prob):
    if prob >= 0.7:
        return "EMERGENCY", "#FF1744", "rgba(255,23,68,0.13)", "EVACUATE IMMEDIATELY"
    elif prob >= 0.4:
        return "WARNING", "#FF6D00", "rgba(255,109,0,0.13)", "PREPARE FOR FLOODING"
    elif prob >= 0.2:
        return "WATCH", "#FFD600", "rgba(255,214,0,0.13)", "MONITOR CONDITIONS"
    else:
        return "NORMAL", "#00E676", "rgba(0,230,118,0.13)", "NO FLOOD RISK DETECTED"


if page == "Live Prediction":
    st.markdown("""
    <div style='margin-bottom:30px;'>
        <div style='font-family:Orbitron;font-size:2em;color:#E0F4FF;letter-spacing:3px;'>LIVE FLOOD PREDICTION</div>
        <div style='font-family:Exo 2;color:#4A8FA8;letter-spacing:2px;margin-top:5px;'>REAL-TIME CNN-LSTM ANALYSIS - JEDDAH, SAUDI ARABIA</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("MODEL", "CNN-LSTM")
    c2.metric("PARAMETERS", "303,745")
    c3.metric("ACCURACY", "99.96%")
    c4.metric("TRAINING DATA", "40 YEARS")
    c5.metric("RECORDS", "350,640")
    st.markdown("<hr style='border-color:#0A3A5C;margin:20px 0;'>", unsafe_allow_html=True)

    if st.button("RUN LIVE PREDICTION", use_container_width=True):
        with st.spinner("Fetching live weather data..."):
            try:
                model = load_model()
                scaler = load_scaler()
                df = fetch_forecast()
                df = engineer_features(df)
                df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0)
                data = scaler.transform(df[FEATURE_COLS].values)
                data = np.nan_to_num(data, nan=0.0)
                if len(data) >= 24:
                    seq = torch.FloatTensor(data[-24:]).unsqueeze(0)
                    with torch.no_grad():
                        prob = model(seq).item()
                    level, color, bg, action = get_alert(prob)
                    latest = df.iloc[-1]
                    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    st.markdown(f"""
                    <div style='background:linear-gradient(135deg,{bg},{color}22);padding:40px;
                        border-radius:12px;text-align:center;margin:20px 0;
                        border:2px solid {color};box-shadow:0 0 40px {color}44;'>
                        <div style='font-family:Orbitron;font-size:3em;font-weight:900;color:{color};letter-spacing:4px;'>{level}</div>
                        <div style='font-family:Orbitron;font-size:1.8em;color:white;margin:10px 0;'>{prob*100:.2f}% FLOOD PROBABILITY</div>
                        <div style='font-family:Orbitron;color:{color};font-size:1em;margin:10px 0;letter-spacing:2px;'>{action}</div>
                        <div style='font-family:Exo 2;color:rgba(255,255,255,0.6);font-size:0.9em;'>UPDATED: {now}</div>
                    </div>""", unsafe_allow_html=True)
                    fig_gauge = go.Figure(go.Indicator(
                        mode="gauge+number", value=prob * 100,
                        title={'text': "FLOOD RISK %", 'font': {'family': 'Orbitron', 'color': '#7EC8E3', 'size': 14}},
                        number={'suffix': "%", 'font': {'family': 'Orbitron', 'color': '#1AA3FF', 'size': 40}},
                        gauge={
                            'axis': {'range': [0, 100], 'tickcolor': '#7EC8E3'},
                            'bar': {'color': color},
                            'bgcolor': '#041628',
                            'bordercolor': '#0A3A5C',
                            'steps': [
                                {'range': [0, 20], 'color': '#00261A'},
                                {'range': [20, 40], 'color': '#1A1A00'},
                                {'range': [40, 70], 'color': '#1A0A00'},
                                {'range': [70, 100], 'color': '#1A0000'}
                            ],
                            'threshold': {'line': {'color': 'white', 'width': 3}, 'thickness': 0.8, 'value': prob * 100}
                        }
                    ))
                    fig_gauge.update_layout(paper_bgcolor='#020B18', font={'color': '#7EC8E3'}, height=280, margin=dict(t=50, b=20))
                    col_g, col_m = st.columns([1, 1])
                    with col_g:
                        st.plotly_chart(fig_gauge, use_container_width=True)
                    with col_m:
                        st.markdown("<div class='section-header'>CURRENT CONDITIONS</div>", unsafe_allow_html=True)
                        m1, m2 = st.columns(2)
                        m1.metric("Temperature", f"{latest['temperature_2m']:.1f}C")
                        m2.metric("Humidity", f"{latest['relativehumidity_2m']:.0f}%")
                        m3, m4 = st.columns(2)
                        m3.metric("Wind Speed", f"{latest['windspeed_10m']:.1f} km/h")
                        m4.metric("3hr Rain", f"{latest['precip_3hr']:.2f} mm")
                        m5, m6 = st.columns(2)
                        m5.metric("24hr Rain", f"{latest['precip_24hr']:.2f} mm")
                        m6.metric("Pressure", f"{latest['surface_pressure']:.0f} hPa")
                    st.markdown("<div class='section-header'>48-HOUR RAINFALL FORECAST</div>", unsafe_allow_html=True)
                    chart_df = df.tail(48)
                    fig_rain = go.Figure()
                    fig_rain.add_trace(go.Bar(x=chart_df['time'], y=chart_df['precipitation'], name='Hourly Rain', marker_color='#1AA3FF', opacity=0.8))
                    fig_rain.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['precip_6hr'], name='6hr Sum', line=dict(color='#FFD600', width=2)))
                    fig_rain.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['precip_24hr'], name='24hr Sum', line=dict(color='#FF6D00', width=2, dash='dash')))
                    fig_rain.add_hrect(y0=70, y1=200, fillcolor="rgba(255,23,68,0.1)", line_width=0, annotation_text="EMERGENCY")
                    fig_rain.add_hrect(y0=40, y1=70, fillcolor="rgba(255,109,0,0.1)", line_width=0, annotation_text="WARNING")
                    fig_rain.add_hrect(y0=20, y1=40, fillcolor="rgba(255,214,0,0.1)", line_width=0, annotation_text="WATCH")
                    fig_rain.update_layout(**DARK, height=380, margin=dict(t=20, b=20), legend=dict(bgcolor='#041628', bordercolor='#0A3A5C'))
                    st.plotly_chart(fig_rain, use_container_width=True)
                    st.markdown("<div class='section-header'>FLOOD THRESHOLDS</div>", unsafe_allow_html=True)
                    t1, t2, t3, t4 = st.columns(4)
                    for col, lv, cl, desc in [
                        (t1, "NORMAL", "#00E676", "Below 20mm/24hr"),
                        (t2, "WATCH", "#FFD600", "20-40mm/24hr"),
                        (t3, "WARNING", "#FF6D00", "40-70mm/3hr"),
                        (t4, "EMERGENCY", "#FF1744", "70mm+/3hr")
                    ]:
                        col.markdown(f"""<div class='threshold-card' style='border-top:3px solid {cl};'>
                            <div style='font-family:Orbitron;color:{cl};font-size:1.1em;'>{lv}</div>
                            <div style='color:#7EC8E3;font-size:0.85em;margin-top:8px;'>{desc}</div>
                        </div>""", unsafe_allow_html=True)
                    st.markdown("<div class='section-header'>ATMOSPHERIC CONDITIONS</div>", unsafe_allow_html=True)
                    w1, w2 = st.columns(2)
                    with w1:
                        fig_wind = go.Figure()
                        fig_wind.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['windspeed_10m'],
                            fill='tozeroy', fillcolor='rgba(26,163,255,0.13)', line=dict(color='#1AA3FF', width=2), name='Wind'))
                        fig_wind.update_layout(**DARK, title='Wind Speed (km/h)', height=250, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_wind, use_container_width=True)
                    with w2:
                        fig_temp = go.Figure()
                        fig_temp.add_trace(go.Scatter(x=chart_df['time'], y=chart_df['temperature_2m'],
                            fill='tozeroy', fillcolor='rgba(255,109,0,0.13)', line=dict(color='#FF6D00', width=2), name='Temp'))
                        fig_temp.update_layout(**DARK, title='Temperature (C)', height=250, margin=dict(t=40, b=20))
                        st.plotly_chart(fig_temp, use_container_width=True)
            except Exception as e:
                st.error(f"Error: {e}")


elif page == "Flood Risk Map":
    st.markdown("""
    <div style='margin-bottom:30px;'>
        <div style='font-family:Orbitron;font-size:2em;color:#E0F4FF;letter-spacing:3px;'>FLOOD RISK MAP</div>
        <div style='font-family:Exo 2;color:#4A8FA8;letter-spacing:2px;margin-top:5px;'>INTERACTIVE RISK ZONES - WADI NETWORK - JEDDAH</div>
    </div>""", unsafe_allow_html=True)
    map_col, info_col = st.columns([3, 1])
    with map_col:
        m = folium.Map(location=[21.5433, 39.1728], zoom_start=11, tiles='CartoDB dark_matter')
        zones = [
            {"name": "Al Nuzha District", "lat": 21.5500, "lon": 39.1800, "risk": "HIGH", "color": "red", "detail": "Catastrophically flooded 2009, 2010, 2011"},
            {"name": "Al Rawdah", "lat": 21.5800, "lon": 39.1600, "risk": "HIGH", "color": "red", "detail": "Low elevation, poor drainage"},
            {"name": "Wadi Al Qaws", "lat": 21.4900, "lon": 39.2100, "risk": "HIGH", "color": "red", "detail": "Main wadi channel - extreme flash flood risk"},
            {"name": "Wadi Bani Malik", "lat": 21.4500, "lon": 39.2500, "risk": "HIGH", "color": "red", "detail": "Major wadi from Hejaz Mountains"},
            {"name": "Al Sharafeyah", "lat": 21.5300, "lon": 39.2000, "risk": "HIGH", "color": "red", "detail": "Dense urban - 2009 catastrophic flooding"},
            {"name": "Al Andalus District", "lat": 21.5200, "lon": 39.1900, "risk": "MEDIUM", "color": "orange", "detail": "Moderate risk near drainage channels"},
            {"name": "Al Zahraa", "lat": 21.5600, "lon": 39.1500, "risk": "MEDIUM", "color": "orange", "detail": "Residential - moderate drainage"},
            {"name": "Al Bawadi", "lat": 21.5100, "lon": 39.2200, "risk": "MEDIUM", "color": "orange", "detail": "Near wadi tributaries"},
            {"name": "Al Balad Old City", "lat": 21.4900, "lon": 39.1900, "risk": "MEDIUM", "color": "orange", "detail": "Historic district - coastal low-lying"},
            {"name": "Corniche Area", "lat": 21.5433, "lon": 39.1200, "risk": "LOW", "color": "green", "detail": "Elevated coastal - good drainage"},
            {"name": "Al Hamra District", "lat": 21.5700, "lon": 39.1300, "risk": "LOW", "color": "green", "detail": "Higher elevation - low flood risk"},
            {"name": "King Abdulaziz Airport", "lat": 21.6796, "lon": 39.1565, "risk": "LOW", "color": "green", "detail": "Northern elevated plateau"},
        ]
        for z in zones:
            r = 25 if z['risk'] == "HIGH" else 18 if z['risk'] == "MEDIUM" else 14
            folium.CircleMarker(
                location=[z['lat'], z['lon']], radius=r, color=z['color'],
                fill=True, fill_opacity=0.55,
                popup=folium.Popup(f"<b>{z['name']}</b><br>Risk: {z['risk']}<br>{z['detail']}", max_width=220),
                tooltip=f"{z['name']} - {z['risk']} RISK"
            ).add_to(m)
        folium.Marker([21.5433, 39.1728], popup="Jeddah City Center", tooltip="Jeddah City Center",
            icon=folium.Icon(color='blue', icon='info-sign')).add_to(m)
        st_folium(m, width=900, height=520)
    with info_col:
        st.markdown("<div class='section-header'>RISK LEGEND</div>", unsafe_allow_html=True)
        for cl, lv, desc in [
            ("#FF1744", "HIGH RISK", "Wadi channels and low-lying districts. Flash flood in under 3 hours."),
            ("#FF6D00", "MEDIUM RISK", "Residential areas near drainage. Flooding possible in heavy rain."),
            ("#00E676", "LOW RISK", "Elevated areas with good drainage. Minimal flood risk.")
        ]:
            st.markdown(f"""<div class='info-card' style='border-left-color:{cl};margin-top:10px;'>
                <div style='font-family:Orbitron;color:{cl};font-size:0.9em;'>{lv}</div>
                <div style='color:#7EC8E3;font-size:0.8em;margin-top:5px;'>{desc}</div>
            </div>""", unsafe_allow_html=True)
        st.markdown("<div class='section-header' style='margin-top:25px;'>GEOGRAPHY</div>", unsafe_allow_html=True)
        st.markdown("""<div style='font-family:Exo 2;color:#7EC8E3;font-size:0.82em;line-height:1.8;'>
        21.54N, 39.17E<br>Hejaz Mountains east<br>Red Sea west<br>
        12 major wadi systems<br>Flash flood in 3hrs<br>Peak risk: Nov-Jan
        </div>""", unsafe_allow_html=True)


elif page == "Historical Analysis":
    st.markdown("""
    <div style='margin-bottom:30px;'>
        <div style='font-family:Orbitron;font-size:2em;color:#E0F4FF;letter-spacing:3px;'>HISTORICAL ANALYSIS</div>
        <div style='font-family:Exo 2;color:#4A8FA8;letter-spacing:2px;margin-top:5px;'>40 YEARS OF JEDDAH FLOOD DATA - 1985-2024</div>
    </div>""", unsafe_allow_html=True)
    floods = pd.DataFrame({
        'date': ['2009-11-20', '2010-01-26', '2011-01-17', '2022-11-24', '2020-11-24', '2015-11-17', '2018-11-21', '1996-11-13'],
        'rainfall_mm': [70, 90, 111, 220, 65, 42, 38, 45],
        'deaths': [113, 122, 10, 0, 0, 0, 0, 0],
        'severity': ['Catastrophic', 'Catastrophic', 'Catastrophic', 'Extreme', 'Severe', 'Moderate', 'Moderate', 'Severe'],
        'duration_hr': [3, 4, 3, 8, 4, 5, 6, 5]
    })
    floods['date'] = pd.to_datetime(floods['date'])
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("DEADLIEST", "2010 - 122 Deaths")
    k2.metric("HEAVIEST RAIN", "220mm (2022)")
    k3.metric("TOTAL DEATHS", f"{floods['deaths'].sum()} recorded")
    k4.metric("MAJOR EVENTS", f"{len(floods)} events")
    st.markdown("<hr style='border-color:#0A3A5C;'>", unsafe_allow_html=True)
    st.markdown("<div class='section-header'>FLOOD EVENT TIMELINE</div>", unsafe_allow_html=True)
    cmap = {'Catastrophic': '#FF1744', 'Extreme': '#FF6D00', 'Severe': '#FFD600', 'Moderate': '#00E676'}
    fig_tl = go.Figure()
    for sev, col in cmap.items():
        ds = floods[floods['severity'] == sev]
        if len(ds) > 0:
            fig_tl.add_trace(go.Scatter(
                x=ds['date'], y=ds['rainfall_mm'], mode='markers+text', name=sev,
                marker=dict(size=ds['rainfall_mm'] / 3 + 10, color=col, opacity=0.8, line=dict(color='white', width=1)),
                text=ds['date'].dt.year.astype(str), textposition='top center',
                textfont=dict(family='Orbitron', color=col, size=10)
            ))
    fig_tl.add_hline(y=70, line_dash='dash', line_color='#FF1744', annotation_text='Emergency 70mm', annotation_font_color='#FF1744')
    fig_tl.add_hline(y=40, line_dash='dash', line_color='#FF6D00', annotation_text='Warning 40mm', annotation_font_color='#FF6D00')
    fig_tl.update_layout(**DARK, height=380, yaxis_title='Rainfall (mm)', margin=dict(t=20, b=20),
        legend=dict(bgcolor='#041628', bordercolor='#0A3A5C'))
    st.plotly_chart(fig_tl, use_container_width=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("<div class='section-header'>MONTHLY RISK</div>", unsafe_allow_html=True)
        months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
        risk = [8, 4, 2, 1, 0, 0, 0, 1, 2, 5, 10, 7]
        colors = ['#FF1744' if r >= 8 else '#FF6D00' if r >= 5 else '#FFD600' if r >= 2 else '#00E676' for r in risk]
        fig_m = go.Figure(go.Bar(x=months, y=risk, marker_color=colors,
            text=risk, textposition='outside', textfont=dict(family='Orbitron', color='#7EC8E3')))
        fig_m.update_layout(**DARK, height=320, yaxis_title='Risk Score', margin=dict(t=20, b=20))
        st.plotly_chart(fig_m, use_container_width=True)
    with col2:
        st.markdown("<div class='section-header'>RAINFALL vs CASUALTIES</div>", unsafe_allow_html=True)
        fig_sc = px.scatter(floods, x='rainfall_mm', y='deaths', size='duration_hr', color='severity',
            color_discrete_map=cmap, hover_data=['date'],
            labels={'rainfall_mm': 'Rainfall (mm)', 'deaths': 'Deaths'})
        fig_sc.update_layout(**DARK, height=320, margin=dict(t=20, b=20),
            legend=dict(bgcolor='#041628', bordercolor='#0A3A5C'))
        st.plotly_chart(fig_sc, use_container_width=True)
    st.markdown("<div class='section-header'>FLOOD DATABASE</div>", unsafe_allow_html=True)
    st.dataframe(floods[['date', 'rainfall_mm', 'duration_hr', 'deaths', 'severity']].sort_values('date', ascending=False),
        use_container_width=True, hide_index=True)


elif page == "Alert Settings":
    st.markdown("""
    <div style='margin-bottom:30px;'>
        <div style='font-family:Orbitron;font-size:2em;color:#E0F4FF;letter-spacing:3px;'>ALERT CONFIGURATION</div>
        <div style='font-family:Exo 2;color:#4A8FA8;letter-spacing:2px;margin-top:5px;'>SMS - WHATSAPP - TWILIO INTEGRATION</div>
    </div>""", unsafe_allow_html=True)
    st.markdown("""<div class='info-card'>
        <div style='font-family:Orbitron;color:#1AA3FF;font-size:0.9em;margin-bottom:8px;'>HOW TO GET TWILIO CREDENTIALS</div>
        <div style='font-family:Exo 2;color:#7EC8E3;font-size:0.85em;line-height:2;'>
        1. Go to twilio.com and create a free account<br>
        2. Get your Account SID and Auth Token from the Console<br>
        3. Get a free Twilio phone number<br>
        4. For WhatsApp: send join keyword to Twilio sandbox<br>
        5. Enter credentials below and send a test alert
        </div></div>""", unsafe_allow_html=True)
    st.markdown("<div class='section-header' style='margin-top:25px;'>TWILIO CONFIGURATION</div>", unsafe_allow_html=True)
    with st.form("alert_form"):
        c1, c2 = st.columns(2)
        with c1:
            sid = st.text_input("Account SID", placeholder="ACxxxxxxxxxxxxxxxx", type="password")
            from_num = st.text_input("Twilio Number", placeholder="+1234567890")
        with c2:
            token = st.text_input("Auth Token", placeholder="your_auth_token", type="password")
            to_num = st.text_input("Your Number", placeholder="+966xxxxxxxxx")
        st.markdown("<div class='section-header'>PREFERENCES</div>", unsafe_allow_html=True)
        p1, p2 = st.columns(2)
        with p1:
            alert_level = st.selectbox("Trigger at:", ["WATCH (20%+)", "WARNING (40%+)", "EMERGENCY (70%+)"])
        with p2:
            channel = st.selectbox("Send via:", ["SMS", "WhatsApp", "Both"])
        submitted = st.form_submit_button("SAVE AND SEND TEST ALERT", use_container_width=True)
        if submitted:
            if sid and token and from_num and to_num:
                try:
                    from twilio.rest import Client
                    client = Client(sid, token)
                    msg = f"JFFEWS TEST - Alerts configured at {alert_level} via {channel}. Built by Komal."
                    if channel in ["SMS", "Both"]:
                        client.messages.create(body=msg, from_=from_num, to=to_num)
                    if channel in ["WhatsApp", "Both"]:
                        client.messages.create(body=msg, from_=f"whatsapp:{from_num}", to=f"whatsapp:{to_num}")
                    st.success("Test alert sent!")
                except Exception as e:
                    st.error(f"Error: {e}")
            else:
                st.warning("Please fill all fields.")


st.markdown("""
<hr style='border-color:#0A3A5C;margin-top:40px;'>
<div style='text-align:center;font-family:Exo 2;color:#2A6A8A;font-size:0.8em;letter-spacing:2px;padding:10px;'>
JEDDAH FLASH FLOOD EARLY WARNING SYSTEM - CNN-LSTM DEEP LEARNING - DATA: OPEN-METEO ERA5 1985-2024 - BUILT BY KOMAL
</div>""", unsafe_allow_html=True)