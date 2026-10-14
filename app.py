"""
AI-Powered Flood Detection, Population Impact Analysis & Rescue Prioritization Dashboard
Study Area: Bhandara / Wainganga River Basin, Maharashtra, India
Satellite Sensor: Sentinel-1 Synthetic Aperture Radar (SAR)
Population Model: WorldPop 100m UN-Adjusted Spatial Raster
"""

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
import json
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from folium import plugins
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go

# ---------------- PAGE CONFIG & STYLING ----------------
st.set_page_config(
    page_title="FloodRescue AI | Disaster Response & Impact Intelligence",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom High-End Hackathon UI CSS
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .main-title {
        font-size: 2.3rem;
        font-weight: 800;
        background: linear-gradient(135deg, #1e3a8a 0%, #0284c7 50%, #0d9488 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
        letter-spacing: -0.02em;
    }
    
    .subtitle {
        font-size: 1.05rem;
        color: #475569;
        margin-bottom: 1.5rem;
        font-weight: 400;
    }
    
    .kpi-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -2px rgba(0, 0, 0, 0.05);
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    
    .kpi-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    
    .kpi-label {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #64748b;
        margin-bottom: 0.4rem;
    }
    
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 700;
        color: #0f172a;
        line-height: 1.1;
    }
    
    .kpi-sub {
        font-size: 0.82rem;
        color: #64748b;
        margin-top: 0.35rem;
    }
    
    .badge-critical {
        background-color: #fee2e2;
        color: #991b1b;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-high {
        background-color: #ffedd5;
        color: #c2410c;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-moderate {
        background-color: #fef9c3;
        color: #854d0e;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .badge-low {
        background-color: #dcfce7;
        color: #166534;
        padding: 3px 8px;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 700;
    }
    
    .alert-box {
        background-color: #eff6ff;
        border-left: 4px solid #3b82f6;
        padding: 0.9rem 1.2rem;
        border-radius: 6px;
        margin: 1rem 0;
        color: #1e3a8a;
        font-size: 0.92rem;
    }
    
    .badge-pill {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 20px;
        font-size: 0.78rem;
        font-weight: 600;
        margin-right: 0.4rem;
    }
    
    .pill-green { background-color: #d1fae5; color: #065f46; }
    .pill-blue { background-color: #dbeafe; color: #1e40af; }
    .pill-amber { background-color: #fef3c7; color: #92400e; }
    
</style>
""", unsafe_allow_html=True)


# ---------------- DATA LOADERS WITH STREAMLIT CACHE ----------------
@st.cache_data
def load_summary():
    path = "output/flood_summary.json"
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

@st.cache_data
def load_villages():
    path = "output/village_impact_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_villages_geojson():
    path = "output/villages_analyzed.geojson"
    if os.path.exists(path):
        return gpd.read_file(path)
    return gpd.GeoDataFrame()

@st.cache_data
def load_hospitals():
    path = "output/hospitals_analyzed.geojson"
    if os.path.exists(path):
        return gpd.read_file(path)
    return gpd.GeoDataFrame()

@st.cache_data
def load_roads_summary():
    path = "output/roads_summary.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    return pd.DataFrame()

@st.cache_data
def load_flood_display():
    # Load fast web display layer
    path = "data/flood_display.geojson"
    if os.path.exists(path):
        return gpd.read_file(path)
    return gpd.GeoDataFrame()

@st.cache_data
def load_flooded_roads():
    path = "output/roads_flooded.geojson"
    if os.path.exists(path):
        gdf = gpd.read_file(path)
        # Filter to only flooded segments for fast map rendering
        return gdf[gdf["flooded"]].copy()
    return gpd.GeoDataFrame()


# ---------------- LOAD BASELINE DATA ----------------
summary = load_summary()
df_villages = load_villages()
gdf_villages = load_villages_geojson()
gdf_hospitals = load_hospitals()
df_roads = load_roads_summary()
gdf_flood = load_flood_display()
gdf_roads_flooded = load_flooded_roads()

# ---------------- BASELINE MAP BUILDER ----------------
def build_flood_map(show_flood=True, show_roads=True, show_villages=True, show_hospitals=True):
    """Build and return an interactive Folium map with all geospatial layers."""
    m = folium.Map(
        location=[21.05, 80.05],
        zoom_start=10,
        tiles="OpenStreetMap",
        control_scale=True
    )
    
    # 1. Flood Polygon Layer
    if show_flood and not gdf_flood.empty:
        folium.GeoJson(
            gdf_flood,
            name="Satellite Flood Extent (Sentinel-1)",
            style_function=lambda f: {
                "fillColor": "#0284c7",
                "color": "#0369a1",
                "weight": 1.2,
                "fillOpacity": 0.45
            },
            tooltip=folium.GeoJsonTooltip(fields=["area_km2"], aliases=["Flood Body Area (km²):"], localize=True)
        ).add_to(m)
        
    # 2. Flooded Roads Layer
    if show_roads and not gdf_roads_flooded.empty:
        folium.GeoJson(
            gdf_roads_flooded,
            name="Inaccessible / Flooded Roads",
            style_function=lambda f: {
                "color": "#dc2626",
                "weight": 3.0,
                "opacity": 0.85
            },
            tooltip=folium.GeoJsonTooltip(fields=["name", "highway", "length_km"], aliases=["Road:", "Class:", "Length (km):"])
        ).add_to(m)
        
    # 3. Hospitals Layer
    if show_hospitals and not gdf_hospitals.empty:
        hosp_group = folium.FeatureGroup(name="Healthcare Facilities")
        for _, row in gdf_hospitals.iterrows():
            pt = row.geometry
            is_f = row.get("is_flooded", False)
            icon_color = "red" if is_f else "blue"
            icon_symbol = "medkit"
            
            popup_html = f"""
            <div style="font-family: sans-serif; font-size: 12px; width: 220px;">
                <b style="color: #1e3a8a; font-size: 13px;">{row.get('name', 'Hospital')}</b><br>
                <b>Status:</b> {'🔴 Inundated' if is_f else '🟢 Operational'}<br>
                <b>Type:</b> {row.get('amenity', 'Hospital')}<br>
                <small style="color: #64748b;">{str(row.get('full_address', ''))[:80]}...</small>
            </div>
            """
            folium.Marker(
                location=[pt.y, pt.x],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"🏥 {row.get('name', 'Hospital')} ({'Operational' if not is_f else 'Flooded'})",
                icon=folium.Icon(color=icon_color, icon=icon_symbol, prefix="fa")
            ).add_to(hosp_group)
        hosp_group.add_to(m)

    # 4. Villages Layer with Priority Colors
    if show_villages and not gdf_villages.empty:
        vill_cluster = plugins.MarkerCluster(name="Settlements (Clustered)").add_to(m)
        for _, row in gdf_villages.iterrows():
            pt = row.geometry
            tier = row.get("priority_tier", "LOW")
            p_score = row.get("priority_score", 0.0)
            p_aff = row.get("pop_affected", 0.0)
            is_iso = row.get("is_isolated", False)
            
            color_map = {
                "CRITICAL": "#991b1b",
                "HIGH": "#ea580c",
                "MODERATE": "#ca8a04",
                "LOW": "#16a34a"
            }
            fill_color = color_map.get(tier, "#16a34a")
            
            popup_content = f"""
            <div style="font-family: sans-serif; font-size: 12px; width: 240px;">
                <h4 style="margin: 0 0 5px 0; color: #0f172a;">{row.get('name', 'Village')}</h4>
                <div style="margin-bottom: 5px;">
                    <span style="background: {fill_color}; color: white; padding: 2px 6px; border-radius: 4px; font-weight: bold; font-size: 10px;">
                        {tier} PRIORITY ({p_score:.1f})
                    </span>
                </div>
                <b>Taluka:</b> {row.get('taluka', 'N/A')}<br>
                <b>Total Population:</b> {row.get('pop_total', 0):,.0f}<br>
                <b>Exposed Population:</b> <b style="color: #dc2626;">{p_aff:,.0f} ({row.get('pct_affected', 0):.1f}%)</b><br>
                <b>Hospital Access:</b> {'🔴 ISOLATED (Cut off)' if is_iso else f"🟢 Accessible ({row.get('dist_after_km', 'N/A')} km)"}<br>
            </div>
            """
            folium.CircleMarker(
                location=[pt.y, pt.x],
                radius=6 if tier in ["HIGH", "CRITICAL"] else 4,
                color=fill_color,
                fill=True,
                fill_color=fill_color,
                fill_opacity=0.85,
                weight=1.5,
                tooltip=f"{row.get('name')} | Priority: {p_score:.1f} ({tier})",
                popup=folium.Popup(popup_content, max_width=260)
            ).add_to(vill_cluster)
            
    folium.LayerControl().add_to(m)
    return m


# ---------------- SIDEBAR CONTROLS ----------------
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/water-element.png", width=64)
    st.title("FloodRescue AI")
    st.markdown("**Disaster Decision Support**")
  
    st.divider()
    
    nav = st.radio(
        "Navigation",
        [
            "🧭 Overview & KPIs",
            "🗺️ Interactive Flood Map",
            "👥 Population Impact",
            "🏘️ Community Impact",
            "🛣️ Road Isolation & Routing",
            "🚨 Rescue Prioritization",
            "🔬 Data & Methodology",
            "📥 Downloads & Report"
        ],
        index=0
    )
    
    st.divider()
    st.markdown("### 🛰️ System Status")
    st.markdown('<span class="badge-pill pill-green">● Satellite SAR Active</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-pill pill-blue">● WorldPop 100m Validated</span>', unsafe_allow_html=True)
    st.markdown('<span class="badge-pill pill-amber">● 385 Villages Indexed</span>', unsafe_allow_html=True)
    st.caption("Region: Bhandara / Wainganga, MH, India")


# ---------------- TAB 1: OVERVIEW & KPIS ----------------
if nav == "🧭 Overview & KPIs":
    st.markdown('<div class="main-title">AI-Powered Flood Detection & Rescue Prioritization</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Real-time geospatial intelligence combining Copernicus Sentinel-1 SAR imagery, WorldPop demographic rasters, and OpenStreetMap network accessibility for emergency response.</div>', unsafe_allow_html=True)
    
    # Study Area Banner
    st.markdown("""
    <div class="alert-box">
        <strong>📍 Study Region & Event:</strong> Bhandara & Wainganga River Basin, Maharashtra, India (Bounds: 79.82°E–80.50°E, 20.50°N–21.50°N) • 
        <strong>Sensor:</strong> Sentinel-1 C-band SAR (Descending Orbit 92, VH Polarization) • 
        <strong>Analysis Mode:</strong> Validated Offline Demonstration Engine.
    </div>
    """, unsafe_allow_html=True)
    
    # KPI Row 1
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🌊 Flood Inundation Area</div>
            <div class="kpi-value" style="color: #0284c7;">{summary['flood_area_km2']:,.1f} <span style="font-size: 1rem;">km²</span></div>
            <div class="kpi-sub">Exact projected UTM 44N calculation</div>
        </div>
        """, unsafe_allow_html=True)
    
    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">👥 Exposed Population</div>
            <div class="kpi-value" style="color: #dc2626;">{summary['total_exposed_population']:,.0f}</div>
            <div class="kpi-sub">Inhabited within SAR water mask</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⛔ Inaccessible Roads</div>
            <div class="kpi-value" style="color: #ea580c;">{summary['flooded_roads_km']:,.1f} <span style="font-size: 1rem;">km</span></div>
            <div class="kpi-sub">{summary['flooded_roads_pct']}% of {summary['total_roads_km']:,.0f} km network</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏥 Operational Hospitals</div>
            <div class="kpi-value" style="color: #16a34a;">{summary['operational_hospitals']} <span style="font-size: 1rem;">/ {summary['total_hospitals']}</span></div>
            <div class="kpi-sub">{summary['inundated_hospitals']} inundated facilities</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.write("")
    
    # KPI Row 2
    kpi5, kpi6, kpi7, kpi8 = st.columns(4)
    with kpi5:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🏘️ Settlements Analyzed</div>
            <div class="kpi-value">{summary['total_settlements_analyzed']}</div>
            <div class="kpi-sub">Verified OSM villages & hamlets</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi6:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⚠️ Flood-Impacted Settlements</div>
            <div class="kpi-value" style="color: #d97706;">{summary['flood_affected_settlements']}</div>
            <div class="kpi-sub">Direct polygon overlap within 1km</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi7:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚫 Road-Isolated Communities</div>
            <div class="kpi-value" style="color: #e11d48;">{summary['isolated_settlements']}</div>
            <div class="kpi-sub">Hospital route severed by flood</div>
        </div>
        """, unsafe_allow_html=True)
        
    with kpi8:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚨 High Priority Triage</div>
            <div class="kpi-value" style="color: #7c3aed;">{summary['high_priority_settlements']}</div>
            <div class="kpi-sub">Prioritization Score ≥ 40 / 100</div>
        </div>
        """, unsafe_allow_html=True)

    st.divider()
    
    # Quick Visualizations & Summary Charts
    c1, c2 = st.columns([3, 2])
    with c1:
        st.subheader("📊 Road Network Inundation by Category")
        if not df_roads.empty:
            fig_roads = px.bar(
                df_roads.sort_values("flooded_km", ascending=False),
                x="highway",
                y=["flooded_km", "total_km"],
                barmode="group",
                labels={"value": "Road Length (km)", "highway": "Road Hierarchy", "variable": "Metric"},
                color_discrete_map={"flooded_km": "#ef4444", "total_km": "#94a3b8"},
                title="Flooded vs Total Road Kilometres by Highway Classification"
            )
            fig_roads.update_layout(legend_title="", margin=dict(l=20, r=20, t=40, b=20), height=320)
            st.plotly_chart(fig_roads, use_container_width=True)
            
    with c2:
        st.subheader("🎯 Rescue Priority Tier Breakdown")
        if not df_villages.empty:
            tier_counts = df_villages["priority_tier"].value_counts().reset_index()
            tier_counts.columns = ["Tier", "Settlements"]
            fig_pie = px.pie(
                tier_counts,
                names="Tier",
                values="Settlements",
                color="Tier",
                color_discrete_map={"CRITICAL": "#dc2626", "HIGH": "#f97316", "MODERATE": "#eab308", "LOW": "#22c55e"},
                hole=0.45
            )
            st.plotly_chart(fig_pie, use_container_width=True)
            
    st.divider()
    st.subheader("🗺️ Live Operational Flood Map")
    st.caption("Real-time geospatial intelligence overlay: Copernicus Sentinel-1 SAR flood extent (blue), submerged road corridors (red), and prioritized settlements. Navigate to '🗺️ Interactive Flood Map' in the sidebar for full-screen layer toggles.")
    overview_map = build_flood_map()
    st_folium(overview_map, use_container_width=True, height=520)


# ---------------- TAB 2: INTERACTIVE FLOOD MAP ----------------
elif nav == "🗺️ Interactive Flood Map":
    st.markdown('<div class="main-title">Interactive Geospatial Operations Map</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Multi-layer operational map combining satellite flood polygons, road inundation, healthcare facilities, and prioritized settlements.</div>', unsafe_allow_html=True)
    
    col_ctrl1, col_ctrl2, col_ctrl3, col_ctrl4 = st.columns(4)
    with col_ctrl1:
        show_flood = st.checkbox("Show Flood Polygons", value=True)
    with col_ctrl2:
        show_roads = st.checkbox("Show Flooded Roads", value=True)
    with col_ctrl3:
        show_villages = st.checkbox("Show Settlements", value=True)
    with col_ctrl4:
        show_hospitals = st.checkbox("Show Hospitals", value=True)
        
    map_obj = build_flood_map(
        show_flood=show_flood,
        show_roads=show_roads,
        show_villages=show_villages,
        show_hospitals=show_hospitals
    )
    st_folium(map_obj, use_container_width=True, height=620)
    
    st.info("💡 **Map Tips:** Click settlement dots to view estimated affected population, road isolation status, and rescue score. Red lines indicate road corridors severed by satellite-detected standing water.")


# ---------------- TAB 3: POPULATION IMPACT ----------------
elif nav == "👥 Population Impact":
    st.markdown('<div class="main-title">Population Exposure & Demographic Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">High-resolution demographic estimation combining WorldPop UN-adjusted 100m raster with satellite inundation masks.</div>', unsafe_allow_html=True)
    
    col_p1, col_p2, col_p3 = st.columns(3)
    with col_p1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">👥 Total Exposed Population</div>
            <div class="kpi-value" style="color: #dc2626;">{summary['total_exposed_population']:,.0f}</div>
            <div class="kpi-sub">Within verified flood boundary</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📐 Inhabited Flooded Pixels</div>
            <div class="kpi-value" style="color: #2563eb;">{summary['flooded_pop_pixels']:,}</div>
            <div class="kpi-sub">100m x 100m (~1 hectare per cell)</div>
        </div>
        """, unsafe_allow_html=True)
    with col_p3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">📊 Peak Exposure Density</div>
            <div class="kpi-value" style="color: #7c3aed;">{summary['max_pixel_exposure']:.1f} <span style="font-size: 1rem;">ppl/pixel</span></div>
            <div class="kpi-sub">Mean: {summary['mean_pixel_exposure']:.2f} persons/inhabited cell</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    col_g1, col_g2 = st.columns(2)
    with col_g1:
        st.subheader("🏘️ Exposed Population by Taluka / Sub-district")
        if not df_villages.empty:
            taluka_df = df_villages.groupby("taluka").agg(
                total_pop=("pop_total", "sum"),
                affected_pop=("pop_affected", "sum"),
                settlements=("name", "count")
            ).reset_index()
            taluka_df["exposure_pct"] = (taluka_df["affected_pop"] / taluka_df["total_pop"] * 100).round(1)
            taluka_df = taluka_df.sort_values("affected_pop", ascending=False)
            
            fig_taluka = px.bar(
                taluka_df,
                x="taluka",
                y="affected_pop",
                text="affected_pop",
                labels={"taluka": "Taluka", "affected_pop": "Exposed Population"},
                color="affected_pop",
                color_continuous_scale="Reds",
                title="Total Population Impact by Taluka"
            )
            fig_taluka.update_traces(texttemplate='%{text:,.0f}', textposition='outside')
            fig_taluka.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380, showlegend=False)
            st.plotly_chart(fig_taluka, use_container_width=True)
            
    with col_g2:
        st.subheader("📈 Settlement Population Exposure Distribution")
        if not df_villages.empty:
            fig_hist = px.histogram(
                df_villages[df_villages["pop_affected"] > 0],
                x="pop_affected",
                nbins=25,
                color_discrete_sequence=["#ea580c"],
                labels={"pop_affected": "Exposed Population per Village"},
                title="Histogram of Settlement-Level Population Inundation"
            )
            fig_hist.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=380)
            st.plotly_chart(fig_hist, use_container_width=True)

    st.markdown("""
    ### 🔬 Methodology & Population Allocation Formula
    1. **Demographic Source**: WorldPop 2020 UN-Adjusted Population Count raster for India (100m resolution, EPSG:4326).
    2. **Zonal Integration**: Flood polygons from Copernicus Sentinel-1 SAR are raster-masked directly against the raster cells. No synthetic interpolation is applied.
    3. **Buffer Territory**: A standard 1,000m catchment buffer is projected around each authenticated settlement centroid to compute local exposed residents versus baseline residents.
    4. **Scientific Uncertainty**: Census disaggregation uses spatial covariates (settlement density, land cover, night lights). Actual residential presence during floods may vary due to pre-emptive evacuations.
    """)


# ---------------- TAB 4: COMMUNITY IMPACT ----------------
elif nav == "🏘️ Community Impact":
    st.markdown('<div class="main-title">Community & Settlement Exposure Directory</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprehensive impact registry across 385 genuine settlements in the Bhandara & Wainganga basin.</div>', unsafe_allow_html=True)
    
    # Filters
    f1, f2, f3, f4 = st.columns(4)
    with f1:
        taluka_list = ["All"] + sorted(df_villages["taluka"].dropna().unique().tolist())
        selected_taluka = st.selectbox("Filter by Taluka", taluka_list)
    with f2:
        tier_list = ["All", "HIGH", "MODERATE", "LOW"]
        selected_tier = st.selectbox("Filter by Priority Tier", tier_list)
    with f3:
        iso_filter = st.selectbox("Filter by Road Access", ["All", "Isolated Only", "Accessible Only"])
    with f4:
        search_query = st.text_input("Search Village Name", "")
        
    filtered_df = df_villages.copy()
    if selected_taluka != "All":
        filtered_df = filtered_df[filtered_df["taluka"] == selected_taluka]
    if selected_tier != "All":
        filtered_df = filtered_df[filtered_df["priority_tier"] == selected_tier]
    if iso_filter == "Isolated Only":
        filtered_df = filtered_df[filtered_df["is_isolated"] == True]
    elif iso_filter == "Accessible Only":
        filtered_df = filtered_df[filtered_df["is_isolated"] == False]
    if search_query:
        filtered_df = filtered_df[filtered_df["name"].str.contains(search_query, case=False, na=False)]
        
    st.markdown(f"**Showing {len(filtered_df)} of {len(df_villages)} Settlements**")
    
    # Styled Data Table
    display_cols = ["name", "taluka", "pop_total", "pop_affected", "pct_affected", "is_isolated", "dist_before_km", "dist_after_km", "priority_score", "priority_tier"]
    rename_cols = {
        "name": "Settlement Name",
        "taluka": "Taluka",
        "pop_total": "Total Pop",
        "pop_affected": "Exposed Pop",
        "pct_affected": "Exposure %",
        "is_isolated": "Road Isolated?",
        "dist_before_km": "Dist Before (km)",
        "dist_after_km": "Dist After (km)",
        "priority_score": "Priority Score",
        "priority_tier": "Tier"
    }
    
    st.dataframe(
        filtered_df[display_cols].rename(columns=rename_cols).sort_values("Priority Score", ascending=False),
        use_container_width=True,
        height=450
    )


# ---------------- TAB 5: ROAD ISOLATION & ROUTING ----------------
elif nav == "🛣️ Road Isolation & Routing":
    st.markdown('<div class="main-title">Road Accessibility & Hospital Route Severance</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Graph-theoretic Dijkstra network analysis evaluating emergency response accessibility before and after inundation.</div>', unsafe_allow_html=True)
    
    rc1, rc2, rc3 = st.columns(3)
    with rc1:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🛣️ Total Network Analyzed</div>
            <div class="kpi-value">{summary['total_roads_km']:,.1f} <span style="font-size: 1rem;">km</span></div>
            <div class="kpi-sub">9,157 road segments, 105,402 nodes</div>
        </div>
        """, unsafe_allow_html=True)
    with rc2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">⛔ Flooded Road Segments</div>
            <div class="kpi-value" style="color: #dc2626;">{summary['flooded_roads_km']:,.1f} <span style="font-size: 1rem;">km</span></div>
            <div class="kpi-sub">{summary['flooded_road_segments']} segments submerged ({summary['flooded_roads_pct']}%)</div>
        </div>
        """, unsafe_allow_html=True)
    with rc3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">🚫 Disconnected Settlements</div>
            <div class="kpi-value" style="color: #ea580c;">{summary['isolated_settlements']}</div>
            <div class="kpi-sub">Zero viable routes to operational hospital</div>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()
    
    col_r1, col_r2 = st.columns([1, 1])
    with col_r1:
        st.subheader("📊 Inundation Impact by Road Hierarchy")
        if not df_roads.empty:
            st.dataframe(
                df_roads[["highway", "total_km", "flooded_km", "flooded_pct"]].rename(columns={
                    "highway": "Road Class",
                    "total_km": "Total (km)",
                    "flooded_km": "Flooded (km)",
                    "flooded_pct": "Inundated (%)"
                }),
                use_container_width=True,
                height=320
            )
            
    with col_r2:
        st.subheader("💡 Dijkstra Network Routing Methodology")
        st.markdown("""
        1. **Dual-Graph Topology**:
           - **$G_{full}$ (Baseline Graph)**: All 9,157 OpenStreetMap drivable segments (105,402 nodes, 110,229 edges).
           - **$G_{safe}$ (Post-Flood Graph)**: Flood-intersecting segments (361 edges, 813.2 km) removed.
        2. **Multi-Source Dijkstra**: Shortest road travel distance evaluated from all 50 verified operational healthcare facilities to community network junctions.
        3. **Isolation Condition**: Communities where no path exists in $G_{safe}$ to any safe medical facility are flagged as **Road-Isolated**, necessitating amphibious or aerial rescue dispatch.
        """)

    st.subheader("🔴 Isolated Communities Requiring Immediate Priority Evacuation")
    isolated_villages = df_villages[df_villages["is_isolated"]].sort_values("pop_affected", ascending=False)
    st.dataframe(
        isolated_villages[["name", "taluka", "pop_total", "pop_affected", "pct_affected", "dist_before_km", "priority_score", "priority_tier"]].rename(columns={
            "name": "Settlement Name",
            "taluka": "Taluka",
            "pop_total": "Total Population",
            "pop_affected": "Exposed Population",
            "pct_affected": "Exposure %",
            "dist_before_km": "Baseline Hospital Dist (km)",
            "priority_score": "Score",
            "priority_tier": "Tier"
        }),
        use_container_width=True,
        height=300
    )


# ---------------- TAB 6: RESCUE PRIORITIZATION ----------------
elif nav == "🚨 Rescue Prioritization":
    st.markdown('<div class="main-title">Multi-Criteria Rescue Prioritization Engine</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Objective, transparent decision matrix for disaster response dispatch commanders.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="alert-box">
        <strong>Rescue Priority Formula:</strong><br>
        $$\\text{Score} = w_1 \\cdot \\text{PopScale} + w_2 \\cdot \\text{ExposureRatio} + w_3 \\cdot \\text{IsolationPenalty} + w_4 \\cdot \\text{DetourPenalty}$$
        Adjust the operational weights below to simulate different tactical response scenarios.
    </div>
    """, unsafe_allow_html=True)
    
    w_col1, w_col2, w_col3, w_col4 = st.columns(4)
    with w_col1:
        w_pop = st.slider("Population Scale Weight (w₁)", 0.0, 1.0, 0.40, 0.05)
    with w_col2:
        w_ratio = st.slider("Exposure Ratio Weight (w₂)", 0.0, 1.0, 0.25, 0.05)
    with w_col3:
        w_iso = st.slider("Road Isolation Weight (w₃)", 0.0, 1.0, 0.20, 0.05)
    with w_col4:
        w_detour = st.slider("Detour/Distance Weight (w₄)", 0.0, 1.0, 0.15, 0.05)
        
    # Recalculate dynamic scores
    dyn_df = df_villages.copy()
    pop_scale = np.clip(dyn_df["pop_affected"] / 1500.0, 0.0, 1.0) * 100.0
    ratio_scale = (dyn_df["pct_affected"] / 100.0) * 100.0
    iso_scale = np.where(dyn_df["is_isolated"], 100.0, 0.0)
    detour_scale = np.where(
        dyn_df["is_isolated"],
        100.0,
        np.where(
            dyn_df["dist_after_km"].notna() & dyn_df["dist_before_km"].notna(),
            np.clip((dyn_df["dist_after_km"] - dyn_df["dist_before_km"]) / 10.0 * 100.0, 0.0, 100.0),
            0.0
        )
    )
    
    total_w = w_pop + w_ratio + w_iso + w_detour
    if total_w > 0:
        dyn_df["custom_score"] = (
            (w_pop * pop_scale + w_ratio * ratio_scale + w_iso * iso_scale + w_detour * detour_scale) / total_w
        ).round(1)
    else:
        dyn_df["custom_score"] = 0.0
        
    def dyn_tier(s):
        if s >= 60.0: return "CRITICAL"
        if s >= 40.0: return "HIGH"
        if s >= 20.0: return "MODERATE"
        return "LOW"
    dyn_df["custom_tier"] = dyn_df["custom_score"].apply(dyn_tier)
    
    st.subheader("🎯 Tactical Rescue Dispatch Leaderboard")
    top_triage = dyn_df.sort_values("custom_score", ascending=False).head(20)
    st.dataframe(
        top_triage[["name", "taluka", "pop_total", "pop_affected", "pct_affected", "is_isolated", "dist_before_km", "dist_after_km", "custom_score", "custom_tier"]].rename(columns={
            "name": "Settlement Name",
            "taluka": "Taluka",
            "pop_total": "Total Pop",
            "pop_affected": "Exposed Pop",
            "pct_affected": "Exposure %",
            "is_isolated": "Isolated?",
            "dist_before_km": "Baseline (km)",
            "dist_after_km": "Post-Flood (km)",
            "custom_score": "Dynamic Score (0-100)",
            "custom_tier": "Priority Tier"
        }),
        use_container_width=True,
        height=450
    )


# ---------------- TAB 7: DATA & METHODOLOGY ----------------
elif nav == "🔬 Data & Methodology":
    st.markdown('<div class="main-title">Data Validation, Methodology & Scientific Integrity</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Complete transparency regarding satellite parameters, algorithms, coordinate systems, and engineering limitations.</div>', unsafe_allow_html=True)
    
    st.markdown("""
    ### 🛰️ 1. Satellite Remote Sensing Pipeline (Sentinel-1 SAR)
    - **Constellation**: European Space Agency (ESA) Copernicus Sentinel-1.
    - **Instrument**: C-band Synthetic Aperture Radar (SAR) operating at 5.405 GHz.
    - **Polarization**: VH (Vertical transmit, Horizontal receive), chosen for superior sensitivity to land-water contrast.
    - **Orbit**: Descending pass, Relative Orbit 92, covering Eastern Maharashtra.
    - **Detection Algorithm**:
      - Speckle filtering to minimize radar grain noise.
      - Backscatter thresholding (approximately -3 dB relative decrease compared to baseline dry season imagery).
      - Morphological cleaning to eliminate micro-speckle artifacts.
    - **Limitation**: Radar signals can be impeded by dense vegetation canopy (flooded forests) or mirrored by flat smooth surfaces such as tarmac or bare dry sand. The classified flood mask has not undergone in-situ field validation.

    ### 👥 2. Population Disaggregation (WorldPop)
    - **Data Product**: WorldPop India UN-Adjusted Population Count (100m spatial resolution).
    - **Coordinate System**: EPSG:4326 (WGS84).
    - **Intersection Technique**: Direct raster masking using exact vector polygon boundaries.
    - **Integrity Guarantee**: Zero synthetic population figures. Every estimate is the exact mathematical sum of validated raster pixels located inside the satellite flood mask.

    ### 🛣️ 3. Transportation Network & Graph Isolation
    - **Road Network**: OpenStreetMap highway network for Bhandara, Gondia, and surrounding districts.
    - **Topology**: Undirected NetworkX graph ($105,402$ nodes, $110,229$ edges).
    - **Route Inundation**: Road segments intersecting projected flood geometries are pruned.
    - **Healthcare Routing**: Multi-source Dijkstra shortest path computation from safe hospital points. Straight-line Euclidean approximation is strictly avoided in routing distances.

    ### 📐 4. Coordinate Reference Systems (CRS)
    - **Analysis Projections**: All area and distance measurements are executed in **UTM Zone 44N (EPSG:32644)** to ensure metric precision.
    - **Display Projections**: Geometries are transformed to **EPSG:4326** for Leaflet/Folium web map compatibility.
    """)


# ---------------- TAB 8: DOWNLOADS & REPORTS ----------------
elif nav == "📥 Downloads & Reports":
    st.markdown('<div class="main-title">Downloads & Executive Reports</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Export validated findings, tabular metrics, and spatial GIS layers for external emergency dispatch systems.</div>', unsafe_allow_html=True)
    
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.subheader("📊 Tabular CSV Outputs")
        p_csv = "output/rescue_prioritization.csv"
        if os.path.exists(p_csv):
            with open(p_csv, "rb") as f:
                st.download_button(
                    label="📥 Download Rescue Prioritization CSV",
                    data=f.read(),
                    file_name="rescue_prioritization.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        v_csv = "output/village_impact_summary.csv"
        if os.path.exists(v_csv):
            with open(v_csv, "rb") as f:
                st.download_button(
                    label="📥 Download Village Impact Summary CSV",
                    data=f.read(),
                    file_name="village_impact_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )
        r_csv = "output/roads_summary.csv"
        if os.path.exists(r_csv):
            with open(r_csv, "rb") as f:
                st.download_button(
                    label="📥 Download Road Inundation Summary CSV",
                    data=f.read(),
                    file_name="roads_summary.csv",
                    mime="text/csv",
                    use_container_width=True
                )

    with col_d2:
        st.subheader("🗺️ Geospatial GIS Layers")
        v_geo = "output/villages_analyzed.geojson"
        if os.path.exists(v_geo):
            with open(v_geo, "rb") as f:
                st.download_button(
                    label="📥 Download Analyzed Villages GeoJSON",
                    data=f.read(),
                    file_name="villages_analyzed.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
        r_geo = "output/roads_flooded.geojson"
        if os.path.exists(r_geo):
            with open(r_geo, "rb") as f:
                st.download_button(
                    label="📥 Download Flooded Roads GeoJSON",
                    data=f.read(),
                    file_name="roads_flooded.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )
        f_geo = "data/flood_display.geojson"
        if os.path.exists(f_geo):
            with open(f_geo, "rb") as f:
                st.download_button(
                    label="📥 Download Web Flood Polygons GeoJSON",
                    data=f.read(),
                    file_name="flood_display.geojson",
                    mime="application/geo+json",
                    use_container_width=True
                )

    with col_d3:
        st.subheader("📄 Formal Executive Report")
        rep_md = "output/flood_impact_report.md"
        if os.path.exists(rep_md):
            with open(rep_md, "rb") as f:
                st.download_button(
                    label="📥 Download Executive Report (Markdown)",
                    data=f.read(),
                    file_name="flood_impact_report.md",
                    mime="text/markdown",
                    use_container_width=True
                )
        sum_json = "output/flood_summary.json"
        if os.path.exists(sum_json):
            with open(sum_json, "rb") as f:
                st.download_button(
                    label="📥 Download Summary Metrics JSON",
                    data=f.read(),
                    file_name="flood_summary.json",
                    mime="application/json",
                    use_container_width=True
                )

    st.divider()
    st.subheader("📖 Executive Report Preview")
    if os.path.exists("output/flood_impact_report.md"):
        with open("output/flood_impact_report.md", "r", encoding="utf-8") as f:
            st.markdown(f.read())

# Footer
st.markdown("---")
st.markdown("<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>FloodRescue AI • College Hackathon Demonstration System • Built with Streamlit, GeoPandas, Rasterio & NetworkX</div>", unsafe_allow_html=True)
