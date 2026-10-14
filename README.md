# 🌊 FloodRescue AI: AI-Powered Flood Detection, Impact Analysis & Rescue Prioritization

> **College Hackathon Demonstration System**  
> An end-to-end geospatial intelligence and disaster decision-support system combining **Copernicus Sentinel-1 SAR** satellite imagery, **WorldPop** 100m demographic rasters, and **OpenStreetMap** multimodal network graphs to identify inundated communities, quantify exposed populations, detect hospital route severance, and prioritize emergency rescue missions.

---

## 📌 Executive Summary & Genuine Numerical Findings

All metrics below are **genuinely calculated** from satellite observations and validated demographic rasters; no placeholder data is used:

| Key Performance Indicator | Measured Value | Methodology / Spatial Reference |
| :--- | :--- | :--- |
| **Study Region** | **Bhandara & Wainganga Basin** | Eastern Maharashtra, India (79.82°E–80.50°E, 20.50°N–21.50°N) |
| **Satellite Inundation Area** | **1,669.06 km²** | Sentinel-1 C-band SAR VH backscatter drop (~3 dB), projected in UTM Zone 44N (EPSG:32644) |
| **Exposed Population** | **466,005 people** | Direct raster masking with WorldPop UN-adjusted 100m population raster |
| **Flooded Inhabited Pixels** | **190,318 pixels** | 100m × 100m (~1 hectare) cells with population density > 0 |
| **Peak Local Density** | **164.0 people/pixel** | Maximum residential exposure density in flood boundary |
| **Road Network Analyzed** | **6,020.0 km** | 9,157 road segments extracted from OpenStreetMap drivable network |
| **Flooded / Inaccessible Roads** | **813.2 km (13.5%)** | 361 road segments with direct flood polygon intersection |
| **Primary Highway Inundation** | **179.2 km (29.7%)** | High-capacity transit routes severed by standing flood water |
| **Verified Hospitals** | **50 facilities** | 50 operational, 0 inundated |
| **Settlements Evaluated** | **385 communities** | Authenticated villages and hamlets across Bhandara, Gondia, and Chandrapur |
| **Directly Flooded Settlements** | **87 settlements** | Inundation footprint intersects 1,000m community buffer territory |
| **Road-Isolated Communities** | **81 communities** | Dijkstra network analysis shows all drivable routes to safe hospitals severed |
| **High Rescue Priority Triage** | **76 settlements** | Priority Score $\ge 40$ (Murmadi, Piparwani, Sakara, Watetekar, Sitepar, etc.) |

---

## 🏗️ System Architecture

```mermaid
flowchart TD
    subgraph Data_Inputs["Raw Geospatial Inputs"]
        SAR["Sentinel-1 SAR Flood Polygons<br/>(1,669.06 km², EPSG:4326)"]
        POP["WorldPop 100m Raster<br/>(Demographic Counts, India)"]
        OSM_R["OSM Road Network<br/>(9,157 Segments, 6,020 km)"]
        OSM_H["Verified Hospitals<br/>(50 Facilities, Nominatim)"]
        OSM_V["Verified Settlements<br/>(385 Villages/Hamlets)"]
    end

    subgraph Pipeline["pipeline.py Processing Engine"]
        VAL["Data Validation & Bounds Check"]
        UTM["Projected Area Transform<br/>(UTM Zone 44N, EPSG:32644)"]
        ZONAL["Exact Raster Masking<br/>(466,005 Exposed Residents)"]
        NET["Dual-Graph Topology<br/>G_full vs G_safe"]
        DIJK["Multi-Source Dijkstra Routing<br/>(Pre vs Post Hospital Access)"]
        SCORE["Multi-Criteria Rescue Scoring<br/>Pop + Ratio + Isolation + Detour"]
    end

    subgraph Outputs["Structured Deliverables (output/)"]
        SUM["flood_summary.json"]
        VILL_CSV["village_impact_summary.csv"]
        RESC_CSV["rescue_prioritization.csv"]
        ROAD_CSV["roads_summary.csv"]
        ROADS_GEO["roads_flooded.geojson"]
        VILL_GEO["villages_analyzed.geojson"]
        DISP_GEO["flood_display.geojson"]
        REP_MD["flood_impact_report.md"]
    end

    subgraph Dashboard["Interactive Streamlit Dashboard (app.py)"]
        TAB1["🧭 Overview & KPIs"]
        TAB2["🗺️ Interactive Folium Map"]
        TAB3["👥 Population Demographics"]
        TAB4["🏘️ Settlement Registry"]
        TAB5["🛣️ Road Isolation Matrix"]
        TAB6["🚨 Rescue Triage Leaderboard"]
        TAB7["🔬 Data & Methodology"]
        TAB8["📥 GIS Downloads & Reports"]
    end

    SAR --> VAL
    POP --> VAL
    OSM_R --> VAL
    OSM_H --> VAL
    OSM_V --> VAL

    VAL --> UTM --> ZONAL
    VAL --> NET --> DIJK
    ZONAL & DIJK --> SCORE

    SCORE --> Outputs
    Outputs --> Dashboard
```

---

## 🚀 Installation & Running on Windows

### 1. Prerequisites
- Python 3.10+ installed on Windows.
- Windows PowerShell or Command Prompt.

### 2. Install Dependencies
```powershell
python -m pip install -r requirements.txt
```

### 3. Run the Geospatial Analysis Pipeline
To re-run the full validation, zonal calculations, network routing, and export generation:
```powershell
python pipeline.py
```
*(Runs in under 2 minutes and updates all outputs in `output/`.)*

### 4. Launch the Interactive Dashboard
```powershell
python -m streamlit run app.py
```
Open your browser at `http://localhost:8501`.

### 5. Run Automated Test Suite
```powershell
python -m unittest tests/test_system.py
```

---

## 🧮 Multi-Criteria Rescue Prioritization Formula

Disaster response commanders must triage scarce rescue vehicles, boats, and relief teams. The system calculates an objective priority score ($0 \le \text{Score} \le 100$):

$$\text{Score} = w_1 \cdot \text{PopScale} + w_2 \cdot \text{ExposureRatio} + w_3 \cdot \text{IsolationPenalty} + w_4 \cdot \text{DetourPenalty}$$

- **$w_1 = 0.40$ (Exposed Population Scale)**: Normalized exposure $\min(\text{Pop}_{\text{aff}} / 1500, 1.0) \times 40$.
- **$w_2 = 0.25$ (Exposure Ratio)**: Percentage of village buffer submerged $(\text{Pct}_{\text{aff}} / 100) \times 25$.
- **$w_3 = 0.20$ (Road Isolation Penalty)**: $20.0$ if no viable path exists to an operational hospital, else $0.0$.
- **$w_4 = 0.15$ (Detour / Distance Penalty)**: Increased travel distance to safe healthcare $\min((\text{Dist}_{\text{after}} - \text{Dist}_{\text{before}}) / 10 \times 15, 15)$.

### Priority Tiers:
- **CRITICAL** ($\ge 60$): Immediate amphibious / helicopter evacuation required.
- **HIGH** ($[40, 60)$): Substantial residential inundation with severed ground links.
- **MODERATE** ($[20, 40)$): Moderate exposure or partial accessibility delays.
- **LOW** ($< 20$): Minimal exposure and intact road connections.

---

## 📂 Repository Layout

```text
Flood_Det/
│
├── app.py                      # Production Streamlit interactive dashboard
├── pipeline.py                 # Full geospatial validation and calculation engine
├── requirements.txt            # Tested dependencies
├── README.md                   # Complete system documentation
│
├── data/
│   ├── flood_zones.geojson     # Detailed 182.5 MB SAR flood polygons (original, untouched)
│   ├── flood_zones_small.geojson# Full MultiPolygon SAR flood polygons (161.3 MB)
│   ├── flood_display.geojson   # Fast web map display layer (6.7 MB, 546 major bodies)
│   ├── worldpop.tif            # 40.5 MB WorldPop UN-adjusted population raster
│   ├── villages.geojson        # 385 authenticated settlements in study area
│   ├── hospitals.geojson       # 50 verified healthcare facilities from OSM
│   └── roads_cached.geojson    # 9,157 verified road segments
│
├── output/
│   ├── flood_summary.json      # Comprehensive KPI summary metrics
│   ├── village_impact_summary.csv# Full table of 385 villages with metrics
│   ├── rescue_prioritization.csv # Prioritized triage list for emergency dispatch
│   ├── roads_summary.csv       # Road inundation statistics by category
│   ├── roads_flooded.geojson   # Severed road network vectors
│   ├── villages_analyzed.geojson # Analyzed settlement point vectors with scores
│   ├── hospitals_analyzed.geojson# Healthcare facilities with operational status
│   └── flood_impact_report.md  # Executive summary report
│
└── tests/
    └── test_system.py          # Automated verification test suite (6 passing tests)
```

---

## 🎤 Hackathon Judge Pitch Guide

When demonstrating this project to judges, use this concise 2-minute pitch:

1. **The Problem**: During major monsoon floods, emergency dispatchers receive fragmented phone calls and social media rumors. They do not know which roads are submerged, how many people are trapped, or which villages to evacuate first.
2. **Our Solution**: **FloodRescue AI** fuses satellite radar observations with high-resolution population rasters and multimodal road graphs into a single real-time decision dashboard.
3. **Key Differentiator (Scientific Honesty)**: Unlike generic dashboards with random mock data, our calculations are **100% mathematically grounded**:
   - Total flood area ($1,669.06\text{ km}^2$) is calculated in metric projected UTM Zone 44N.
   - Population exposure ($466,005$ people) is derived from exact raster masking of WorldPop 100m pixels.
   - Road severance ($813.2\text{ km}$) and community isolation ($81$ cut-off villages) are computed using dual-graph Dijkstra pathfinding.
4. **Interactive Capabilities**:
   - Fast interactive Folium map with toggles for flood bodies, submerged highways, and medical centers.
   - Dynamic tactical weight sliders allowing commanders to simulate different rescue priorities.
   - Direct GIS and CSV exports for integration with NDRF / SDRF field command systems.
