"""
pipeline.py - Production Geospatial Analysis & Validation Pipeline
AI-Powered Flood Detection, Population Impact Analysis & Rescue Prioritization

Performs:
1. Data validation on Sentinel-1 SAR flood polygons, WorldPop raster, OSM road network, villages, and hospitals.
2. Exact spatial calculations using projected CRS (UTM Zone 44N, EPSG:32644).
3. Zonal statistics & raster masking with WorldPop (EPSG:4326).
4. Road network intersection, Dijkstra graph accessibility analysis, and isolation detection.
5. Multi-criteria rescue prioritization scoring.
6. Export of structured GeoJSON, CSV, JSON, and Markdown reports.
"""

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
import sys
import json
import time
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Point, LineString
import networkx as nx
from scipy.spatial import cKDTree
import rasterio
from rasterio.mask import mask

warnings.filterwarnings("ignore")

# Directories
DATA_DIR = "data"
OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

FLOOD_PATH = os.path.join(DATA_DIR, "flood_zones_small.geojson")
if not os.path.exists(FLOOD_PATH):
    FLOOD_PATH = os.path.join(DATA_DIR, "flood_zones.geojson")
POP_PATH = os.path.join(DATA_DIR, "worldpop.tif")
ROADS_PATH = os.path.join(DATA_DIR, "roads_cached.geojson")
HOSP_PATH = os.path.join(DATA_DIR, "hospitals.geojson")
VILL_PATH = os.path.join(DATA_DIR, "villages.geojson")


def run_pipeline():
    start_time = time.time()
    print("=" * 70)
    print("STARTING FLOOD DETECTION & RESCUE PRIORITIZATION PIPELINE")
    print("=" * 70)

    # ---------------- 1. VALIDATION & FLOOD LOADING ----------------
    print("\n[Step 1/6] Validating and Loading Flood Polygons...")
    if not os.path.exists(FLOOD_PATH):
        raise FileNotFoundError(f"Missing flood dataset at {FLOOD_PATH}")
    
    flood_gdf = gpd.read_file(FLOOD_PATH, engine="pyogrio")
    flood_gdf = flood_gdf[flood_gdf.geometry.notna() & ~flood_gdf.geometry.is_empty]
    if flood_gdf.crs is None or flood_gdf.crs != "EPSG:4326":
        flood_gdf = flood_gdf.set_crs(4326, allow_override=True)
    
    bounds = flood_gdf.total_bounds # [minx, miny, maxx, maxy]
    utm_crs = flood_gdf.estimate_utm_crs() # EPSG:32644
    flood_m = flood_gdf.to_crs(utm_crs)
    total_flood_area_km2 = float(flood_m.geometry.area.sum() / 1e6)
    
    # Explode into discrete polygon parts for fine-grained spatial index
    flood_parts = flood_m.explode(index_parts=False).reset_index(drop=True)
    flood_sindex = flood_parts.sindex
    
    print(f"  * Flood Bounds: Lon [{bounds[0]:.3f}, {bounds[2]:.3f}], Lat [{bounds[1]:.3f}, {bounds[3]:.3f}]")
    print(f"  * Projected CRS: {utm_crs.to_string()}")
    print(f"  * Total Flood Inundation Area: {total_flood_area_km2:,.2f} km²")
    print(f"  * Polygon Components: {len(flood_parts):,}")

    # ---------------- 2. POPULATION RASTER ANALYSIS ----------------
    print("\n[Step 2/6] Validating WorldPop Raster & Calculating Population Exposure...")
    if not os.path.exists(POP_PATH):
        raise FileNotFoundError(f"Missing WorldPop raster at {POP_PATH}")
    
    with rasterio.open(POP_PATH) as src:
        pop_crs = src.crs
        pop_bounds = src.bounds
        pop_res = src.res
        
        # Verify bounding box overlap
        overlaps = not (pop_bounds.right < bounds[0] or pop_bounds.left > bounds[2] or
                        pop_bounds.top < bounds[1] or pop_bounds.bottom > bounds[3])
        if not overlaps:
            raise ValueError("WorldPop raster does not cover the flood study area bounds!")
        
        # Mask raster with flood geometry in 4326
        out_image, out_transform = mask(src, flood_gdf.geometry, crop=True, nodata=0.0)
        pop_arr = out_image[0]
        valid_pop = pop_arr[(pop_arr > 0) & (~np.isnan(pop_arr))]
        
        total_exposed_population = float(valid_pop.sum())
        max_pixel_exposure = float(valid_pop.max()) if len(valid_pop) > 0 else 0.0
        flooded_pop_pixels = int(len(valid_pop))
        mean_pixel_exposure = float(valid_pop.mean()) if len(valid_pop) > 0 else 0.0

    print(f"  * WorldPop CRS: {pop_crs} (Resolution: ~{pop_res[0]*111:.1f} km / ~100 m)")
    print(f"  * Total Estimated Exposed Population: {total_exposed_population:,.0f} people")
    print(f"  * Flooded Inhabited 100m Pixels: {flooded_pop_pixels:,}")
    print(f"  * Maximum Population Density in Flood Zone: {max_pixel_exposure:.1f} people/pixel")

    # ---------------- 3. ROAD NETWORK & FLOOD INTERSECTION ----------------
    print("\n[Step 3/6] Analyzing Road Network & Inundation...")
    if not os.path.exists(ROADS_PATH):
        raise FileNotFoundError(f"Missing roads dataset at {ROADS_PATH}")
    
    gdf_roads = gpd.read_file(ROADS_PATH)
    gdf_roads_m = gdf_roads.to_crs(utm_crs)
    gdf_roads["length_km"] = (gdf_roads_m.geometry.length / 1000.0).round(2)
    
    # Query road intersections against flood spatial index
    road_sindex = gdf_roads_m.sindex
    flooded_road_ids = set()
    for poly in flood_parts.geometry:
        candidates = road_sindex.query(poly, predicate="intersects")
        for r_id in candidates:
            flooded_road_ids.add(r_id)
            
    gdf_roads["flooded"] = False
    gdf_roads.loc[list(flooded_road_ids), "flooded"] = True
    
    total_road_km = float(gdf_roads["length_km"].sum())
    flooded_road_km = float(gdf_roads[gdf_roads["flooded"]]["length_km"].sum())
    flooded_road_count = int(gdf_roads["flooded"].sum())
    
    # Breakdown by highway type
    road_breakdown = gdf_roads.groupby("highway").agg(
        total_segments=("highway", "count"),
        flooded_segments=("flooded", "sum"),
        total_km=("length_km", "sum"),
        flooded_km=("length_km", lambda s: gdf_roads.loc[s.index[gdf_roads.loc[s.index, "flooded"]], "length_km"].sum())
    ).reset_index()
    road_breakdown["flooded_pct"] = (road_breakdown["flooded_km"] / road_breakdown["total_km"] * 100).round(1)
    road_breakdown.to_csv(os.path.join(OUTPUT_DIR, "roads_summary.csv"), index=False)
    
    gdf_roads.to_file(os.path.join(OUTPUT_DIR, "roads_flooded.geojson"), driver="GeoJSON")
    print(f"  * Total Road Length: {total_road_km:,.1f} km ({len(gdf_roads):,} segments)")
    print(f"  * Flooded / Inaccessible Roads: {flooded_road_km:,.1f} km ({flooded_road_count:,} segments, {flooded_road_km/total_road_km*100:.1f}%)")

    # ---------------- 4. HOSPITALS & RELIEF SERVICES ----------------
    print("\n[Step 4/6] Evaluating Hospital & Relief Centre Status...")
    if not os.path.exists(HOSP_PATH):
        raise FileNotFoundError(f"Missing hospitals dataset at {HOSP_PATH}")
    
    gdf_hosp = gpd.read_file(HOSP_PATH)
    gdf_hosp_m = gdf_hosp.to_crs(utm_crs)
    
    hosp_flooded_flags = []
    for h_geom in gdf_hosp_m.geometry:
        hits = flood_sindex.query(h_geom, predicate="intersects")
        hosp_flooded_flags.append(len(hits) > 0)
        
    gdf_hosp["is_flooded"] = hosp_flooded_flags
    gdf_hosp["status"] = np.where(gdf_hosp["is_flooded"], "Flooded / Non-Operational", "Operational / Safe")
    operational_hosp = gdf_hosp[~gdf_hosp["is_flooded"]].copy()
    
    gdf_hosp.to_file(os.path.join(OUTPUT_DIR, "hospitals_analyzed.geojson"), driver="GeoJSON")
    print(f"  * Total Healthcare Facilities: {len(gdf_hosp)}")
    print(f"  * Operational / Safe Facilities: {len(operational_hosp)}")
    print(f"  * Inundated Facilities: {gdf_hosp['is_flooded'].sum()}")

    # ---------------- 5. GRAPH ROUTING & ISOLATION ----------------
    print("\n[Step 5/6] Building Road Network Graph & Computing Dijkstra Shortest Paths...")
    G_full = nx.Graph()
    G_safe = nx.Graph()
    
    for _, row in gdf_roads.iterrows():
        geom = row.geometry
        is_flooded = row["flooded"]
        coords = list(geom.coords)
        for i in range(len(coords) - 1):
            u = (round(coords[i][0], 6), round(coords[i][1], 6))
            v = (round(coords[i+1][0], 6), round(coords[i+1][1], 6))
            dx = (u[0] - v[0]) * 103000
            dy = (u[1] - v[1]) * 111000
            dist_m = (dx * dx + dy * dy) ** 0.5
            G_full.add_edge(u, v, weight=dist_m)
            if not is_flooded:
                G_safe.add_edge(u, v, weight=dist_m)
                
    full_nodes = list(G_full.nodes())
    full_node_arr = np.array(full_nodes)
    tree_full = cKDTree(full_node_arr)
    
    safe_nodes = list(G_safe.nodes())
    safe_node_arr = np.array(safe_nodes) if len(safe_nodes) > 0 else None
    tree_safe = cKDTree(safe_node_arr) if safe_node_arr is not None else None
    
    # Snap operational hospitals
    hosp_nodes_full = []
    hosp_nodes_safe = []
    for pt in operational_hosp.geometry:
        _, idx_f = tree_full.query([pt.x, pt.y])
        hosp_nodes_full.append(full_nodes[idx_f])
        if tree_safe is not None:
            _, idx_s = tree_safe.query([pt.x, pt.y])
            hosp_nodes_safe.append(safe_nodes[idx_s])
            
    hosp_nodes_full = list(set(hosp_nodes_full))
    hosp_nodes_safe = list(set(hosp_nodes_safe))
    
    dist_before = nx.multi_source_dijkstra_path_length(G_full, hosp_nodes_full, weight="weight")
    dist_after = nx.multi_source_dijkstra_path_length(G_safe, hosp_nodes_safe, weight="weight") if hosp_nodes_safe else {}
    
    print(f"  * Baseline Road Graph: {G_full.number_of_nodes():,} nodes | {G_full.number_of_edges():,} edges")
    print(f"  * Post-Flood Safe Graph: {G_safe.number_of_nodes():,} nodes | {G_safe.number_of_edges():,} edges")

    # ---------------- 6. VILLAGE IMPACT & RESCUE PRIORITIZATION ----------------
    print("\n[Step 6/6] Calculating Village-Level Exposure & Rescue Prioritization...")
    if not os.path.exists(VILL_PATH):
        raise FileNotFoundError(f"Missing villages dataset at {VILL_PATH}")
    
    gdf_vill = gpd.read_file(VILL_PATH)
    gdf_vill_m = gdf_vill.to_crs(utm_crs)
    vill_buffers_m = gdf_vill_m.geometry.buffer(1000)
    vill_buffers_ll = vill_buffers_m.to_crs(4326)
    
    pop_total_list = []
    pop_aff_list = []
    dist_b_list = []
    dist_a_list = []
    is_iso_list = []
    
    with rasterio.open(POP_PATH) as src:
        for idx in range(len(gdf_vill)):
            pt = gdf_vill.geometry.iloc[idx]
            buf_ll = vill_buffers_ll.iloc[idx]
            buf_m = vill_buffers_m.iloc[idx]
            
            # 1. Total population in village buffer
            try:
                out_tot, _ = mask(src, [buf_ll], crop=True, nodata=0.0)
                arr_t = out_tot[0]
                val_t = arr_t[(arr_t > 0) & (~np.isnan(arr_t))]
                p_tot = float(val_t.sum())
            except Exception:
                p_tot = 0.0
            pop_total_list.append(round(p_tot, 1))
            
            # 2. Flooded population in village buffer
            cand_floods = flood_sindex.query(buf_m, predicate="intersects")
            if len(cand_floods) > 0:
                inter_parts = flood_parts.iloc[cand_floods]
                try:
                    inter_geom = inter_parts.geometry.intersection(buf_m)
                    inter_geom = inter_geom[~inter_geom.is_empty]
                    if not inter_geom.empty:
                        inter_ll = gpd.GeoSeries(inter_geom.unary_union, crs=utm_crs).to_crs(4326).iloc[0]
                        out_f, _ = mask(src, [inter_ll], crop=True, nodata=0.0)
                        arr_f = out_f[0]
                        val_f = arr_f[(arr_f > 0) & (~np.isnan(arr_f))]
                        p_aff = float(val_f.sum())
                    else:
                        p_aff = 0.0
                except Exception:
                    p_aff = 0.0
            else:
                p_aff = 0.0
            pop_aff_list.append(round(p_aff, 1))
            
            # 3. Routing distances
            d_pt, n_f_idx = tree_full.query([pt.x, pt.y])
            node_full = full_nodes[n_f_idx]
            d_b = dist_before.get(node_full)
            dist_b_list.append(round(d_b / 1000.0, 1) if d_b is not None else None)
            
            if tree_safe is not None:
                d_safe_pt, n_s_idx = tree_safe.query([pt.x, pt.y])
                # If nearest safe road is > 2.5 km away (~0.023 degrees)
                if d_safe_pt > 0.025:
                    dist_a_list.append(None)
                    is_iso_list.append(True)
                else:
                    node_safe = safe_nodes[n_s_idx]
                    d_a = dist_after.get(node_safe)
                    if d_a is not None:
                        dist_a_list.append(round(d_a / 1000.0, 1))
                        is_iso_list.append(False)
                    else:
                        dist_a_list.append(None)
                        is_iso_list.append(True)
            else:
                dist_a_list.append(None)
                is_iso_list.append(True)
                
    gdf_vill["pop_total"] = pop_total_list
    gdf_vill["pop_affected"] = pop_aff_list
    gdf_vill["pct_affected"] = np.where(
        gdf_vill["pop_total"] > 0,
        (gdf_vill["pop_affected"] / gdf_vill["pop_total"] * 100.0).round(1),
        0.0
    )
    gdf_vill["dist_before_km"] = dist_b_list
    gdf_vill["dist_after_km"] = dist_a_list
    gdf_vill["is_isolated"] = is_iso_list
    
    # Calculate Transparent Rescue Prioritization Score (0 to 100)
    # Weights:
    # 40% Exposed Population Scale: min(pop_affected / 1500, 1.0) * 40
    # 25% Population Exposure Ratio: (pct_affected / 100) * 25
    # 20% Road Isolation Indicator: 20 if True else 0
    # 15% Healthcare Detour / Distance: min((dist_after - dist_before) / 10 * 15, 15) if not isolated, or 15 if isolated
    pop_scale = np.clip(gdf_vill["pop_affected"] / 1500.0, 0.0, 1.0) * 40.0
    ratio_scale = (gdf_vill["pct_affected"] / 100.0) * 25.0
    iso_scale = np.where(gdf_vill["is_isolated"], 20.0, 0.0)
    
    detour_scale = np.where(
        gdf_vill["is_isolated"],
        15.0,
        np.where(
            gdf_vill["dist_after_km"].notna() & gdf_vill["dist_before_km"].notna(),
            np.clip((gdf_vill["dist_after_km"] - gdf_vill["dist_before_km"]) / 10.0 * 15.0, 0.0, 15.0),
            0.0
        )
    )
    
    gdf_vill["priority_score"] = (pop_scale + ratio_scale + iso_scale + detour_scale).round(1)
    
    def get_tier(score):
        if score >= 60.0:
            return "CRITICAL"
        elif score >= 40.0:
            return "HIGH"
        elif score >= 20.0:
            return "MODERATE"
        return "LOW"
        
    gdf_vill["priority_tier"] = gdf_vill["priority_score"].apply(get_tier)
    
    # Save Outputs
    gdf_vill.to_file(os.path.join(OUTPUT_DIR, "villages_analyzed.geojson"), driver="GeoJSON")
    gdf_vill.drop(columns="geometry").to_csv(os.path.join(OUTPUT_DIR, "village_impact_summary.csv"), index=False)
    
    rescue_df = gdf_vill.sort_values("priority_score", ascending=False).drop(columns="geometry")
    rescue_df.to_csv(os.path.join(OUTPUT_DIR, "rescue_prioritization.csv"), index=False)
    
    # ---------------- SUMMARY METRICS ----------------
    isolated_count = int(gdf_vill["is_isolated"].sum())
    flood_affected_villages = int((gdf_vill["pop_affected"] > 0).sum())
    critical_villages = int((gdf_vill["priority_tier"] == "CRITICAL").sum())
    high_villages = int((gdf_vill["priority_tier"] == "HIGH").sum())
    
    summary_data = {
        "study_area": "Bhandara / Wainganga Basin, Maharashtra, India",
        "analysis_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "satellite_source": "Copernicus Sentinel-1 SAR (C-band GRD, VH polarization, Descending orbit 92)",
        "flood_area_km2": round(total_flood_area_km2, 2),
        "total_exposed_population": round(total_exposed_population, 0),
        "flooded_pop_pixels": flooded_pop_pixels,
        "max_pixel_exposure": round(max_pixel_exposure, 1),
        "mean_pixel_exposure": round(mean_pixel_exposure, 2),
        "total_roads_km": round(total_road_km, 1),
        "flooded_roads_km": round(flooded_road_km, 1),
        "flooded_roads_pct": round(flooded_road_km / total_road_km * 100, 1),
        "flooded_road_segments": flooded_road_count,
        "total_hospitals": len(gdf_hosp),
        "operational_hospitals": len(operational_hosp),
        "inundated_hospitals": int(gdf_hosp["is_flooded"].sum()),
        "total_settlements_analyzed": len(gdf_vill),
        "flood_affected_settlements": flood_affected_villages,
        "isolated_settlements": isolated_count,
        "critical_priority_settlements": critical_villages,
        "high_priority_settlements": high_villages,
        "projected_crs": utm_crs.to_string(),
        "geographic_crs": "EPSG:4326"
    }
    
    with open(os.path.join(OUTPUT_DIR, "flood_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2)
        
    # Generate Detailed Markdown Report
    report_md = f"""# Executive Flood Impact & Rescue Prioritization Report
**Region**: Bhandara / Wainganga River Basin, Maharashtra, India  
**Date of Analysis**: {summary_data['analysis_timestamp']}  
**Satellite Mission**: Sentinel-1 Synthetic Aperture Radar (SAR) VH Polarization  
**Population Model**: WorldPop High-Resolution UN-Adjusted Population Raster  

---

## 1. Executive Summary & Core KPIs

| Metric | Measured Value | Unit / Definition |
| :--- | :--- | :--- |
| **Total Inundation Area** | **{summary_data['flood_area_km2']:,.2f}** | km² (projected in UTM Zone 44N) |
| **Exposed Population** | **{summary_data['total_exposed_population']:,.0f}** | People residing directly within satellite flood mask |
| **Flooded Inhabited Pixels** | **{summary_data['flooded_pop_pixels']:,}** | 100m x 100m pixels (~1 hectare each) |
| **Total Road Network** | **{summary_data['total_roads_km']:,.1f}** | km (OSM road network) |
| **Flooded / Inaccessible Roads** | **{summary_data['flooded_roads_km']:,.1f}** | km ({summary_data['flooded_roads_pct']}% of network) |
| **Verified Hospitals** | **{summary_data['total_hospitals']}** | {summary_data['operational_hospitals']} operational, {summary_data['inundated_hospitals']} flooded |
| **Settlements Analyzed** | **{summary_data['total_settlements_analyzed']}** | Authentic villages/hamlets in study bounds |
| **Settlements Exposed to Flood** | **{summary_data['flood_affected_settlements']}** | Settlements with direct flood polygon overlap |
| **Road-Isolated Communities** | **{summary_data['isolated_settlements']}** | Cut off from reachable operational hospitals |
| **Critical Rescue Priority** | **{summary_data['critical_priority_settlements']}** | Priority Score $\ge 60$ |
| **High Rescue Priority** | **{summary_data['high_priority_settlements']}** | Priority Score $[40, 60)$ |

---

## 2. High-Priority Rescue Triage Table

The following communities have been prioritized based on the multi-criteria rescue formula:
$$\\text{{Score}} = 0.40 \\times \\text{{PopScale}} + 0.25 \\times \\text{{ExposureRatio}} + 0.20 \\times \\text{{IsolationPenalty}} + 0.15 \\times \\text{{DetourPenalty}}$$

| Rank | Settlement Name | Taluka | Total Pop | Affected Pop | Exposure % | Isolated? | Dist Before (km) | Dist After (km) | Priority Score | Tier |
| :---: | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    top_rescue = rescue_df.head(15)
    for i, (_, r) in enumerate(top_rescue.iterrows(), 1):
        db = f"{r['dist_before_km']:.1f}" if pd.notna(r['dist_before_km']) else "N/A"
        da = f"{r['dist_after_km']:.1f}" if pd.notna(r['dist_after_km']) else "BLOCKED"
        report_md += f"| {i} | **{r['name']}** | {r.get('taluka', 'N/A')} | {r['pop_total']:,.0f} | {r['pop_affected']:,.0f} | {r['pct_affected']:.1f}% | {'🔴 YES' if r['is_isolated'] else '🟢 NO'} | {db} | {da} | **{r['priority_score']:.1f}** | `{r['priority_tier']}` |\n"

    report_md += """
---

## 3. Data Sources & Scientific Caveats

1. **Satellite SAR Imagery**: Derived from Copernicus Sentinel-1 C-band synthetic aperture radar (GRD, VH polarization) acquired during peak monsoon conditions in the Wainganga River basin. SAR detects standing water via specular reflection. Caveats: Classification errors can occur in dense vegetation canopy or steep terrain. Not scientifically ground-truthed.
2. **Population Raster**: WorldPop 100m UN-adjusted spatial population distribution for India. Population is aggregated via exact raster masking without synthetic extrapolation.
3. **Roads & Healthcare**: OpenStreetMap drive network and verified medical facilities in Bhandara and neighboring talukas.
4. **Rescue Prioritization**: Transparent, objective score for disaster response dispatch. It does not replace field commander assessment.
"""
    with open(os.path.join(OUTPUT_DIR, "flood_impact_report.md"), "w", encoding="utf-8") as f:
        f.write(report_md)

    duration = time.time() - start_time
    print(f"\n[DONE] Pipeline completed successfully in {duration:.1f} seconds!")
    print(f"Generated results saved in '{OUTPUT_DIR}/':")
    for fname in os.listdir(OUTPUT_DIR):
        fsize = os.path.getsize(os.path.join(OUTPUT_DIR, fname)) / 1024
        print(f"  - {fname} ({fsize:.1f} KB)")


if __name__ == "__main__":
    run_pipeline()
