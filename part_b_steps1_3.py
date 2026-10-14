
# Part B: Flood impact and road isolation analysis
# Original flood data is preserved. No geometry simplification is performed.

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

import warnings
import geopandas as gpd
import networkx as nx
import osmnx as ox
ox.settings.overpass_url = "https://overpass.kumi.systems/api"
ox.settings.timeout = 180
import pandas as pd
from rasterstats import zonal_stats

warnings.filterwarnings("ignore")
os.makedirs("output", exist_ok=True)

# ---------------- CONFIG ----------------
PLACE = "Bhandara district, Maharashtra, India"
FLOOD_PATH = "data/flood_zones.geojson"
POP_PATH = "data/worldpop.tif"
VILLAGE_RADIUS_M = 1000

# ---------------- HELPERS ----------------
def union_all(gdf):
    try:
        return gdf.geometry.union_all()
    except AttributeError:
        return gdf.geometry.unary_union


# ---------------- STEP 1: LOAD DATA ----------------
print("Loading detailed flood polygons...", flush=True)

flood = gpd.read_file(FLOOD_PATH)
flood = flood[
    flood.geometry.notna() & ~flood.geometry.is_empty
].to_crs(4326)

print(f"Loaded {len(flood)} flood features.", flush=True)

utm = flood.estimate_utm_crs()
flood_m = flood.to_crs(utm).reset_index(drop=True)

# Build a spatial index; do not merge the full flood dataset.
print("Building flood spatial index...", flush=True)
flood_index = flood_m.sindex
print("Flood spatial index ready.", flush=True)


def intersects_flood(geometry):
    """Return True when a geometry intersects a flood polygon."""
    if geometry is None or geometry.is_empty:
        return False

    candidate_ids = flood_index.query(geometry)

    for i in candidate_ids:
        if flood_m.geometry.iloc[i].intersects(geometry):
            return True

    return False


def flooded_intersection(geometry):
    """Return only the part of a geometry intersecting flood polygons."""
    if geometry is None or geometry.is_empty:
        return None

    candidate_ids = flood_index.query(geometry)
    parts = []

    for i in candidate_ids:
        flood_geom = flood_m.geometry.iloc[i]

        if flood_geom.intersects(geometry):
            part = geometry.intersection(flood_geom)
            if not part.is_empty:
                parts.append(part)

    if not parts:
        return None

    return union_all(gpd.GeoDataFrame(
        geometry=parts, crs=utm
    ))


# Download roads and map features from OpenStreetMap.
print("Downloading road network for Bhandara...", flush=True)
G = ox.graph_from_place(PLACE, network_type="drive")
G = ox.truncate.largest_component(G, strongly=True)

print("Downloading villages and hospitals...", flush=True)
villages = ox.features_from_place(
    PLACE, tags={"place": ["village", "hamlet"]}
).reset_index()

hospitals = ox.features_from_place(
    PLACE, tags={"amenity": "hospital"}
).reset_index()

if villages.empty:
    raise SystemExit("No villages found. Check the PLACE setting.")

villages_m = villages.to_crs(utm)
villages_m["geometry"] = villages_m.geometry.centroid
villages_m = villages_m.reset_index(drop=True)

if "name" not in villages_m.columns:
    villages_m["name"] = "Unnamed village"
else:
    villages_m["name"] = villages_m["name"].fillna("Unnamed village")

villages_m["village_id"] = villages_m.index
villages_m = villages_m[["village_id", "name", "geometry"]]

if hospitals.empty:
    raise SystemExit(
        "No hospitals found in OpenStreetMap. "
        "Add a verified safe relief centre as a destination."
    )

hospitals_m = hospitals.to_crs(utm)
hospitals_m["geometry"] = hospitals_m.geometry.centroid
hospitals_m = hospitals_m[
    hospitals_m.geometry.notna() & ~hospitals_m.geometry.is_empty
].copy()

print(
    f"Villages: {len(villages_m)} | Hospitals: {len(hospitals_m)} "
    f"| Road nodes: {len(G.nodes)}",
    flush=True
)

# ---------------- STEP 2: PEOPLE AFFECTED ----------------
print("Creating village analysis areas...", flush=True)

areas = villages_m.copy()
areas["geometry"] = areas.geometry.buffer(VILLAGE_RADIUS_M)

print("Finding flood intersections for village areas...", flush=True)
areas["flooded_part"] = areas.geometry.apply(flooded_intersection)


def population_sum(geometry):
    if geometry is None or geometry.is_empty:
        return 0.0

    geom_ll = gpd.GeoSeries([geometry], crs=utm).to_crs(4326).iloc[0]
    result = zonal_stats(
        [geom_ll], POP_PATH, stats=["sum"], nodata=None
    )[0]

    value = result.get("sum")
    return float(value) if value is not None else 0.0


print("Summing WorldPop pixels for each village...", flush=True)

villages_m["people_total"] = [
    population_sum(geom) for geom in areas.geometry
]

villages_m["people_affected"] = [
    population_sum(geom) for geom in areas["flooded_part"]
]

villages_m["people_affected"] = villages_m["people_affected"].round(0)

print("\nTop 10 villages by estimated affected population:")
print(
    villages_m[["name", "people_total", "people_affected"]]
    .sort_values("people_affected", ascending=False)
    .head(10).to_string(index=False)
)

# ---------------- STEP 3: ROAD ISOLATION ----------------
print("\nChecking road segments against flood polygons...", flush=True)

edges = ox.graph_to_gdfs(G, nodes=False)
edges_m = edges.to_crs(utm)
edges_m["flooded"] = edges_m.geometry.apply(intersects_flood)

flooded_keys = list(edges_m.index[edges_m["flooded"]])
H = G.copy()
H.remove_edges_from(flooded_keys)

print(
    f"Flooded road segments: {len(flooded_keys)} / {len(edges_m)}",
    flush=True
)

# Find hospitals outside the flood area.
safe_hosp = hospitals_m[
    ~hospitals_m.geometry.apply(intersects_flood)
].copy()

if safe_hosp.empty:
    raise SystemExit(
        "No hospital outside the mapped flood area. "
        "Check hospital data or add a verified safe relief centre."
    )

# Snap safe hospitals to the road graph.
hosp_ll = safe_hosp.to_crs(4326)
hosp_nodes = list(set(ox.nearest_nodes(
    G, X=hosp_ll.geometry.x.to_numpy(),
    Y=hosp_ll.geometry.y.to_numpy()
)))

# Snap villages to their nearest road nodes.
vill_ll = villages_m.to_crs(4326)
villages_m["node"] = ox.nearest_nodes(
    G, X=vill_ll.geometry.x.to_numpy(),
    Y=vill_ll.geometry.y.to_numpy()
)

# Distances to the nearest reachable hospital.
before = nx.multi_source_dijkstra_path_length(
    G.reverse(copy=False), hosp_nodes, weight="length"
)
after = nx.multi_source_dijkstra_path_length(
    H.reverse(copy=False), hosp_nodes, weight="length"
)

villages_m["dist_before_km"] = villages_m["node"].map(
    lambda n: round(before[n] / 1000, 1) if n in before else None
)
villages_m["dist_after_km"] = villages_m["node"].map(
    lambda n: round(after[n] / 1000, 1) if n in after else None
)

villages_m["is_isolated"] = villages_m["dist_after_km"].isna()

print(
    f"\nPotentially isolated villages: "
    f"{int(villages_m['is_isolated'].sum())} of {len(villages_m)}"
)

print("\nVillage impact and accessibility summary:")
print(
    villages_m.sort_values("people_affected", ascending=False)[
        ["name", "people_affected", "is_isolated",
         "dist_before_km", "dist_after_km"]
    ].head(10).to_string(index=False)
)

# ---------------- SAVE RESULTS ----------------
print("\nSaving results...", flush=True)

villages_out = villages_m.drop(columns=["node"]).to_crs(4326)
villages_out.to_file(
    "output/villages_step3.geojson", driver="GeoJSON"
)

edges_m[["flooded", "geometry"]].to_crs(4326).to_file(
    "output/roads_flooded.geojson", driver="GeoJSON"
)

villages_out.drop(columns="geometry").to_csv(
    "output/village_impact_summary.csv", index=False
)

print("Finished. Results saved in the output folder.")
