# download_data.py
# Run ONCE. Downloads roads/villages/hospitals for the flood's bounding box,
# saves them to data/, and checks your flood + WorldPop files.
# If a file already exists in data/, that step is skipped, so you can re-run safely.
import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"
import os
import time
import warnings
import geopandas as gpd
import osmnx as ox
import rasterio
from rasterio.windows import from_bounds

warnings.filterwarnings("ignore")

FLOOD_PATH = "data/flood_zones_small.geojson"   # use the small file for testing
POP_PATH = "data/worldpop.tif"
PAD_DEG = 0.02                                  # padding around the flood (~2 km)

OVERPASS_SERVERS = [
    "https://overpass-api.de/api",
    "https://overpass.kumi.systems/api",
]
ox.settings.requests_timeout = 300
ox.settings.use_cache = True
ox.settings.log_console = False

os.makedirs("data", exist_ok=True)

# ---------- 1. Flood file ----------
print("\n[1/4] Checking flood file...")
flood = gpd.read_file(FLOOD_PATH).to_crs(4326)
flood = flood[flood.geometry.notna() & ~flood.geometry.is_empty]
minx, miny, maxx, maxy = flood.total_bounds
print(f"  features: {len(flood)} | bounds: {minx:.3f}, {miny:.3f}, {maxx:.3f}, {maxy:.3f}")
w, h = maxx - minx, maxy - miny
print(f"  box size: {w:.2f} x {h:.2f} degrees (about {w*111:.0f} x {h*111:.0f} km)")
if w > 0.6 or h > 0.6:
    print("  WARNING: this area is large. Road download may fail or be very slow.")
    print("  Consider clipping to a smaller district or sub-area.")

# One merged flood shape is much faster for later analysis than thousands of pieces
if not os.path.exists("data/flood_union.geojson"):
    try:
        merged = flood.geometry.union_all()
    except AttributeError:
        merged = flood.geometry.unary_union
    gpd.GeoDataFrame({"id": [1]}, geometry=[merged.buffer(0)], crs=4326).to_file(
        "data/flood_union.geojson", driver="GeoJSON")
    print("  saved data/flood_union.geojson (single merged flood shape)")

bbox = (minx - PAD_DEG, miny - PAD_DEG, maxx + PAD_DEG, maxy + PAD_DEG) 
BBOX_OVERRIDE = (80.0, 20.75, 80.25, 21.0)   # replace with your top cell
if BBOX_OVERRIDE:
    bbox = BBOX_OVERRIDE # left,bottom,right,top

# ---------- 2. WorldPop check ----------
print("\n[2/4] Checking WorldPop raster...")
with rasterio.open(POP_PATH) as src:
    print(f"  CRS: {src.crs} | size: {src.width}x{src.height} | nodata: {src.nodata}")
    b = src.bounds
    overlaps = not (b.right < minx or b.left > maxx or b.top < miny or b.bottom > maxy)
    print(f"  raster bounds: {b.left:.2f}, {b.bottom:.2f}, {b.right:.2f}, {b.top:.2f}")
    print(f"  covers flood area: {overlaps}")
    if overlaps:
        win = from_bounds(minx, miny, maxx, maxy, src.transform)
        arr = src.read(1, window=win, masked=True)
        print(f"  people in flood bounding box: {float(arr.sum()):,.0f} | max pixel: {float(arr.max()):.1f}")
        print("  (if this is 0 or nan, the raster does not match your area)")

# ---------- helper: retry across servers ----------
def with_retries(fn, label):
    for attempt in range(1, 4):
        for server in OVERPASS_SERVERS:
            ox.settings.overpass_url = server
            try:
                print(f"  {label}: attempt {attempt} via {server}")
                return fn()
            except Exception as e:
                print(f"    failed: {type(e).__name__}: {str(e)[:100]}")
                time.sleep(5)
    raise SystemExit(f"Could not download {label}. Try a smaller area or try later.")

# ---------- 3. Roads ----------
print("\n[3/4] Roads...")
if os.path.exists("data/roads.graphml"):
    print("  already saved, skipping")
else:
    G = with_retries(lambda: ox.graph_from_bbox(bbox, network_type="drive"), "roads")
    ox.save_graphml(G, "data/roads.graphml")
    print(f"  saved data/roads.graphml ({len(G.nodes)} nodes, {len(G.edges)} edges)")

# ---------- 4. Villages and hospitals ----------
print("\n[4/4] Villages and hospitals...")

def grab(tags, out, label):
    if os.path.exists(out):
        print(f"  {label}: already saved, skipping")
        return
    gdf = with_retries(lambda: ox.features_from_bbox(bbox, tags=tags), label)
    gdf = gdf.reset_index()
    keep = [c for c in ["name", "geometry"] if c in gdf.columns]
    gdf[keep].to_file(out, driver="GPKG")
    print(f"  saved {out} ({len(gdf)} features)")

grab({"place": ["village", "hamlet"]}, "data/villages.gpkg", "villages")
grab({"amenity": "hospital"}, "data/hospitals.gpkg", "hospitals")

print("\nDone. Files in data/: roads.graphml, villages.gpkg, hospitals.gpkg, flood_union.geojson")