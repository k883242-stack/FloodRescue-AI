# find_hotspot.py
# Splits your big flood area into grid cells and ranks them by flooded area,
# so you can pick a smaller sub-area (about 25 x 25 km) to analyse.

import os
os.environ["OGR_GEOJSON_MAX_OBJ_SIZE"] = "0"

import warnings
import numpy as np
import geopandas as gpd

warnings.filterwarnings("ignore")

FLOOD_PATH = "data/flood_zones_small.geojson"
CELL_DEG = 0.25          # about 27 km per cell

flood = gpd.read_file(FLOOD_PATH).to_crs(4326)
parts = flood.explode(index_parts=False).reset_index(drop=True)   # split multipolygon
print(f"flood pieces: {len(parts)}")

utm = parts.estimate_utm_crs()
parts_m = parts.to_crs(utm)
area_km2 = parts_m.geometry.area / 1e6
cent = parts.geometry.centroid

minx, miny, maxx, maxy = parts.total_bounds
cx = np.floor((cent.x - minx) / CELL_DEG).astype(int)
cy = np.floor((cent.y - miny) / CELL_DEG).astype(int)

df = gpd.pd.DataFrame({"cx": cx, "cy": cy, "km2": area_km2.values})
grid = df.groupby(["cx", "cy"])["km2"].sum().reset_index()
grid["left"] = minx + grid.cx * CELL_DEG
grid["bottom"] = miny + grid.cy * CELL_DEG
grid["right"] = grid.left + CELL_DEG
grid["top"] = grid.bottom + CELL_DEG
grid = grid.sort_values("km2", ascending=False).head(6)

print("\nTop 6 most flooded cells (flooded km2 per cell):\n")
for _, r in grid.iterrows():
    print(f"  flooded {r.km2:8.1f} km2   BBOX = "
          f"({r.left:.3f}, {r.bottom:.3f}, {r.right:.3f}, {r.top:.3f})")
print("\nCopy the BBOX of the top cell into download_data.py (BBOX_OVERRIDE).")