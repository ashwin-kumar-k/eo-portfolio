import rasterio
from rasterio.warp import transform_bounds
from pathlib import Path

S2_SAFE = Path("/Users/ashwinkumar/Downloads/S2C_MSIL2A_20260225T053811_N0512_R005_T43QBB_20260225T103810.SAFE")
granule_folder = next(f for f in (S2_SAFE / "GRANULE").iterdir() if not f.name.startswith("."))
R10m = granule_folder / "IMG_DATA" / "R10m"
S1_GEOREF = Path("/Users/ashwinkumar/Downloads/S1_georeferenced.tif")

s2_path = R10m / "T43QBB_20260225T053811_B04_10m.jp2"

#Get S2 bounds (already in EPSG:32643)
with rasterio.open(s2_path) as src:
    s2_crs    = src.crs
    s2_bounds = src.bounds
    print("S2 bounds (EPSG:32643 — metres):")
    print(f"  left   : {s2_bounds.left}")
    print(f"  right  : {s2_bounds.right}")
    print(f"  bottom : {s2_bounds.bottom}")
    print(f"  top    : {s2_bounds.top}")

# Get S1 bounds (in EPSG:4326 — degrees)
# Then convert to EPSG:32643 so both are in same units
with rasterio.open(S1_GEOREF) as src:
    s1_bounds_deg = src.bounds
    print("\nS1 bounds (EPSG:4326 — degrees):")
    print(f"  left   : {s1_bounds_deg.left}")
    print(f"  right  : {s1_bounds_deg.right}")
    print(f"  bottom : {s1_bounds_deg.bottom}")
    print(f"  top    : {s1_bounds_deg.top}")

    # Convert S1 bounds from degrees to metres (EPSG:32643)
    s1_bounds_utm = transform_bounds(
        src.crs,      # from EPSG:4326
        s2_crs,       # to   EPSG:32643
        s1_bounds_deg.left,
        s1_bounds_deg.bottom,
        s1_bounds_deg.right,
        s1_bounds_deg.top
    )
    print("\nS1 bounds converted to EPSG:32643 (metres):")
    print(f"  left   : {s1_bounds_utm[0]:.1f}")
    print(f"  bottom : {s1_bounds_utm[1]:.1f}")
    print(f"  right  : {s1_bounds_utm[2]:.1f}")
    print(f"  top    : {s1_bounds_utm[3]:.1f}")

# Calculate intersection mathematically
# Overlap = the area that exists in BOTH datasets
overlap_left   = max(s2_bounds.left,   s1_bounds_utm[0])
overlap_bottom = max(s2_bounds.bottom, s1_bounds_utm[1])
overlap_right  = min(s2_bounds.right,  s1_bounds_utm[2])
overlap_top    = min(s2_bounds.top,    s1_bounds_utm[3])

print("\n" + "="*50)
print("CALCULATED OVERLAP BOUNDING BOX (EPSG:32643)")
print("="*50)
print(f"  left   : {overlap_left:.1f}")
print(f"  bottom : {overlap_bottom:.1f}")
print(f"  right  : {overlap_right:.1f}")
print(f"  top    : {overlap_top:.1f}")

# Validate — check overlap is valid
width_km  = (overlap_right - overlap_left) / 1000
height_km = (overlap_top - overlap_bottom) / 1000

if overlap_right > overlap_left and overlap_top > overlap_bottom:
    print(f"\nOverlap size: {width_km:.1f} km x {height_km:.1f} km")
    print("Overlap is VALID!")
    print("\nCopy these values into your fusion script:")
    print(f'OVERLAP = {{')
    print(f'    "left":   {overlap_left:.1f},')
    print(f'    "bottom": {overlap_bottom:.1f},')
    print(f'    "right":  {overlap_right:.1f},')
    print(f'    "top":    {overlap_top:.1f},')
    print(f'}}')
else:
    print("ERROR: No overlap found between S1 and S2!")
