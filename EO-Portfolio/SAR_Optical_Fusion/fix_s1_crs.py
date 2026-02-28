import rasterio
from rasterio.crs import CRS
from rasterio.transform import from_gcps
from pathlib import Path

S1_SAFE = Path("/Users/ashwinkumar/Downloads/S1A_IW_GRDH_1SDV_20260219T010312_20260219T010337_063281_07F235_63D3.SAFE")
s1_path = next((S1_SAFE / "measurement").glob("*-vv-*.tiff"))
OUTPUT = Path("/Users/ashwinkumar/Downloads/S1_georeferenced.tif")

print("Reading S1 GCPs...")

with rasterio.open(s1_path) as src:
    gcps, gcp_crs = src.gcps

    print(f"  Found {len(gcps)} GCPs")
    print(f"  GCP CRS: {gcp_crs}")
    print(f"  S1 shape: {src.height} x {src.width}")

    if len(gcps) == 0:
        print("ERROR: No GCPs found!")
    else:
        # Show first 3 GCPs
        for g in gcps[:3]:
            print(f"  GCP: pixel({g.col:.0f},{g.row:.0f}) -> geo({g.x:.4f},{g.y:.4f})")

        transform = from_gcps(gcps)

        profile = src.profile.copy()
        profile.update({
            "crs": CRS.from_epsg(4326),
            "transform": transform,
            "driver": "GTiff",
        })

        print(f"\nWriting georeferenced S1 to: {OUTPUT.name}")
        data = src.read(1)

        with rasterio.open(OUTPUT, "w", **profile) as dst:
            dst.write(data, 1)

print("\nVerifying output...")
with rasterio.open(OUTPUT) as src:
    print(f"  CRS    : {src.crs}")
    print(f"  Bounds : {src.bounds}")
    print(f"  Shape  : {src.height} x {src.width}")
