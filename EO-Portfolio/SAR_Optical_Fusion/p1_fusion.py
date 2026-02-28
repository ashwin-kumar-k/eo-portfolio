
import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from rasterio.warp import calculate_default_transform, reproject, Resampling
from pathlib import Path

S2_SAFE = Path("/Users/ashwinkumar/Downloads/S2C_MSIL2A_20260225T053811_N0512_R005_T43QBB_20260225T103810.SAFE")
granule_folder = next(f for f in (S2_SAFE / "GRANULE").iterdir() if not f.name.startswith("."))
R10m = granule_folder / "IMG_DATA" / "R10m"

S1_GEOREF = Path("/Users/ashwinkumar/Downloads/S1_georeferenced.tif")
OUTPUT_DIR = Path("/Users/ashwinkumar/Downloads/P1_SAR_Optical_Fusion")
OUTPUT_DIR.mkdir(exist_ok=True)

S2_BANDS = {
    "B02": R10m / "T43QBB_20260225T053811_B02_10m.jp2",
    "B03": R10m / "T43QBB_20260225T053811_B03_10m.jp2",
    "B04": R10m / "T43QBB_20260225T053811_B04_10m.jp2",
    "B08": R10m / "T43QBB_20260225T053811_B08_10m.jp2",
}

#calculated overlap (from calculate_overlap.py)
OVERLAP = {
    "left":   199980.0,
    "bottom": 2090220.0,
    "right":  309780.0,
    "top":    2195885.3,
}

def load_s2_crop(band_path, overlap):
    with rasterio.open(band_path) as src:
        window = rasterio.windows.from_bounds(
            overlap["left"], overlap["bottom"],
            overlap["right"], overlap["top"],
            transform=src.transform
        )
        data = src.read(1, window=window).astype(np.float32)
        transform = src.window_transform(window)
        crs = src.crs
    return data, transform, crs

def reproject_s1_to_overlap(s1_path, ref_transform, ref_crs, ref_shape, output_path):
    print("Reprojecting S1 to match S2 overlap zone...")
    with rasterio.open(s1_path) as s1_src:
        profile = s1_src.profile.copy()
        profile.update({
            "crs":     ref_crs,
            "transform": ref_transform,
            "width":   ref_shape[1],
            "height":  ref_shape[0],
            "driver":  "GTiff",
            "dtype":   "float32",
            "nodata":  0,
            "count":   1,
        })
        with rasterio.open(output_path, "w", **profile) as dst:
            reproject(
                source=rasterio.band(s1_src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=s1_src.transform,
                src_crs=s1_src.crs,
                dst_transform=ref_transform,
                dst_crs=ref_crs,
                resampling=Resampling.bilinear
            )
    print(f"  S1 reprojected: {output_path.name}")

def normalize(data, low=2, high=98):
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data)
    p_low  = np.percentile(valid, low)
    p_high = np.percentile(valid, high)
    return np.clip((data - p_low) / (p_high - p_low + 1e-10), 0, 1)

def run_fusion():
    print("\n" + "="*55)
    print("  P1: SAR-OPTICAL FUSION — FINAL PIPELINE")
    print("="*55)

    print("\nLoading S2 bands over overlap zone...")
    s2_red,  transform, crs = load_s2_crop(S2_BANDS["B04"], OVERLAP)
    s2_nir,  _,         _   = load_s2_crop(S2_BANDS["B08"], OVERLAP)
    s2_blue, _,         _   = load_s2_crop(S2_BANDS["B02"], OVERLAP)
    s2_grn,  _,         _   = load_s2_crop(S2_BANDS["B03"], OVERLAP)

    print(f"  S2 crop shape : {s2_red.shape}")

    s1_out = OUTPUT_DIR / "S1_aligned.tif"
    reproject_s1_to_overlap(S1_GEOREF, transform, crs, s2_red.shape, s1_out)

    with rasterio.open(s1_out) as src:
        s1_vv = src.read(1).astype(np.float32)

    print("\nNormalizing all bands...")
    s2_red_n  = normalize(s2_red)
    s2_nir_n  = normalize(s2_nir)
    s2_blue_n = normalize(s2_blue)
    s2_grn_n  = normalize(s2_grn)
    s1_vv_n   = normalize(s1_vv)

    s2_rgb   = np.stack([s2_red_n, s2_grn_n, s2_blue_n], axis=-1)
    s2_false = np.stack([s2_nir_n, s2_red_n, s2_blue_n], axis=-1)
    fusion   = np.stack([s2_red_n, s2_nir_n, s1_vv_n],   axis=-1)

    print("\nGenerating final 4-panel visualization...")
    fig = plt.figure(figsize=(20, 7))
    fig.patch.set_facecolor("#0D1117")
    gs = gridspec.GridSpec(1, 4, figure=fig, wspace=0.04,
                           left=0.01, right=0.99, top=0.88, bottom=0.02)

    panels = [
        (s2_rgb,   "Sentinel-2\nTrue Colour (RGB)",           None),
        (s1_vv_n,  "Sentinel-1\nSAR Backscatter (VV)",        "gray"),
        (s2_false, "Sentinel-2\nFalse Colour (NIR-Red-Blue)",  None),
        (fusion,   "SAR-Optical FUSION\nR:S2Red G:S2NIR B:S1VV", None),
    ]

    for i, (data, title, cmap) in enumerate(panels):
        ax = fig.add_subplot(gs[i])
        if cmap:
            ax.imshow(data, cmap=cmap)
        else:
            ax.imshow(np.clip(data, 0, 1))
        ax.set_title(title, color="white", fontsize=10, pad=6, fontweight="bold")
        ax.axis("off")

    fig.suptitle(
    "Project 1 — SAR-Optical Fusion  |  Sentinel-1 (IW-GRD VV) + Sentinel-2 (MSI)  |  Mumbai Coast, India",
    color="white", fontsize=12, fontweight="bold"
    )

    output_plot = OUTPUT_DIR / "P1_SAR_Optical_Fusion_FINAL.png"
    plt.savefig(output_plot, dpi=200, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()

    print(f"\nFinal image saved: {output_plot}")
    print("\n" + "="*55)
    print("  PROJECT 1 COMPLETE!")
    print("="*55)

run_fusion()
