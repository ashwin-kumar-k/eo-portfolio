import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
from rasterio.windows import from_bounds

# ============================================================
# PATHS
# ============================================================
SAR_TC  = Path("/Users/ashwinkumar/Downloads/SAR_TC.tif")
S2_SAFE = Path("/Users/ashwinkumar/Downloads/S2C_MSIL2A_20260225T053811_N0512_R005_T43QBB_20260225T103810.SAFE")
granule_folder = next(f for f in (S2_SAFE/"GRANULE").iterdir()
                      if not f.name.startswith("."))
R10m = granule_folder / "IMG_DATA" / "R10m"
S2_BANDS = {
    "B02": R10m / "T43QBB_20260225T053811_B02_10m.jp2",
    "B03": R10m / "T43QBB_20260225T053811_B03_10m.jp2",
    "B04": R10m / "T43QBB_20260225T053811_B04_10m.jp2",
    "B08": R10m / "T43QBB_20260225T053811_B08_10m.jp2",
}
OUTPUT_DIR = Path("/Users/ashwinkumar/Downloads/P3_Backscatter_Analysis")
OUTPUT_DIR.mkdir(exist_ok=True)

# Small 20km x 20km crop
OVERLAP = {
    "left":   280000.0,
    "bottom": 2095000.0,
    "right":  300000.0,
    "top":    2115000.0,
}

CLASS_WINDOWS = [
    (200,  200,  150),
    (800,  800,  150),
    (400, 1400,  150),
    (1400, 600,  150),
]
CLASS_NAMES  = ["Water", "Urban", "Vegetation", "Bare Soil"]
CLASS_COLORS = ["#3498DB", "#E74C3C", "#2ECC71", "#E67E22"]

# ============================================================
# FUNCTIONS
# ============================================================
def load_crop(path, overlap, band=1):
    with rasterio.open(path) as src:
        window = from_bounds(
            overlap["left"], overlap["bottom"],
            overlap["right"], overlap["top"],
            transform=src.transform
        )
        return src.read(band, window=window).astype(np.float32)

def to_db(data):
    return 10 * np.log10(np.where(data <= 0, 1e-10, data))

def normalize(data, low=2, high=98):
    valid = data[np.isfinite(data) & (data > -50)]
    if len(valid) == 0:
        return np.zeros_like(data)
    p_low  = np.percentile(valid, low)
    p_high = np.percentile(valid, high)
    return np.clip((data - p_low) / (p_high - p_low + 1e-10), 0, 1)

def normalize_s2(data):
    valid = data[data > 0]
    if len(valid) == 0:
        return np.zeros_like(data)
    p2  = np.percentile(valid, 2)
    p98 = np.percentile(valid, 98)
    return np.clip((data - p2) / (p98 - p2 + 1e-10), 0, 1)

def calculate_rvi(vv_db, vh_db):
    vv_lin = np.power(10, vv_db / 10)
    vh_lin = np.power(10, vh_db / 10)
    rvi = (4 * vh_lin) / (vv_lin + vh_lin + 1e-10)
    return np.where((vv_lin < 1e-8) | (vh_lin < 1e-8), np.nan, rvi).astype(np.float32)

def extract_stats(data_db, windows, names):
    results = []
    for name, (r, c, size) in zip(names, windows):
        patch = data_db[r:r+size, c:c+size]
        valid = patch[np.isfinite(patch) & (patch > -40)]
        if len(valid) == 0:
            print(f"  WARNING: No valid pixels for {name}")
            continue
        results.append({
            "Class":   name,
            "Mean dB": round(float(np.mean(valid)), 2),
            "Std dB":  round(float(np.std(valid)),  2),
            "Min dB":  round(float(np.min(valid)),  2),
            "Max dB":  round(float(np.max(valid)),  2),
        })
    return results

# ============================================================
# MAIN
# ============================================================
def run():
    print("\n" + "="*60)
    print("  P3: SAR BACKSCATTER + SPECTRAL ANALYSIS (FAST)")
    print("="*60)

    print("\nLoading SAR bands (20km crop)...")
    vv_db = to_db(load_crop(SAR_TC, OVERLAP, band=1))
    vh_db = to_db(load_crop(SAR_TC, OVERLAP, band=2))
    print(f"  Shape        : {vv_db.shape}")
    print(f"  Valid pixels : {np.sum(vv_db > -40):,}")

    print("Calculating RVI...")
    rvi = calculate_rvi(vv_db, vh_db)
    valid_rvi = rvi[np.isfinite(rvi) & (rvi >= 0) & (rvi <= 1)]
    print(f"  RVI mean : {np.nanmean(valid_rvi):.3f}")

    print("Loading Sentinel-2 bands...")
    s2_blue  = load_crop(S2_BANDS["B02"], OVERLAP)
    s2_green = load_crop(S2_BANDS["B03"], OVERLAP)
    s2_red   = load_crop(S2_BANDS["B04"], OVERLAP)
    s2_nir   = load_crop(S2_BANDS["B08"], OVERLAP)

    print("Extracting stats per class...")
    vv_stats = extract_stats(vv_db, CLASS_WINDOWS, CLASS_NAMES)
    vh_stats = extract_stats(vh_db, CLASS_WINDOWS, CLASS_NAMES)

    print(f"\n  VV BACKSCATTER PER CLASS (dB):")
    print(f"  {'Class':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("  " + "-"*47)
    for r in vv_stats:
        print(f"  {r['Class']:<15} {r['Mean dB']:>8.2f} "
              f"{r['Std dB']:>8.2f} {r['Min dB']:>8.2f} {r['Max dB']:>8.2f}")

    print("\nExtracting spectral signatures...")
    band_data = [s2_blue, s2_green, s2_red, s2_nir]
    band_wl   = [490, 560, 665, 842]
    spectral  = {}
    for name, (r, c, size) in zip(CLASS_NAMES, CLASS_WINDOWS):
        sig = []
        for band in band_data:
            patch = band[r:r+size, c:c+size]
            valid = patch[patch > 0]
            sig.append(float(np.mean(valid)) if len(valid) > 0 else 0)
        spectral[name] = sig
        print(f"  {name:<15} B:{sig[0]:.0f} G:{sig[1]:.0f} "
              f"R:{sig[2]:.0f} NIR:{sig[3]:.0f}")

    # Save CSVs
    df_vv = pd.DataFrame(vv_stats); df_vv["Pol"] = "VV"
    df_vh = pd.DataFrame(vh_stats); df_vh["Pol"] = "VH"
    pd.concat([df_vv, df_vh]).to_csv(
        OUTPUT_DIR/"backscatter_stats.csv", index=False)
    pd.DataFrame(spectral, index=["Blue","Green","Red","NIR"]).T.to_csv(
        OUTPUT_DIR/"spectral_signatures.csv")
    print("\n  CSVs saved!")

    # ============================================================
    # PLOT
    # ============================================================
    print("\nGenerating dashboard...")
    fig = plt.figure(figsize=(22, 16))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "Project 3 — SAR Backscatter + Spectral Signatures  |  Mumbai Coast, India",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 4, figure=fig,
                           hspace=0.45, wspace=0.35,
                           left=0.06, right=0.97,
                           top=0.93, bottom=0.06)

    # Row 1 — SAR maps
    for i, (data, title, cmap) in enumerate([
        (normalize(vv_db),  "VV Backscatter (dB)",       "gray"),
        (normalize(vh_db),  "VH Backscatter (dB)",        "gray"),
        (np.where(np.isfinite(rvi), rvi, 0), "Radar Vegetation Index (RVI)", "RdYlGn"),
    ]):
        ax = fig.add_subplot(gs[0, i])
        ax.set_facecolor("#1A1A2E")
        im = ax.imshow(data, cmap=cmap)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_title(title, color="white", fontsize=10, pad=6, fontweight="bold")
        ax.axis("off")

    # Row 1 Panel 4 — bar chart
    ax4 = fig.add_subplot(gs[0, 3])
    ax4.set_facecolor("#1A1A2E")
    if vv_stats and vh_stats:
        names_f = [r["Class"] for r in vv_stats]
        x = np.arange(len(names_f))
        ax4.bar(x-0.175, [r["Mean dB"] for r in vv_stats], 0.35,
                label="VV", color="#3498DB", alpha=0.85)
        ax4.bar(x+0.175, [r["Mean dB"] for r in vh_stats], 0.35,
                label="VH", color="#E74C3C", alpha=0.85)
        ax4.set_xticks(x)
        ax4.set_xticklabels(names_f, color="white", fontsize=7, rotation=15)
        ax4.set_ylabel("Mean Backscatter (dB)", color="white", fontsize=8)
        ax4.legend(facecolor="#0D1117", labelcolor="white", fontsize=8)
    ax4.set_title("VV vs VH by Land Cover",
                  color="white", fontsize=10, fontweight="bold")
    ax4.tick_params(colors="gray")
    for sp in ax4.spines.values(): sp.set_color("#333333")
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # Row 2 — S2 bands
    for i, (data, title, cmap) in enumerate([
        (normalize_s2(s2_blue),  "S2 Blue (B02)",  "Blues"),
        (normalize_s2(s2_green), "S2 Green (B03)", "Greens"),
        (normalize_s2(s2_red),   "S2 Red (B04)",   "Reds"),
        (normalize_s2(s2_nir),   "S2 NIR (B08)",   "RdPu"),
    ]):
        ax = fig.add_subplot(gs[1, i])
        ax.set_facecolor("#1A1A2E")
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color="white", fontsize=10, pad=6, fontweight="bold")
        ax.axis("off")

    # Row 3 — Spectral signatures
    ax_spec = fig.add_subplot(gs[2, :2])
    ax_spec.set_facecolor("#1A1A2E")
    for (name, sig), color in zip(spectral.items(), CLASS_COLORS):
        ax_spec.plot(band_wl, sig, color=color, linewidth=2.5,
                     marker="o", markersize=7, label=name)
    ax_spec.set_xlabel("Wavelength (nm)", color="white", fontsize=9)
    ax_spec.set_ylabel("Mean Reflectance (DN)", color="white", fontsize=9)
    ax_spec.set_title("Spectral Signature Profiles by Land Cover Class",
                      color="white", fontsize=10, fontweight="bold")
    ax_spec.legend(facecolor="#0D1117", labelcolor="white", fontsize=9)
    ax_spec.set_xticks(band_wl)
    ax_spec.set_xticklabels(["490nm\nBlue","560nm\nGreen",
                              "665nm\nRed","842nm\nNIR"], color="gray")
    ax_spec.tick_params(colors="gray")
    for sp in ax_spec.spines.values(): sp.set_color("#333333")

    # Row 3 — RVI histogram
    ax_rvi = fig.add_subplot(gs[2, 2:])
    ax_rvi.set_facecolor("#1A1A2E")
    ax_rvi.hist(valid_rvi, bins=80, color="#2ECC71",
                alpha=0.8, edgecolor="none")
    ax_rvi.axvline(x=0.3, color="#E74C3C", linestyle="--",
                   linewidth=1.5, label="RVI=0.3 (veg threshold)")
    ax_rvi.set_xlabel("RVI Value (0=bare, 1=dense veg)",
                      color="white", fontsize=9)
    ax_rvi.set_ylabel("Pixel Count", color="white", fontsize=9)
    ax_rvi.set_title("RVI Distribution Histogram",
                     color="white", fontsize=10, fontweight="bold")
    ax_rvi.legend(facecolor="#0D1117", labelcolor="white", fontsize=8)
    ax_rvi.tick_params(colors="gray")
    for sp in ax_rvi.spines.values(): sp.set_color("#333333")

    out = OUTPUT_DIR / "P3_Backscatter_Analysis.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n  Saved: {out}")
    print("\n" + "="*60)
    print("  PROJECT 3 COMPLETE!")
    print("="*60)

run()
