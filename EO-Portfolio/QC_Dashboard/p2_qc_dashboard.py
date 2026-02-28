import rasterio
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from pathlib import Path
from datetime import datetime

# ============================================================
# PATHS — YOUR EXACT LOCATIONS
# ============================================================
S2_SAFE = Path("/Users/ashwinkumar/Downloads/S2C_MSIL2A_20260225T053811_N0512_R005_T43QBB_20260225T103810.SAFE")
granule_folder = next(f for f in (S2_SAFE / "GRANULE").iterdir() if not f.name.startswith("."))
R10m = granule_folder / "IMG_DATA" / "R10m"
S1_GEOREF = Path("/Users/ashwinkumar/Downloads/S1_georeferenced.tif")
OUTPUT_DIR = Path("/Users/ashwinkumar/Downloads/P2_QC_Dashboard")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# BAND DEFINITIONS
# ============================================================
S2_BANDS = {
    "B02 (Blue)":  R10m / "T43QBB_20260225T053811_B02_10m.jp2",
    "B03 (Green)": R10m / "T43QBB_20260225T053811_B03_10m.jp2",
    "B04 (Red)":   R10m / "T43QBB_20260225T053811_B04_10m.jp2",
    "B08 (NIR)":   R10m / "T43QBB_20260225T053811_B08_10m.jp2",
}

# QC Thresholds — these are the rules for PASS/FAIL
SATURATION_THRESHOLD  = 1.0    # flag if more than 1% pixels are saturated
NODATA_THRESHOLD      = 5.0    # flag if more than 5% pixels are zero/nodata
SNR_THRESHOLD         = 5.0    # flag if signal-to-noise ratio is below 5
S2_MAX_VALID          = 10000  # S2 L2A reflectance values above this = saturated
S1_MAX_VALID          = 65000  # S1 raw DN values above this = saturated

# ============================================================
# CORE QC FUNCTION — runs health checks on one band
# ============================================================
def run_qc_checks(name, data, max_valid):
    total_pixels = data.size

    # Check 1: No-data pixels (value = 0)
    nodata_pixels = np.sum(data == 0)
    nodata_pct    = (nodata_pixels / total_pixels) * 100

    # Check 2: Saturated pixels (value above max valid)
    saturated_pixels = np.sum(data >= max_valid)
    saturated_pct    = (saturated_pixels / total_pixels) * 100

    # Check 3: Basic statistics
    valid_data = data[(data > 0) & (data < max_valid)]
    mean_val   = np.mean(valid_data)
    std_val    = np.std(valid_data)
    min_val    = np.min(valid_data)
    max_val    = np.max(valid_data)

    # Check 4: Signal-to-Noise Ratio (SNR)
    # SNR = mean / std — higher is better (less noisy)
    snr = mean_val / (std_val + 1e-10)

    # Check 5: Determine overall status
    flags = []
    if saturated_pct > SATURATION_THRESHOLD:
        flags.append("HIGH SATURATION")
    if nodata_pct > NODATA_THRESHOLD:
        flags.append("HIGH NODATA")
    if snr < SNR_THRESHOLD:
        flags.append("LOW SNR")

    status = "FAIL" if flags else "PASS"

    return {
        "Band":           name,
        "Mean":           round(mean_val, 2),
        "Std Dev":        round(std_val, 2),
        "Min":            round(min_val, 2),
        "Max":            round(max_val, 2),
        "SNR":            round(snr, 2),
        "Nodata %":       round(nodata_pct, 3),
        "Saturated %":    round(saturated_pct, 3),
        "Flags":          ", ".join(flags) if flags else "None",
        "Status":         status,
        "valid_data":     valid_data,  # kept for histogram plotting
    }

# ============================================================
# PRINT REPORT TO TERMINAL
# ============================================================
def print_report(results, sensor_name):
    print("\n" + "="*60)
    print(f"  QC HEALTH REPORT — {sensor_name}")
    print(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)

    for r in results:
        status_symbol = "✓" if r["Status"] == "PASS" else "✗"
        print(f"\n  [{status_symbol}] {r['Band']}")
        print(f"      Mean          : {r['Mean']}")
        print(f"      Std Deviation : {r['Std Dev']}")
        print(f"      Min / Max     : {r['Min']} / {r['Max']}")
        print(f"      SNR           : {r['SNR']}")
        print(f"      No-data       : {r['Nodata %']}%")
        print(f"      Saturated     : {r['Saturated %']}%")
        print(f"      Flags         : {r['Flags']}")
        print(f"      Status        : {r['Status']}")

    passed = sum(1 for r in results if r["Status"] == "PASS")
    print(f"\n  Overall: {passed}/{len(results)} bands passed QC")
    print("="*60)

# ============================================================
# SAVE CSV LOG
# ============================================================
def save_csv_log(results, sensor_name, output_path):
    log_data = [{k: v for k, v in r.items() if k != "valid_data"}
                for r in results]
    df = pd.DataFrame(log_data)
    df["Sensor"]    = sensor_name
    df["Timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(output_path, index=False)
    print(f"\n  CSV log saved: {output_path.name}")
    return df

# ============================================================
# PLOT DASHBOARD — histograms + status cards
# ============================================================
def plot_dashboard(s2_results, s1_results, output_path):
    total_panels = len(s2_results) + len(s1_results)
    fig = plt.figure(figsize=(22, 12))
    fig.patch.set_facecolor("#0D1117")

    # Title
    fig.suptitle(
        "Project 2 — Automated Image QC Dashboard  |  Sentinel-2 (MSI) + Sentinel-1 (SAR)  |  Mumbai Coast",
        color="white", fontsize=13, fontweight="bold", y=0.98
    )

    # Create grid: 2 rows (S2 top, S1 bottom), N columns
    n_s2 = len(s2_results)
    n_s1 = len(s1_results)
    gs   = gridspec.GridSpec(2, max(n_s2, n_s1), figure=fig,
                             hspace=0.5, wspace=0.35,
                             left=0.05, right=0.97,
                             top=0.90, bottom=0.08)

    band_colours_s2 = ["#4A90D9", "#4CAF50", "#E74C3C", "#9B59B6"]
    band_colours_s1 = ["#F39C12"]

    def plot_band_histogram(ax, result, colour, row_label):
        valid = result["valid_data"]
        status = result["Status"]
        bg_colour = "#1A2A1A" if status == "PASS" else "#2A1A1A"
        ax.set_facecolor(bg_colour)

        # Plot histogram
        ax.hist(valid, bins=80, color=colour, alpha=0.85, edgecolor="none")

        # Status badge
        badge_col = "#2ECC71" if status == "PASS" else "#E74C3C"
        ax.text(0.97, 0.95, status,
                transform=ax.transAxes,
                color=badge_col, fontsize=9, fontweight="bold",
                ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117", alpha=0.8))

        # Stats text
        stats = (f"Mean: {result['Mean']:.0f}\n"
                 f"SNR:  {result['SNR']:.1f}\n"
                 f"Sat:  {result['Saturated %']:.2f}%")
        ax.text(0.03, 0.95, stats,
                transform=ax.transAxes,
                color="white", fontsize=7.5, va="top", family="monospace",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#0D1117", alpha=0.8))

        # Flag annotation
        if result["Flags"] != "None":
            ax.text(0.5, 0.5, result["Flags"],
                    transform=ax.transAxes,
                    color="#E74C3C", fontsize=8, ha="center", va="center",
                    alpha=0.6, fontweight="bold")

        ax.set_title(f"{row_label}\n{result['Band']}",
                     color="white", fontsize=9, pad=4)
        ax.tick_params(colors="gray", labelsize=7)
        ax.set_xlabel("Pixel Value", color="gray", fontsize=7)
        ax.set_ylabel("Count", color="gray", fontsize=7)
        for spine in ax.spines.values():
            spine.set_edgecolor("#333333")

    # Plot S2 bands — top row
    for i, (result, colour) in enumerate(zip(s2_results, band_colours_s2)):
        ax = fig.add_subplot(gs[0, i])
        plot_band_histogram(ax, result, colour, "Sentinel-2")

    # Plot S1 band — bottom row
    for i, (result, colour) in enumerate(zip(s1_results, band_colours_s1)):
        ax = fig.add_subplot(gs[1, i])
        plot_band_histogram(ax, result, colour, "Sentinel-1 SAR")

    # Add sensor labels on left
    fig.text(0.01, 0.72, "SENTINEL-2\nMSI", color="#4A90D9",
             fontsize=9, fontweight="bold", va="center", rotation=90)
    fig.text(0.01, 0.30, "SENTINEL-1\nSAR", color="#F39C12",
             fontsize=9, fontweight="bold", va="center", rotation=90)

    plt.savefig(output_path, dpi=180, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()
    print(f"\n  Dashboard saved: {output_path.name}")

# ============================================================
# MAIN PIPELINE
# ============================================================
def run_qc_pipeline():
    print("\n" + "="*60)
    print("  P2: AUTOMATED IMAGE QC PIPELINE")
    print("="*60)

    # --- SENTINEL-2 QC ---
    print("\nRunning QC on Sentinel-2 bands...")
    s2_results = []
    for name, path in S2_BANDS.items():
        print(f"  Checking {name}...")
        with rasterio.open(path) as src:
            # Read centre crop for speed (3000x3000 pixels)
            w = rasterio.windows.Window(3990, 3990, 3000, 3000)
            data = src.read(1, window=w).astype(np.float32)
        result = run_qc_checks(name, data, S2_MAX_VALID)
        s2_results.append(result)

    # --- SENTINEL-1 QC ---
    print("\nRunning QC on Sentinel-1 VV band...")
    with rasterio.open(S1_GEOREF) as src:
        # Read centre crop
        w = rasterio.windows.Window(
            src.width  // 2 - 1500,
            src.height // 2 - 1500,
            3000, 3000
        )
        s1_data = src.read(1, window=w).astype(np.float32)
    s1_result = run_qc_checks("VV Backscatter", s1_data, S1_MAX_VALID)
    s1_results = [s1_result]

    # --- PRINT REPORTS ---
    print_report(s2_results, "Sentinel-2 MSI")
    print_report(s1_results, "Sentinel-1 SAR")

    # --- SAVE CSV LOGS ---
    s2_df = save_csv_log(s2_results, "Sentinel-2",
                         OUTPUT_DIR / "S2_QC_log.csv")
    s1_df = save_csv_log(s1_results, "Sentinel-1",
                         OUTPUT_DIR / "S1_QC_log.csv")

    # --- COMBINED LOG ---
    combined = pd.concat([s2_df, s1_df], ignore_index=True)
    combined.to_csv(OUTPUT_DIR / "COMBINED_QC_log.csv", index=False)
    print(f"  Combined log saved: COMBINED_QC_log.csv")

    # --- PLOT DASHBOARD ---
    print("\nGenerating QC Dashboard...")
    plot_dashboard(s2_results, s1_results,
                   OUTPUT_DIR / "P2_QC_Dashboard.png")

    print("\n" + "="*60)
    print("  PROJECT 2 COMPLETE!")
    print("="*60)
    print(f"\n  Output folder: {OUTPUT_DIR}")
    print("  Files saved:")
    print("    - P2_QC_Dashboard.png")
    print("    - S2_QC_log.csv")
    print("    - S1_QC_log.csv")
    print("    - COMBINED_QC_log.csv")

run_qc_pipeline()
