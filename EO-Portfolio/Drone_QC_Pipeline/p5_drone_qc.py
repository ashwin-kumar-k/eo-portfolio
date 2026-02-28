import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image
import exifread
from datetime import datetime

# ============================================================
# PATHS
# ============================================================
DRONE_FOLDER = Path("/Users/ashwinkumar/Downloads/4thAve")
OUTPUT_DIR   = Path("/Users/ashwinkumar/Downloads/P5_Drone_QC")
OUTPUT_DIR.mkdir(exist_ok=True)

# ============================================================
# QC THRESHOLDS
# ============================================================
BLUR_THRESHOLD        = 400.0   # Was 100 — now stricter
OVEREXPOSE_THRESHOLD  = 1.0     # Was 5.0 — now stricter  
UNDEREXPOSE_THRESHOLD = 1.0     # Was 5.0 — now stricter

# ============================================================
# CHECK 1: BLUR DETECTION
# Uses Laplacian filter — measures edge sharpness
# Sharp image = high variance, Blurry = low variance
# ============================================================
def check_blur(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, "UNREADABLE"
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    variance = cv2.Laplacian(gray, cv2.CV_64F).var()
    status = "BLURRY" if variance < BLUR_THRESHOLD else "SHARP"
    return round(variance, 2), status

# ============================================================
# CHECK 2: EXPOSURE DETECTION
# Counts pixels near 0 (dark) and near 255 (bright)
# ============================================================
def check_exposure(image_path):
    img = cv2.imread(str(image_path))
    if img is None:
        return 0.0, 0.0, "UNREADABLE"
    
    total = img.size
    overexposed  = np.sum(img >= 250) / total * 100
    underexposed = np.sum(img <= 10)  / total * 100

    if overexposed > OVEREXPOSE_THRESHOLD:
        status = "OVEREXPOSED"
    elif underexposed > UNDEREXPOSE_THRESHOLD:
        status = "UNDEREXPOSED"
    else:
        status = "NORMAL"

    return round(overexposed, 3), round(underexposed, 3), status

# ============================================================
# CHECK 3: EXIF METADATA VALIDATION
# Checks GPS, altitude, timestamp from image EXIF
# ============================================================
def check_metadata(image_path):
    with open(image_path, 'rb') as f:
        tags = exifread.process_file(f, details=False)

    metadata = {
        "has_gps":       False,
        "has_altitude":  False,
        "has_timestamp": False,
        "latitude":      None,
        "longitude":     None,
        "altitude_m":    None,
        "timestamp":     None,
        "camera_model":  None,
    }

    # GPS Latitude
    if "GPS GPSLatitude" in tags and "GPS GPSLatitudeRef" in tags:
        metadata["has_gps"] = True
        lat = tags["GPS GPSLatitude"].values
        lat_ref = str(tags["GPS GPSLatitudeRef"])
        # Convert from degrees/minutes/seconds to decimal
        lat_dec = float(lat[0]) + float(lat[1])/60 + float(lat[2])/3600
        if lat_ref == "S":
            lat_dec = -lat_dec
        metadata["latitude"] = round(lat_dec, 6)

    # GPS Longitude
    if "GPS GPSLongitude" in tags and "GPS GPSLongitudeRef" in tags:
        lon = tags["GPS GPSLongitude"].values
        lon_ref = str(tags["GPS GPSLongitudeRef"])
        lon_dec = float(lon[0]) + float(lon[1])/60 + float(lon[2])/3600
        if lon_ref == "W":
            lon_dec = -lon_dec
        metadata["longitude"] = round(lon_dec, 6)

    # Altitude
    if "GPS GPSAltitude" in tags:
        metadata["has_altitude"] = True
        alt = tags["GPS GPSAltitude"].values[0]
        metadata["altitude_m"] = round(float(alt), 1)

    # Timestamp
    if "Image DateTime" in tags:
        metadata["has_timestamp"] = True
        metadata["timestamp"] = str(tags["Image DateTime"])

    # Camera model
    if "Image Model" in tags:
        metadata["camera_model"] = str(tags["Image Model"])

    # Metadata status
    missing = []
    if not metadata["has_gps"]:       missing.append("GPS")
    if not metadata["has_altitude"]:  missing.append("Altitude")
    if not metadata["has_timestamp"]: missing.append("Timestamp")
    meta_status = "MISSING: " + ", ".join(missing) if missing else "COMPLETE"

    return metadata, meta_status

# ============================================================
# QUALITY SCORE CALCULATOR
# Combines all checks into a 0-100 score
# ============================================================
def calculate_score(blur_status, exposure_status, meta_status):
    score = 100

    # Deduct for blur
    if blur_status == "BLURRY":
        score -= 40
    elif blur_status == "UNREADABLE":
        score -= 100

    # Deduct for exposure
    if exposure_status in ["OVEREXPOSED", "UNDEREXPOSED"]:
        score -= 30

    # Deduct for missing metadata
    if "MISSING" in meta_status:
        missing_count = meta_status.count(",") + 1
        score -= (10 * missing_count)

    return max(0, score)

# ============================================================
# MAIN QC PIPELINE
# ============================================================
def run_qc_pipeline():
    print("\n" + "="*60)
    print("  P5: DRONE IMAGE QC PIPELINE")
    print(f"  Dataset: DJI Phantom 3 — Cedaredge Colorado Reservoir")
    print(f"  Images : {len(list(DRONE_FOLDER.glob('*.JPG')))}")
    print("="*60)

    images = sorted(DRONE_FOLDER.glob("*.JPG"))
    results = []

    for i, img_path in enumerate(images):
        print(f"  Checking {img_path.name} ({i+1}/{len(images)})...")

        # Run all 3 checks
        blur_var,  blur_status     = check_blur(img_path)
        over_pct,  under_pct, exp_status = check_exposure(img_path)
        metadata,  meta_status     = check_metadata(img_path)

        # Calculate quality score
        score = calculate_score(blur_status, exp_status, meta_status)

        # Overall pass/fail
        overall = "PASS" if score >= 70 else "FAIL"

        results.append({
            "Filename":       img_path.name,
            "Blur Variance":  blur_var,
            "Blur Status":    blur_status,
            "Overexposed %":  over_pct,
            "Underexposed %": under_pct,
            "Exposure Status":exp_status,
            "GPS Present":    metadata["has_gps"],
            "Altitude (m)":   metadata["altitude_m"],
            "Timestamp":      metadata["timestamp"],
            "Camera":         metadata["camera_model"],
            "Meta Status":    meta_status,
            "Quality Score":  score,
            "Overall":        overall,
        })

    # Convert to DataFrame
    df = pd.DataFrame(results)

    # Print summary
    print("\n" + "="*60)
    print("  QC SUMMARY REPORT")
    print("="*60)
    print(f"\n  Total images   : {len(df)}")
    print(f"  PASS           : {len(df[df['Overall']=='PASS'])}")
    print(f"  FAIL           : {len(df[df['Overall']=='FAIL'])}")
    print(f"  Blurry         : {len(df[df['Blur Status']=='BLURRY'])}")
    print(f"  Overexposed    : {len(df[df['Exposure Status']=='OVEREXPOSED'])}")
    print(f"  Underexposed   : {len(df[df['Exposure Status']=='UNDEREXPOSED'])}")
    print(f"  Missing GPS    : {len(df[df['GPS Present']==False])}")
    print(f"\n  Mean Quality Score : {df['Quality Score'].mean():.1f}/100")
    print(f"  Min Quality Score  : {df['Quality Score'].min()}/100")

    # Print per-image table
    print("\n  PER IMAGE RESULTS:")
    print(f"  {'Filename':<18} {'Score':>6} {'Blur':>10} {'Exposure':>12} {'Meta':>10} {'Result':>6}")
    print("  " + "-"*66)
    for _, row in df.iterrows():
        print(f"  {row['Filename']:<18} {row['Quality Score']:>6} "
              f"{row['Blur Status']:>10} {row['Exposure Status']:>12} "
              f"{row['Meta Status']:>10} {row['Overall']:>6}")

    # Save CSV
    csv_path = OUTPUT_DIR / "drone_qc_report.csv"
    df.to_csv(csv_path, index=False)
    print(f"\n  CSV saved: {csv_path}")

    # ============================================================
    # VISUALIZATION
    # ============================================================
    print("\nGenerating QC Dashboard...")
    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor("#0D1117")
    fig.suptitle(
        "Project 5 — Drone Image QC Pipeline  |  DJI Phantom 3 FC300S  |  Cedaredge Reservoir, Colorado",
        color="white", fontsize=12, fontweight="bold", y=0.98
    )
    gs = gridspec.GridSpec(3, 3, figure=fig,
                           hspace=0.5, wspace=0.35,
                           left=0.07, right=0.97,
                           top=0.93, bottom=0.07)

    # Panel 1: Quality score per image
    ax1 = fig.add_subplot(gs[0, :])
    ax1.set_facecolor("#1A1A2E")
    colors = ["#2ECC71" if s >= 70 else "#E74C3C"
              for s in df["Quality Score"]]
    bars = ax1.bar(range(len(df)), df["Quality Score"],
                   color=colors, alpha=0.85, width=0.8)
    ax1.axhline(y=70, color="white", linestyle="--",
                linewidth=1.5, label="Pass threshold (70)")
    ax1.set_xticks(range(len(df)))
    ax1.set_xticklabels([f.replace("DJI_","").replace(".JPG","")
                         for f in df["Filename"]],
                        rotation=45, color="gray", fontsize=7)
    ax1.set_ylabel("Quality Score", color="white", fontsize=9)
    ax1.set_title("Per-Image Quality Score  (Green=PASS  Red=FAIL)",
                  color="white", fontsize=10, fontweight="bold")
    ax1.set_ylim(0, 110)
    ax1.legend(facecolor="#0D1117", labelcolor="white", fontsize=8)
    ax1.tick_params(colors="gray")
    for sp in ax1.spines.values(): sp.set_color("#333333")
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # Panel 2: Blur variance distribution
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.set_facecolor("#1A1A2E")
    ax2.hist(df["Blur Variance"], bins=20,
             color="#3498DB", alpha=0.85, edgecolor="none")
    ax2.axvline(x=BLUR_THRESHOLD, color="#E74C3C",
                linestyle="--", linewidth=1.5,
                label=f"Blur threshold ({BLUR_THRESHOLD})")
    ax2.set_xlabel("Laplacian Variance", color="white", fontsize=8)
    ax2.set_ylabel("Image Count", color="white", fontsize=8)
    ax2.set_title("Blur Variance Distribution",
                  color="white", fontsize=10, fontweight="bold")
    ax2.legend(facecolor="#0D1117", labelcolor="white", fontsize=8)
    ax2.tick_params(colors="gray")
    for sp in ax2.spines.values(): sp.set_color("#333333")
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # Panel 3: Exposure status pie chart
    ax3 = fig.add_subplot(gs[1, 1])
    ax3.set_facecolor("#1A1A2E")
    exp_counts = df["Exposure Status"].value_counts()
    pie_colors = {"NORMAL": "#2ECC71",
                  "OVEREXPOSED": "#E74C3C",
                  "UNDEREXPOSED": "#E67E22"}
    ax3.pie(exp_counts.values,
            labels=exp_counts.index,
            colors=[pie_colors.get(k, "#888") for k in exp_counts.index],
            autopct="%1.0f%%",
            textprops={"color": "white", "fontsize": 9},
            startangle=90)
    ax3.set_title("Exposure Status Distribution",
                  color="white", fontsize=10, fontweight="bold")

    # Panel 4: GPS coverage
    ax4 = fig.add_subplot(gs[1, 2])
    ax4.set_facecolor("#1A1A2E")
    gps_counts = df["GPS Present"].value_counts()
    gps_labels = ["GPS Present" if k else "GPS Missing"
                  for k in gps_counts.index]
    gps_colors = ["#2ECC71" if k else "#E74C3C"
                  for k in gps_counts.index]
    ax4.pie(gps_counts.values,
            labels=gps_labels,
            colors=gps_colors,
            autopct="%1.0f%%",
            textprops={"color": "white", "fontsize": 9},
            startangle=90)
    ax4.set_title("GPS Metadata Coverage",
                  color="white", fontsize=10, fontweight="bold")

    # Panel 5: Altitude profile
    ax5 = fig.add_subplot(gs[2, :2])
    ax5.set_facecolor("#1A1A2E")
    altitudes = df["Altitude (m)"].fillna(0)
    ax5.plot(range(len(df)), altitudes,
             color="#F39C12", linewidth=2,
             marker="o", markersize=4)
    ax5.fill_between(range(len(df)), altitudes,
                     alpha=0.2, color="#F39C12")
    ax5.set_xticks(range(len(df)))
    ax5.set_xticklabels([f.replace("DJI_","").replace(".JPG","")
                         for f in df["Filename"]],
                        rotation=45, color="gray", fontsize=7)
    ax5.set_ylabel("Elevation ASL (metres)", color="white", fontsize=8)
    ax5.set_title("Image Capture Elevation — Cedaredge CO (~1994m ASL)",
                  color="white", fontsize=10, fontweight="bold")
    ax5.tick_params(colors="gray")
    for sp in ax5.spines.values(): sp.set_color("#333333")
    ax5.spines["top"].set_visible(False)
    ax5.spines["right"].set_visible(False)

    # Panel 6: Summary stats card
    ax6 = fig.add_subplot(gs[2, 2])
    ax6.set_facecolor("#1A1A2E")
    ax6.axis("off")
    pass_count = len(df[df["Overall"] == "PASS"])
    fail_count = len(df[df["Overall"] == "FAIL"])
    summary_text = (
        f"DATASET SUMMARY\n"
        f"{'─'*22}\n"
        f"Sensor      : DJI FC300S\n"
        f"Date        : 14 Nov 2017\n"
        f"Location    : Cedaredge CO\n"
        f"Total Images: {len(df)}\n\n"
        f"PASS  : {pass_count} ({pass_count/len(df)*100:.0f}%)\n"
        f"FAIL  : {fail_count} ({fail_count/len(df)*100:.0f}%)\n\n"
        f"Blurry      : {len(df[df['Blur Status']=='BLURRY'])}\n"
        f"Overexposed : {len(df[df['Exposure Status']=='OVEREXPOSED'])}\n"
        f"Missing GPS : {len(df[df['GPS Present']==False])}\n\n"
        f"Mean Score  : {df['Quality Score'].mean():.1f}/100\n"
    )
    ax6.text(0.05, 0.95, summary_text,
             transform=ax6.transAxes,
             color="white", fontsize=9,
             verticalalignment="top",
             fontfamily="monospace",
             bbox=dict(boxstyle="round,pad=0.5",
                       facecolor="#0D1117", alpha=0.8))
    ax6.set_title("QC Summary", color="white",
                  fontsize=10, fontweight="bold")

    out = OUTPUT_DIR / "P5_Drone_QC_Dashboard.png"
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.show()

    print(f"\n  Dashboard saved: {out}")
    print("\n" + "="*60)
    print("  PROJECT 5 COMPLETE!")
    print("="*60)
    print(f"\n  Output folder: {OUTPUT_DIR}")
    print("  Files saved:")
    print("    - P5_Drone_QC_Dashboard.png")
    print("    - drone_qc_report.csv")

run_qc_pipeline()
