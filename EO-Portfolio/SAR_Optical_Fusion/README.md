# SAR + Optical Fusion

I wanted to understand how multi-sensor satellites combine radar and optical data into a single image. The idea of two completely different sensors looking at the same place — one using light, one using radio waves — and then aligning them pixel-by-pixel felt like an interesting technical challenge.

So I tried to do it myself using free Sentinel data over Mumbai.

---

## The Problem

Sentinel-1 (radar) and Sentinel-2 (optical) are two separate satellites with different orbits, different sensors, and — critically — **different coordinate systems**. Before you can combine them, you have to solve a geometry problem.

Sentinel-2 uses a projected UTM coordinate system in metres (EPSG:32643).
Sentinel-1 GRD stores its position as Ground Control Points in geographic degrees (EPSG:4326).

They're speaking different languages. The fusion only works once you translate them into the same one.

---

## What I Did

1. Downloaded Sentinel-1 IW-GRD and Sentinel-2 L2A from ESA Copernicus
2. Extracted the 210 Ground Control Points from the raw S1 file and built a proper affine transform
3. Calculated the spatial intersection of both sensor footprints mathematically
4. Reprojected S1 into S2's UTM coordinate system using bilinear resampling
5. Generated a 4-panel comparison image

---

## Output

![Fusion Result](P1_SAR_Optical_Fusion_FINAL.png)

**Panel 1 — True Colour:** Mumbai coastline in natural colour
**Panel 2 — SAR Backscatter:** Radar return values — water is dark, buildings are bright
**Panel 3 — False Colour NIR:** Vegetation appears red, water appears blue
**Panel 4 — Fusion Composite:** R=Red, G=NIR, B=SAR — you can see information from both sensors simultaneously

---

## What I Found Interesting

The SAR panel was the most revealing. Water bodies appear almost black because radar reflects away from the sensor at flat surfaces (specular reflection). Urban areas appear very bright because buildings create a double-bounce effect — radar bounces off the ground, then off a wall, back to the sensor.

Once you know this, you can read a SAR image like a map even without the optical reference.

---

## Scripts

| File | What it does |
|------|-------------|
| `fix_s1_crs.py` | Extracts GCPs from raw S1 and writes a georeferenced GeoTIFF |
| `calculate_overlap.py` | Finds the intersection of S1 and S2 footprints |
| `p1_final_fusion.py` | Main pipeline — reprojects, aligns, and visualises both sensors |

```bash
python3.11 fix_s1_crs.py
python3.11 calculate_overlap.py
python3.11 p1_final_fusion.py
```

---

## Data

- Sentinel-2 L2A — Tile T43QBB, 25 Feb 2026 — https://dataspace.copernicus.eu
- Sentinel-1 IW-GRD — 19 Feb 2026 — https://dataspace.copernicus.eu
