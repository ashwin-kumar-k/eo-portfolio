# Earth Observation Projects

A collection of hands-on projects I built while learning remote sensing and satellite image analysis. Everything here uses freely available data from ESA Copernicus and other open sources.

I started with a simple question — *how do satellites actually see the Earth differently depending on the sensor?* — and ended up going deeper than I expected.

---

## Projects

| Project | What I explored |
|---------|----------------|
| [SAR + Optical Fusion](./P1_SAR_Optical_Fusion/) | Combining radar and optical satellite imagery at pixel level |
| [Automated Image QC](./P2_QC_Dashboard/) | Writing a script to check satellite image quality automatically |
| [SAR Backscatter Analysis](./P3_Backscatter_Analysis/) | Understanding how radar signals behave over different surfaces |
| [Drone Image QC](./P5_Drone_QC_Pipeline/) | Automating quality checks on a real drone survey dataset |

---

## Data Sources

All data is free and publicly available:

- **Sentinel-1 SAR + Sentinel-2 MSI** — ESA Copernicus Open Access Hub (https://dataspace.copernicus.eu)
- **DJI Drone Survey** — DroneMapper sample dataset (https://dronemapper.com/sample_data)

Study area for satellite projects: **Mumbai Coast, India** (Sentinel tile T43QBB)

---

## Tools

```
Python      Rasterio    NumPy       Pandas
Matplotlib  OpenCV      GDAL        PyProj
ESA SNAP    QGIS        Pillow      ExifRead
```

---

## What I Learned

The biggest insight was understanding that **satellite images are just arrays of numbers** — and once you see them that way, the analysis becomes intuitive. Each project built on the previous one.

SAR was the most surprising. I expected radar imagery to look like a blurry version of an optical image. It doesn't — it sees the world completely differently, responding to surface roughness and geometry rather than colour. Water appears almost black. Metal rooftops are extremely bright. It took a while to develop the intuition for reading it.

---

## Notes

- All code is written in Python 3.11
- Scripts are documented with comments explaining the reasoning, not just the syntax
- Output images and CSV logs are included in each project folder
