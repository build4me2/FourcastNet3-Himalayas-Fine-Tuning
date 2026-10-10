"""Run one forecast (or reuse an existing NetCDF) and plot raw vs corrected t2m over the box.

    pip install "fcn3-himalayas[plot] @ git+https://github.com/build4me2/FourcastNet3-Himalayas-Fine-Tuning@v0.1.0"
    python examples/plot_forecast.py --init 2024-07-01T00 --lead 120
"""
import argparse
import os

import matplotlib.pyplot as plt
import xarray as xr

ap = argparse.ArgumentParser()
ap.add_argument("--init", default="2024-07-01T00")
ap.add_argument("--lead", type=int, default=120)
ap.add_argument("--nc", default="forecast.nc")
a = ap.parse_args()

if not os.path.exists(a.nc):
    from fcn3_himalayas.pipeline import run_forecast

    run_forecast(a.init, a.lead, a.nc)

ds = xr.open_dataset(a.nc).sel(lead_time=a.lead)
raw, cor = ds.t2m_raw - 273.15, ds.t2m_corrected - 273.15
vmin, vmax = float(min(raw.min(), cor.min())), float(max(raw.max(), cor.max()))
fig, ax = plt.subplots(1, 3, figsize=(15, 3.6), constrained_layout=True)
raw.plot(ax=ax[0], vmin=vmin, vmax=vmax, cmap="RdYlBu_r", cbar_kwargs={"label": "°C"})
cor.plot(ax=ax[1], vmin=vmin, vmax=vmax, cmap="RdYlBu_r", cbar_kwargs={"label": "°C"})
(cor - raw).plot(ax=ax[2], cmap="RdBu_r", center=0, cbar_kwargs={"label": "K"})
for x, t in zip(ax, ["FCN3 raw t2m", "FCN3 + adapter t2m", "correction"]):
    ds.elevation.plot.contour(ax=x, levels=[1500, 3000, 4500], colors="k", linewidths=0.5)
    x.set_title(f"{t}  (+{a.lead} h from {ds.attrs['init_time']})")
fig.savefig("t2m_raw_vs_corrected.png", dpi=120)
print("wrote t2m_raw_vs_corrected.png")
