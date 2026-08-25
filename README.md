# climatewatches

`climatewatches` is a small Python toolkit for climate bulletin monitoring and
visualization. It includes station monitoring plots, rainfall and temperature
maps, climatology maps, tropical cyclone monitoring, and large-scale climate
driver diagnostics.

## Installation

From PyPI, after release:

```bash
pip install climatewatches
```

From a local clone:

```bash
pip install .
```

For development:

```bash
pip install -e .
```

## Examples

```python
from climatewatches import plot_precip_climatology_map

plot_precip_climatology_map(
    file="RR_monthly_BOENY_198101-202507.nc",
    shapefile="Boeny-district_new2025.shp",
    analysis="season",
    timescale="ASO",
)
```

```python
from climatewatches import plot_stn_precip_monitoring

plot_stn_precip_monitoring(
    "DATA_MAHAJANGA_AERO_81-25.csv",
    language="fr",
)
```

```python
from climatewatches import plot_climate_indices

plot_climate_indices(
    sst_file="sst.mnmean.nc",
    soi_file="soi_monthly.txt",
    include_roni=True,
    include_oni=True,
    include_iod=False,
    include_siod=True,
    include_soi=True,
)
```

## Data

Large local data files are not included in the package. Keep data directories,
CSV files, and NetCDF files outside version control.

## Build

```bash
python -m build
```

Upload with Twine after checking the generated files:

```bash
twine check dist/*
twine upload dist/*
```
