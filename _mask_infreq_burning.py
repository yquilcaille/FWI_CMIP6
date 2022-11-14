import csv
import os
import warnings

import cartopy.crs as ccrs
import matplotlib.animation as anim
import matplotlib.colors as plcol
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import xarray as xr
from matplotlib import cm
from mpl_toolkits.axes_grid1 import inset_locator, make_axes_locatable

from functions_support_plots import get_spatial_info

# from statsmodels.nonparametric.smoothers_lowess import lowess


# ============================================================================================
# 0. OPTIONS
# ============================================================================================
# INPUTS
grids = {
    "cmip6ng": "cmip6_ng_master/grids/g025.txt",
    "sensitivity": ["ACCESS-CM2", "gn"],
}
file_esa_cci_lc = "/net/exo/landclim/data/dataset/ESA-CCI-LC_Land-Cover-Maps/v2.1.1/300m_plate-carree_1y/original/C3S-LC-L4-LCCS-Map-300m-P1Y-2016-v2.1.1.nc"

# FLAGS of ESA-CCI-LC that will be used for categories
list_infreq_burning = [
    "bare_areas",
    "bare_areas_consolidated",
    "bare_areas_unconsolidated",
    "water",
    "snow_and_ice",
    "sparse_vegetation",
    "sparse_tree",
    "sparse_shrub",
    "sparse_herbaceous",
]
list_water = ["water"]  #'snow_and_ice'

# OUTPUT
file_save = {
    "cmip6ng": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/spatial_info.nc",
    "sensitivity": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/spatial_info_sensitivity-ACCESS-CM2.nc",
}
# ============================================================================================
# ============================================================================================


# ============================================================================================
# 1. PREPARING INPUT GRIDs
# ============================================================================================
DATA_MASK = {}
for grid in grids.keys():
    # preparing
    DATA_MASK[grid] = xr.Dataset()
    DATA_MASK[grid].coords["bnds"] = [0, 1]
    tmp = {}

    if grid == "cmip6ng":
        # loading the grid of CMIP6-ng
        with open(grids[grid], "r") as fd:
            reader = csv.reader(fd)
            grids_cmip6ng = {}
            for row in reader:
                grids_cmip6ng[str.split(row[0], " ")[0]] = str.split(row[0], " ")[-1]

        # deduce latitude and longitude
        tmp["lon"] = float(grids_cmip6ng["xfirst"]) + float(
            grids_cmip6ng["xinc"]
        ) * np.arange(int(grids_cmip6ng["xsize"]))
        tmp["lat"] = float(grids_cmip6ng["yfirst"]) + float(
            grids_cmip6ng["yinc"]
        ) * np.arange(int(grids_cmip6ng["ysize"]))

        # bounds
        tmp["lon_bnds"] = np.array(
            [
                [
                    l - 0.5 * (tmp["lon"][1] - tmp["lon"][0]),
                    l + 0.5 * (tmp["lon"][1] - tmp["lon"][0]),
                ]
                for l in tmp["lon"]
            ]
        )
        tmp["lat_bnds"] = np.array(
            [
                [
                    l - 0.5 * (tmp["lat"][1] - tmp["lat"][0]),
                    l + 0.5 * (tmp["lat"][1] - tmp["lat"][0]),
                ]
                for l in tmp["lat"]
            ]
        )

    else:
        # loading
        spa_dat = xr.open_dataset(
            os.path.join(
                "/net/atmos/data/cmip6",
                "historical",
                "fx",
                "areacella",
                grids[grid][0],
                "r1i1p1f1",
                grids[grid][1],
                "areacella_fx_"
                + grids[grid][0]
                + "_historical_r1i1p1f1_"
                + grids[grid][1]
                + ".nc",
            )
        )
        for var in ["lat", "lon", "lat_bnds", "lon_bnds"]:
            tmp[var] = spa_dat[var].values

    # OUTPUT: coordinates
    for var in ["lat", "lon"]:
        DATA_MASK[grid].coords[var] = tmp[var]

    # OUTPUT: variables
    for var in ["lat_bnds", "lon_bnds"]:
        v = str.split(var, "_")[0]
        DATA_MASK[grid][var] = xr.DataArray(
            tmp[var], coords={v: tmp[v], "bnds": [0, 1]}, dims=(v, "bnds")
        )
# ============================================================================================
# ============================================================================================


# ============================================================================================
# 2. PREPARING ESA-CCI-LC_Land-Cover-Maps
# ============================================================================================
# loading ESA-CCI-LC_Land-Cover-Maps data
data_esa_cci = xr.open_dataset(file_esa_cci_lc)
data_esa_cci = data_esa_cci.isel(time=0).drop("time")

# checks
if (data_esa_cci["lat"].units not in ["degrees_north"]) or (
    data_esa_cci["lon"].units not in ["degrees_east"]
):
    raise Exception("Incorrect units for grid")

# surface
data_esa_cci["surf"] = np.cos(np.deg2rad(data_esa_cci.lat.values))[
    :, np.newaxis
] * xr.ones_like(data_esa_cci["lccs_class"])
data_esa_cci["surf"] *= 510072000 * 1.0e6 / data_esa_cci["surf"].sum()  # m2

# preparing flags
flag_names = str.split(data_esa_cci["lccs_class"].flag_meanings, " ")
dico_names_values = {
    flag_names[i]: val for i, val in enumerate(data_esa_cci["lccs_class"].flag_values)
}

# preparing variables, all grid cells initiated with zeros
data_esa_cci["surf_infreq_burning"] = xr.zeros_like(data_esa_cci["lccs_class"])
data_esa_cci["surf_sea"] = xr.zeros_like(data_esa_cci["lccs_class"])
data_esa_cci["surf_tot"] = xr.zeros_like(data_esa_cci["lccs_class"])

# masking the full dataset
for name in flag_names:
    print("masking using " + name, end="\n")
    mask_flag = xr.where(data_esa_cci["lccs_class"] == dico_names_values[name], 1, 0)
    if name in list_infreq_burning:
        data_esa_cci["surf_infreq_burning"] = (
            data_esa_cci["surf_infreq_burning"] + mask_flag * data_esa_cci["surf"]
        )  # only one value per grid-cell: ok!
    if name in list_water:
        data_esa_cci["surf_sea"] = (
            data_esa_cci["surf_sea"] + mask_flag * data_esa_cci["surf"]
        )  # only one value per grid-cell: ok!
# ============================================================================================
# ============================================================================================


# ============================================================================================
# 3. AGGREGATING
# ============================================================================================
for grid in grids.keys():
    print("Aggregating " + grid)
    new_data = xr.Dataset()

    # matching for aggregation
    new_data["match_lat"] = xr.DataArray(
        [
            np.where(
                (DATA_MASK[grid]["lat_bnds"][:, 0] <= lat)
                & (lat <= DATA_MASK[grid]["lat_bnds"][:, 1])
            )[0][0]
            for lat in data_esa_cci.lat.values
        ],
        coords={
            "lat": data_esa_cci.lat.values,
        },
        dims=("lat",),
    )
    new_data["match_lon"] = xr.DataArray(
        [
            np.where(
                (DATA_MASK[grid]["lon_bnds"][:, 0] <= lon)
                & (lon <= DATA_MASK[grid]["lon_bnds"][:, 1])
                | (DATA_MASK[grid]["lon_bnds"][:, 0] <= lon + 360)
                & (lon + 360 <= DATA_MASK[grid]["lon_bnds"][:, 1])
            )[0][0]
            for lon in data_esa_cci.lon.values
        ],
        coords={
            "lon": data_esa_cci.lon.values,
        },
        dims=("lon",),
    )

    # aggregating
    new_data["surf_infreq_burning"] = (
        data_esa_cci["surf_infreq_burning"]
        .groupby(new_data["match_lat"])
        .sum("lat")
        .groupby(new_data["match_lon"])
        .sum("lon")
        .rename({"match_lat": "lat_new", "match_lon": "lon_new"})
    )
    new_data["surf_sea"] = (
        data_esa_cci["surf_sea"]
        .groupby(new_data["match_lat"])
        .sum("lat")
        .groupby(new_data["match_lon"])
        .sum("lon")
        .rename({"match_lat": "lat_new", "match_lon": "lon_new"})
    )
    new_data["surf_tot"] = (
        data_esa_cci["surf"]
        .groupby(new_data["match_lat"])
        .sum("lat")
        .groupby(new_data["match_lon"])
        .sum("lon")
        .rename({"match_lat": "lat_new", "match_lon": "lon_new"})
    )

    # coordinates, giving corresponding values
    new_data = new_data.drop(("match_lat", "match_lon", "lat", "lon"))
    new_data = new_data.rename({"lat_new": "lat", "lon_new": "lon"})
    new_data.coords["lat"] = DATA_MASK[grid]["lat"]
    new_data.coords["lon"] = DATA_MASK[grid]["lon"]

    # archiving
    DATA_MASK[grid]["area_total"] = new_data["surf_tot"]
    DATA_MASK[grid]["area_land"] = new_data["surf_tot"] - new_data["surf_sea"]
    DATA_MASK[grid]["area_infreq_burning"] = new_data["surf_infreq_burning"]
    DATA_MASK[grid]["fraction_infreq_burning"] = (
        new_data["surf_infreq_burning"] / new_data["surf_tot"]
    )
    DATA_MASK[grid]["fraction_water"] = new_data["surf_sea"] / new_data["surf_tot"]
# ============================================================================================
# ============================================================================================


# ============================================================================================
# 4. SAVING
# ============================================================================================
for grid in grids.keys():
    # general attributes
    if grid == "cmip6ng":
        DATA_MASK[grid].attrs["info"] = "Spatial information relative to regridded FWI."
        DATA_MASK[grid].attrs["grid"] = "CMIP6-ng grid"

    else:
        DATA_MASK[grid].attrs[
            "info"
        ] = "Spatial information relative to NON-regridded FWI. Used for sensitivity analysis."
        DATA_MASK[grid].attrs["source_id"] = grid[0]
        DATA_MASK[grid].attrs["grid_label"] = grid[1]
    if (
        file_esa_cci_lc
        == "/net/exo/landclim/data/dataset/ESA-CCI-LC_Land-Cover-Maps/v2.1.1/300m_plate-carree_1y/original/C3S-LC-L4-LCCS-Map-300m-P1Y-2016-v2.1.1.nc"
    ):
        DATA_MASK[grid].attrs["source_dataset"] = "ESA-CCI-LC_Land-Cover-Maps"
        DATA_MASK[grid].attrs[
            "source_file"
        ] = "C3S-LC-L4-LCCS-Map-300m-P1Y-2016-v2.1.1.nc"
    else:
        raise Exception("adapt here...")
    DATA_MASK[grid].attrs["contact"] = "Yann Quilcaille <yann.quilcaille@env.ethz.ch>"

    # attributes on variables
    DATA_MASK[grid]["area_total"].attrs["unit"] = "m2"
    DATA_MASK[grid]["area_total"].attrs[
        "description"
    ] = "Total surface of the grid cell"

    DATA_MASK[grid]["area_land"].attrs["unit"] = "m2"
    DATA_MASK[grid]["area_land"].attrs[
        "description"
    ] = "Land surface of the grid cell. Only water is excluded here, not water and ice."

    DATA_MASK[grid]["area_infreq_burning"].attrs["unit"] = "m2"
    DATA_MASK[grid]["area_infreq_burning"].attrs[
        "description"
    ] = "Surface of the grid cell considered as infrequent burning."
    DATA_MASK[grid]["area_infreq_burning"].attrs[
        "list_flags"
    ] = "Flags considered for infrequent burning: " + ", ".join(list_infreq_burning)

    DATA_MASK[grid]["fraction_infreq_burning"].attrs["unit"] = "1"
    DATA_MASK[grid]["fraction_infreq_burning"].attrs[
        "description"
    ] = "Areal fraction of the grid cell considered as infrequent burning."
    DATA_MASK[grid]["fraction_infreq_burning"].attrs[
        "list_flags"
    ] = "Flags considered for infrequent burning: " + ", ".join(list_infreq_burning)

    DATA_MASK[grid]["fraction_water"].attrs["unit"] = "1"
    DATA_MASK[grid]["fraction_water"].attrs[
        "description"
    ] = "Areal fraction of the grid cell covered with water. Snow and ice are excluded."

    # saving
    DATA_MASK[grid].to_netcdf(
        file_save[grid],
        encoding={var: {"zlib": True} for var in DATA_MASK[grid].variables},
    )
# ============================================================================================
# ============================================================================================


# ============================================================================================
# 5. PLOT
# ============================================================================================
from functions_support_plots import func_map

for grid in grids.keys():
    tmp_vars = ["surf_tot", "surf_sea", "surf_infreq_burning"]
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle(grid)
    spec = gridspec.GridSpec(ncols=len(tmp_vars), nrows=2, figure=fig)
    for var in tmp_vars:
        ax = plt.subplot(spec[0, tmp_vars.index(var)], projection=ccrs.Robinson())
        plt.title(var, size=14)
        func_map(
            data_plot=new_data[var],
            ax=ax,
            fontsize_colorbar=12,
            spatial_info=new_data,
            type_plot="default",
        )  # , vmin=0, vmax=1 )
    ax = plt.subplot(spec[1, 1], projection=ccrs.Robinson())
    plt.title("fraction_water", size=14)
    func_map(
        data_plot=DATA_MASK[grid]["fraction_water"],
        ax=ax,
        fontsize_colorbar=12,
        spatial_info=DATA_MASK[grid],
        type_plot="default",
        vmin=0,
        vmax=1,
    )
    ax = plt.subplot(spec[1, 2], projection=ccrs.Robinson())
    plt.title("fraction_infreq_burning", size=14)
    func_map(
        data_plot=DATA_MASK[grid]["fraction_infreq_burning"],
        ax=ax,
        fontsize_colorbar=12,
        spatial_info=DATA_MASK[grid],
        type_plot="default",
        vmin=0,
        vmax=1,
    )
# ============================================================================================
# ============================================================================================
