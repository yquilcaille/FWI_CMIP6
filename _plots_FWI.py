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
import seaborn as sns  # # for colors
import xarray as xr
from matplotlib import cm
from mpl_toolkits.axes_grid1 import inset_locator, make_axes_locatable

# from statsmodels.nonparametric.smoothers_lowess import lowess

CB_color_cycle = sns.color_palette("colorblind", n_colors=10000)
import regionmask as regionmask

from functions_calc_FWI import *
from functions_load_CMIP6 import *
from functions_support import *
from functions_support_plots import *

# =====================================================================
# 0. GENERAL STUFF
# =====================================================================
# --------------------------------------------
# General parameters
# --------------------------------------------
# Absolute path where the figures for the daily FWI will be saved
path_cmip6 = "/net/atmos/data/cmip6"
# path_save = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/'
path_save_plotsFWI = "/home/yquilcaille/FWI/figures/"

esm_sensitivity = "ACCESS-CM2"
# --------------------------------------------
# --------------------------------------------


# --------------------------------------------
# Colors, labels, etc, common to plots
# --------------------------------------------
cols_scen = {"historical": "black"} | {
    xp: CB_color_cycle[i_xp]
    for i_xp, xp in enumerate(
        [
            "ssp245",
            "ssp370",
            "ssp126",
            "ssp585",
            "ssp119",
            "ssp460",
            "ssp434",
            "ssp534-over",
        ]
    )
}
dico_indics = {
    "fwixx": "Extreme value of the FWI",
    "fwils": "Length of the fire season",
    "fwixd": "Number of days with extreme fire weather",
    "fwisa": "Seasonal average of the FWI",
}
list_indics = ["fwixx", "fwixd", "fwils", "fwisa"]
# --------------------------------------------
# --------------------------------------------
# =====================================================================
# =====================================================================


# =====================================================================
# 1. PREPARATION
# =====================================================================
treshold_infreq_burning = 0.8
mask = {}

# loading mask for regridded data
data_mask_regrid = xr.open_dataset(
    "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/spatial_info.nc"
)
mask["regridded"] = (
    data_mask_regrid["fraction_infreq_burning"] <= treshold_infreq_burning
)

# loading mask for native data
data_mask_native = xr.open_dataset(
    "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/spatial_info_sensitivity-"
    + esm_sensitivity
    + ".nc"
)
mask[esm_sensitivity] = (
    data_mask_native["fraction_infreq_burning"] <= treshold_infreq_burning
)
# =====================================================================
# =====================================================================


# =====================================================================
# 2. FIGURE TREE FOR AVAILABLE DATA (figures paper: )
# =====================================================================
# WARNING: plotting this tree requires several dependencies in plotly and kaleido...
if False:
    # path_in = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwixx/ann/g025'
    # label_plot = 'tree_FWI-CMIP6_hurs-tasmax.pdf'
    path_in = "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hursmin_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwixx/ann/g025"
    label_plot = "tree_FWI-CMIP6_hursmin-tasmax.pdf"

    # preparing list of files:
    list_files = os.listdir(path_in)

    # prepare tree
    tree = tree_FWI_CMIP6(list_files)
    colors = {
        "lines": "rgb(200,200,200)",
        "nodes": "rgb(250,180,30)",
        "edges": "rgb(100,100,100)",
        "background": "rgb(248,248,248)",
        "text": "rgb(0,0,0)",
    }
    sizes = {"dots": 15, "CMIP6": 14, "scen": 12, "esm": 7, "member": 9}

    # prepare positions
    tree.calculate_positions_nodes(layout="rt_circular", fact_rad=1)
    # plot tree
    fig = tree.plot(figsize=(1000, 1000), colors=colors, sizes=sizes)
    fig.show()
    fig.write_image(os.path.join(path_save_plotsFWI, label_plot))
# =====================================================================
# =====================================================================


# =====================================================================
# 3. SENSITIVITY ANALYSIS
# =====================================================================
if False:
    # ---------------------------------------------------------
    # OPTIONS
    # ---------------------------------------------------------
    # Overall options
    esm = esm_sensitivity
    memb = "r1i1p1f1"
    axis = "DayLength"  # 'DayLength' | 'DryingFactor' | 'overwinterDC' | 'variables'

    # Options maps
    date_maps = "2014-01-01T12:00:00"  # '2014-01-01T12:00:00'  |  '2014-07-01T12:00:00'

    # Options timeseries
    lat_bands = {
        "Northern\nland": [20, 90],
        "Tropical\nland": [-20, 20],
        "Southern\nland": [-90, -20],
    }
    period_timeseries = {
        "historical": ("1995-01-01T12:00:00", "2014-12-31T12:00:00"),
        "ssp585": ("2081-01-01T12:00:00", "2100-12-31T12:00:00"),
    }
    resol_timeseries = "day"  # 'day' | 'month'
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # PREPARATION
    # ---------------------------------------------------------
    # preparing
    if axis == "DryingFactor":
        options = ["original", "NSH", "NSHeq"]
        dico_labels = {
            "original": "original",
            "NSH": "two hemi.",
            "NSHeq": "two hemi. & tropics",
        }
        VARS = ["dc", "fwi"]
        prop_calc_FWI = {
            "path_cmip6": "/net/atmos/data/cmip6",
            "path_save": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/",
            "type_variables": "hurs-tasmax",
            "adjust_DryingFactor": "original",
            "adjust_DayLength": "original",
            "adjust_overwinterDC": "original",
            "option_full_outputs": True,
        }
        name_axis = "adjust_DryingFactor"

    elif axis == "DayLength":
        options = ["original", "bins", "continuous"]
        VARS = ["dmc", "fwi"]
        dico_labels = {
            "original": "original",
            "bins": "bins of lat.",
            "continuous": "continuous lat.",
        }
        prop_calc_FWI = {
            "path_cmip6": "/net/atmos/data/cmip6",
            "path_save": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/",
            "type_variables": "hurs-tasmax",
            "adjust_DryingFactor": "original",
            "adjust_DayLength": "original",
            "adjust_overwinterDC": "original",
            "option_full_outputs": True,
        }
        name_axis = "adjust_DayLength"

    elif axis == "overwinterDC":
        options = ["original", "wDC"]
        VARS = ["dc", "fwi"]
        dico_labels = {"original": "original", "wDC": "overwintering"}
        prop_calc_FWI = {
            "path_cmip6": "/net/atmos/data/cmip6",
            "path_save": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/",
            "type_variables": "hurs-tasmax",
            "adjust_DryingFactor": "original",
            "adjust_DayLength": "original",
            "adjust_overwinterDC": "original",
            "option_full_outputs": True,
        }
        name_axis = "adjust_overwinterDC"

    elif axis == "variables":
        options = ["hurs-tasmax", "hursmin-tasmax"]
        dico_labels = {
            "hurs-tasmax": "RH",
            "hursmin-tasmax": "RH$_{min}$",
        }  # {'hurs-tasmax':'RH & T$_{max}$', 'hursmin-tasmax':'RH$_{min}$ & T$_{max}$'}
        VARS = ["dmc", "ffmc", "fwi"]  #'dc'
        prop_calc_FWI = {
            "path_cmip6": "/net/atmos/data/cmip6",
            "path_save": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/",
            "type_variables": "hurs-tasmax",
            "adjust_DryingFactor": "NSHeq",
            "adjust_DayLength": "continuous",
            "adjust_overwinterDC": "wDC",
            "option_full_outputs": True,
        }
        name_axis = "type_variables"

    # loading
    limits = {"esm": [esm], "scen": list(period_timeseries.keys()), "memb": [memb]}
    results = func_load_sensitivity(
        limits,
        prop_calc_FWI,
        name_axis,
        options,
        period_timeseries,
        date_maps,
        subset_vars=VARS,
        mask_FWI=mask[esm],
    )

    # Calculating monthly average // daily average
    for opt in options:
        for _var in VARS:
            print("Climatology (" + resol_timeseries + ") for " + _var + " in " + opt)
            for scen in period_timeseries.keys():
                # if method_time_average == 'resample':# faster
                # tt = {'month':'1MS', 'day':'1D'}[resol_timeseries]
                # tmp = results[opt]['time_map_'+_var+'_'+scen].resample(time=tt).mean()
                # results[opt]['time_map_'+_var+'_'+scen+'_mean'] = tmp.groupby('time.'+resol_timeseries).mean('time')
                tt = {"month": "time.month", "day": "time.dayofyear"}[resol_timeseries]
                results[opt]["time_map_" + _var + "_" + scen + "_mean"] = (
                    results[opt]["time_map_" + _var + "_" + scen]
                    .groupby(tt)
                    .mean("time")
                )
                results[opt]["time_map_" + _var + "_" + scen + "_stddev"] = (
                    results[opt]["time_map_" + _var + "_" + scen]
                    .groupby(tt)
                    .std("time")
                )

    # COMPUTING actual timeseries
    area_burning = (
        data_mask_native["area_total"] - data_mask_native["area_infreq_burning"]
    )
    for opt in options:
        for _var in VARS:
            print("Latidunal averages of climatology for " + _var + " in " + opt)
            for scen in period_timeseries.keys():
                for type_data in ["mean", "stddev"]:
                    name_var_in = "time_map_" + _var + "_" + scen + "_" + type_data
                    for i_band, band in enumerate(lat_bands):
                        name_var_out = (
                            "time_" + band + "_" + _var + "_" + scen + "_" + type_data
                        )
                        # selecting over band of latitudes
                        i_lat_min = np.argmin(
                            np.abs(
                                results[opt][name_var_in].lat.values
                                - lat_bands[band][0]
                            )
                        )
                        i_lat_max = np.argmin(
                            np.abs(
                                results[opt][name_var_in].lat.values
                                - lat_bands[band][1]
                            )
                        )
                        results[opt][name_var_out] = (
                            results[opt][name_var_in] * area_burning
                        ).isel(lat=slice(i_lat_min, i_lat_max)).sum(
                            ("lat", "lon")
                        ) / area_burning.isel(
                            lat=slice(i_lat_min, i_lat_max)
                        ).sum(
                            ("lat", "lon")
                        )
    # ---------------------------------------------------------
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # PLOT
    # ---------------------------------------------------------
    # Properties of plot
    width_figure = 20
    wspace, hspace = 0.18, 0.02
    ratio_height_time_map = 0.5
    height_figure = (
        (
            width_figure * 0.4 * len(options)
            + width_figure * len(lat_bands) * ratio_height_time_map
        )
        / len(VARS)
        * (1 + hspace * (len(options) + len(lat_bands)))
        / (1 + wspace * len(VARS))
        * 0.9
    )  # last factor to condense a bit
    size_text = 20 * np.sqrt(height_figure / 22)

    # preparing
    dico_vars = {
        "dc": "Drought Code",
        "fwi": "Fire Weather Index",
        "ffmc": "Fine Fuel Moisture Code",
        "dmc": "Duff Moisture Code",
    }
    label_plot = "_".join(
        [
            "sensitivity_FWI",
            esm,
            memb,
            axis,
            resol_timeseries,
            date_maps[: len("2014-08-01")],
        ]
    )
    fig = plt.figure(figsize=(width_figure, height_figure))  # , dpi=dpi
    spec = gridspec.GridSpec(
        ncols=len(VARS),
        nrows=len(options) + len(lat_bands),
        figure=fig,
        width_ratios=list(np.ones(len(VARS))),
        height_ratios=list(np.ones(len(options)))
        + list(ratio_height_time_map * np.ones(len(lat_bands))),
        left=0.08,
        right=0.95,
        bottom=0.10,
        top=0.97,
        wspace=wspace,
        hspace=hspace,
    )
    counter_letter = 0

    # PLOT maps:
    for i_opt, opt in enumerate(options):
        for VAR in VARS:
            ax = plt.subplot(spec[i_opt, VARS.index(VAR)], projection=ccrs.Robinson())
            if i_opt == 0:
                func_map(
                    results[opt]["map_" + VAR].values,
                    ax,
                    spatial_info=data_mask_native,
                    type_plot="default",
                    fontsize_colorbar=size_text * 0.9,
                    n_levels=8,
                )
                plt.title(dico_vars[VAR], size=size_text)
                if VAR == VARS[0]:
                    ax.text(
                        -0.05,
                        0.55,
                        s=dico_labels[options[0]]
                        + "\n in "
                        + date_maps[: len("2014-08-01")],
                        fontdict={"size": size_text},
                        color="k",
                        va="bottom",
                        ha="center",
                        rotation="vertical",
                        rotation_mode="anchor",
                        transform=ax.transAxes,
                    )

            else:
                # to_plot = 100 * ( (results[opt]['map_'+VAR] - results[options[0]]['map_'+VAR]) / results[options[0]]['map_'+VAR]).values
                # treshold_nan = {'dc':40, 'fwi':5, 'ffmc':5, 'dmc':10}[VAR]
                # to_plot[np.where( results[options[0]]['map_'+VAR].values < treshold_nan )] = np.nan
                to_plot = results[opt]["map_" + VAR] - results[options[0]]["map_" + VAR]
                func_map(
                    to_plot,
                    ax,
                    spatial_info=data_mask_native,
                    type_plot="symetric",
                    fontsize_colorbar=size_text * 0.9,
                    n_levels=9,
                )
                if VAR == VARS[0]:
                    ax.text(
                        -0.05,
                        0.55,
                        s=dico_labels[opt]
                        + " - "
                        + dico_labels[options[0]]
                        + "\n in "
                        + date_maps[: len("2014-08-01")],
                        fontdict={"size": size_text},
                        color="k",
                        va="bottom",
                        ha="center",
                        rotation="vertical",
                        rotation_mode="anchor",
                        transform=ax.transAxes,
                    )
            plt.text(
                x=ax.get_xlim()[0] + 0.05 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                y=ax.get_ylim()[0] + 0.90 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                s=list_letters[counter_letter],
                fontdict={"size": 0.8 * size_text},
            )
            counter_letter += 1

    # ploting latitudinal averages of climatologies:
    dico_col = {
        opt: CB_color_cycle[options.index(opt)] if opt not in ["original"] else "k"
        for opt in options
    }
    dico_ls = {"historical": "-", "ssp585": "--"}
    # different markers for options, different colors for scenarios
    for i_band, band in enumerate(lat_bands):
        for VAR in VARS:
            ax = plt.subplot(spec[len(options) + i_band, VARS.index(VAR)])
            for i_opt, opt in enumerate(options):
                for scen in period_timeseries.keys():
                    plot_mean = results[opt][
                        "time_" + band + "_" + VAR + "_" + scen + "_mean"
                    ]
                    plot_std = results[opt][
                        "time_" + band + "_" + VAR + "_" + scen + "_stddev"
                    ]
                    tt = np.arange(1, len(plot_mean) + 1)
                    plt.plot(
                        tt,
                        plot_mean.values,
                        color=dico_col[opt],
                        ls=dico_ls[scen],
                        lw=3,
                        label=dico_labels[opt] + " (" + scen + ")",
                    )
                    plt.fill_between(
                        tt,
                        (plot_mean - plot_std).values,
                        (plot_mean + plot_std).values,
                        facecolor=dico_col[opt],
                        edgecolor=dico_col[opt],
                        alpha=0.25,
                        ls=dico_ls[scen],
                        lw=1.5,
                    )

            # improving plot
            box = ax.get_position()
            ax.set_position([box.x0, box.y0, box.width, box.height * 0.8])
            plt.text(
                x=ax.get_xlim()[0] + 0.075 * (ax.get_xlim()[1] - ax.get_xlim()[0]),
                y=ax.get_ylim()[0]
                + 0.975 * 0.8 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                s=list_letters[counter_letter],
                fontdict={"size": 0.8 * size_text},
            )
            counter_letter += 1
            plt.grid()
            plt.yticks(size=size_text * 0.8)
            plt.xlim({"day": [1, 365], "month": [1, 12]}[resol_timeseries])
            # plt.ylim( np.min(aver_climato[VAR])-0.05*(np.max(aver_climato[VAR])-np.min(aver_climato[VAR])), np.max(aver_climato[VAR])+0.05*(np.max(aver_climato[VAR])-np.min(aver_climato[VAR])) )
            m = [
                "Jan",
                "Feb",
                "Mar",
                "Apr",
                "May",
                "Jun",
                "Jul",
                "Aug",
                "Sep",
                "Oct",
                "Nov",
                "Dec",
            ]
            d = np.array([31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31])
            plt.xticks(
                {
                    "day": [sum(d[:i]) + int(d[i] / 2) for i in range(len(d))],
                    "month": np.arange(1, 12 + 0.1, 1),
                }[resol_timeseries],
                m,
                size=size_text * 0.9,
                rotation=45,
            )
            if i_band != len(lat_bands) - 1:
                ax.tick_params(axis="x", label1On=False)
            if VAR == VARS[0]:
                plt.ylabel(band, size=size_text)
            if (i_band == len(lat_bands) - 1) and (VAR == VARS[len(VARS) // 2]):
                plt.legend(
                    bbox_to_anchor=[
                        0.5 + ((len(VARS) % 2 - 1)) * (0.5 + wspace / 2),
                        -0.7,
                    ],
                    loc="center",
                    prop={"size": size_text * 0.8},
                    ncol=len(options),
                )  # ncol=len(options)*len(period_timeseries)

    # fig.savefig(path_save_plotsFWI+'/'+label_plot, dpi=400)
    fig.savefig(path_save_plotsFWI + "/" + label_plot + ".pdf")
    # ---------------------------------------------------------
    # ---------------------------------------------------------
# =====================================================================
# =====================================================================


# =====================================================================
# 4. FIGURE ON RESULTS: ANNUAL INDICATORS
# =====================================================================
# ---------------------------------------------------------
# 4.1 ENSEMBLE MEMBERS
# ---------------------------------------------------------
if False:
    ##### METHOD #####
    # for 1 ESM, 1 historical + 1 ssp, represent maps of mean and standard deviation on members   AND   below timeseries on bands of latitude
    # n_members = 50
    # list_runs = [ [scen,esm] for scen in xp_avail_esms.keys() for esm in xp_avail_esms[scen].keys() if len(xp_avail_esms[scen][esm]) >= n_members ]
    # len(set( [run[1] for run in list_runs] ))

    xps = ["historical", "ssp585"]  # MORE THAN 2 IS USELESS HERE!
    list_years = [1850, 2100]
    lat_bands = {
        "Northern land": [20, 90],
        "Tropical land": [-20, 20],
        "Southern land": [-90, -20],
    }

    # running these ones:
    for indic in ["fwixx", "fwixd", "fwils", "fwisa"]:
        for esm in ["MIROC6", "CanESM5"]:
            print("Ploting " + indic + " for " + esm)

            # preparing runs
            path_in = (
                "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/"
                + indic
                + "/ann/g025"
            )
            # preparing list of files:
            list_files = os.listdir(path_in)
            xp_avail_esms = matching_scenarios(list_files)
            tmp_files = [
                os.path.join(path_in, file_W)
                for file_W in list_files
                if (str.split(file_W, "_")[3] in xps)
                and (str.split(file_W, "_")[2] == esm)
            ]
            DATA = xr.open_mfdataset(
                [os.path.join(path_in, file_W) for file_W in tmp_files],
                preprocess=func_preprocess_annual,
            )
            DATA = DATA.compute()

            # ploting
            fig = plot_maps_timeseries(
                DATA,
                indic,
                xps,
                list_years,
                lat_bands,
                axis_comparison="member",
                value_ref=esm,
                name_figure=indic + "_members-" + esm + ".pdf",
            )
            plt.close(fig)
# ---------------------------------------------------------
# ---------------------------------------------------------


# ---------------------------------------------------------
# 4.2 ESMs
# ---------------------------------------------------------
if False:
    ##### METHOD #####
    # for 1 member, 1 historical + 1 ssp, represent maps of mean and standard deviation on members   AND   below timeseries on bands of latitude

    member = "r1i1p1f1"
    xps = ["historical", "ssp585"]  # MORE THAN 2 IS USELESS HERE!
    list_years = [1850, 2100]
    lat_bands = {
        "Northern land": [20, 90],
        "Tropical land": [-20, 20],
        "Southern land": [-90, -20],
    }

    # running these ones:
    for indic in ["fwixx", "fwixd", "fwils", "fwisa"]:
        print("Ploting " + indic + " for " + member)

        # preparing runs
        path_in = (
            "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/"
            + indic
            + "/ann/g025"
        )
        # preparing list of files:
        list_files = os.listdir(path_in)
        xp_avail_esms = matching_scenarios(list_files)

        # selecting ESMs on this member with both experiments
        total_esms = set([esm for scen in xps for esm in xp_avail_esms[scen].keys()])
        list_esms = []
        for esm in total_esms:
            tmp = []
            for scen in xps:
                if esm in xp_avail_esms[scen]:
                    tmp.append(member in xp_avail_esms[scen][esm])
                else:
                    tmp.append(False)
            if np.all(tmp):
                list_esms.append(esm)
        del esm
        list_esms.sort()

        # loading
        tmp_files = [
            os.path.join(path_in, file_W)
            for file_W in list_files
            if (str.split(file_W, "_")[3] in xps)
            and (str.split(file_W, "_")[2] in list_esms)
            and (str.split(file_W, "_")[4] == member)
        ]
        DATA = xr.open_mfdataset(
            [os.path.join(path_in, file_W) for file_W in tmp_files],
            preprocess=func_preprocess_annual,
        )
        DATA = DATA.compute()

        # ploting
        fig = plot_maps_timeseries(
            DATA,
            indic,
            xps,
            list_years,
            lat_bands,
            axis_comparison="esm",
            value_ref=member,
            name_figure=indic + "_esms-" + member + ".pdf",
        )
        plt.close(fig)
# ---------------------------------------------------------
# ---------------------------------------------------------


# ---------------------------------------------------------
# 4.3 GLOBAL WARMING LEVELS
# ---------------------------------------------------------
if False:
    ##### METHOD #####
    # Identification of GWL:
    # - load tas from CMIP6-ng, excludes ssp534-over and scenarios without ssp245
    # - rolling mean over n_years = 20, historical extended forward with ssp245, ssp scenarios extended backward with historical
    # - remove reference period ref_period = (1850,1900)
    # - for each run, first year that a Global Warming Level is crossed, select this year. archive positions of crossing.
    # GWLs on FWI
    # - load the maps of annual indicators of FWI for the same runs
    # - select the maps at same positions -10 years and +9 years. if needed, historical extended forward with ssp245 and ssp scenarios extended backward with historical.
    # - average the maps of each run
    # - average over ensemble members
    # Representation of uncertainties with method of IPCC
    # evaluate internal variability:
    # - from preindustrial control
    #    - detrend the pre-industrial control using quadratic fit
    #    - calculating its local standard deviation of 20-year mean over non-overlapping periods in preindustrial control: sigma_20yr of the ESM x member
    # - if preindustrial not available:
    #    - interannual standard deviation over linearly detrend modern periods: sigma_1yr of the ESM x member
    #    - sigma_20yr = sigma_1yr / sqrt(20)

    # check if 50 years?

    # deduce internal variability as: gamma = sqrt(2) * 1.645 * sigma_20yr
    # if more than 66% of models have a change greater than gamma:
    # -yes- if more than 80% of models agree on the signe of the change:
    #   -yes-> Robust signal: colour only, nothing else
    #   -no -> Conflicting signal: colour and crossed lines
    # -no:
    #   --> No change or no robust signal: colour and Reverse diagonal
    # (dont say hatching, but diagonal lines for non-expert audiences)
    # (include these patterns in the legend)

    # ----------------------------------
    # CALCULATION
    # ----------------------------------
    ref_period = (1851, 1900)
    list_GWLs = [1.0, 1.5, 2, 3]  # , 4]
    approach_ipcc = "B"
    gamma_period = (1851, 1870)  # used only if approach_ipcc='C'
    option_common_set = True
    type_vars = "hurs-tasmax"
    option_calc_on_GWL = "mean"  # 'mean', '90percentile'
    option_plot_diff_GWLs = True
    load_GWLs = True

    # Preparing list of global temperatures of runs necessary:
    path_data_plot = {
        "hursmin-tasmax": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hursmin_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/",
        "hurs-tasmax": "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/",
    }[type_vars]
    # WARMING!!  REMOVING ssp534-over  AND  KEEPING ONLY THOSE WITH A ssp245 FOR EXTENSION OF HISTORICAL FOR ROLLING MEAN FOR GLOBAL WARMING LEVEL!!
    list_files = os.listdir(os.path.join(path_data_plot, list_indics[0], "ann/g025"))
    xp_avail_esms = matching_scenarios(list_files)
    files_cmip6ng, to_remove = [], []
    for scen in xp_avail_esms:
        for esm in xp_avail_esms[scen]:
            for member in xp_avail_esms[scen][esm]:
                file_W = os.sep.join(
                    [
                        "/net/atmos/data/cmip6-ng/",
                        "tas",
                        "ann",
                        "g025",
                        "tas_ann_" + esm + "_" + scen + "_" + member + "_g025.nc",
                    ]
                )
                file_W_245 = os.sep.join(
                    [
                        "/net/atmos/data/cmip6-ng/",
                        "tas",
                        "ann",
                        "g025",
                        "tas_ann_" + esm + "_" + "ssp245" + "_" + member + "_g025.nc",
                    ]
                )  # for rolling mean over historical
                if (
                    os.path.isfile(file_W)
                    and os.path.isfile(file_W_245)
                    and scen not in ["ssp534-over"]
                ):
                    files_cmip6ng.append(file_W)
                else:
                    to_remove.append([scen, esm, member])

    # load formerly calculated GWLs?
    if load_GWLs:
        maps_GWLs = {}
        for indic in list_indics:
            maps_GWLs[indic] = xr.open_dataset(
                path_save_plotsFWI
                + "/data-plots_"
                + type_vars
                + "_"
                + indic
                + "-GWLs-"
                + option_calc_on_GWL
                + option_common_set * "-common"
                + ".nc"
            )

    else:
        # Parameters for calculation of GWL
        n_years = 20  # for rolling mean

        # loading data from CMIP6-ng
        GWL_FWI = GWL()
        GWL_FWI.prep_cmip6ng(
            files_cmip6ng=files_cmip6ng, n_years=n_years, ref_period=ref_period
        )

        # calculating positions of GWLs
        GWL_FWI.find_position_GWLs(
            list_GWLs=list_GWLs, option_common_set=option_common_set
        )

        # Calculating GWL-maps
        maps_GWLs = {}
        for indic in list_indics:
            print("Calculating GWL maps of " + indic)

            # preparing runs
            path_in = os.path.join(path_data_plot, indic, "ann/g025")
            # preparing list of files:
            list_files = os.listdir(path_in)

            # loading
            tmp_files = [
                os.path.join(path_in, file_W)
                for file_W in list_files
                if [
                    str.split(file_W, "_")[3],
                    str.split(file_W, "_")[2],
                    str.split(file_W, "_")[4],
                ]
                not in to_remove
            ]
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                DATA = xr.open_mfdataset(
                    [os.path.join(path_in, file_W) for file_W in tmp_files],
                    preprocess=func_preprocess_annual,
                )

            # get maps to use
            maps_GWLs[indic] = GWL_FWI.apply_GWL(
                data_in=DATA[indic], option_calc_on_GWL=option_calc_on_GWL
            )

            # saving for future figures
            maps_GWLs[indic].attrs["n_years"] = str(n_years)
            maps_GWLs[indic].attrs["ref_period"] = (
                str(ref_period[0]) + "-" + str(ref_period[0])
            )
            maps_GWLs[indic].attrs["option_calc_on_GWL"] = option_calc_on_GWL
            maps_GWLs[indic].to_netcdf(
                path_save_plotsFWI
                + "/data-plots_"
                + type_vars
                + "_"
                + indic
                + "-GWLs-"
                + option_calc_on_GWL
                + option_common_set * "-common"
                + ".nc",
                encoding={var: {"zlib": True} for var in maps_GWLs[indic].variables},
            )
    # ----------------------------------
    # ----------------------------------

    # ----------------------------------
    # PLOT
    # ----------------------------------
    dico_indics_short = {
        "fwixx": "Extreme value\nof the FWI",
        "fwils": "Length of the\nfire season",
        "fwixd": "Number of days with\nextreme fire weather",
        "fwisa": "Seasonal average\nof the FWI",
    }
    width_figure = 20
    wspace, hspace = 0.075, 0.10

    # preparing figure
    height_figure = (
        (width_figure * 0.5)
        * len(list_indics)
        / (1 + len(list_GWLs))
        * (1 + hspace)
        / (1 + wspace)
    )
    fig = plt.figure(figsize=(width_figure, height_figure))
    spec = gridspec.GridSpec(
        ncols=1 + len(list_GWLs),
        nrows=len(list_indics),
        figure=fig,
        left=0.03,
        right=0.98,
        bottom=0.05,
        top=0.94,
        wspace=wspace,
        hspace=hspace,
    )

    # looping on indicators
    plot_maps_gwl = {}
    for i_indic, indic in enumerate(list_indics):
        # preparing runs
        path_in = os.path.join(path_data_plot, indic, "ann/g025")
        # preparing list of files:
        list_files = os.listdir(path_in)

        # loading
        tmp_files = [
            os.path.join(path_in, file_W)
            for file_W in list_files
            if [
                str.split(file_W, "_")[3],
                str.split(file_W, "_")[2],
                str.split(file_W, "_")[4],
            ]
            not in to_remove
        ]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            DATA = xr.open_mfdataset(
                [os.path.join(path_in, file_W) for file_W in tmp_files],
                preprocess=func_preprocess_annual,
            )

        # preparation
        plot_maps_gwl[indic] = maps_mean_uncertainties()
        # calculating how robust or certain is everything
        plot_maps_gwl[indic].eval_robust_certain(
            map_ref=DATA[indic].sel(
                scen="historical", time=slice(ref_period[0], ref_period[1])
            ),
            maps_to_plot=maps_GWLs[indic]["maps_GWL"],
            mask_map=mask["regridded"],
            dim_plot="GWL",
            approach_ipcc=approach_ipcc,
            data_gamma=DATA[indic].sel(
                scen="historical", time=slice(gamma_period[0], gamma_period[1])
            ),
            limit_certainty_members=0.66,
            limit_certainty_ESMs=0.66,
            limit_robustness_members=0.80,
            limit_robustness_ESMs=0.80,
        )

        # ploting map for each indic
        legend_val_dim = (
            list_GWLs[int(len(list_GWLs) / 2) - 1] if indic == list_indics[-1] else None
        )
        plot_maps_gwl[indic].plot(
            fig,
            spec,
            ind_row=i_indic,
            label_row=dico_indics_short[indic],
            do_title=(i_indic == 0),
            unit_dim="K",
            plot_diff_GWLs=option_plot_diff_GWLs,
            vmin=None,
            vmax=None,
            levels_cmap=7,
            fontsize={"colorbar": 12, "title": 12, "legend": 12, "label_row": 12},
            density_visual_code={"x": 5, "\\": 3, "/": 3},
            margin_colorbar_pct=2.5,
            legend_val_dim=legend_val_dim,
        )

    # saving figure
    name_plot = (
        option_plot_diff_GWLs * "diff-"
        + "plot_FWI_"
        + type_vars
        + "-GWL-"
        + option_calc_on_GWL
        + option_common_set * "-common"
        + "_approach-"
        + approach_ipcc
        + "_gamma-"
        + str(gamma_period[0])
        + "-"
        + str(gamma_period[1])
    )
    fig.savefig(path_save_plotsFWI + "/" + name_plot + ".pdf")
    # ----------------------------------
    # ----------------------------------
# ---------------------------------------------------------
# ---------------------------------------------------------
# =====================================================================
# =====================================================================


# =====================================================================
# 5. TEST for PLOTTING ANNUAL INDICATORS (not part of paper)
# =====================================================================
if False:
    # looping on indicators
    fig = plt.figure(figsize=(20, 10))
    point_plot = -16, 360 - 56

    for indic in list_indics:
        path_in = (
            "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/"
            + indic
            + "/ann/g025"
        )
        list_files = os.listdir(path_in)

        ax = plt.subplot(2, 2, list_indics.index(indic) + 1)

        # preparing mapping
        xp_avail_esms = matching_scenarios(list_files)

        # loading for each scenario
        for scen in xp_avail_esms.keys():
            print("plotting " + indic + " on " + scen)
            tmp_files = [
                os.path.join(path_in, file_W)
                for file_W in list_files
                if str.split(file_W, "_")[3] == scen
            ]
            DATA = xr.open_mfdataset(
                [os.path.join(path_in, file_W) for file_W in tmp_files],
                preprocess=func_preprocess_annual,
            )  # , concat_dim=('scen','member','esm') )
            i_lat, i_lon = np.argmin(
                np.abs(DATA.lat.values - point_plot[0])
            ), np.argmin(np.abs(DATA.lon.values - point_plot[1]))

            # looping on each esm
            memb_mean = (
                DATA[indic]
                .sel(scen=scen)
                .isel(lat=i_lat, lon=i_lon)
                .mean(("member"))
                .compute()
            )
            plot_mean = memb_mean.mean("esm")
            plot_stdd = memb_mean.std("esm")

            # plotting
            plt.plot(
                DATA.time, plot_mean.values, lw=2, color=cols_scen[scen], label=scen
            )
            plt.fill_between(
                DATA.time,
                (plot_mean - plot_stdd).values,
                (plot_mean + plot_stdd).values,
                lw=2,
                facecolor=cols_scen[scen],
                edgecolor=None,
                alpha=0.3,
            )

        # polishing
        plt.grid()
        plt.legend(loc=0, prop={"size": 10})
        plt.title(dico_indics[indic], size=14)

    # save
    plt.suptitle(
        "Point represented at:"
        + str(DATA.lat.values[i_lat])
        + "N, "
        + str(DATA.lon.values[i_lon])
        + "E"
    )
    fig.savefig(path_save_plotsFWI + "/test_plot_annual_indicators.pdf")

# =====================================================================
# =====================================================================


# =====================================================================
# X. STASH
# =====================================================================
# --------------------------------------------
# plotting annual indicators
# --------------------------------------------
if False:
    # PREPARING DATA
    list_files = os.listdir(path_FWI_annual)

    def add_xpmemb_dim(data):
        data = data.expand_dims(xp=[data.attrs["experiment"]])
        data = data.expand_dims(ensemble_member=[data.attrs["ensemble_member"]])
        return data

    print("temporary plot for fire workshop")
    list_files_CanESM5 = [
        file_W for file_W in list_files if str.split(file_W, "_")[1] == "CanESM5"
    ]
    files_to_get = [
        os.path.join(path_FWI_annual, file_W) for file_W in list_files_CanESM5
    ]
    DATA = xr.open_mfdataset(
        files_to_get, preprocess=add_xpmemb_dim, concat_dim=("xp", "memb")
    )  # , chunks={'time':1}

    lat_fwi, lon_fwi = data.lat.values, data.lon.values
    lon_fwi_mesh, lat_fwi_mesh = np.meshgrid(lon_fwi, lat_fwi)
    cmap = cm.get_cmap("Reds", 100)


# --------------------
# MAPS
# --------------------
if False:

    def plot_maps(data, var, xp, range_mean, years=[1850, 2000, 2100]):
        fig = plt.figure(figsize=(20, 10))  # , dpi=dpi
        spec = gridspec.GridSpec(
            ncols=len(years),
            nrows=2,
            figure=fig,
            left=0.05,
            right=0.95,
            bottom=0.08,
            top=0.975,
            wspace=0.15,
            hspace=0.07,
        )

        for i_yr, year in enumerate(years):
            # doing first row: mean on ensemble members
            ax = plt.subplot(spec[0, i_yr], projection=ccrs.Robinson())
            ax.coastlines()
            if year > 2014:
                to_plot = data[var].sel(xp=xp, time=year).mean("ensemble_member")
            else:
                to_plot = (
                    data[var].sel(xp="historical", time=year).mean("ensemble_member")
                )
            pmesh = ax.pcolormesh(
                lon_fwi_mesh,
                lat_fwi_mesh,
                to_plot,
                transform=ccrs.PlateCarree(),
                rasterized=True,
                cmap=cmap,
                vmin=range_mean[0],
                vmax=range_mean[1],
            )
            cbar = plt.colorbar(
                pmesh, fraction=0.0235, pad=0.04, orientation="horizontal"
            )
            plt.title(year, size=15)

            # doing second row: mean on ensemble members
            ax = plt.subplot(spec[1, i_yr], projection=ccrs.Robinson())
            ax.coastlines()
            if year > 2014:
                to_plot = data[var].sel(xp=xp, time=year).std("ensemble_member")
            else:
                to_plot = (
                    data[var].sel(xp="historical", time=year).std("ensemble_member")
                )
            pmesh = ax.pcolormesh(
                lon_fwi_mesh,
                lat_fwi_mesh,
                to_plot,
                transform=ccrs.PlateCarree(),
                rasterized=True,
                cmap=cmap,
            )
            # vmin=0, vmax=1
            cbar = plt.colorbar(
                pmesh, fraction=0.0235, pad=0.04, orientation="horizontal"
            )
        plt.grid()
        fig.savefig(path_save_plotsFWI + "/" + "tmp_maps_" + var + "_" + xp)

    dico_range = {
        "FWI_Xx": [0, 150],
        "FWI_LS": [0, 365],
        "FWI_XD": [0, 365],
        "FWI_SA": [0, 150],
    }
    for var in ["FWI_Xx", "FWI_LS", "FWI_XD", "FWI_SA"]:
        plot_maps(data=DATA, var=var, xp="ssp585", range_mean=dico_range[var])
# --------------------
# --------------------


# --------------------
# RELATIONSHIPs to tas
# --------------------
if False:
    # need to run MESMER-X quickly to have temperatures.
    pred = "tas"
    gen = 6
    dir_cmipng = "/net/atmos/data/cmip" + str(gen) + "-ng/"
    dir_cmip_X = (
        "/net/cfc/landclim1/mathause/projects/IPCC_AR6_CH11/IPCC_AR6_CH11/data/"
    )
    dir_obs = (
        "/net/exo/landclim/yquilcaille/mesmer-x/data/observations/"  # observations
    )
    dir_aux = "/net/exo/landclim/yquilcaille/mesmer-x/data/auxiliary/"  # auxiliary data
    dir_mesmer_params = "/net/exo/landclim/yquilcaille/mesmer-x/calibrated_parameters/"
    dir_mesmer_emus = "/net/exo/landclim/yquilcaille/mesmer-x/emulations/"
    dir_stats = (
        "/net/exo/landclim/yquilcaille/mesmer-x/statistics/"  # emulation statistics
    )
    dir_plots = "/net/exo/landclim/yquilcaille/mesmer-x/plots/"  # plots
    cfg = config_mesmer(
        gen=gen,
        paths={
            "dir_cmipng": dir_cmipng,
            "dir_cmip_X": dir_cmip_X,
            "dir_obs": dir_obs,
            "dir_aux": dir_aux,
            "dir_mesmer_params": dir_mesmer_params,
            "dir_mesmer_emus": dir_mesmer_emus,
            "dir_stats": dir_stats,
            "dir_plots": dir_plots,
        },
        esms="all",
    )

    ## load in the ESM runs
    esms = cfg.esms
    print("just taking CanESM5")
    esms = ["CanESM5"]

    ## initiate TEMPORARY dictionaries
    pred_g_dict = {
        esm: {} for esm in esms
    }  ## predictor with global coverage (dict[esm][scen][run]: array Time x Lat x Lon)
    PRED_dict = {
        esm: {} for esm in esms
    }  ## global mean predictor (dict[esm][scen][run]: array Time)

    ## initiate dictionnaries
    time = {esm: {} for esm in esms}  ## time axis (dict[esm][scen]: array Time)
    pred_g = (
        {}
    )  ## predictor with global coverage (dict[esm][scen]: array Run x Time x Lat x Lon)
    PRED = {}  ## global mean tas (dict[esm][scen]: array Run x Time x Lat x Lon)

    dico_runs = {}
    for esm in esms:
        for scen in cfg.scenarios:
            ## checking if this (esm,scen) combination has compatible runs.
            dico_runs[scen], _ = test_combination_vars([pred], esm, scen, cfg)
            if len(dico_runs[scen]) > 0:
                (
                    pred_g_dict[esm][scen],
                    PRED_dict[esm][scen],
                    lon,
                    lat,
                    time[esm][scen],
                ) = load_cmip(pred, esm, scen, cfg)
        ## grouping the level [run] of dict[esm][scen][run] into a single array
        pred_g[esm] = convert_dict_to_arr(pred_g_dict[esm])
        PRED[esm] = convert_dict_to_arr(PRED_dict[esm])
    ## clean temporary files
    del pred_g_dict, PRED_dict

    ## load in the constant files
    reg_dict, ls, wgt_g, lon, lat = load_regs_ls_wgt_lon_lat(
        cfg.reg_type, lon_fwi, lat_fwi
    )

    ## extract land
    land_pred, reg_dict, ls = extract_land(
        pred_g, reg_dict, wgt_g, ls, threshold_land=cfg.threshold_land
    )

    ## adding few lines for future regional calculations (used for tests)
    ind = np.where(ls["idx_grid_l"])
    gp2reg = reg_dict["grids"][:, ind[0], ind[1]]  ## grid points to regions
    ww_reg = np.nansum((ls["wgt_gp_l"] * gp2reg).T, axis=0)

    def add_subplot_axes(ax, rect, facecolor="w", proj=ccrs.Robinson()):
        ## cannot use inset_locator.inset_axes with projection Robinson, creating this function to account for it.
        fig = plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]
        subax = fig.add_axes(
            [x, y, width, height], facecolor=facecolor, projection=proj
        )
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2] ** 0.5
        y_labelsize *= rect[3] ** 0.5
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
        return subax

    # PLOTTING RELATIONSHIP
    dico_names_vars = {
        "FWI_Xx": "Annual maximum of FWI",
        "FWI_LS": "Length of the fire season (days)",
        "FWI_XD": "Number of extreme fire weather days (days)",
        "FWI_SA": "Seasonal average of FWI",
    }
    colors_scen = {
        "historical": "k",
        "ssp119": (0 / 255.0, 170 / 255.0, 208 / 255.0),
        "ssp126": (0 / 255.0, 52 / 255.0, 102 / 255.0),
        "ssp245": (239 / 255.0, 85 / 255.0, 15 / 255.0),
        "ssp370": (224 / 255.0, 0 / 255.0, 0 / 255.0),
        "ssp434": (255 / 255.0, 169 / 255.0, 0 / 255.0),
        "ssp460": (196 / 255.0, 121 / 255.0, 0 / 255.0),
        "ssp534-over": (127 / 255.0, 0 / 255.0, 110 / 255.0),
        "ssp585": (153 / 255.0, 0 / 255.0, 2 / 255.0),
    }
    # NOT aggregating regions, different maps, hard to get proper one cf code of Lea
    ar6 = regionmask.defined_regions.ar6.land

    def plot_relationship(var, list_lat_lon):
        fig = plt.figure(figsize=(20, 10))
        for xp in COMMON["xps"][::-1]:
            if xp in DATA.xp.values:
                if xp == "historical":
                    common_members = set(DATA["ensemble_member"].values).intersection(
                        dico_runs["h-ssp585"]
                    )
                    ind = [dico_runs["h-ssp585"].index(memb) for memb in common_members]
                    xx = PRED[esm]["h-ssp585"][ind][:, : 2014 + 1 - 1850]
                    yy = (
                        DATA[var]
                        .sel(xp=xp, ensemble_member=list(common_members))
                        .values
                    )  # Memb x Time x Lat x Lon
                    yy = yy[:, : 2014 + 1 - 1850, ...]

                else:
                    common_members = set(DATA["ensemble_member"].values).intersection(
                        dico_runs["h-" + xp]
                    )
                    ind = [dico_runs["h-" + xp].index(memb) for memb in common_members]
                    xx = PRED[esm]["h-" + xp][ind][:, 2014 + 1 - 1850 :]  # Memb x Time
                    yy = (
                        DATA[var]
                        .sel(xp=xp, ensemble_member=list(common_members))
                        .values
                    )  # Memb x Time x Lat x Lon
                    yy = yy[:, 2014 + 1 - 1850 :, ...]

                for i, coo in enumerate(list_lat_lon):
                    ax = plt.subplot(2, 2, i + 1)
                    i_lat, i_lon = np.argmin(np.abs(lat_fwi - coo[0])), np.argmin(
                        np.abs(lon_fwi - coo[1])
                    )
                    plt.scatter(
                        xx.flatten(),
                        yy[:, :, i_lat, i_lon].flatten(),
                        color=colors_scen[xp],
                        marker="x",
                        s=8,
                        label=xp,
                    )
        for i, coo in enumerate(list_lat_lon):
            i_lat, i_lon = np.argmin(np.abs(lat_fwi - coo[0])), np.argmin(
                np.abs(lon_fwi - coo[1])
            )
            ax = plt.subplot(2, 2, i + 1)
            plt.grid()
            plt.legend(loc="bottom right", prop={"size": 10})
            # position
            subax = add_subplot_axes(
                ax=ax, rect=[0.02, 0.37, 0.3, 0.9], facecolor="w", proj=ccrs.Robinson()
            )  # x0,y0,D_x,(D_y)
            ar6.plot(
                ax=subax,
                add_label=False,
                label="abbrev",
                proj=ccrs.Robinson(),
                line_kws={"lw": 0.5, "color": "0.35"},
            )
            pos_geodetic = lat_fwi[i_lat], lon_fwi[i_lon]
            pos_robinson = ccrs.Robinson().transform_point(
                pos_geodetic[1], pos_geodetic[0], ccrs.Geodetic()
            )
            subax.scatter(
                pos_robinson[0],
                pos_robinson[1],
                marker="x",
                color="red",
                edgecolor=None,
                s=50,
            )
            if i in [2, 3]:
                ax.set_xlabel("$\Delta$T (K)", size=15)
            if i in [0, 2]:
                ax.set_ylabel(dico_names_vars[var], size=15)
        fig.savefig(path_save_plotsFWI + "/" + "tmp_relationship_" + var)

    list_lat_lon = [[47.5, 8.5], [-22.5, 125], [-19.5, 47], [36, -118.5]]
    for var in dico_names_vars.keys():
        plot_relationship(var, list_lat_lon)
# --------------------
# --------------------


# --------------------------------------------
# Former test on trees
# --------------------------------------------
if False:
    # handling level 0
    root = Node(name="CMIP6")

    # handling xps
    first_level = [Node(name=xp) for xp in COMMON["xps"]]
    root.children = {item for item in first_level}

    # handling ESMs
    second_level = []
    for i_xp, xp in enumerate(COMMON["xps"]):
        second_level.append([Node(name=esm) for esm in COMMON["esms"]])
        first_level[i_xp].add_children(second_level[i_xp])

    # add_path_of_children?

    # creating the tree object
    tree = Tree(root=root)

    # ploting
    create_tree_diagram(
        tree=tree, webshot=True, verbose=True, save_path=path_save_treeFWI
    )
    # tree.show(verbose=True, save_path=path_save_treeFWI)


if False:
    tree = Tree()
    tree.create_node("CMIP6", "CMIP6")  # No parent means its the root node
    for i_xp, xp in enumerate(COMMON["xps"]):
        tree.create_node(xp, xp, parent="CMIP6")
        for i_esm, esm in enumerate(COMMON["esms"]):
            tree.create_node(esm, xp + "_" + esm, parent=xp)
    tree.show()
# --------------------------------------------
# --------------------------------------------
# =====================================================================
# =====================================================================
