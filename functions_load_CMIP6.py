import os

import cftime as cft
import numpy as np
import xarray as xr
from pandas import DatetimeIndex

dico_descrip = {
    "fwi": "Canadian Fire Weather Index.",
    "ffmc": "The FFMC is a numeric rating of the moisture content of litter and other cured fine fuels. This code is an indicator of the relative ease of ignition and the flammability of fine fuel.",
    "dmc": "The DMC is a numeric rating of the average moisture content of loosely compacted organic layers of moderate depth. This code gives an indication of fuel consumption in moderate duff layers and medium-size woody material.",
    "dc": "The DC is a numeric rating of the average moisture content of deep, compact organic layers. This code is a useful indicator of seasonal drought effects on forest fuels and the amount of smoldering in deep duff layers and large logs.",
    "isi": "The ISI is a numeric rating of the expected rate of fire spread. It is based on wind speed and FFMC. Like the rest of the FWI system components, ISI does not take fuel type into account. Actual spread rates vary between fuel types at the same ISI.",
    "bui": "The BUI is a numeric rating of the total amount of fuel available for combustion. It is based on the DMC and the DC. The BUI is generally less than twice the DMC value, and moisture in the DMC layer is expected to help prevent burning in material deeper down in the available fuel.",
    "TEMP": "Temperature fed as input",
    "RH": "Relative humidity as input",
    "RAIN": "Precipitations as input",
    "WIND": "Wind as input",
}
dico_names = {
    "fwi": "Canadian Fire Weather Index",
    "ffmc": "Fine Fuel Moisture Code",
    "dmc": "Duff Moisture Code",
    "dc": "Drought Code",
    "isi": "Initial Spread Index",
    "bui": "Built Up Index",
    "TEMP": "Temperature fed as input",
    "RH": "Relative humidity as input",
    "RAIN": "Precipitations as input",
    "WIND": "Wind as input",
}

dico_warnings_corrections = {
    "correction_ssp534-over_ssp585-2039": [
        "warning_time_init",
        "This run starts in CMIP6 files in 2040. Using its corresponding ssp585 to branch from 2039 to 2040.",
    ],
    "correction_ssp534-over_ssp585-2040": [
        "warning_time_init",
        "This run starts in CMIP6 files in 2041. Using its corresponding ssp585 to branch from 2040 to 2041.",
    ],
    "correction_cut-common-time": [
        "warning_common_timeline",
        "This run has not the same timeline on all variables, using the common one.",
    ],
    "correction_take-one-lon": [
        "warning_common_longitude",
        "This run has not the same longitude provided on all variables, but still compatible.",
    ],
    "correction_take-one-lat": [
        "warning_common_latitude",
        "This run has not the same latitude provided on all variables, but still compatible.",
    ],
    "correction_shift-one-day": [
        "warning_shift_1_day",
        "This run had their timeline shifted by 1 day too early.",
    ],
    "correction_shift-one-half-day": [
        "warning_shift_1_half_day",
        "This run had their timeline shifted by 1 half day too early.",
    ],
    "correction_drop-2014": [
        "warning_drop_2014",
        "This run had 2014 included, but was removed because it is exactly the same of its historical.",
    ],
    "correction_tasmax-KIOST": [
        "warning_files_tasmax_2061",
        "One of the files in the repository for tasmax didnt make any sense: for 2061, with a non monotonous time axis, while 2061 was already in the main file. Here, only the main file was used.",
    ],
}


def func_prepare_datasets(esm, xp, member, grid, cfg):

    # loading required variables
    DATA = {var: load_dataset(esm, xp, member, grid, var, cfg) for var in cfg.list_vars}

    # ---------------------------
    # Preparing coordinates
    # ---------------------------
    FWI = xr.Dataset()
    test_bnds_dims = False
    # checking that for each coord (time, lat, lon), all files agree:
    for coo in ["time", "lat", "lon", "time_bnds", "lat_bnds", "lon_bnds"]:

        # comparing coordinates
        tmp = []
        for var in cfg.list_vars:
            # checking name of coordinates
            if (coo not in DATA[var]) and (var in ["time", "lat", "lon"]):
                ok_coord = False
            elif (coo not in DATA[var]) and coo in [
                "time_bnds",
                "lat_bnds",
                "lon_bnds",
            ]:
                if coo.replace("bnds", "bounds") in DATA[var]:
                    coo = coo.replace("bnds", "bounds")
                    ok_coord = True
                else:
                    ok_coord = False
            else:
                ok_coord = True

            # getting coordinate
            if ok_coord:
                tmp.append(DATA[var][coo].values)
            elif coo in ["lat_bnds", "lon_bnds"]:
                FWI.attrs["warning_" + coo] = (
                    "No " + coo + " was provided in initial dataset."
                )
            else:
                raise Exception(
                    "Coordinate " + coo + " may have a different name here? Check."
                )

        # Checking coordinates
        if len(tmp) > 0:

            # some very special case: boundaries with unnecessary complexity
            str_coo = str.split(coo, "_")
            if (len(str_coo) > 1) and (str_coo[1] in ["bnds", "bounds"]):
                for var in cfg.list_vars:
                    if DATA[var][coo].ndim > 2:
                        # taking only necessary axis: (time OR lat OR lon) and (time), but not time AND lat AND lon AND time...
                        # too many files had this exception (mostly from EC-Earth3 and ACCESS-CM2, but not all), then transformed into a systematic check/correction
                        dims_drop = [
                            dd for dd in DATA[var][coo].dims if dd not in str_coo
                        ]
                        vals_dims = {dd: DATA[var][dd].values[0] for dd in dims_drop}
                        DATA[var][coo] = DATA[var][coo].loc[vals_dims].drop(dims_drop)
                        test_bnds_dims = True

            if (
                ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys())
                and (
                    "correction_cut-common-time"
                    in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
                )
                and (coo in ["time", "time_bnds", "time_bounds"])
            ):
                ind_tmp = np.argmin([len(tt) for tt in tmp])

            elif (
                ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys())
                and (
                    "correction_take-one-lat"
                    in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
                )
                and (coo in ["lat", "lat_bnds", "lat_bounds"])
            ):
                ind_tmp = 0

            elif (
                ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys())
                and (
                    "correction_take-one-lon"
                    in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
                )
                and (coo in ["lon", "lon_bnds", "lon_bounds"])
            ):
                ind_tmp = 0
                # EC-Earth3,ssp126, multiple ensemble members ,gr has its lon shifted by half a degree on sfcWind... assuming it is alright here

            else:
                for ii in range(1, len(cfg.list_vars)):
                    if (tmp[0].shape != tmp[ii].shape) or (
                        np.array_equal(tmp[0], tmp[ii]) == False
                    ):
                        raise Exception(
                            "May have different sets of coordinates on "
                            + coo
                            + ", check."
                        )
                ind_tmp = 0
                # tmp0 = np.unique(tmp) # not working with _bnds variables

            if ("bnds" in coo) or ("bounds" in coo):
                # preparing the variable
                FWI[coo] = DATA[cfg.list_vars[ind_tmp]][coo]
            else:
                # preparing the coordinate
                FWI.coords[coo] = tmp[ind_tmp]

    # adding some coordinates if necessary
    if cfg.adjust_overwinterDC == "wDC":
        FWI.coords["days_wDC"] = ["day-2", "day-1", "day", "day+1", "day+2"]
    # ---------------------------
    # ---------------------------

    # ---------------------------
    # creating variables of interest
    # ---------------------------
    for var in ["fwi"] + cfg.option_full_outputs * [
        "ffmc",
        "dmc",
        "dc",
        "isi",
        "bui",
        "TEMP",
        "RH",
        "RAIN",
        "WIND",
    ]:
        FWI[var] = xr.DataArray(
            np.nan,
            coords={
                "time": FWI.time.values,
                "lat": FWI.lat.values,
                "lon": FWI.lon.values,
            },
            dims=("time", "lat", "lon"),
        )
        FWI[var].attrs["name"] = dico_names[var]
        FWI[var].attrs["description"] = dico_descrip[var]
        if var in ["fwi", "ffmc", "dc", "dmc", "isi", "bui"]:
            FWI[var].attrs["units"] = "1"

    # copying attributes of coordinates (for some reason, removed after creation of variable fwi...)
    for coo in ["time", "lat", "lon"]:
        for att in DATA[cfg.list_vars[0]][coo].attrs:
            FWI[coo].attrs[att] = DATA[cfg.list_vars[0]][coo].attrs[att]
    # ---------------------------
    # ---------------------------

    # ---------------------------
    # Corrections
    # ---------------------------
    # removing useless stuff poping
    for stf in ["type", "height"]:
        if stf in FWI:
            FWI = FWI.drop(stf)

    # converting immediately all time formats into cftime, because of the inherent limitation of datetime64 to 2262 AND https://xarray.pydata.org/en/v0.11.3/time-series.html#cftimeindex
    for var in cfg.list_vars:
        DATA[var] = update_time_format(DATA[var])
    FWI = update_time_format(FWI)

    # Correcting the time axis ife required
    if ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys()) and (
        "correction_shift-one-day"
        in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
    ):
        FWI, DATA = shift_time(FWI, DATA, "one-day")

    if ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys()) and (
        "correction_shift-one-half-day"
        in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
    ):
        FWI, DATA = shift_time(FWI, DATA, "one-half-day")

    if ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys()) and (
        "correction_drop-2014"
        in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
    ):
        FWI = FWI.isel(time=slice(365, len(FWI.time)))
    # ---------------------------
    # ---------------------------

    # ---------------------------
    # Attributes
    # ---------------------------
    # General attributes of this dataset
    for att in [
        "experiment_id",
        "table_id",
        "source_id",
        "variant_label",
        "grid_label",
    ]:
        FWI.attrs[att] = DATA[cfg.list_vars[0]].attrs[att]
    FWI.attrs["variable_id"] = "fwi"
    FWI.attrs["info"] = "Canadian Fire Weather Index calculated for CMIP6."
    FWI.attrs[
        "method"
    ] = "Computation is based on the equations from Wang et al, 2015 (https://cfs.nrcan.gc.ca/publications?id=36461), adapted here for CMIP6 by Yann Quilcaille."
    FWI.attrs["contact"] = "Yann Quilcaille <yann.quilcaille@env.ethz.ch>"
    FWI.attrs["warning"] = "This variable is not an official CMIP6 variable."

    # Attribute: historical initialization
    FWI.attrs[
        "warning_method_init"
    ] = "The FWI has been initialized by the common way: https://cwfis.cfs.nrcan.gc.ca/background/dsm/normals"

    # Attributes for exceptions:
    if "_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys():
        for excep in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]:
            if excep not in dico_warnings_corrections:
                raise Exception(
                    "The name of this exception is unknown, please check: " + excep
                )
            else:
                FWI.attrs[
                    dico_warnings_corrections[excep][0]
                ] = dico_warnings_corrections[excep][1]

    # Attribute: boundaries
    if test_bnds_dims:
        FWI.attrs[
            "warning_boundaries"
        ] = "The variables for boundaries had additional dimensions (eg. time for lat_bnds), and it has been removed here."
    # ---------------------------
    # ---------------------------

    return DATA, FWI


def load_dataset(esm, xp, member, grid, var, cfg):
    print(
        "loading files of " + esm + ", " + xp + ", " + member + ", " + grid + ", " + var
    )
    path_files = os.path.join(cfg.path_cmip6, xp, "day", var, esm, member, grid)
    list_files = os.listdir(path_files)
    list_files_W = [
        os.path.join(path_files, file_W) for file_W in list_files if ".nc" in file_W
    ]
    if (
        ("_".join([esm, xp, member, grid]) in cfg.runs_exceptions_okay.keys())
        and (
            "correction_tasmax-KIOST"
            in cfg.runs_exceptions_okay["_".join([esm, xp, member, grid])]
        )
        and var == "tasmax"
    ):
        list_files_W = [file_W for file_W in list_files_W if "2061" not in file_W]
    OUT = xr.open_mfdataset(list_files_W, use_cftime=True)
    # checking the unit
    expected_unit = {
        "hursmin": ["%"],
        "hurs": ["%"],
        "hursmax": ["%"],
        "tasmax": ["K"],
        "tasm": ["K"],
        "tasmin": ["K"],
        "sfcWind": ["m s-1"],
        "pr": ["kg m-2 s-1"],
    }[var]
    if OUT[var].attrs["units"] not in expected_unit:
        raise Exception("Incorrect unit here.")

    return OUT


def update_time_format(data):
    if type(data.time.values[0]) in [np.datetime64]:
        # doing time
        new_time = []
        for val in data.time.values:
            val = DatetimeIndex([val])
            new_time.append(
                cft.DatetimeGregorian(
                    val[0].year,
                    val[0].month,
                    val[0].day,
                    val[0].hour,
                    val[0].minute,
                    val[0].second,
                )
            )
        data.coords["time"] = new_time

        # doing time_bnds
        new_time = []
        for val in data.time_bnds.values:
            val = DatetimeIndex(val)
            new_time.append(
                [
                    cft.DatetimeGregorian(
                        val[0].year,
                        val[0].month,
                        val[0].day,
                        val[0].hour,
                        val[0].minute,
                        val[0].second,
                    ),
                    cft.DatetimeGregorian(
                        val[1].year,
                        val[1].month,
                        val[1].day,
                        val[1].hour,
                        val[1].minute,
                        val[1].second,
                    ),
                ]
            )
        data["time_bnds"] = new_time

    elif type(data.time.values[0]) in [
        cft._cftime.DatetimeGregorian,
        cft._cftime.DatetimeNoLeap,
        cft.Datetime360Day,
        cft.DatetimeProlepticGregorian,
        cft.DatetimeJulian,
    ]:
        pass  # leave it that way

    else:
        raise Exception("This time format is not correctly handled.")

    return data


def shift_time(data_fwi, data, mode):
    # TIME
    tt = data_fwi.time.values
    format_tt = type(tt[0])
    if mode == "one-day":
        s = DatetimeIndex([np.datetime64(tt[0])]).shift(periods=1, freq="D")[0]
        e = DatetimeIndex([np.datetime64(tt[-1])]).shift(periods=1, freq="D")[0]
    elif mode == "one-half-day":
        s = DatetimeIndex([np.datetime64(tt[0])]).shift(periods=12, freq="H")[0]
        e = DatetimeIndex([np.datetime64(tt[-1])]).shift(periods=12, freq="H")[0]
    else:
        raise Exception("Unknown type of shift")
    shifted_time = xr.cftime_range(
        start=format_tt(s.year, s.month, s.day, s.hour, s.minute, s.second),
        end=format_tt(e.year, e.month, e.day, e.hour, e.minute, e.second),
        freq="1D",
        calendar=format_tt,
    )
    # shifting
    data_fwi.coords["time"] = shifted_time
    for var in data.keys():
        data[var].coords["time"] = shifted_time

    # TIME_BNDS
    tt = data_fwi.time_bnds.values
    if mode == "one-day":
        s = [
            DatetimeIndex([np.datetime64(tt[0][i])]).shift(periods=1, freq="D")[0]
            for i in range(len(tt[0]))
        ]
        e = [
            DatetimeIndex([np.datetime64(tt[-1][i])]).shift(periods=1, freq="D")[0]
            for i in range(len(tt[-1]))
        ]
    elif mode == "one-half-day":
        s = [
            DatetimeIndex([np.datetime64(tt[0][i])]).shift(periods=12, freq="H")[0]
            for i in range(len(tt[0]))
        ]
        e = [
            DatetimeIndex([np.datetime64(tt[-1][i])]).shift(periods=12, freq="H")[0]
            for i in range(len(tt[-1]))
        ]
    else:
        raise Exception("Unknown type of shift")
    shifted_time = np.array(
        [
            xr.cftime_range(
                start=format_tt(
                    s[i].year, s[i].month, s[i].day, s[i].hour, s[i].minute, s[i].second
                ),
                end=format_tt(
                    e[i].year, e[i].month, e[i].day, e[i].hour, e[i].minute, e[i].second
                ),
                freq="1D",
                calendar=format_tt,
            )
            for i in range(len(s))
        ]
    ).T
    # shifting
    data_fwi["time_bnds"] = xr.DataArray(shifted_time, dims=("time", "bnds"))
    for var in data.keys():
        data[var]["time_bnds"] = xr.DataArray(shifted_time, dims=("time", "bnds"))

    return data_fwi, data
