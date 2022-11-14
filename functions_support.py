import csv
import datetime as dt
import os

import numpy as np
import xarray as xr


class configuration_FWI_CMIP6:
    """
    Class used for the calculation of the FWI on CMIP6 data.

    Parameters
    ----------
    type_variables: str
        String to identify the combination of variables to use. Structure must be VAR1 + '-' + VAR2, with:
            - VAR1 in ["hurs", "hursmin", "hursmax"]
            - VAR2 in ["tas", "tasmin", "tasmax"]
    adjust_DryingFactor: str
        Option for the "drying factor" used in calculation of DC. The name in the original algorithm is "Day-length adjustment in DC". It is changed here to avoid confusion with adjustments on effective day length, used in DMC.
            - 'original': values for the Northern hemisphere applied everywhere (Wagner et al, 1987: https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19927.pdf)
            - 'NSH': the values for the Southern hemisphere are those for the northern shifted by 6 months (https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py)
            - 'NSHeq': the same idea is applied, but near the equator, one same value is applied for all months (https://rdrr.io/rforge/cffdrs/src/R/dcCalc.R)
    adjust_DayLength: str
        Option for the effective day length, used in calculation of DMC.
            - 'original': values adapted for Canadian latitudes, depends on the month (Wagner et al, 1987: https://cfs.nrcan.gc.ca/pubwarehouse/pdfs/19927.pdf)
            - 'bins': depends on 4 bins of latitudes and the month (https://github.com/buckinha/pyfwi/blob/master/pyFWI/FWIFunctions.py)
            - 'continuous': depends continuously on latitude and the day of the year (https://github.com/NCAR/fire-indices & https://www.ncl.ucar.edu/Document/Functions/Crop/daylight_fao56.shtml)
    adjust_overwinterDC: str
        Option for the overwintering of DC.
            - 'original': without.
            - 'wDC': with overwintering (code based on Lawson & Armitage (2008) (http://cfs.nrcan.gc.ca/pubwarehouse/pdfs/29152.pdf) and https://rdrr.io/github/jordan-evens-nrcan/cffdrs/src/R/fireSeason.r)
    limits_on_data: dict
        Dictionnary informing which experiments, members and esms to take.
            Example 1: only historical and ssp585 of ACCESS-CM2 on r1i1p1f1 would be: {'xps':['historical','ssp585'], 'members':['r1i1p1f1'], 'esms':['ACCESS-CM2']}
            Example 2: everything would be: {'xps':None, 'members':None, 'esms':None}
    path_cmip6: str
        Path of CMIP6 data. Please mind that the code may have to be adapted depending on the architecture of your repositories.
    path_save: str
        Path of where to save the outputs. A repository will be created inside, its name depending on the options chosen for the configuration.
    overwrite_path_saveFWI: str
        If not None, path that overwrites where to save the FWI.
    option_overwrite: bool
        If the run has already been computed, option to confirm whether it needs to be run and overwriten again.
    overwrite_before: str
        If "option_overwrite" is True, will run & overwrite only those before this date. Format of this date must be: "YYYY-MM-DDTHH:MM:SS"  # Year, Month, Day, Hour, Minute, Second
    option_load_available: bool
        Whether need to produce the list of available files, or load the former saved list
    option_calc_size: bool
        Whether need to calculate the size of all files for all variables / esm / xp / member / common  and  those that are present for all variables. Warning, takes some time.
    option_full_outputs: bool
        WARNING, TO USE *VERY* CAUTIOUSLY!! TAKES A LOT OF MEMORY, BOTH IN RAM AND STORAGE.
        If True, returns all intermediary variables (DC, DMC, ...), used for debugging or sensitivity analysis.
    """

    # --------------------
    # INITIALIZATION
    def __init__(
        self,
        type_variables,
        adjust_DryingFactor,
        adjust_DayLength,
        adjust_overwinterDC,
        limits_on_data,
        path_cmip6,
        path_save,
        overwrite_path_saveFWI,
        option_overwrite,
        overwrite_before,
        option_load_available,
        option_calc_size,
        option_full_outputs,
    ):
        self.type_variables = type_variables
        self.adjust_DryingFactor = adjust_DryingFactor
        self.adjust_DayLength = adjust_DayLength
        self.adjust_overwinterDC = adjust_overwinterDC
        self.limits_on_data = limits_on_data
        self.path_cmip6 = path_cmip6
        self.path_save = path_save
        self.overwrite_path_saveFWI = overwrite_path_saveFWI
        self.option_overwrite = option_overwrite
        self.overwrite_before = overwrite_before
        self.option_load_available = option_load_available
        self.option_calc_size = option_calc_size
        self.option_full_outputs = option_full_outputs

    # --------------------

    # --------------------
    # Preparation
    def prepare_variables(self):
        # list of the variables to that will be used
        vars_ = str.split(self.type_variables, "-")

        # checking/filling
        if len(vars_) != 2:
            raise Exception('Format of "type_variables" is not recognize.')

        elif vars_[0] not in ["hurs", "hursmin", "hursmax"]:
            raise Exception(
                'First one in "type_variables" should be in hurs, hursmin, hursmax.'
            )

        elif vars_[1] not in ["tas", "tasmin", "tasmax"]:
            raise Exception(
                'First one in "type_variables" should be in tas, tasmin or tasmax.'
            )

        else:
            self.list_vars = [vars_[0], vars_[1], "sfcWind", "pr"]

    def prepare_common_files(self):
        # Available files
        if self.option_load_available:
            self.COMMON = {}
            for file_W in os.listdir("available_files"):
                if self.type_variables + ".csv" in file_W:
                    with open(
                        "available_files" + os.sep + file_W, newline=""
                    ) as csvfile:
                        read = csv.reader(csvfile, delimiter=" ")
                        tmp = []
                        for row in read:
                            if len(row) == 1:
                                tmp.append(row[0])
                            else:
                                tmp.append(row)
                        self.COMMON[
                            file_W[: -len("_" + self.type_variables + ".csv")]
                        ] = tmp

        else:
            # identifying all available runs
            dico_available_runs = self.find_all_available_members()

            # but only those that are available simultaneously for the selected variables
            self.COMMON = self.find_common_available_members()

            # writing these files
            for key in self.COMMON.keys():
                with open(
                    "available_files"
                    + os.sep
                    + key
                    + "_"
                    + self.type_variables
                    + ".csv",
                    "w",
                    newline="",
                ) as csvfile:
                    writ = csv.writer(csvfile, delimiter=" ")
                    for line in self.COMMON[key]:
                        if type(line) == str:
                            writ.writerow([line])
                        else:
                            writ.writerow(line)

    def prepare_exceptions(self):
        # Handling runs with problems

        # Problems with some runs with an easy fix:
        with open(
            "available_files" + os.sep + "exceptions_" + self.type_variables + ".csv",
            newline="",
        ) as csvfile:
            read = csv.reader(csvfile, delimiter=",")
            self.runs_exceptions_okay = {}
            for row in read:
                item = "_".join(row[: 3 + 1])
                if item not in self.runs_exceptions_okay:
                    self.runs_exceptions_okay[item] = []
                self.runs_exceptions_okay[item].append(row[4])

        # Problems with some runs that are not yet solved:
        with open(
            "available_files" + os.sep + "exclusions_" + self.type_variables + ".csv",
            newline="",
        ) as csvfile:
            read = csv.reader(csvfile, delimiter=",")
            self.runs_to_avoid = []
            for row in read:
                self.runs_to_avoid.append(row[: 3 + 1])

    def prepare_repository(self):
        if self.overwrite_path_saveFWI is None:
            # normal mode
            name_folder_variables = "_".join(self.list_vars)
            name_folder_adjusts = "_".join(
                [
                    "Drying-" + self.adjust_DryingFactor,
                    "Day-" + self.adjust_DayLength,
                    "Owinter-" + self.adjust_overwinterDC,
                ]
            )
            self.path_saveFWI = os.path.join(
                self.path_save,
                name_folder_variables,
                name_folder_adjusts,
                "intermediary",
            )

        else:
            # debug mode
            self.path_saveFWI = self.overwrite_path_saveFWI

        # checking if the repository needs to be created
        os.makedirs(self.path_saveFWI, exist_ok=True)

    def func_prepare_files(self):
        # preparing additional variables
        if self.limits_on_data["xps"] is None:
            self.list_xps = [
                "historical",
                "ssp119",
                "ssp126",
                "ssp245",
                "ssp370",
                "ssp434",
                "ssp460",
                "ssp585",
                "ssp534-over",
            ]
        else:
            self.list_xps = self.limits_on_data["xps"]

        # preparing everything
        self.prepare_variables()
        self.prepare_common_files()
        self.prepare_exceptions()
        self.prepare_repository()

        # removing those that cant be used
        for item in self.runs_to_avoid:
            if item in self.COMMON["runs"]:
                # removing the run, be it historical or not
                self.COMMON["runs"].remove(item)
            # checking if need to remove associated scenarios
            if item[1] in ["historical"]:
                for xp in self.COMMON["xps"]:
                    if ([item[0], xp, item[2], item[3]] in self.COMMON["runs"]) and (
                        xp not in ["historical"]
                    ):
                        self.COMMON["runs"].remove([item[0], xp, item[2], item[3]])

        # limiting number of ESMs for sensitivity analysis
        if self.limits_on_data["esms"] is not None:
            TMP = []
            for item in self.COMMON["runs"]:
                if item[0] in self.limits_on_data["esms"]:
                    TMP.append(item)
            self.COMMON["runs"] = TMP

        # limiting number of experiments for sensitivity analysis
        # option_members
        TMP = []
        for item in self.COMMON["runs"]:
            if item[1] in self.list_xps:
                TMP.append(item)
        self.COMMON["runs"] = TMP

        # limiting number of runs for sensitivity analysis
        if self.limits_on_data["members"] is not None:
            TMP = []
            for item in self.COMMON["runs"]:
                if item[2] in self.limits_on_data["members"]:
                    TMP.append(item)
            self.COMMON["runs"] = TMP

        # making sure that historical are done before, for matter of initialization. Same goes for ssp585, some runs need it for their ssp534-over
        prepared_order_xp = [
            "historical",
            "ssp119",
            "ssp126",
            "ssp245",
            "ssp370",
            "ssp434",
            "ssp460",
            "ssp585",
            "ssp534-over",
        ]
        self.list_sorted_common = []
        for xp in prepared_order_xp:
            for item in self.COMMON["runs"]:
                if item[1] == xp:
                    self.list_sorted_common.append(item)
                elif item[1] not in prepared_order_xp:
                    raise Exception("Must list all experiments to sort them.")

        # preparing which variables will be transmitted between timesteps
        self.vars_transmit_timesteps = ["ffmcPREV", "dcPREV", "dmcPREV"] + (
            self.adjust_overwinterDC == "wDC"
        ) * ["TEMP_wDC", "SeasonActive", "DCf", "rw", "CounterSeasonActive"]
        self.list_xp_init = ["historical", "ssp585"]

    # --------------------

    # --------------------
    # FINDING CMIP6 files if asked for
    def func_test_vars(self, xp, esm, member, grid):
        avail_all = []
        for var in self.list_vars:
            test = False
            # variable available for this xp / esm?
            if esm in self.dico_available_runs[xp][var]:
                # variable available for this xp / esm / member?
                if member in self.dico_available_runs[xp][var][esm]:
                    # variable available for this xp / esm / member / grid?
                    if grid in self.dico_available_runs[xp][var][esm][member]:
                        test = True
            avail_all.append(test)
        return avail_all

    def find_all_available_members(self):
        # finding available runs
        self.dico_available_runs = {}
        counter_before = 0  # rough estimation of how data it represents
        for xp in self.list_xps:
            print("Checking available runs on " + xp)
            self.dico_available_runs[xp] = {}
            for var in self.list_vars:
                self.dico_available_runs[xp][var] = {}
                path_esms = os.path.join(self.path_cmip6, xp, "day", var)
                if os.path.isdir(path_esms):
                    list_esms = os.listdir(path_esms)
                    for esm in list_esms:
                        path_members = os.path.join(path_esms, esm)
                        self.dico_available_runs[xp][var][esm] = {}
                        list_members = os.listdir(path_members)
                        for member in list_members:
                            path_grids = os.path.join(path_members, member)
                            # edit of the code: append the grid only if contains files:
                            # self.dico_available_runs[xp][var][esm][member] = os.listdir( path_grids )
                            self.dico_available_runs[xp][var][esm][member] = []
                            for tmp in os.listdir(path_grids):
                                if len(os.listdir(os.path.join(path_grids, tmp))) > 0:
                                    self.dico_available_runs[xp][var][esm][
                                        member
                                    ].append(tmp)

                            # do we calculate sizes?
                            if self.option_calc_size:
                                for grid in self.dico_available_runs[xp][var][esm][
                                    member
                                ]:
                                    # summing size of files
                                    path_files = os.path.join(path_grids, grid)
                                    list_files = os.listdir(path_files)
                                    for file_W in list_files:
                                        if ".nc" in file_W:
                                            counter_before += os.path.getsize(
                                                os.path.join(path_files, file_W)
                                            )
        if self.option_calc_size:
            print(
                "initial size of ALL the data: "
                + str(np.round(counter_before / (1024) ** 4, 1))
                + "Tb"
            )

        return self.dico_available_runs

    def find_common_available_members(self):
        # producing full list of ESMs, members and grid:
        list_esms_all, list_members_all, list_grids_all = [], [], []
        for xp in self.dico_available_runs:
            for var in self.dico_available_runs[xp]:
                for esm in self.dico_available_runs[xp][var]:
                    if esm not in list_esms_all:
                        list_esms_all.append(esm)
                    for member in self.dico_available_runs[xp][var][esm]:
                        if member not in list_members_all:
                            list_members_all.append(member)
                        for grid in self.dico_available_runs[xp][var][esm][member]:
                            if grid not in list_grids_all:
                                list_grids_all.append(grid)

        # selecting data that will actually be selected:
        #  - need to have the same variables together on the same xp / esm / member / grid
        #  - if scenario, need to have the corresponding historical
        list_runs_common, list_esms_common, list_members_common, list_grids_common = (
            [],
            [],
            [],
            [],
        )
        for esm in list_esms_all:
            for xp in self.list_xps:
                for member in list_members_all:
                    for grid in list_grids_all:
                        # checking for each variable if the xp / esm / member / grid is available
                        avail_all = self.func_test_vars(xp, esm, member, grid)
                        # checking that it is available for all variables
                        if np.all(avail_all):
                            # before adding this run, checking thatthe FWI of scenarios can be initialized by the FWI of historical
                            if xp in ["historical"]:
                                # adding
                                list_runs_common.append([esm, xp, member, grid])
                                if esm not in list_esms_common:
                                    list_esms_common.append(esm)
                                if member not in list_members_common:
                                    list_members_common.append(member)
                                if grid not in list_grids_common:
                                    list_grids_common.append(grid)
                            elif xp in [
                                "ssp119",
                                "ssp126",
                                "ssp245",
                                "ssp370",
                                "ssp434",
                                "ssp460",
                                "ssp585",
                                "ssp534-over",
                            ]:
                                avail_all_hist = self.func_test_vars(
                                    "historical", esm, member, grid
                                )
                                if np.all(avail_all_hist):
                                    # adding
                                    list_runs_common.append([esm, xp, member, grid])
                                    if esm not in list_esms_common:
                                        list_esms_common.append(esm)
                                    if member not in list_members_common:
                                        list_members_common.append(member)
                                    if grid not in list_grids_common:
                                        list_grids_common.append(grid)
                            else:
                                raise Exception(
                                    "Unprepared experience, need to decide about its continuity to a historical run for the initialization of the FWI."
                                )
        print(
            "Number of ESMs used: "
            + str(len(list_esms_common))
            + " (initially in data: "
            + str(len(list_esms_all))
            + ")"
        )
        print(
            "Number of members used: "
            + str(len(list_members_common))
            + " (initially in data: "
            + str(len(list_members_all))
            + ")"
        )
        print(
            "Number of grids used: "
            + str(len(list_grids_common))
            + " (initially in data: "
            + str(len(list_grids_all))
            + ")"
        )

        if self.option_calc_size:
            counter_after = 0  # rough estimation of how data it represents
            for item in list_runs_common:
                esm, xp, member, grid = item
                for var in list_vars:
                    path_files = os.path.join(
                        self.path_cmip6, xp, "day", var, esm, member, grid
                    )
                    list_files = os.listdir(path_files)
                    for file_W in list_files:
                        if ".nc" in file_W:
                            counter_after += os.path.getsize(
                                os.path.join(path_files, file_W)
                            )
            print(
                "Size of the data that will actually be used: "
                + str(np.round(counter_after / (1024) ** 4, 1))
                + "Tb"
            )

        return {
            "runs": list_runs_common,
            "esms": list_esms_common,
            "members": list_members_common,
            "grids": list_grids_common,
            "xps": self.list_xps,
        }

    # --------------------


def convert_wind(wind_cmip6, k_time):
    """
    Converts m s-1 to kph.

    Parameters
    ----------
    wind_cmip6: DataArray
        data for the wind in CMIP6. Must have a 'time' axis.
    k_time: float
        Index of a timestep
    """
    if wind_cmip6.units not in ["m s-1"]:
        raise Exception("Unprepared unit for the wind: " + wind_cmip6.units)
    else:
        return wind_cmip6.isel(time=k_time).values * 3600 / 1000


def convert_temp(temp_cmip6, k_time):
    """
    Converts K to degC.

    Parameters
    ----------
    temp_cmip6: DataArray
        data for the temperature in CMIP6. Must have a 'time' axis.
    k_time: float
        Index of a timestep
    """
    if temp_cmip6.units not in ["K"]:
        raise Exception("Unprepared unit for the temperature: " + temp_cmip6.units)
    else:
        return temp_cmip6.isel(time=k_time).values - 273.15


def convert_rain(rain_cmip6, k_time):
    """
    Converts kg m-2 s-1 to 24-hour accumulated rainfall in mm.

    Parameters
    ----------
    rain_cmip6: DataArray
        data for the precipitations in CMIP6. Must have a 'time' axis.
    k_time: float
        Index of a timestep
    """
    if rain_cmip6.units not in ["kg m-2 s-1"]:
        raise Exception("Unprepared unit for the rain: " + rain_cmip6.units)
    else:
        return rain_cmip6.isel(time=k_time).values * 24 * 3600 / 1.0


def convert_rh(rh_cmip6, k_time):
    """
    Just a check of unit and selection of year.

    Parameters
    ----------
    rh_cmip6: DataArray
        data for the relative humidity in CMIP6. Must have a 'time' axis.
    k_time: float
        Index of a timestep
    """
    #
    if rh_cmip6.units not in ["%"]:
        raise Exception("Unprepared unit for the relative humidity: " + rh_cmip6.units)
    else:
        return rh_cmip6.isel(time=k_time).values


def prepare_variables_FWI(former_calc, DATA, k_time, cfg):
    """
    Function used to prepare everything required for calculation of the FWI on this current day

    Parameters
    ----------
    former_calc: dictionary
        Variables for computation of the FWI at the former day
    DATA: xarrays
        Data produced by "func_prepare_datasets".
    k_time: int
        position of the timestep (better than value of the timestep, because of overwintering DC)
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    calc: dictionary
        Variables for computation of the FWI at the current day
    """
    # Preparing the dictionary that will be returned:
    calc = {}

    # basic variables that will be returned: (MONTH, NUMB_DAY)
    vv = list(DATA.keys())
    val_time = DATA[vv[0]].time.values[k_time]
    calc["MONTH"] = val_time.month
    calc[
        "NUMB_DAY"
    ] = (
        val_time.dayofyr
    )  # TO DO: adapt by factor depending on calendar for number of days in year --> 365

    # Transmitting the codes from former timesteps: (ffmcPREV, dcPREV, dmcPREV)
    for var in ["ffmcPREV", "dcPREV", "dmcPREV"]:
        if var in former_calc:
            calc[var] = np.copy(former_calc[var])

    # preparing next ones: (LAT)
    _, lat_mesh = np.meshgrid(DATA["pr"].lon.values, DATA["pr"].lat.values)
    calc["LAT"] = lat_mesh

    # Doing this step before to minimize I/O time on loading twice tasmax: (TEMP_wDC, SeasonActive, DCf, rw, CounterSeasonActive)
    if cfg.adjust_overwinterDC == "wDC":
        if "tasmax" not in DATA:
            raise Exception(
                "The detection of the fire season is meant to be used with the daily maximum temperature (tasmax)."
            )
        else:
            # adding 2 days before, the current day and the next 2 days of tasmax if need to overwinter DC
            if (former_calc["CounterSeasonActive"] is None) or (
                former_calc["CounterSeasonActive"].ndim == 0
            ):  # corresponds to (k_time == 0) and xp in ['historical']:
                # Approximation: initialization of this variable, with 3 times the first timestep
                calc["TEMP_wDC"] = convert_temp(
                    DATA["tasmax"]["tasmax"], [0, 0, 0, 1, 2]
                )
            else:
                # adding 2 days before, the current day and the next 2 days of tasmax if need to overwinter DC
                # Approximation: last timesteps of the run are repeated instead of using scenarios.
                temp_new = convert_temp(
                    DATA["tasmax"]["tasmax"],
                    np.min([DATA["tasmax"].time.size - 1, k_time + 2]),
                )
                calc["TEMP_wDC"] = np.vstack(
                    [former_calc["TEMP_wDC"][1:, ...], temp_new[np.newaxis, ...]]
                )

            # immediately saving the TEMP: actual day of TEMP_wDC
            calc["TEMP"] = calc["TEMP_wDC"][2, ...]

            # also adding 3 variables to handle the fire season:
            if (former_calc["CounterSeasonActive"] is None) or (
                former_calc["CounterSeasonActive"].ndim == 0
            ):  # corresponds to (k_time == 0) and xp in ['historical']:
                calc["SeasonActive"] = False * np.empty(
                    calc["LAT"].shape, dtype=bool
                )  # SeasonActive: Boolean to check whether the fire season is active or not
                calc["DCf"] = np.nan * np.ones(
                    calc["LAT"].shape
                )  # DCf: Final fall DC value
                calc["rw"] = np.zeros(
                    calc["LAT"].shape
                )  # rw: Precipitations accumulated over winter
                calc["CounterSeasonActive"] = np.zeros(
                    calc["LAT"].shape
                )  # CounterSeasonActive: counts how many fire seasons there has been. Required for the case of the first active fire season.
            else:
                # carries same values from former day, they will be updated in correct grid cells when calculating FWI
                for var in ["SeasonActive", "DCf", "rw", "CounterSeasonActive"]:
                    calc[var] = former_calc[var]

    elif cfg.adjust_overwinterDC != "original":
        raise Exception(
            "Wrong name for adjust_overwinterDC, only wDC (=True) and original (=False) are possible values."
        )

    # doing crucial variables. No need to do TEMP if overwintering DC: (WIND, RAIN, RH, TEMP)
    for var in ["WIND", "RAIN", "RH"] + (cfg.adjust_overwinterDC != "wDC") * ["TEMP"]:
        # function to use for conversion
        fct_convert = {
            "WIND": convert_wind,
            "RAIN": convert_rain,
            "TEMP": convert_temp,
            "RH": convert_rh,
        }[var]

        # variable to use for conversion
        if var == "WIND":
            var_cmip6 = "sfcWind"
        elif var == "RAIN":
            var_cmip6 = "pr"
        elif var == "TEMP":
            vars_tmp = [
                var for var in DATA.keys() if var in ["tas", "tasmax", "tasmin"]
            ]
            if len(vars_tmp) > 1:
                raise Exception(
                    "It seems that several variables could be interpreted as temperature."
                )
            elif len(vars_tmp) == 0:
                raise Exception(
                    "It seems that no variables could be interpreted as temperature."
                )
            else:
                var_cmip6 = vars_tmp[0]
        elif var == "RH":
            vars_tmp = [
                var for var in DATA.keys() if var in ["hurs", "hursmax", "hursmin"]
            ]
            if len(vars_tmp) > 1:
                raise Exception(
                    "It seems that several variables could be interpreted as relative humidity."
                )
            elif len(vars_tmp) == 0:
                raise Exception(
                    "It seems that no variables could be interpreted as relative humidity."
                )
            else:
                var_cmip6 = vars_tmp[0]

        # converting
        calc[var] = fct_convert(DATA[var_cmip6][var_cmip6], k_time)
    return calc


def function_check_run_file(name_file, cfg):
    """
    Checking if a file has been run and if it is too recent to rerun

    Parameters
    ----------
    name_file: string
        name of the file to check. Its path is in 'cfg.path_saveFWI'
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    run_file: bool
        If True, this file needs to be run
    """
    if os.path.isfile(cfg.path_saveFWI + os.sep + name_file) == False:
        # file not existing: need to run it
        run_file = True
    else:
        if cfg.option_overwrite == False:
            # file existing and no overwrite asked: no re-run
            run_file = False
        else:
            t_f = np.datetime64(
                dt.date.fromtimestamp(
                    os.path.getmtime(cfg.path_saveFWI + os.sep + name_file)
                )
            )
            # f = np.datetime64(dt.date.fromtimestamp(os.path.getmtime(name_file)))
            if t_f > np.datetime64(cfg.overwrite_before):
                # file existing, overwrite, but file too recent to be overwritten: no re-run
                run_file = False
            else:
                run_file = True
    return run_file


def func_init_from_scen(item, full_dates, date0, cfg):
    """
    Initializing the FWI from a scenario

    Parameters
    ----------
    item: list of string
        ESM, Experiment, Ensemble member, Grid
    full_dates: dict
        Dictionnary of dates created in '_calc_FWI-CMIP6.py'
    date0: timestep, type depends on the CMIP6 run
        First timestep of the run
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.

    Returns
    ----------
    out: dict
        Dictionnary with necessary values for initialization
    """

    # dictionary that will be returned:
    out = {}

    # dates that will used for initialization in this experiment
    if ("_".join(item) in cfg.runs_exceptions_okay.keys()) and (
        "correction_ssp534-over_ssp585-2039" in cfg.runs_exceptions_okay["_".join(item)]
    ):
        xp_init = (
            "ssp585"  # corresponding warning already embedded in the dataset run_FWI.
        )
        date_init = full_dates[2039]
    elif ("_".join(item) in cfg.runs_exceptions_okay.keys()) and (
        "correction_ssp534-over_ssp585-2040" in cfg.runs_exceptions_okay["_".join(item)]
    ):
        xp_init = (
            "ssp585"  # corresponding warning already embedded in the dataset run_FWI.
        )
        date_init = full_dates[2040]
    else:
        xp_init = "historical"
        date_init = full_dates[2014]

    # loading former computation for initialization
    esm, xp, member, grid = item
    if cfg.option_full_outputs:
        init_file = (
            "fwi_day_"
            + esm
            + "_"
            + xp_init
            + "_"
            + member
            + "_"
            + grid
            + "_full-outputs.nc"
        )
    else:
        init_file = "fwi_day_" + esm + "_" + xp_init + "_" + member + "_" + grid + ".nc"
    init = xr.open_dataset(cfg.path_saveFWI + os.sep + init_file, use_cftime=True)

    # checking continuity of dates
    if (date0 - init.time.sel(time=date_init).values).days == 1:
        for kk in cfg.vars_transmit_timesteps:
            # checking date of save
            if init[kk + "_" + str(date_init.year)].attrs["time_values"] != str(
                date_init
            ):
                raise Exception(
                    "The initialization seems not to have been saved at the correct date."
                )
            else:
                out[kk] = init[kk + "_" + str(date_init.year)].values
    else:
        raise Exception("Timelines for initialization dont correspond.")

    return out


def adhoc_concat2040(item, FWI, cfg):
    """
    Creating a dirty function to conserve attributes, without merge that takes wayyy too long

    Parameters
    ----------
    item: list of string
        ESM, Experiment, Ensemble member, Grid
    FWI: Dataset
        FWI
    cfg: class configuration_FWI_CMIP6
        Class used to carry information on which calculations have to be performed, specifically on options for adjustments.
    """

    # loading file for initialization
    esm, xp, member, grid = item
    if cfg.option_full_outputs:
        init_file = (
            "fwi_day_"
            + esm
            + "_"
            + "ssp585"
            + "_"
            + member
            + "_"
            + grid
            + "_full-outputs.nc"
        )
    else:
        init_file = (
            "fwi_day_" + esm + "_" + "ssp585" + "_" + member + "_" + grid + ".nc"
        )
    init = xr.open_dataset(cfg.path_saveFWI + os.sep + init_file, use_cftime=True)

    # prepare out dataset
    out = xr.Dataset()

    # preparing 2040
    t2040 = [tt for tt in init.time.values if tt.year == 2040]

    # preparing coordinates
    for coord in FWI.coords:
        if coord == "time":  # can be /bnds/bounds
            out.coords[coord] = xr.concat(
                [init[coord].sel(time=t2040), FWI[coord]], dim="time"
            )
        else:
            out.coords[coord] = FWI[coord]

    # preparing variables
    for var in FWI.variables:
        if var == "fwi":
            out["fwi"] = xr.concat(
                [init["fwi"].sel(time=t2040), FWI["fwi"]], dim="time"
            )
        elif var in ["time_bnds", "time_bounds"]:
            out[var] = xr.concat(
                [init[var].sel(time=t2040).compute(), FWI[var].compute()], dim="time"
            )
        elif var not in FWI.coords:
            out[var] = FWI[var]

    # handling general attributes
    out.attrs = FWI.attrs

    return out
