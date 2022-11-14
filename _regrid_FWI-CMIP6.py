# usual ones
import csv
import os

# importing necessary scripts from the cmip6-ng processor
import sys
import time as ti
from datetime import datetime as dt

import numpy as np
import xarray as xr

sys.path.append("/home/yquilcaille/FWI/cmip6_ng_master/cmip6_ng")
import logging
import traceback

from core import core_functions_fwi as cf
from core import grid_functions_fwi as gf
from utils import string_functions_fwi as sf

from functions_support import *

# =====================================================================
# 0. GENERAL PARAMETERS
# =====================================================================
# Absolute path to get CMIP6 data: native daily hursmin-tasmax
time_res = "day"
# path_fwi_cmip6_native = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/intermediary'
# SAVE_PATH = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded'
path_fwi_cmip6_native = "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hursmin_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/intermediary"
SAVE_PATH = "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hursmin_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded"

# To control if run everything or using a bunch of processes
run_on_server = False
# Priority of this script on the server. Value in [-20,19], default at 0, higher is nicer to others
nice_value = 19

# Option: overwrite existing files? If True, will overwrite ONLY THOSE BEFORE THE PROVIDED DATE.
option_overwrite = False
overwrite_before = "2022-01-01T00:00:00"  # Year, Month, Day, Hour, Minute, Second
# =====================================================================
# =====================================================================


# =====================================================================
# 1. PREPARATION FILES
# =====================================================================
# Only if debug mode
SAVE_PATH_TMP = "/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/debug_regridding"

# checking path
if not os.path.exists(SAVE_PATH):
    os.makedirs(SAVE_PATH)

if run_on_server and (nice_value != 19):
    print(
        "THIS PROCESS IS QUITE RESOURCE INTENSIVE, PLEASE CONSIDER SETTING A HIGHER NICE SCORE TO REDUCE THE DISTURBANCE TO OTHER USERS: at the moment, asked for "
        + str(nice_value)
        + " instead of 19."
    )
os.nice(nice_value)

# list of files that will be regridded
list_files = [
    os.path.join(path_fwi_cmip6_native, file_W)
    for file_W in os.listdir(path_fwi_cmip6_native)
    if "full-outputs" not in file_W
]

# removing files to regrid that have same ESM, experiment, ensemble member, but different grids: taking only the one with the finest resolution
# must be ordered by "quality" of grid. Will take the first best one encountered.
to_remove = {
    "fwi_day_GFDL-CM4_historical_r1i1p1f1": ["gr1", "gr2"],
    "fwi_day_GFDL-CM4_ssp585_r1i1p1f1": ["gr1", "gr2"],
    "fwi_day_GFDL-CM4_ssp245_r1i1p1f1": ["gr1", "gr2"],
}
for file_remove in to_remove:
    test = {
        grid: os.sep.join([path_fwi_cmip6_native, file_remove + "_" + grid + ".nc"])
        in list_files
        for grid in to_remove[file_remove]
    }
    if sum(test.values()) > 1:
        run_to_rm = False
        for grid in to_remove[file_remove]:
            if (
                test[grid] and run_to_rm
            ):  # other grid present in files, assumed not the best, removing it
                pth = os.sep.join(
                    [path_fwi_cmip6_native, file_remove + "_" + grid + ".nc"]
                )
                print("Removing " + pth + ' because another grid is "better".')
                list_files.remove(pth)
            elif test[grid]:  # first grid present in files, assumed best, keeping it
                pth = os.sep.join(
                    [path_fwi_cmip6_native, file_remove + "_" + grid + ".nc"]
                )
                print("Keeping " + pth + ' because this grid is "better".')
                run_to_rm = True
        print(" ")

# quick check trying to find any other:
dic_full = {}
for file_W in list_files:
    esm, xp, memb, grid = file_W.split(os.sep)[-1][:-3].split("_")[2 : 5 + 1]
    if esm + "_" + xp + "_" + memb not in dic_full:
        dic_full[esm + "_" + xp + "_" + memb] = []
    dic_full[esm + "_" + xp + "_" + memb].append(grid)
for item in dic_full:
    if len(dic_full[item]) > 1:
        print(
            "Warning: several runs still share the same ESM, experiment and ensemble member, but differ by their grid. Please choose the initial grid for regridding. Happens in    "
            + item
        )
# =====================================================================
# =====================================================================


# =====================================================================
# 2. PREPARING REGRIDDING
# =====================================================================
# GIT INFO
# print('git_info would be: git_info = cf.get_git_info(). Here, adapting a lot...')
curr_time = ti.strftime("%Y:%m:%d at %H:%M:%S")
rep_name = "https://git.iac.ethz.ch/cmip6-ng/cmip6-ng"
branch_name = "master"
revision = "adapted for FWI"
git_info = f"{curr_time} {rep_name}: {branch_name} {revision}"

# Dirty definition of args, normally from console
args = type(
    "cfg_check",
    (object,),
    {"overwrite": False, "log_level": 20, "log_file": None, "debug": False},
)()
# overwrite: choices=['all', 'update', 'false']
# log_level: default 20, choices=[10, 20, 30, 40]


def get_path_out_regrid(files):
    # extracted from cf.process_native_grid to identify files path
    metadata, (starttime, endtime) = sf.check_filename_vs_path([files], 7)
    save_path = sf.get_ng_path(
        varn=metadata["varn"], time_res=time_res, SAVE_PATH=SAVE_PATH
    )
    os.makedirs(save_path, exist_ok=True)
    save_filename = sf.get_ng_filename(
        varn=metadata["varn"],
        time_res=time_res,
        model=metadata["model"],
        run_type=metadata["run_type"],
        ensemble=metadata["ensemble"],
    )
    save_path = save_path.replace("native", "g025")
    save_file = save_filename.replace("native", "g025")
    return [save_path, save_file, os.path.join(save_path, save_file)]


# =====================================================================
# =====================================================================


# =====================================================================
# 3. REGRIDDING
# =====================================================================
# used to launch several calculations in the meantime, remains fair to other thanks to the high nice value.
subindex_csl = int(sys.argv[1]) if run_on_server else None
runs_per_process = 1
# implementing a counter: if missing file, +1, and will have each python script for a given subindex_csl
counter_missing = -1


for files in list_files:

    # Controlling manually the overwrite, not yet comfortable with the processor.
    path_tmp = get_path_out_regrid(files)
    cfg = type(
        "cfg_check",
        (object,),
        {
            "path_saveFWI": path_tmp[0],
            "option_overwrite": option_overwrite,
            "overwrite_before": overwrite_before,
        },
    )()
    if function_check_run_file(path_tmp[1], cfg):
        counter_missing += 1

        # checking if running these ones
        if (run_on_server == False) or (
            counter_missing
            in range(
                subindex_csl * runs_per_process, (subindex_csl + 1) * runs_per_process
            )
        ):
            # preparing regrid
            filename = None
            format_ = (
                "%(asctime)s - %(levelname)s - %(funcName)s() %(lineno)s: %(message)s"
            )
            logging.basicConfig(
                level=args.log_level, filename=args.log_file, format=format_
            )
            logger = logging.getLogger("regrid_CMIP6")

            # trying the regrid
            try:
                filename, updated = cf.process_native_grid(
                    [files],
                    time_res,
                    git_info=git_info,
                    overwrite=args.overwrite,
                    debug=args.debug,
                    SAVE_PATH=SAVE_PATH,
                )
                if filename is not None:  # checks passed -> regrid from native-ng
                    gf.regrid_cdo(filename, "g025", args.overwrite, method="con2")
            except Exception:
                if filename is not None:  # failed after native processing
                    source = filename
                else:
                    source = os.path.dirname(files[0])

                if not gf.delete_corrupt_files(source):  # check/delete corrupt source
                    errmsg = "\n".join(
                        [
                            f"Uncovered exception while processing source",
                            f"  {source}",
                            f"  {traceback.format_exc()}",
                        ]
                    )
                    logger.error(errmsg)
print(
    "****************************************************************************************************"
)
print(
    "*  FOR MEMORY REASONS, DO NOT FORGET TO DELETE THE INTERMEDIARY REPOSITORY AND ALL FILES INSIDE!!  *"
)
print(
    "****************************************************************************************************"
)
# --------------------------------------------
# --------------------------------------------
