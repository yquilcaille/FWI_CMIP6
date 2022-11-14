import os
import numpy as np
import xarray as xr
import sys # used for running all at once
import time
import warnings

from functions_calc_FWI import *
from functions_load_CMIP6 import *
from functions_support import *


# THIS SCRIPT SELECTS CMIP6 DATA TO CALCULATE THE CANADIAN FIRE WEATHER INDEX.
# - Only a selection of experiments are used for the calculation
# - The only runs used are those that have the 4 required variables on the same runs (= ESM / xp / grid / member)
# - The only scenarios used are those for which a corresponding historical has been run: because the FWI at t depends on values at t-1, hence the need for the correct historical.


#=====================================================================
# 0. PREPARATION
#=====================================================================
#--------------------------------------------
# General options
#--------------------------------------------
# Which variables are used: 'hursmin-tasmax', 'hurs-tasmax'
type_variables = 'hursmin-tasmax'
# Which type of drying factor: 'original', 'NSH', 'NSHeq'
adjust_DryingFactor = 'NSHeq' 
# Which type of DayLength: 'original', 'bins', 'continuous'
adjust_DayLength = 'continuous' 
# Adjustment with overwintering DC: 'original', 'wDC'
adjust_overwinterDC = 'wDC'

# For sensitivity analysis, one may decide to calculate less data. None means no limit. Otherwise, write a subset of what is sought. For example: 
# -> Full: {'xps':None, 'members':None, 'esms':None}
# -> Subset: {'xps':['historical','ssp585'], 'members':['r1i1p1f1'], 'esms':['ACCESS-CM2']}
limits_on_data = {'xps':None, 'members':None, 'esms':None}

# Absolute path to get CMIP6 data
path_cmip6 = '/net/atmos/data/cmip6'

# Absolute path where the daily FWI will be saved: adds on top of it a repository structure accounting for choices of variables and adjustments
path_save = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/'

# Option: overwrite existing files? If True, will overwrite ONLY THOSE BEFORE THE PROVIDED DATE.
option_overwrite = False
overwrite_before = '2022-09-06T00:00:00' # Year, Month, Day, Hour, Minute, Second
#--------------------------------------------
#--------------------------------------------




#--------------------------------------------
# Options for the server
#--------------------------------------------
# Run from JupyterLab or using the script '_run_all_calcFWI_CMIP6.sh' for exo?
run_on_server = False

# Changing the nice value: the higher the value, the lower the priority of this script on the server. Meant not to bother other users. Value must be in [-20,19], default is at 0, negative values can only set by super-users.
nice_value = 19
#--------------------------------------------
#--------------------------------------------



#--------------------------------------------
# Options for the search for available files
#--------------------------------------------
# Option: need to produce the list of available files, or load the former saved list
option_load_available = True
#--------------------------------------------
#--------------------------------------------



#--------------------------------------------
# Hard options: not recommended to change them
#--------------------------------------------
# TO USE *VERY* CAUTIOUSLY: returns all intermediary variables (DC, DMC, ...), used for debugging or sensitivity analysis. !!WARNING!!, avoid running too many in parallel if True.
option_full_outputs = False

# Yet, for debugging purposes, some value could be provided here. Otherwise, keep it None.
overwrite_path_saveFWI = None

# Option: to calculate the size of all files for all variables / esm / xp / member / common  and  those that are present for all variables. Warning, takes some time.
option_calc_size = False
#--------------------------------------------
#--------------------------------------------
#=====================================================================
#=====================================================================









#=====================================================================
# 1. PREPARATION: NICE VALUE, AVAILABLE FILES, CALCULATING SIZE OF DATA
#=====================================================================
# Changing nice value
if run_on_server and (nice_value != 19):
    print('THIS PROCESS IS QUITE RESOURCE INTENSIVE, PLEASE CONSIDER SETTING A HIGHER NICE SCORE TO REDUCE THE DISTURBANCE TO OTHER USERS: at the moment, asked for '+str(nice_value)+' instead of 19.')
os.nice(nice_value)

# prepare configuration
cfg = configuration_FWI_CMIP6(type_variables, adjust_DryingFactor, adjust_DayLength, adjust_overwinterDC, \
                              limits_on_data, path_cmip6, path_save, overwrite_path_saveFWI, \
                              option_overwrite, overwrite_before,\
                              option_load_available, option_calc_size, option_full_outputs)

# prepare files
cfg.func_prepare_files()
#=====================================================================
#=====================================================================





#=====================================================================
# 2. CALCULATING FWI at DAILY RESOLUTION
#=====================================================================
# used to launch several calculations in the meantime, remains fair to other thanks to the high nice value.
subindex_csl = int(sys.argv[1]) if run_on_server else None
runs_per_process = 1
# implementing a counter: if missing file, +1, and will have each python script for a given subindex_csl
counter_missing= -1


for item in cfg.list_sorted_common:
    ti_start = time.time()
    esm,xp,member,grid = item
    if cfg.option_full_outputs:
        name_file = 'fwi_day_'+esm+'_'+xp+'_'+member+'_'+grid+'_full-outputs.nc'
    else:
        name_file = 'fwi_day_'+esm+'_'+xp+'_'+member+'_'+grid+'.nc'
    
    # Checking if need to run this file
    if function_check_run_file( name_file, cfg ):
        counter_missing += 1
        if (run_on_server==False)  or  (counter_missing in range(subindex_csl*runs_per_process,(subindex_csl+1)*runs_per_process)):
            # DATA for inputs; prepared dataset for FWI (with land fraction)
            DATA, run_FWI = func_prepare_datasets(esm, xp, member, grid, cfg)
            
            # handling dates for initialization: those saved and those used
            type_time = type(run_FWI.time.values[0])

            # dates that will have to be saved in this experiment: used later
            full_dates = {2014:type_time(2014, 12, 31-1*(type_time == cft.Datetime360Day), 12,0,0,0),\
                          2039:type_time(2039, 12, 31-1*(type_time == cft.Datetime360Day), 12,0,0,0),\
                          2040:type_time(2040, 12, 31-1*(type_time == cft.Datetime360Day), 12,0,0,0)}
                    
            # initialization
            vars_calc = init_prev_values( DATA, cfg ) if xp in ['historical'] else func_init_from_scen( item, full_dates, run_FWI.time.values[0], cfg )
                    
            # looping over time
            t0 = time.time()
            # day values for FWI that will have to be saved at the end
            dico_mem_PREV = {}
            # just for printing purposes
            year_mem = -np.inf
            # looping (finally.)
            for (k_time,val_time) in enumerate(run_FWI.time.values):
                    
                # preparing all variables for FWI
                former_vars_calc = {var:np.copy(vars_calc[var]) for var in cfg.vars_transmit_timesteps}
                vars_calc = prepare_variables_FWI( former_vars_calc, DATA, k_time, cfg )

                # printing only if new year
                if val_time.year > year_mem:
                    print('Calculating FWI on: '+esm+', '+xp+', '+member+', '+grid+', '+str(val_time.year), end={True:'\n', False:'\r'}[run_on_server] )
                    year_mem = val_time.year

                # calculating FWI: 'calcFWI' wraps every significative steps for the calculation of the FWI.
                vars_calc = calcFWI( vars_calc, cfg )
                
                # checking that everything looks good
                if np.any(np.isnan(vars_calc['fwi'][vars_calc['ind_calc_FWI'][0],vars_calc['ind_calc_FWI'][1]])):
                    raise Exception('stop, NaN value!')

                # archiving value (not optimized here)
                for var in ['fwi'] + cfg.option_full_outputs * ['dc', 'dmc', 'ffmc', 'isi', 'bui', 'TEMP', 'RH', 'RAIN', 'WIND']:
                    run_FWI[var].loc[{'time':val_time}] = vars_calc[var]
                
                # checking if need to save these values for future runs
                if (xp in cfg.list_xp_init) and (val_time in full_dates.values()):
                    for key in cfg.vars_transmit_timesteps:
                        dico_mem_PREV[key+'_'+str(val_time.year)] = np.copy(vars_calc[key])

            # Saving values of FFMC, DMC and DC, which may be used later (historical--> ssps, ssp585-->ssp534-over)
            if xp in cfg.list_xp_init:
                # attributes removed on lat and lon by following step (issue of xarray not solved yet:  https://github.com/pydata/xarray/issues/2245)
                mem_attrs = { coo:run_FWI[coo].attrs for coo in ['lat', 'lon'] }
                # new variables
                for key in dico_mem_PREV.keys():
                    if dico_mem_PREV[key].ndim == 3:
                        run_FWI[key] = xr.DataArray( dico_mem_PREV[key], dims=('days_wDC', 'lat','lon') )
                    else:
                        run_FWI[key] = xr.DataArray( dico_mem_PREV[key], dims=('lat','lon') )                    
                    run_FWI[key].attrs['info'] = 'Simply saving this variable to initialize the scenarios using historical values'
                    run_FWI[key].attrs['time_values'] = str(full_dates[int(str.split(key,'_')[-1])])
                # reset attributes
                for coo in mem_attrs.keys():
                    run_FWI[coo].attrs = mem_attrs[coo]
                    
            # end of computation of FWI
            ti_end = time.time()
            print( 'Computation time: '+str(np.round( (ti_end - ti_start)/60,1 ))+'min' )
            
            # for some ssp534-over, need to add the year 2040 of ssp585 before regrid
            if ('_'.join(item) in cfg.runs_exceptions_okay.keys())  and  ('correction_ssp534-over_ssp585-2040' in cfg.runs_exceptions_okay['_'.join(item)]):
                run_FWI = adhoc_concat2040(item, run_FWI, cfg)
                
            # Saving the dataset
            #if cfg.type_variables in ['hurs-tasmax']:
            #    run_FWI.attrs['WARNING'] = 'This file is for sensitivity analysis, using these variables: '+str(cfg.list_vars)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore") # to turn off warning on unit time
                run_FWI.to_netcdf( cfg.path_saveFWI+'/'+name_file, encoding={var:{'zlib':True} for var in run_FWI.variables} )

            # thorough cleaning
            for var in cfg.list_vars:
                DATA[var].close()
            run_FWI.close()
            del DATA, run_FWI
print('Finished!')
#=====================================================================
#=====================================================================

