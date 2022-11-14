import os
import numpy as np
import xarray as xr
import warnings
import csv
import sys
import time

from functions_calc_indicators import *
from functions_support import *


#=====================================================================
# 0. PARAMETERS
#=====================================================================
#--------------------------------------------
# Path
#--------------------------------------------
# Absolute path for input: daily FWI. Outputs will be in regridded/fwixx/ann/g025
#path_FWInative_day = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hurs_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwi/day/g025'
path_FWInative_day = '/net/exo/landclim_nobackup/yquilcaille/FWI_CMIP6/hursmin_tasmax_sfcWind_pr/Drying-NSHeq_Day-continuous_Owinter-wDC/regridded/fwi/day/g025'
#--------------------------------------------
#--------------------------------------------

#--------------------------------------------
# Options for calculation
#--------------------------------------------
list_indics = ['fwixx', 'fwils', 'fwixd','fwisa']
# ref_period = (1965,2014,['historical']) # important for fwils & fwixd
ref_period = (1850,1899,['historical']) # important for fwils & fwixd
method_fwixd = 'percentile-95'
check_NaN = False # off because of overwintering
#--------------------------------------------
#--------------------------------------------


#--------------------------------------------
# Options for computation & server
#--------------------------------------------
# To control if run everything or using a bunch of processes
run_on_server = False
# Priority of this script on the server. Value in [-20,19], default at 0, higher is nicer to others
nice_value = 19

# Option: overwrite existing files? If True, will overwrite ONLY THOSE BEFORE THE PROVIDED DATE.
option_overwrite = False
overwrite_before = '2022-10-01T22:00:00' # Year, Month, Day, Hour, Minute, Second
#--------------------------------------------
#--------------------------------------------
#=====================================================================
#=====================================================================








#=====================================================================
# 1. PREPARATION
#=====================================================================
#--------------------------------------------
# Preparation of paths and run
#--------------------------------------------
if run_on_server and (nice_value != 19):
    print('THIS PROCESS IS QUITE RESOURCE INTENSIVE, PLEASE CONSIDER SETTING A HIGHER NICE SCORE TO REDUCE THE DISTURBANCE TO OTHER USERS: at the moment, asked for '+str(nice_value)+' instead of 19.')
os.nice(nice_value)

# Controlling manually the overwrite, because of the structure of scenarios for indicators. 
# used to launch several calculations in the meantime, remains fair to other thanks to the high nice value.
subindex_csl = int(sys.argv[1]) if run_on_server else None
runs_per_process = 30

# preparing final path
path_FWInative_annual = os.sep.join( path_FWInative_day.split(os.sep)[:-4+1] )

# time resolution of new indicators
current_time_res = 'day'
new_time_res = 'ann'

# preparing folders for save
for indic in list_indics:
    os.makedirs( os.path.join(path_FWInative_annual, indic, new_time_res, 'g025'), exist_ok=True)
#--------------------------------------------
#--------------------------------------------


#--------------------------------------------
# Preparation of files
#--------------------------------------------
# preparing total list of files
list_files = os.listdir( path_FWInative_day )

# preparing matching of scenarios
dico_items = {}
for file_W in list_files:
    var, _, esm, xp, memb, grid = str.split( file_W[:-len('.nc')], '_' ) # no need for "time_res"
    name_item = '/'.join([esm,memb,grid])
    if name_item not in dico_items.keys():
        dico_items[name_item] = []
    dico_items[name_item].append(xp)

# preparing list of files to run
list_tmp = []
# implementing a counter: if missing file, +1, and will have each python script for a given subindex_csl
counter_missing = -1
for name_item in dico_items.keys():
    esm,memb,grid = name_item.split('/')
    
    # checking if fully computed
    test_NeedRun = {}
    for scen in dico_items[name_item]:
        for indic in list_indics:
            # time_res = str.split( path_FWInative_day, os.sep)[-2]
            path_tmp = os.path.join( path_FWInative_annual, indic, new_time_res, grid )
            cfg = type('cfg_check', (object,), {'path_saveFWI':path_tmp, 'option_overwrite':option_overwrite, 'overwrite_before':overwrite_before})()
            name_file = '_'.join([indic,new_time_res,esm,scen,memb,grid])+'.nc'
            test_NeedRun[name_file] = function_check_run_file(name_file,cfg)

    if np.any( list(test_NeedRun.values()) ):# if so, adding them all: required because of historical period as reference for scenarios
        counter_missing += len( dico_items[name_item] ) # counting how many scenarios to treat here
        
        # taking them only if required
        if (run_on_server==False)  or  (counter_missing in range(subindex_csl*runs_per_process,(subindex_csl+1)*runs_per_process)):
            list_tmp.append( name_item )
            
# preparing exceptions
type_variables = '-'.join( str.split( path_FWInative_day[path_FWInative_day.find('hurs'):], '_')[:1+1] )
with open('available_files/'+'exceptions_'+type_variables+'.csv', newline='') as csvfile:
    read = csv.reader(csvfile, delimiter=',')
    runs_exceptions_okay = {}
    for row in read:
        item = '_'.join( row[:3+1] )
        if item not in runs_exceptions_okay:
            runs_exceptions_okay[item] = []
        runs_exceptions_okay[item].append( row[4] )
#--------------------------------------------
#--------------------------------------------
#=====================================================================
#=====================================================================








#=====================================================================
# 2. CALCULATION
#=====================================================================
# running through files to calculate indicators
for name_item in list_tmp:
    ti_start = time.time()
    esm,memb,grid = name_item.split('/')

    # identifying which files to do
    name_files_W = [ '_'.join([var, current_time_res, esm,scen,memb,grid])+'.nc' for scen in dico_items[name_item] ]
    files_to_get = [os.path.join( path_FWInative_day, W) for W in name_files_W]

    # loading datasets:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        DATA = xr.open_mfdataset(files_to_get, preprocess=add_xp_dim, combine='nested', concat_dim='xp' )#, chunks={'time':1}
        
    # calculating annual indicators
    to_save = calc_indicators( DATA, ref_period, indics=list_indics, method_fwixd=method_fwixd )

    # saving each xp in different files
    for scen in dico_items[name_item]:
        print( 'Calculating and saving '+esm+', '+memb+' on '+scen )#+' for '+indic ) 
        for indic in list_indics:
            name_tmp = '_'.join([indic,new_time_res,esm,scen,memb,grid])+'.nc'
            path_save = os.path.join(path_FWInative_annual, indic, new_time_res, 'g025', name_tmp)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                func_save_xp( to_save, scen, indic, path_save, attrs_out=DATA.attrs, check_NaN=check_NaN )
    ti_end = time.time()
    print( 'Computation time: '+str(np.round( (ti_end - ti_start)/60,1 ))+'min' )

    # cleaning
    DATA.close()
    del DATA
    to_save.close()
    del to_save
print('Finished!')
#=====================================================================
#=====================================================================


