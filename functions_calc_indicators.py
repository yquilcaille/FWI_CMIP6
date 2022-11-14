import os
import numpy as np
import xarray as xr
import warnings
import cftime as cft
from scipy import interpolate




dico_bounds_dangers = {'very low':(0,5.2), 'low':(5.2,11.2), 'moderate':(11.2,21.3), 'high':(21.3,38.0), 'very high':(38.0,50.0), 'extreme':(50.0,np.inf)}



def add_xp_dim(data):
    data = data.expand_dims( xp = [data.attrs['experiment_id']] )
    return data




def calc_indicators( data, ref_period, indics=['fwixx', 'fwils', 'fwixd','fwisa'], method_fwixd='percentile-95' ):
    '''
        Computes annual indicators of the FWI from daily FWI values over several experiences.
        
        dat: DataArray
            daily FWI (xp,time,lat,lon)
            
        ref_period: tuple
            (start_date, end_date, ref_experiment) defining the reference period. Used for fwils and fwixd.

        indics: list of str
            Indicators that will he computed: 'fwixx', 'fwils', 'fwixd','fwisa'
            
        method_fwixd: str
            defines the method used for the calculation of fwixd:
                'percentile-XX': calculation of the local threshold based on the XX percentile over the reference period
                'category-YY': pre-defined threshold from dico_bounds_dangers
    '''
    # prepare dataset
    OUT = xr.Dataset()    
    
    # handling reference period
    ref_xp = ref_period[2]
    type_time = type(data.time.values[0])
    if type_time in [np.datetime64]:
        day_start = np.datetime64(str(ref_period[0])+'-01-01T12:00:00')
        day_end = np.datetime64(str(ref_period[1])+'-12-31T12:00:00')
    else:
        day_start = type_time(ref_period[0], 1, 1, 12,0,0,0)
        day_end = type_time(ref_period[1], 12, 31-1*(type_time == cft.Datetime360Day), 12,0,0,0)

    # checking methodfwixd
    if ('category' not in method_fwixd) and ('percentile' not in method_fwixd):
        raise Exception('Unknown method for fwixd')
    
    # calculating
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
    
        if 'fwixx' in indics: # annual maxima
            OUT['fwixx'] = calc_indic_fwixx( data['fwi'] )
            
            # Attributes
            OUT['fwixx'].attrs['long_name'] = 'Annual Maximum Of The Fire Weather Index'
            OUT['fwixx'].attrs['standard_name'] = 'fire_weather_index'
            OUT['fwixx'].attrs['method_fwixx'] = 'Annual maximum over each year of the FWI'
            OUT['fwixx'].attrs['units'] = data['fwi'].attrs['units']
            
            
        if 'fwils' in indics: # length fire weather season
            OUT['fwils'] = calc_indic_fwils(data['fwi'], day_start, day_end, ref_xp)
            
            # Attributes
            OUT['fwils'].attrs['long_name'] = 'Length Of The Fire Weather Season'
            OUT['fwils'].attrs['standard_name'] = 'fire_weather_index'
            OUT['fwils'].attrs['method_fwils'] = 'Number of days for which the FWI is above the mean of minimum and maximum of the FWI over the reference period for this single ensemble member.'
            OUT['fwils'].attrs['units'] = 'day'
            OUT['fwils'].attrs['ref_period'] = str(day_start)+' - '+str(day_end)

        if 'fwixd' in indics: # nb of days extreme fire weather
            OUT['fwixd'] = calc_indic_fwixd(data['fwi'], day_start, day_end, ref_xp, method=method_fwixd)
            
            # Attributes
            OUT['fwixd'].attrs['long_name'] = 'Number Of Days With Extreme Fire Weather'
            OUT['fwixd'].attrs['standard_name'] = 'fire_weather_index'
            OUT['fwixd'].attrs['units'] = 'day'
            OUT['fwixd'].attrs['ref_period'] = str(day_start)+' - '+str(day_end)
            lvl = str.split( method_fwixd, '-' )[1]
            if 'category' in method_fwixd:
                OUT['fwixd'].attrs['method_fwixd'] = 'Counted days above a threshold defined as the '+str(dico_bounds_dangers[lvl][0])+' - '+str(dico_bounds_dangers[lvl][1])+' fire risk in the Canadian Fire Weather Index categories'
            elif 'percentile' in method_fwixd:
                OUT['fwixd'].attrs['method_fwixd'] = 'Counted days above a threshold defined as the '+lvl+'% percentile of the reference period'
            
        if 'fwisa' in indics: # seasonal average
            OUT['fwisa'] = calc_indic_fwisa(data['fwi'])
        
            # Attributes
            OUT['fwisa'].attrs['long_name'] = 'Seasonal Average Of The Fire Weather Index'
            OUT['fwisa'].attrs['standard_name'] = 'fire_weather_index'
            OUT['fwisa'].attrs['method_fwisa'] = 'Annual peak of a 90 days average'
            OUT['fwisa'].attrs['units'] = data['fwi'].attrs['units']
            
    # correcting years --> not to do for the regrid --> doing this AFTER the regrid, all okay.
    #OUT.coords['time'] = np.array([val.year for val in OUT.time.values])
    
    # adding more attributes
    for var in ['time', 'lat', 'lon']:
        OUT[var].attrs = data[var].attrs

    # quick fix of a former version to avoid reproducing full data:
    if data['fwi'].attrs['description'] != 'Very low danger: FWI < 5.2; Low danger: 5.2 <= FWI < 11.2; Moderate danger: 11.2 <= FWI < 21.3; High danger: 21.3 <= FWI < 38.0; Very high danger: 38.0 <= FWI < 50; Extreme danger: 50 < FWI':
        for indic in indics:
            OUT[indic].attrs['description'] = data['fwi'].attrs['description']
        
    return OUT



def dangers_classes_fwi( fwi , categ ):
    '''
        fwi: DataArray
            daily FWI
        
        category: str
            must be within [ALL, very low, low, moderate, high, very high, extreme]
    '''
    if categ == 'ALL':
        raise Exception('Way too long at the moment, rewrite for xarray')
        danger = np.nan * np.ones( fwi.shape )
        for i,risk in enumerate( list(dico_bounds_dangers.keys()) ):
            m1,m2 = dico_bounds_dangers[risk]
            danger[np.where( (m1 <= fwi) & (fwi < m2) )] = i
        #categories = { 'very low':0, 'low':1, 'moderate':2, 'high':3, 'very high':4, 'extreme':5 }
        
    else:
        m1,m2 = dico_bounds_dangers[categ]
        danger = xr.where( (m1 <= fwi) & (fwi < m2), 1, 0 )
    return danger



def calc_indic_fwixx(dat):
    '''
        Calculate the annual maxima of the FWI.
        Similar to https://doi.org/10.1029/2018GL080959: just take max over the year
        
        dat: DataArray
            daily FWI
    '''
    return dat.resample(time='1Y').max()




def calc_indic_fwils(dat, day_start, day_end, ref_xp):
    '''
        Calculate the length of the fire season of the FWI.
        https://www.nature.com/articles/ncomms8537 & its SPM (ie same than https://doi.org/10.1029/2018GL080959)
        1. At each location, normalize fwi over reference period, min and max: fwi_norm = (fwi-min) / (max-min)
        2. Checking locally:
            - if fwi_norm < 0.5: not fire weather.
            - if fwi_norm > 0.5: fire weather!
        3. sum over the days for each year.
        
        NB: this definition of the fire season is independant from the definition used for overwintering DC.
        
        dat: DataArray
            daily FWI
            
        day_start: same type as time in dat
            day in dat for the beginning of the reference period

        day_end: same type as time in dat
            day in dat for the end of the reference period
            
        ref_xp: list of str
            experiment(s) to consider for the reference period
    '''
    fwi_min = dat.sel(time=slice(day_start,day_end), xp=ref_xp).min( ('xp','time') )
    fwi_max = dat.sel(time=slice(day_start,day_end), xp=ref_xp).max( ('xp','time') )
    fwi_norm = (dat - fwi_min) / (fwi_max - fwi_min)
    fws_days = xr.where( fwi_norm > 0.5, 1, 0)
    return fws_days.resample(time='1Y').sum()




def calc_indic_fwixd(dat, day_start, day_end, ref_xp, method):
    '''
        Calculate the number of extreme fire days.
        Two options:
            - using preset categories of the FWI
            - number of days above the 95th percentile: https://doi.org/10.1029/2018GL080959
        
        dat: DataArray
            daily FWI
            
        day_start: same type as time in dat
            day in dat for the beginning of the reference period

        day_end: same type as time in dat
            day in dat for the end of the reference period
            
        ref_xp: list of str
            experiment(s) to consider for the reference period
            
        method: str
            either 'category' or 'percentile'
    '''
    if 'category' in method:
        lvl = str.split( method, '-' )[1]
        lvl_danger = dangers_classes_fwi( dat, categ=lvl )

    elif 'percentile' in method:
        lvl = float( str.split( method, '-' )[1] )
        tmp = dat.sel(time=slice(day_start,day_end), xp=ref_xp)
        local_threshold = xr.where(np.isnan(tmp)==False, tmp, 0).quantile( q=lvl/100, dim=('xp','time') ).drop('quantile') # need to identify NaN because of overwintering
        lvl_danger = xr.where( (local_threshold <= dat), 1, 0 )
        
    return lvl_danger.resample(time='1Y').sum()




def calc_indic_fwisa(dat):
    '''
        Calculate the seasonal average of the FWI, defined as the annual peak 90-day mean.
        Similar to https://doi.org/10.1029/2018GL080959
        
        dat: DataArray
            daily FWI
    '''
    rollmean = xr.where(np.isnan(dat)==False, dat, 0).rolling(time=90, center=True).mean()# need to identify NaN because of overwintering
    return rollmean.resample(time='1Y').max()

    



        
def np_interpolate_nan_grid( array, lat_mesh, lon_mesh, method='linear' ):
    '''
        Fill in the NaN of a numpy array
        
            array: numpy array
                2D data having NaN
                
            lat_mesh: numpy array
                latitude with the same shape as array
                
            lon_mesh: numpy array
                longitude with the same shape as array
                
            method: str
                method for interpolation: 'nearest', 'linear', 'cubic'
    '''
    # mask invalid values
    array = np.ma.masked_invalid(array)

    # get only the valid values
    lat_mesh_mask = lat_mesh[~array.mask]
    lon_mesh_mask = lon_mesh[~array.mask]
    newarr = array[~array.mask]

    # interpolation
    return interpolate.griddata((lon_mesh_mask, lat_mesh_mask), newarr.ravel(), (lon_mesh, lat_mesh), method=method)
    
    
    
def xr_interpolate_nan_grid(data, method='linear'):
    '''
        Fill in the NaN in a DataArray using interpolation.
        
            data: DataArray
                Data having at least the coordinates 'lat' and 'lon'
                
            method: str
                method for interpolation: 'nearest', 'linear', 'cubic'
    '''
    
    # preparing coordinates
    la = np.arange(0, data.lat.size)
    lo = np.arange(0, data.lon.size)
    lo_mesh, la_mesh = np.meshgrid(lo, la)
    
    # making sure that coordinates are in the correct order, especially lat before lon
    data = data.transpose( 'time', 'lat', 'lon' )
    out = np.nan * xr.ones_like( data )
    
    # checking for the time axis
    if 'time' in data.dims:
        for t in data.time:
            out.loc[{'time':t}] = np_interpolate_nan_grid( array=data.sel(time=t).values, lat_mesh=la_mesh, lon_mesh=lo_mesh, method=method)
    else:
        out[...] = np_interpolate_nan_grid( array=data.values, lat_mesh=la_mesh, lon_mesh=lo_mesh, method=method)
        
    return out
            
    
    
    
def func_save_xp(full_data, xp, indic, path_save, attrs_out, check_NaN=True):
    '''
        Function saving the required data
            full_data: Dataset
                full dataset of outputs
            
            xp: str
                specific experiment to save
            
            indic: str
                specific annual indicator to save
            
            path_save: str
                full path where the file will be saved. Name of the file included.
                
            attrs_out: dict
                global attributes to transfer
                
            check_NaN: str
                what to do with NaN. If True, exception if find some. If False, interpolate to fill in these ones.
    '''
    
    # adapting the time axis
    type_time = type(full_data.time.values[0])
    if type_time in [np.datetime64]:
        new_time = [np.datetime64(str(yr)+'-07-01T06:00:00') for yr in full_data.time.dt.year.values]
    else:
        new_time = [type_time(yr, 7, 1, 6,0,0,0) for yr in full_data.time.dt.year.values]
    full_data.coords['time'] = new_time

    # preparing the correct period
    period = {**{'historical':(1850,2014), 'ssp534-over':(2040,)}, \
              **{ss:(2015,) for ss in ['ssp119', 'ssp126', 'ssp245', 'ssp370', 'ssp434', 'ssp460', 'ssp585']} }[xp]
    
    # getting required data
    y0 = full_data.time.dt.year.values[0]
    if len(period) == 2:
        data_save = full_data.sel(xp=xp).isel(time=slice(period[0]-y0, period[1]-y0+1)).drop('xp')
    else:
        end = full_data.time.dt.year.values[-1]
        # warning, if not all the xp have been extended, it will introduce NaN.
        data_save = full_data.sel(xp=xp).isel(time=slice(period[0]-y0, end-y0+1)).drop('xp')
        
    # must drop indicators not required. Watch out to keep other variables (eg bounds) for the regrid
    for ind in ['fwixx', 'fwils', 'fwixd','fwisa']:
        if (ind in data_save.variables) and (ind != indic):
            data_save = data_save.drop(ind)
    data_save = data_save.compute()

    # controlling data
    warning_nan = False
    for var in data_save.variables:
        if var not in data_save.coords:
            if np.any(np.isnan(data_save[var])):
                if (check_NaN == False) and (var in [indic]):
                    warning_nan = True
                    tmp = data_save[var].attrs
                    data_save[var] = xr_interpolate_nan_grid( data_save[var] , method='linear' )
                    data_save[var].attrs = tmp
                else:
                    raise Exception("NaN detected in"+var)

    # handling attributes
    data_save.attrs = attrs_out
            
    # MUST edit some attributes for regrid
    data_save.attrs['experiment_id'] = xp # dont forget it, we are saving each experiment separately, while the esm and member attributes are unchanged.
    data_save.attrs['variable_id'] = indic.lower()
    data_save.attrs['table_id'] = 'ann'
    data_save.attrs['grid_label'] = 'g025'
    if 'WARNING' in data_save.attrs:
        del data_save.attrs['WARNING']
    if warning_nan:
        data_save[indic].attrs['handling_NaN'] = 'Some points had NaN values, they have been filled using 2D linear interpolation.'

    # actually saving
    data_save.to_netcdf( path_save, encoding={var:{'zlib':True} for var in data_save.variables} )
    
    # cleaning
    data_save.close()
    del data_save
