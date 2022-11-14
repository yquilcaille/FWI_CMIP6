Scripts used for the computation of the Canadian Fire Weather Index (FWI) under data from the Climate Model Intercomparison Project Phase 6 (CMIP6). These scripts are developed in Python 3.7. The primary reason for sharing this code is replication by other users. No support of any kind is guaranteed, but feel free to ask. Any feedbacks and contributions are welcome.

Using this code is encouraged, but please acknowledge the authors.

This code comes into several components:
 * Computation of the FWI:
	- "_calc_FWI-CMIP6.py": calculates the FWI on CMIP6 runs.
	- "functions_calc_FWI.py": functions used for calculation of the FWI. The original algorithm for the FWI (Simard, 1970: https://cfs.nrcan.gc.ca/publications?id=36178) and has been updated in several codes. The basis of this code is pyfwi (https://github.com/buckinha/pyfwi). It has been adapted to handle arrays, not only scalars. Besides, some exceptions are now handled. The code has been extended by integrating improvements from NCAR (https://github.com/NCAR/fire-indices) and from cffdrs (https://rdrr.io/rforge/cffdrs).
	- "functions_load_CMIP6.py": functions used to load CMIP6 data.
	- "functions_support.py": functions used throughout components.
	- "available_files": repository that gather information on the runs. In particular, CMIP6 runs may have to undergo corrections on their reported grids. Worse cases, some CMIP6 runs have to be excluded.
	- "_run_calc_FWI-CMIP6.sh": simply a script to run an ensemble of jobs "_calc_FWI-CMIP6.py" on a server.
 * Regrid the FWI: the FWI from CMIP6 were regridded to a single grid using the CMIP6-ng processor (https://git.iac.ethz.ch/cmip6-ng/cmip6-ng), although adapted to integrate the variable FWI, which was not in CMIP6. Nevertheless, the following step can be performed on the native grid of the outputs without regridding. Otherwise, the user can use its own favorite tool for regridding (e.g. CDO).
 * Annual indicators of the FWI:
	- "_extract_annual_indicators.py": calculates annual indicators of the FWI from daily FWI.
	- "functions_calc_indicators.py": functions used for the calculation of annual indicators.
 * Mask for infrequent burning:
	- "_mask_infreq_burning.py": calculates the mask for chosen grids.
 * Plots of FWI on CMIP6 data:
	- "_plots_FWI.py": plots used for representation of the FWI in CMIP6 runs.
	- "functions_support_plot.py": functions used for the plots.


Requirements on packages: cartopy (0.20.2), cftime (1.6.1), matplotlib (3.5.2), numpy (1.23.1), pandas (1.4.3), regionmask (0.9.0), seaborn (0.11.2), xarray (2022.6.0)
