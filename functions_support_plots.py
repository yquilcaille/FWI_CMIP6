import os
import numpy as np
import xarray as xr
import csv
import warnings
from sklearn.linear_model import LinearRegression

import igraph
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import matplotlib.animation as anim
import matplotlib.colors as plcol
import matplotlib.gridspec as gridspec
import matplotlib.ticker as ticker
from matplotlib.text import Text
from mpl_toolkits.axes_grid1 import make_axes_locatable, inset_locator
from matplotlib import cm
import seaborn as sns ## for colors
import cartopy.crs as ccrs
CB_color_cycle = sns.color_palette( 'colorblind', n_colors=10000 )
import regionmask as regionmask

from functions_calc_FWI import *
from functions_load_CMIP6 import *
from functions_support import *





list_letters = ['(a)','(b)','(c)','(d)','(e)','(f)','(g)','(h)','(i)','(j)','(k)','(l)','(m)','(n)','(o)','(p)','(q)','(r)','(s)','(t)','(u)','(v)','(w)','(x)','(y)','(z)']



#============================================================================================
# TREE
#============================================================================================
class tree_FWI_CMIP6:
    '''
    The initialization of this class gathers all options used for calculation of FWI on CMIP6 data.
    'func_prepare_files' is then used for preparing all the files that will be used, such as the runs to use, the exceptions, etc.
    
    NB: Used during calculation of the FWI from CMIP6 data.
    '''
    #--------------------
    # INITIALIZATION    
    #--------------------
    def __init__(self, list_files):
        self.list_files = list_files
        self.list_files.sort()
        self.prep_runs()
    #--------------------
    #--------------------

    
    
    #--------------------
    # PREPARE RUNS
    #--------------------
    def prep_runs( self ):
        self.xp_avail_esms = matching_scenarios( self.list_files )

        # sorting some elements
        self.list_xps = list(self.xp_avail_esms.keys())
        self.list_xps.sort()
    #--------------------
    #--------------------

    
    
    #--------------------
    # POSITIONS OF NODES
    #--------------------
    def edit_radius_historical(self, XX, YY, fact_rad):
        XX_cp,YY_cp = np.copy(XX), np.copy(YY)
        new_pos = lambda fact, start, end: (XX[start] + fact * (XX_cp[end]-XX_cp[start])  ,  YY[start] + fact * (YY_cp[end]-YY_cp[start]))
        XX[self.dico_mem['historical']], YY[self.dico_mem['historical']] = new_pos( fact_rad, self.dico_mem['CMIP6'], self.dico_mem['historical'] )
        for esm in self.list_xps:
            if 'historical/'+esm in self.dico_mem:
                # esm
                start = self.dico_mem['historical']
                end = self.dico_mem['historical/'+esm]
                XX[end], YY[end] = new_pos( fact_rad, start, end )
                # memb
                start = self.dico_mem['historical/'+esm]
                end = self.dico_mem['historical/'+esm+'/'+'memb']
                XX[end], YY[end] = new_pos( 1, start, end )
        return XX,YY

    
    def edit_angle_ssp434(self, XX, YY):
        ind = self.labels.index('ssp434')
        cm = self.labels.index('CMIP6')
        angm1 = self.find_angle( XX[ind-1]-XX[cm], YY[ind-1]-XX[cm] )
        angp1 = self.find_angle( XX[ind+1]-XX[cm], YY[ind+1]-XX[cm] )
        R = np.sqrt( (XX[ind]-XX[cm])**2 + (YY[ind]-YY[cm])**2 )
        XX[ind] = XX[cm] + R*np.cos( (angm1+angp1/2) )
        YY[ind] = YY[cm] + R*np.sin( (angm1+angp1/2) )
        return XX, YY
    
    
    @staticmethod
    def find_angle( x, y ):
        angle = np.arccos( x / np.sqrt(x**2 + y**2) )
        if y < 0:
            angle *= -1
        return angle

    
    def calculate_positions_nodes(self, layout, fact_rad=1, margin_angle=5/360*2*np.pi, radius={'xps':1,'esms':1,'membs':1}, **args):
        # layout:: auto, bipartite, circle, dh, drl, drl_3d, fr, fr_3d, grid, grid_3d, graphopt, kk, kk_3d, lgl, mds, random, random_3d, rt, rt_circular, sphere, star, sugiyama
        # doing personal on the way if required
        
        # preparing tree
        self.labels = ['CMIP6'] # central node
        self.list_links = [] # tells which nodes are linked
        self.dico_mem,counter = { 'CMIP6':0 }, 0 # used to identify nodes
        if layout == 'personal':
            self.lay = [ [0,0] ] # will be coordinates of nodes

        # preparing first level
        for i_xp, xp in enumerate(self.list_xps):
            counter += 1
            self.dico_mem[xp] = counter
            self.list_links.append( (0,self.dico_mem[xp]) )
            self.labels.append( xp+'<br>('+str(sum([len(self.xp_avail_esms[xp][esm]) for esm in self.xp_avail_esms[xp]]))+')' )

        if layout == 'personal':
            # adding coordinates on first level: xps
            self.lay.append( self.func_coord(origin=self.lay[0], n_nodes=len(self.list_xps), radius=radius['xps'], available_angle=[0,2*np.pi]) )

        # preparing second level: esms
        for i_xp, xp in enumerate(self.list_xps):
            for i_esm, esm in enumerate(self.xp_avail_esms[xp].keys()):
                if esm in self.xp_avail_esms[xp]:
                    counter += 1
                    self.dico_mem[xp+'/'+esm] = counter
                    self.list_links.append( (self.dico_mem[xp], self.dico_mem[xp+'/'+esm]) )
                    self.labels.append( esm )
            if layout == 'personal':
                # adding coordinates on second level (esms) of this first node
                range_angle = self.find_range_angle( self.lay[1], i_xp, margin_angle )
                self.lay.append( self.func_coord(origin=self.lay[1][i_xp,:], n_nodes=int(len(self.xp_avail_esms[xp])), radius=radius['esms'], available_angle=range_angle) )

        # preparing third level: members
        for i_xp, xp in enumerate(self.list_xps):
            for i_esm, esm in enumerate(self.xp_avail_esms[xp].keys()):
                counter += 1
                self.dico_mem[xp+'/'+esm+'/'+'memb'] = counter
                self.list_links.append( (self.dico_mem[xp+'/'+esm], self.dico_mem[xp+'/'+esm+'/'+'memb']) )
                self.labels.append( str(len(self.xp_avail_esms[xp][esm])) )
                if layout == 'personal':
                    # adding coordinates on third level (members) of this second node
                    range_angle = self.find_range_angle( self.lay[1+1+i_xp], i_esm, margin_angle ) # will have only one output from here
                    self.lay.append( self.func_coord(origin=self.lay[1+i_xp+1][i_esm,:], n_nodes=1, radius=radius['membs'], available_angle=range_angle) )
                    
        # dealing with layout
        nr_vertices = len(self.list_links)+1
        G = igraph.Graph(self.list_links)
        if layout == 'personal':
            self.lay = np.vstack( self.lay )

        else:
            self.lay = G.layout(layout, **args) # lots of layouts to try

            if layout in ['rt_circular']:
                # edits
                XX = [self.lay[k][0] for k in range(len(self.lay))]
                YY = [self.lay[k][1] for k in range(len(self.lay))]
                # increasing radius on historical by a factor:
                XX, YY = self.edit_radius_historical(XX,YY,fact_rad)
                # shifting the angle for ssp434
                if False:
                    XX, YY = self.edit_angle_ssp434(XX,YY)
                # saving edited layout
                self.lay = [ [XX[k],YY[k]] for k in range(len(XX))]

        # positions
        positions = {k: self.lay[k] for k in range(nr_vertices)}
        #self.Msens = max( [self.lay[k][1] for k in range(nr_vertices)] )

        # edges
        es = igraph.EdgeSeq(G) # sequence of edges
        E = [e.tuple for e in G.es] # list of edges

        # final preparation
        self.Xn = [positions[k][0] for k in range(len(positions))]
        #self.Yn = [2*self.Msens-positions[k][1] for k in range(len(positions))]
        self.Yn = [positions[k][1] for k in range(len(positions))]
        self.Xe, self.Ye = [], []
        for edge in E:
            self.Xe += [positions[edge[0]][0], positions[edge[1]][0], None]
            #self.Ye+=[2*self.Msens-positions[ edge[0] ][1], 2*self.Msens-positions[ edge[1] ][1], None]
            self.Ye += [positions[edge[0]][1], positions[edge[1]][1], None]
    #--------------------
    #--------------------
    
    
    
    #--------------------
    # OWN FUNCTION FOR POSITIONS OF NODES
    #--------------------
    # following functions are only meant for my own layout
    def find_range_angle( self, coords, i_xp, margin_angle ):
        # given cartesian coordinates of the former level, given a point on this level, returns which angles to stay in for the next level starting at this point, while ensuring an angular margin with neighbors.

        # minimum angle of cone
        if i_xp == 0:
            min_angle = self.find_angle( x=coords[-1,0], y=coords[-1,1] )
        else:
            min_angle = self.find_angle( x=coords[i_xp-1,0], y=coords[i_xp-1,1] )

        # maximum angle of cone
        if i_xp == coords.shape[0]-1:
            max_angle = self.find_angle( x=coords[0,0], y=coords[0,1] )
        else:
            max_angle = self.find_angle( x=coords[i_xp+1,0], y=coords[i_xp+1,1] )

        # central angle of cone
        central_angle = self.find_angle( x=coords[i_xp,0], y=coords[i_xp,1] )

        # size_range
        size_range = [np.abs(max_angle-central_angle), np.abs(central_angle-min_angle)]
        for i in [0,1]:
            if size_range[i] > np.pi:
                size_range[i] = 2 * np.pi - size_range[i]

        # deduce range
        max_range = central_angle + 0.5*size_range[0]
        min_range = central_angle - 0.5*size_range[1]
        if min_range < max_range:
            return min_range+ 0.5*margin_angle, max_range - 0.5*margin_angle
        else:
            return max_range+ 0.5*margin_angle, min_range - 0.5*margin_angle


    @staticmethod
    def func_coord( origin, n_nodes, radius, available_angle=[0,2*np.pi], weights=None ):
        out = np.nan*np.ones( (n_nodes,2) )

        # handling weights
        if weights is None:
            weights = np.ones(n_nodes)
        tmp_weights = np.array(weights)

        # deduce angles
        angles = available_angle[0] + (available_angle[1] - available_angle[0]) * np.cumsum(tmp_weights) / sum(tmp_weights)

        # give simply cartesian coordinates of equally spaced points in a given portion of an angle
        for i in range(n_nodes):
            out[i,:] = [origin[0] + radius*np.cos(angles[i]) , origin[1] + radius*np.sin(angles[i])]
        return out
    #--------------------
    #--------------------
    
    
    
    #--------------------
    # ANNOTATIONS
    #--------------------
    def make_annotations(self, font_size, font_color='rgb(0,0,0)'):
        # central node
        annotations = [dict( text='CMIP6', x=self.Xn[0], y=self.Yn[0], xref='x',yref='y', font=dict(color=font_color, size=font_size['CMIP6']), textangle=0, showarrow=False) ]
        
        # nodes with links
        for k in range(len(self.list_links)):
            # start & end of the link
            start,end = self.list_links[k]
            
            # properties
            if self.labels[end] in self.list_xps:
                ftsz = font_size['scen']
                txt = self.labels[end]
            elif self.labels[end] in [self.xp_avail_esms[xp] for xp in self.list_xps]:
                ftsz = font_size['esm']
                txt = self.labels[end]
            else:
                ftsz = font_size['member']
                txt = "<b>"+self.labels[end]+"</b>"
            # rotation of the text
            ang = self.find_angle( x=self.Xn[end]-self.Xn[start], y=self.Yn[end]-self.Yn[start] ) * 360 / (2 * np.pi)
            # crazy angles in update_layout... (>_<)!
            if (-90 <= ang) and (ang < 90):
                ang = - ang
            else:
                ang = 180 - ang
            annotations.append( dict(text=txt, x=self.Xn[end], y=self.Yn[end], xref='x',yref='y', font=dict(color=font_color, size=ftsz), textangle=ang, showarrow=False) )
        return annotations
    # rotation_txt!        
    #--------------------
    #--------------------
    
    
    
    #--------------------
    # PLOT
    #--------------------
    def plot(self, figsize=(1000,1000), \
             colors={'lines':'rgb(200,200,200)', 'nodes':'rgb(250,180,30)', 'edges':'rgb(100,100,100)', 'background':'rgb(248,248,248)', 'text':'rgb(0,0,0)'}, \
             sizes={'dots':15, 'CMIP6':14, 'scen':12, 'esm':7, 'member':9} ):
        '''
            Plot the tree
            
            args:
                figsize: tuple, list or numpy.array
                    (width,height)
                    
                colors: dict
                    Colors for lines, nodes, edges, background, text
                    
                sizes: dict
                    Sizes for dots, CMIP6, scenarios, ESMs, ensemble members
        '''
        # ploting
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=self.Xe,
                           y=self.Ye,
                           mode='lines',
                           line=dict(color=colors['lines'], width=1),
                           hoverinfo='none'
                           ))
        fig.add_trace(go.Scatter(x=self.Xn,
                          y=self.Yn,
                          mode='markers',
                          marker=dict(symbol='circle',
                                        size=sizes['dots'],
                                        color=colors['nodes'],
                                        line=dict(color=colors['edges'], width=1)
                                        ),
                          text=self.labels,
                          hoverinfo='text',
                          opacity=0.8
                          ))

        # Actualizing plot
        axis = dict(showline=False, # hide axis line, grid, ticklabels and  title
                    zeroline=False,
                    showgrid=False,
                    showticklabels=False,
                    )

        self.annotations = self.make_annotations(font_size=sizes, font_color=colors['text'])
        fig.update_layout(annotations=self.annotations,
                          font_size=10, # not required, hiding this one
                          showlegend=False,
                          xaxis=axis,
                          yaxis=axis,
                          width=figsize[0],
                          height=figsize[1],
                          margin=dict(l=40, r=40, b=40, t=40),
                          hovermode='closest',
                          plot_bgcolor=colors['background']
                         )
        
        return fig
    #--------------------
    #--------------------
#============================================================================================
#============================================================================================













#============================================================================================
# CLASS FOR GLOBAL WARMING LEVELs
#============================================================================================
class GWL:
    '''
    The initialization of this class gathers all options used for calculation of FWI on CMIP6 data.
    'func_prepare_files' is then used for preparing all the files that will be used, such as the runs to use, the exceptions, etc.
    
    NB: Used during calculation of the FWI from CMIP6 data.
    '''
    #--------------------
    # INITIALIZATION    
    #--------------------
    def __init__(self):
        # preparation of extensions: used for rolling mean   and   average over maps
        self.dico_extensions = {'historical':'ssp245'} | {ss:'historical' for ss in ['ssp245', 'ssp370', 'ssp126', 'ssp585', 'ssp119', 'ssp460', 'ssp434']}
        self.dico_periods = {'historical':(1850,2014)} | {ss:(2015,2100) for ss in ['ssp245', 'ssp370', 'ssp126', 'ssp585', 'ssp119', 'ssp460', 'ssp434']}
    #--------------------
    #--------------------
    
    
    
    #--------------------
    # PREPARATION OF DATA CMIP6-ng
    #--------------------
    def prep_cmip6ng(self, files_cmip6ng, n_years, ref_period):
        self.files_cmip6ng = files_cmip6ng
        self.n_years = n_years
        self.ref_period = ref_period
        
        # loading the spatial average
        print('Loading the CMIP6-ng data to compute positions of warming levels')
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.DATA_cmip6ng = xr.open_mfdataset( self.files_cmip6ng, preprocess=self.func_preprocess_annual_spatial_average )
        
        print('Preparing the CMIP6-ng data to compute the positions of warming levels')
        self.concat_historical()
        self.rolling_mean()
        self.remove_ref_period()
        return
    

    @staticmethod
    def func_preprocess_annual_spatial_average(data):
        '''
            Short function for preprocessing of annual tas of CMIP6-ng, meant to be used either within 'xarray.open_mfdataset' or directly after loading a file.

            arguments:
                data: xarray.Dataset
                    annual indicator of the CMIP6 FWI
        '''
        # passing attributes as coordinates
        dico_match = {'experiment_id':'scen', 'variant_label':'member', 'source_id':'esm'}
        for var in dico_match:
            data = data.expand_dims( {dico_match[var]:[data.attrs[var]]} )
            del data.attrs[var]

        # transforming the type of time axis (not transformed previously according to cmip6-ng conventions)
        data.coords['time'] = data.time.dt.year

        # spatial average
        weights_area = np.cos(np.deg2rad(data.lat.values))

        # spatial average of maps
        data['tas'] = ( data['tas'] * weights_area[np.newaxis,:,np.newaxis] / (data.lon.size*sum(weights_area)) ).sum(('lat','lon'))
        return data

    def concat_historical(self):
        tm = slice(self.dico_periods['historical'][0],self.dico_periods['historical'][1])
        sc_nh = [scen for scen in self.DATA_cmip6ng.scen.values if scen != 'historical']
        for scen in sc_nh:
            self.DATA_cmip6ng['tas'].loc[{'scen':scen, 'time':tm}] = self.DATA_cmip6ng['tas'].sel(scen='historical',time=tm)
        self.DATA_cmip6ng = self.DATA_cmip6ng.sel(scen=sc_nh)

        
    def rolling_mean(self):
        if True:
            self.DATA_cmip6ng['tas'] = self.DATA_cmip6ng['tas'].rolling(time=self.n_years, center=True).mean('time')
        else:
            # running mean: must append scenarios together to avoid losing central years
            for scen in self.DATA_cmip6ng.scen.values:
                ext = self.dico_extensions[scen]
                time_ext = slice(self.dico_periods[ext][0], self.dico_periods[ext][1])
                time_scen = slice(self.dico_periods[scen][0], self.dico_periods[scen][1])
                # merging historical and ssp245   OR    historical and scenario
                tmp = xr.merge( [self.DATA_cmip6ng['tas'].sel(scen=scen,time=time_scen).drop('scen'), self.DATA_cmip6ng['tas'].sel(scen=ext,time=time_ext).drop('scen')] )
                # rolling mean
                self.DATA_cmip6ng['tas'].loc[{'scen':scen, 'time':time_scen}] = tmp['tas'].rolling(time=self.n_years, center=True).mean('time').sel(time=time_scen)
        return
    
    
    def remove_ref_period(self):
        # removing preindustrial
        if True:
            self.DATA_cmip6ng['tas'] -= self.DATA_cmip6ng['tas'].sel(time=slice(self.ref_period[0],self.ref_period[1])).mean('time')
        else:
            self.DATA_cmip6ng['tas'] -= self.DATA_cmip6ng['tas'].sel(time=slice(self.ref_period[0],self.ref_period[1]),scen='historical').mean('time').drop('scen')
        return
    #--------------------
    #--------------------

    
    
    #--------------------
    # Identifying dates where GWL are reached
    #--------------------
    def find_position_GWLs(self, list_GWLs, option_common_set=False):
        self.list_GWLs = list_GWLs
        
        print('Calculating positions of warming levels')
        self.DATA_cmip6ng.coords['GWL'] = list_GWLs
        self.DATA_cmip6ng['year_GWL'] = xr.DataArray(np.nan, coords={d:self.DATA_cmip6ng[d] for d in ['esm', 'member', 'scen', 'GWL']}, dims=('esm', 'member', 'scen', 'GWL') )
        self.dico_ind_GWL = {GWL:[] for GWL in self.list_GWLs}
        for GWL in self.list_GWLs:
            sel = (self.DATA_cmip6ng['tas'] - GWL) > 0.
            index_gwl = sel.argmax(dim='time').values
            ind = np.where( index_gwl>0 )
            added = []
            # creating a list of coordinates to use. Don't use indexes, because the order of load may differ in CMIP6-ng & FWI files, leading to different coordinates.
            for i in range(len(ind[0])):
                pos = {}
                for i_dd,dd in enumerate(self.DATA_cmip6ng['tas'].dims):
                    yr0 = self.DATA_cmip6ng['time'].values[ index_gwl[ind][i] ]
                    if dd == 'time':
                        pos[dd] = yr0
                    elif (dd == 'scen') and (yr0 <= self.dico_periods['historical'][1]): # due to the merge of historical
                        pos[dd] = 'historical'#self.DATA_cmip6ng[dd].values[ ind[i_dd][i] ]#  'historical' # /!\: KEEPING THAT ONE, OTHERWISE CAUSING ISSUES, PARTICULARLY WITH option_common_set
                    else:
                        pos[dd] = self.DATA_cmip6ng[dd].values[ ind[i_dd][i] ]
                run = pos['esm']+'/'+pos['scen']+'/'+pos['member']
                if run not in added:
                    added.append( run )
                    self.dico_ind_GWL[GWL].append( pos )
                    
        if option_common_set:
            # preparing
            lst = {GWL:[[pos['scen'], pos['esm'], pos['member']] for pos in self.dico_ind_GWL[GWL]] for GWL in self.list_GWLs}

            # common set: looping over each GWL to check that each ESM x scen x member is available in all GWLs:
            #common = [item for item in lst[GWL_FWI.list_GWLs[0]] if np.all( [item in lst[GWL] for GWL in GWL_FWI.list_GWLs] )] --> warning, problem with historical...
            common = []
            for pos in lst[self.list_GWLs[-1]]:
                test = True
                for GWL in self.list_GWLs:
                    if (pos not in lst[GWL])  and  (['historical', pos[1], pos[2]] not in lst[GWL]):
                        test = False
                if test:
                    common.append( pos )
            
            # new dico_ind_GWL: for each GWL, getting the correct year over the common set
            new = {GWL:[] for GWL in self.list_GWLs}
            for GWL in self.list_GWLs:
                # AGAIN the problem with historical: at high GWL, it is a scenario, at low it is historical
                for pos in self.dico_ind_GWL[GWL]:
                    if pos['time'] <= self.dico_periods['historical'][1]:
                        test = np.any( [scen, pos['esm'], pos['member']] in self.dico_ind_GWL[GWL] for scen in list(self.DATA_cmip6ng.scen.values)+['historical'] )
                    else:
                        test = [pos['scen'], pos['esm'], pos['member']] in common
                    if test:
                        new[GWL].append( pos )
            self.dico_ind_GWL = new
        return
    #--------------------
    #--------------------

    
    
    #--------------------
    # using GWLs
    #--------------------
    def apply_GWL( self, data_in, option_calc_on_GWL='mean' ):
        # checking which operation will be performed on maps of GWLs
        self.option_calc_on_GWL = option_calc_on_GWL
        
        # creating dataset that will be output:
        data_out = xr.Dataset()
        for coo in ['esm', 'member', 'scen', 'lat', 'lon']:
            data_out.coords[coo] = data_in[coo]
        data_out.coords['GWL'] = self.list_GWLs
        data_out['maps_GWL'] = xr.DataArray(np.nan, coords={d:data_out[d] for d in ['esm', 'member', 'scen', 'lat', 'lon', 'GWL']}, dims=('esm', 'member', 'scen', 'lat', 'lon', 'GWL') )

        for gwl in self.list_GWLs:
            for i_pos, pos in enumerate( self.dico_ind_GWL[gwl] ):
                print( 'Selecting maps for GWL '+str(gwl)+'K: '+str(i_pos+1)+'/'+str(len(self.dico_ind_GWL[gwl])), end='\r' )
                # identifying
                scen, esm, member, yr0 = pos['scen'], pos['esm'], pos['member'], pos['time']
                time_scen = np.arange(self.dico_periods[scen][0], self.dico_periods[scen][1]+1)

                # getting data
                if (yr0 - self.n_years/2 < time_scen[0])  or (yr0 + self.n_years/2-1 > time_scen[-1]):
                    # need an extension
                    ext = self.dico_extensions[scen]
                    time_ext = np.arange(self.dico_periods[ext][0], self.dico_periods[ext][1]+1)

                    # identifying smaller periods to load
                    if (yr0 - self.n_years/2 < time_scen[0]):# ssp scenario with part of the data to select BEFORE in historical
                        tt_ext = np.arange( yr0 - self.n_years/2, time_ext[-1]+1 )
                        tt_scen = np.arange( time_scen[0], yr0 + self.n_years/2 )
                    else:# historical with part of the data to select AFTER in ssp245
                        tt_scen = np.arange( yr0 - self.n_years/2, time_scen[-1]+1 )
                        tt_ext = np.arange( time_ext[0], yr0 + self.n_years/2 )

                    # selecting data
                    tmp = xr.merge( [data_in.sel(esm=esm, member=member, scen=scen, time=tt_scen).drop('scen'), data_in.sel(esm=esm, member=member, scen=ext,time=tt_ext).drop('scen')] )[data_in.name]

                else:
                    # no need for extension: directly selecting data
                    tmp = data_in.sel(esm=esm, member=member, scen=scen,time=slice(yr0 - self.n_years/2, yr0 + self.n_years/2-1) ).drop('scen')

                # average over time:
                if option_calc_on_GWL == 'mean':
                    data_out['maps_GWL'].loc[{'GWL':gwl, 'esm':esm, 'scen':scen, 'member':member}] = tmp.mean('time') # ('lat','lon')
                elif ('percentile' in self.option_calc_on_GWL):
                    lvl = float(self.option_calc_on_GWL[:-len('percentile')])
                    data_out['maps_GWL'].loc[{'GWL':gwl, 'esm':esm, 'scen':scen, 'member':member}] = tmp.quantile( q=lvl/100, dim=('time') ).drop('quantile') # ('lat','lon')
                else:
                    raise Exception("Unprepared option_calc_on_GWL")
            print( 'Selecting maps for GWL '+str(gwl)+'K: '+str(i_pos+1)+'/'+str(len(self.dico_ind_GWL[gwl])), end='\n' )
                
        return data_out
    #--------------------
    #--------------------
#============================================================================================
#============================================================================================









#============================================================================================
# CLASS TO REPRESENT MAPS WITH UNCERTAINTIES ACCORDING TO IPCC STANDARDS
#============================================================================================
class maps_mean_uncertainties:
    '''
        evaluate internal variability:
         - from preindustrial control
           - detrend the pre-industrial control using quadratic fit
           - calculating its local standard deviation of 20-year mean over non-overlapping periods in preindustrial control: sigma_20yr of the ESM x member
         - if preindustrial not available:
           - interannual standard deviation over linearly detrend moder periods: sigma_1yr of the ESM x member
           - sigma_20yr = sigma_1yr / sqrt(20)
        deduce internal variability as: gamma = sqrt(2) * 1.645 * sigma_20yr ---> depends on the ESM x member
        if more than 66% of models have a change greater than gamma:
         -yes- if more than 80% of models agree on the signe of the change:
           -yes-> Robust signal: colour only, nothing else
           -no -> Conflicting signal: colour and crossed lines
         -no:
           --> No change or no robust signal: colour and Reverse diagonal 
        (dont say hatching, but diagonal lines for non-expert audiences)
        (include these patterns in the legend)
    
    '''
    #--------------------
    # INITIALIZATION    
    #--------------------
    def __init__(self):
        # nothing to initialize
        pass
    #--------------------
    #--------------------


    
    #--------------------
    # TESTS ON DATA
    #--------------------
    def eval_robust_certain( self, map_ref, maps_to_plot, dim_plot, approach_ipcc, data_gamma, limit_certainty_members, limit_certainty_ESMs, limit_robustness_members, limit_robustness_ESMs, mask_map=None ):
        # archiving
        self.approach_ipcc = approach_ipcc
        self.mask_map = mask_map
        self.reference_period = (map_ref.time.values[0],map_ref.time.values[-1])
        self.map_ref = map_ref.mean('time')
        self.maps_to_plot = maps_to_plot
        self.dim_plot = dim_plot
        
        # check where are the runs that will be ploted
        print('Verifying which runs to plot')
        self.where_runs = { val_dim: self.check_where_runs( maps_to_plot.loc[{self.dim_plot:val_dim}] ) for val_dim in maps_to_plot[self.dim_plot].values }
        
        # checking where the signs are robust
        self.robust_change = self.is_robust( limit_robustness_members, limit_robustness_ESMs )
        
        # checking where the changes are certain
        if approach_ipcc == 'C':
            self.calc_gamma( data_in=data_gamma , mask_map=mask_map )
            self.certain = self.is_certain( limit_certainty_members, limit_certainty_ESMs )

        # masking
        if self.mask_map is not None:
            self.robust_change = xr.where( self.mask_map, self.robust_change, np.nan )
            if approach_ipcc == 'C':
                self.certain = xr.where( self.mask_map, self.certain, np.nan )
        
    
    @staticmethod
    def check_where_runs( map_check ):
        vals00 = map_check.isel(lat=0,lon=0).compute()
        out = {}
        for esm in map_check.esm.values:
            for scen in map_check.scen.values:
                for member in map_check.member.values:
                    if np.isnan( vals00.sel(esm=esm, scen=scen, member=member).values )==False:
                        if esm not in out:
                            out[esm] = {}
                        if scen not in out[esm]:
                            out[esm][scen] = []
                        out[esm][scen].append( member )
        return out

    
    @staticmethod
    def handle_test( test, where_runs, val_dim, limit_members, limit_ESMs ):
        # selecting only the ESMs where there is something to plot here:
        esms = list( where_runs[val_dim].keys() )

        # For each available ESM, what is the fraction of scenarios x members that respects the test?
        tot_numb = xr.DataArray( 0, coords={'esm':test['esm'].sel(esm=esms)}, dims=('esm',))
        for esm in esms:
            for scen in where_runs[val_dim][esm].keys():
                tot_numb.loc[{'esm':esm}] += len(where_runs[val_dim][esm][scen])
        fraction_members = test.sel(esm=esms).sum( ('scen','member') ) / tot_numb

        # For each ESM, if more than a threshold of its members agrees on the test, takes it as True for the ESM.
        fraction_ESMs = fraction_members > limit_members
        
        # If more than a threshold of the ESMs agrees on the test, takes it as True in the grid cell of this map.
        return fraction_ESMs.sum('esm')/len(esms) > limit_ESMs
    
    
    def is_certain( self, limit_certainty_members=0.66, limit_certainty_ESMs=0.66 ):
        certain = xr.DataArray( np.nan, coords={dd:self.maps_to_plot[dd].values for dd in [self.dim_plot, 'lat', 'lon']}, dims=(self.dim_plot, 'lat', 'lon') )
        
        for val_dim in self.maps_to_plot[self.dim_plot].values:
            print('Verifying where the signal is certain in '+self.dim_plot+': '+str(val_dim), end='\r')
            # is the change from the reference to the evaluated map above the internal variability threshold?
            test = (self.maps_to_plot.loc[{self.dim_plot:val_dim}] - self.map_ref) > self.gamma
            certain.loc[{self.dim_plot:val_dim}] = self.handle_test( test, self.where_runs, val_dim, limit_certainty_members, limit_certainty_ESMs )
            
        print('Verifying where the signal is certain in '+self.dim_plot+': done.', end='\n')
            
        return certain

    
    def is_robust( self, limit_robustness_members=0.80, limit_robustness_ESMs=0.80 ):
        robustness = xr.DataArray( np.nan, coords={dd:self.maps_to_plot[dd].values for dd in [self.dim_plot, 'lat', 'lon']}, dims=(self.dim_plot, 'lat', 'lon') )
        
        for val_dim in self.maps_to_plot[self.dim_plot].values:
            print('Verifying where the changes are robust in '+self.dim_plot+': '+str(val_dim), end='\r')
            # is the change from the reference to the evaluated map positive?
            test = (self.maps_to_plot.loc[{self.dim_plot:val_dim}] - self.map_ref) > 0
            robustness.loc[{self.dim_plot:val_dim}] = self.handle_test( test, self.where_runs, val_dim, limit_robustness_members, limit_robustness_ESMs )
        print('Verifying where the changes are robust in '+self.dim_plot+': done                ', end='\n')
            
        return robustness
    #--------------------
    #--------------------
    
    
    #--------------------
    # INTERNAL VARIABILITY
    #--------------------
    def calc_gamma( self, data_in ):
        # just saving the period
        self.period_calc_gamma = data_in.time.values[0], data_in.time.values[-1]
        if self.period_calc_gamma[0] < 1850 or self.period_calc_gamma[1] > 2014:
            raise Exception('Select the period within 1850-2014')
        
        self.calc_sigma_1yr( data_in )
        
        # internal variability threshold
        self.gamma = np.sqrt( 2 / 20 ) * 1.645 * self.sigma_1yr

        
    def calc_sigma_1yr( self, data_in ):
        print('Calculating standard deviation over '+str(self.period_calc_gamma[0])+'-'+str(self.period_calc_gamma[1]))
        
        # preparing output variable
        self.sigma_1yr = xr.DataArray( np.nan, coords={'esm':data_in.esm.values,'member':data_in.member.values, 'lat':data_in.lat.values, 'lon':data_in.lon.values}, dims=('esm','member', 'lat', 'lon') )
        
        # preparing linear detrend
        x = np.vstack([ np.ones(self.period_calc_gamma[1]-self.period_calc_gamma[0]+1), data_in.time.values ]).T
        y_all = data_in.compute()
        
        # looping on checking where need
        for esm in data_in.esm:
            for member in data_in.member:
                if np.any(np.isnan( y_all.sel(esm=esm,member=member).values ))==False:
                    y = y_all.sel(esm=esm,member=member).transpose( 'time','lat','lon' ).values
                    shp_y = y.shape

                    # reshaping and linear trend
                    y = y.reshape( (shp_y[0], shp_y[1]*shp_y[2]) )
                    local_trends = LinearRegression(fit_intercept=False).fit(x, y)
                    y_fit = local_trends.predict( x )
                    y = y.reshape( (shp_y[0], shp_y[1], shp_y[2]) )
                    y_fit = y_fit.reshape( (shp_y[0], shp_y[1], shp_y[2]) )

                    # linear detrend and standard deviation
                    self.sigma_1yr.loc[{'esm':esm, 'member':member}] = np.std(y - y_fit, axis=0) # ('lat','lon')
                    
        # masking
        if self.mask_map is not None:
            self.sigma_1yr *= self.mask_map
    #--------------------
    #--------------------
    
    
    
    #--------------------
    # PLOT
    #--------------------
    def plot(self, fig, spec, ind_row, do_title, unit_dim, label_row, plot_diff_GWLs=False, \
             vmin=None, vmax=None, levels_cmap=9, fontsize={'colorbar':14, 'title':14, 'legend':12, 'label_row':14}, density_visual_code={'x':2,'\\':2,'/':2}, margin_colorbar_pct=1, legend_val_dim=None):
        # overall preparation
        counter_letter = ind_row * len(self.maps_to_plot[self.dim_plot]) + (ind_row!=0)*1
        
        # preparation
        lon_mesh, lat_mesh = np.meshgrid(self.maps_to_plot.lon.values, self.maps_to_plot.lat.values)
        self.data_plot = {}
        self.esms_plot = {}
        self.tot_numb_plot = {}

        # the first column is the reference map
        ax = plt.subplot( spec[ind_row,0], projection=ccrs.Robinson() )

        # preparing data, computing ONLY required ones:
        self.data_plot['ref'], self.esms_plot['ref'], self.tot_numb_plot['ref'] = self.average_map(dat_in=self.map_ref, val_dim=self.maps_to_plot[ self.dim_plot ].values[0], option_multiple_scens=False )
        for val_dim in self.maps_to_plot[ self.dim_plot ].values:
            # preparing data, computing ONLY required ones:
            self.data_plot[val_dim], self.esms_plot[val_dim], self.tot_numb_plot[val_dim] = self.average_map(dat_in=self.maps_to_plot.loc[{self.dim_plot:val_dim}], val_dim=val_dim, option_multiple_scens=True )
        
        # preparing figure
        ax.coastlines()
        cmaps = self.prepare_colormaps(vmin, vmax, levels_cmap, margin_colorbar_pct, plot_diff_GWLs)
       
        # ploting mean signal
        pmesh = ax.pcolormesh(lon_mesh, lat_mesh, self.data_plot['ref'], transform=ccrs.PlateCarree(), rasterized=True, cmap=cmaps['cmap'], vmin=cmaps['vmin'], vmax=cmaps['vmax'] )
        if plot_diff_GWLs:
            cbar = plt.colorbar(pmesh,fraction=0.05, pad=0.04, orientation='vertical')# fraction=0.0235, default 0.15
            cbar.ax.tick_params(labelsize=fontsize['colorbar'])

        # finishing subplot
        ax.grid()
        if do_title:plt.title( str(self.reference_period[0])+'-'+str(self.reference_period[1])+'\n('+str(self.tot_numb_plot['ref'])+' runs over '+str(len(self.esms_plot['ref']))+' ESMs)' , size=fontsize['title'])
        ax.text(-0.05, 0.55, s=label_row, fontdict={'size':fontsize['label_row']}, color='k', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes)
        plt.text( x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.90*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter_letter],fontdict={'size':0.8*fontsize['title']} )
        counter_letter += 1

        # each other column will be a different value of dim_plot
        for i_dim, val_dim in enumerate( self.maps_to_plot[ self.dim_plot ].values ):
            ax = plt.subplot( spec[ind_row,1+i_dim], projection=ccrs.Robinson() )

            # preparing figure
            ax.coastlines()

            if plot_diff_GWLs == False:
                # ploting signal
                pmesh = ax.pcolormesh(lon_mesh, lat_mesh, self.data_plot[val_dim], transform=ccrs.PlateCarree(), rasterized=True, cmap=cmaps['cmap'], vmin=cmaps['vmin'], vmax=cmaps['vmax'] )
            else:
                # ploting difference of signal to preindustrial
                pmesh = ax.pcolormesh(lon_mesh, lat_mesh, self.data_plot[val_dim]-self.data_plot['ref'], transform=ccrs.PlateCarree(), rasterized=True, cmap=cmaps['cmap_diff'], vmin=cmaps['vmin_diff'], vmax=cmaps['vmax_diff'] )
            # ghost plot for legend
            plt.fill_between( lon_mesh[0,:], np.nan*lon_mesh[0,:], np.nan*lon_mesh[0,:], facecolor='darkorchid', edgecolor=None, label='Robust signal')

            if self.approach_ipcc == 'B':
                # adding reverse diagonal where no change or no robust signal
                ax.contourf(lon_mesh, lat_mesh, xr.where( (self.robust_change==0).loc[{self.dim_plot:val_dim}], 1, np.nan), transform=ccrs.PlateCarree(), colors='none', hatches=[density_visual_code['/']*'/'])
                # ghost plot for legend
                plt.fill_between( lon_mesh[0,:], np.nan*lon_mesh[0,:], np.nan*lon_mesh[0,:], facecolor=None, edgecolor=None, alpha=0, hatch=density_visual_code['/']*'/', label='Low model agreement')
                
            elif self.approach_ipcc == 'C':
                # adding reverse diagonal where no change or no robust signal
                ax.contourf(lon_mesh, lat_mesh, xr.where( (self.certain==0).loc[{self.dim_plot:val_dim}], 1, np.nan), transform=ccrs.PlateCarree(), colors='none', hatches=[density_visual_code['\\']*'\\'])
                # ghost plot for legend
                plt.fill_between( lon_mesh[0,:], np.nan*lon_mesh[0,:], np.nan*lon_mesh[0,:], facecolor=None, edgecolor=None, alpha=0, hatch=density_visual_code['\\']*'\\', label='No change or no robust signal')

                # adding crossed lines where conflicting signals
                ax.contourf(lon_mesh, lat_mesh, xr.where( ((self.certain==1) * (self.robust_change==0)).loc[{self.dim_plot:val_dim}], 1, np.nan), transform=ccrs.PlateCarree(), colors='none', hatches=[density_visual_code['x']*'x'])
                # ghost plot for legend
                plt.fill_between( lon_mesh[0,:], np.nan*lon_mesh[0,:], np.nan*lon_mesh[0,:], facecolor=None, edgecolor=None, alpha=0, hatch=density_visual_code['x']*'x', label='Conflicting signal')

            # finishing subplot
            if i_dim == len(self.maps_to_plot[self.dim_plot].values)-1:
                cbar = plt.colorbar(pmesh,fraction=0.05, pad=0.04, orientation='vertical')# fraction=0.0235, default 0.15
                cbar.ax.tick_params(labelsize=fontsize['colorbar'])
            gridlines = ax.gridlines(draw_labels=False,zorder=0)
            if do_title:plt.title( self.dim_plot+' = '+str(val_dim)+' '+unit_dim+'\n('+str(self.tot_numb_plot[val_dim])+' runs over '+str(len(self.esms_plot[val_dim]))+' ESMs)' , size=fontsize['title'])
            if (legend_val_dim is not None) and (legend_val_dim==val_dim):plt.legend(loc='center',bbox_to_anchor=[0.5, -0.11], prop={'size':fontsize['legend']}, ncol=3)
            plt.text( x=ax.get_xlim()[0]+0.05*(ax.get_xlim()[1]-ax.get_xlim()[0]),y=ax.get_ylim()[0]+0.90*(ax.get_ylim()[1]-ax.get_ylim()[0]),s=list_letters[counter_letter],fontdict={'size':0.8*fontsize['title']} )
            counter_letter += 1
            
            
    def average_map( self, dat_in, val_dim, option_multiple_scens ):
        esms = list( self.where_runs[val_dim].keys() )
        TMP = xr.DataArray( np.nan, coords={'esm':esms, 'lat':dat_in['lat'].values, 'lon':dat_in['lon'].values}, dims=('esm', 'lat', 'lon') )
        tot_numb = 0
        for esm in esms:
            
            if option_multiple_scens:
                scens = list( self.where_runs[val_dim][esm].keys() )
                tmp = xr.DataArray( np.nan, coords={'scen':scens, 'lat':dat_in['lat'].values, 'lon':dat_in['lon'].values}, dims=('scen', 'lat', 'lon') )
                for scen in scens:
                    # averaging over members
                    tmp.loc[{'scen':scen}] = dat_in.loc[{'esm':esm, 'scen':scen, 'member':self.where_runs[val_dim][esm][scen]}].mean('member')
                    tot_numb += len(self.where_runs[val_dim][esm][scen])
                # averaging over scenarios
                TMP.loc[{'esm':esm}] = tmp.mean('scen')
                
            else:
                # averaging over members, but not on scenarios. Example: for reference map, just historical.
                members = list(set([ memb for scen in self.where_runs[val_dim][esm].keys() for memb in self.where_runs[val_dim][esm][scen]] ))
                TMP.loc[{'esm':esm}] = dat_in.loc[{'esm':esm, 'member':members}].mean('member') # no scen dimension, because in historical
                tot_numb += len(members)
                
        # averaging over ESMs
        dat_out = TMP.mean('esm')
        
        # masking
        if self.mask_map is not None:
            dat_out = xr.where( self.mask_map, dat_out, np.nan )
            
        return dat_out, esms, tot_numb
    
    
    def prepare_colormaps(self, vmin, vmax, levels_cmap, margin_colorbar_pct, plot_diff_GWLs):
        cmap = cm.get_cmap('Reds', levels_cmap)
        if plot_diff_GWLs == False:
            if vmin is None:
                vmin = min( [np.nanpercentile( self.data_plot['ref'], margin_colorbar_pct)] + [np.nanpercentile(self.data_plot[val_dim], margin_colorbar_pct) for val_dim in self.maps_to_plot[self.dim_plot].values] )
            if vmax is None:
                vmax = max( [np.nanpercentile( self.data_plot['ref'], 100-margin_colorbar_pct)] + [np.nanpercentile(self.data_plot[val_dim], 100-margin_colorbar_pct) for val_dim in self.maps_to_plot[self.dim_plot].values] )
            return {'cmap':cmap, 'vmin':vmin, 'vmax':vmax}
            
        else:
            if vmin is None:
                vmin = np.nanpercentile( self.data_plot['ref'], margin_colorbar_pct)
                vmin_diff = min( [np.nanpercentile(self.data_plot[val_dim]-self.data_plot['ref'], margin_colorbar_pct) for val_dim in self.maps_to_plot[self.dim_plot].values] )
                cmap_diff = cm.get_cmap('Reds', levels_cmap)
            if vmax is None:
                vmax = np.nanpercentile( self.data_plot['ref'], 100-margin_colorbar_pct)
                vmax_diff = max( [np.nanpercentile(self.data_plot[val_dim]-self.data_plot['ref'], 100-margin_colorbar_pct) for val_dim in self.maps_to_plot[self.dim_plot].values] )
                cmap_diff = cm.get_cmap('Reds', levels_cmap)

            # finding better ones
            order_mag = max( [np.int32(np.log10(np.abs(vmin_diff))), np.int32(np.log10(np.abs(vmax_diff))) ])
            vmin_diff = np.round( vmin_diff, 1-order_mag )
            vmax_diff = np.round( vmax_diff, 1-order_mag )
            if vmin_diff < 0:
                size_bins = np.round( (vmax_diff-vmin_diff) / levels_cmap, 1-order_mag)
                vmin_diff = -size_bins * (np.round(-vmin_diff/size_bins,0) + 0.5)
                vmax_diff = size_bins * (np.round(vmax_diff/size_bins,0) + 0.5)
                cvals  = [np.floor(vmin_diff), 0, (np.ceil(vmax_diff)+1)/2, np.ceil(vmax_diff)+1]
                colors = ["green", "white", "orchid", "darkorchid"]
                levels_cmap = np.int32( np.ceil( (vmax_diff-vmin_diff)/size_bins ) ) - 1
            else:
                cvals  = [0, (np.ceil(vmax_diff)+1)/2, np.ceil(vmax_diff)+1]
                colors = ["white", "orchid", "darkorchid"]
                size_bins = np.round( vmax_diff / levels_cmap, 1-order_mag)
                vmin_diff = -0.5*size_bins
                levels_cmap = np.int32( np.ceil( vmax_diff/size_bins ) )
                vmax_diff = (levels_cmap+0.5) * size_bins

            norm_diff=plt.Normalize(min(cvals),max(cvals))
            cmap_diff = plcol.LinearSegmentedColormap.from_list("new_colormap", list(zip(map(norm_diff,cvals), colors)),levels_cmap+1)
            
            return {'cmap':cmap, 'vmin':vmin, 'vmax':vmax, 'cmap_diff':cmap_diff, 'vmin_diff':vmin_diff, 'vmax_diff':vmax_diff}
    #--------------------
    #--------------------
#============================================================================================
#============================================================================================










#============================================================================================
# ADVANCED FUNCTIONS FOR PLOTS - VERY SPECIFIC
#============================================================================================
def plot_maps_timeseries( DATA, indic, xps, list_years, lat_bands, axis_comparison, value_ref, name_figure, window_average = 20 ):
    '''
        Function plotting map at selected years with running mean with average & standard deviation and below timeseries with running mean over different bands of latitudes

        args:
            DATA: xr.Dataset
                Data to plot

            indic: str
                Name of the indicator

            list_years: list
                list of the years to plot

            lat_bands: dictionary
                dict with keys as names of bands and values the boundaries of the bands

            axis_comparison: str
                'member' or 'esm':axis over which mean and deviation will be computed => switch type of figure

            value_ref: str
                if axis_comparison is 'esm', must be a member run by enough ESMs (eg 'r1i1p1f1'). Otherwise, must be an ESM with enough members (eg 'CanESM5' or 'MIROC6').

            name_figure: str
                Well, the name of the figure.
                
            window_average: int or float
                left to right width of the window for the running mean

    '''
    
    # Checking over years
    years_bnds = {}
    for yr in list_years:
        if yr - int(window_average/2) < DATA.time.values[0]:
            years_bnds[yr] = (DATA.time.values[0], DATA.time.values[0]+window_average-1)
        elif yr + int(window_average/2) > DATA.time.values[-1]:
            years_bnds[yr] = (DATA.time.values[-1]-window_average+1, DATA.time.values[-1])
        else:
            years_bnds[yr] = (yr-int(window_average/2), yr+int(window_average/2)-1)
    
    # axis of reference
    axis_ref = {'esm':'member', 'member':'esm'}[axis_comparison]
    
    # Calculation over years
    dic_TMP_yrs = {}
    for i_yr, year in enumerate(list_years):
        scen = xps[1] if year > 2014 else xps[0]
        TMP = DATA[indic].sel(scen=scen,time=slice(years_bnds[year][0],years_bnds[year][1])).loc[{axis_ref:value_ref}].mean('time')
        plot_mean = TMP.mean( (axis_comparison) )
        plot_stdd = TMP.std( (axis_comparison) )
        dic_TMP_yrs[year] = [plot_mean,plot_stdd]
    vmin_mean = max([np.nanpercentile( TMP[0], 5) for TMP in dic_TMP_yrs.values()])
    vmax_mean = max([np.nanpercentile( TMP[0], 95) for TMP in dic_TMP_yrs.values()])
    vmin_stdd = max([np.nanpercentile( TMP[1], 5) for TMP in dic_TMP_yrs.values()])
    vmax_stdd = max([np.nanpercentile( TMP[1], 95) for TMP in dic_TMP_yrs.values()])

    # adding spatial information for timeseries on bands of latitudes
    reg = regionmask.defined_regions.ar6.land # regionmask.defined_regions.natural_earth_v5_0_0.land_10
    mask_reg = mask_percentage( reg, DATA.lon.values, DATA.lat.values )
    weights_area = np.cos(np.deg2rad(DATA.lat.values))
    weights = np.sum(mask_reg,axis=0) * weights_area[:,np.newaxis]

    # Calculation over bands of latitude
    dic_TMP_lat = {}
    for i_band,band in enumerate(lat_bands):
        # selecting over band of latitudes
        i_lat_min = np.argmin( np.abs(DATA.lat.values - lat_bands[band][0]) )
        i_lat_max = np.argmin( np.abs(DATA.lat.values - lat_bands[band][1]) )
        TMP = weights[i_lat_min:i_lat_max,:] * DATA[indic].loc[{axis_ref:value_ref}].isel( lat=slice(i_lat_min, i_lat_max) )
        TMP = TMP.sum(('lat','lon')) / np.sum( weights[i_lat_min:i_lat_max,:] )
        TMP = xr.concat( [TMP.sel(scen=xps[0],time=slice(1850,2014)), TMP.sel(scen=xps[1],time=slice(2015,2100))], dim='time' )
        TMP = TMP.rolling(time=window_average, center=True).mean()
        dic_TMP_lat[band] = TMP
    xmin = min([TMP.min() for TMP in dic_TMP_lat.values()])
    xmax = max([TMP.max() for TMP in dic_TMP_lat.values()])


    # PLOT
    # Properties of plot
    width_figure = 20
    wspace, hspace = 0.1,0.03
    height_figure = (width_figure/3) * (2+len(lat_bands)) / len(list_years) * (1+wspace)/(1+hspace)
    size_text = 20 * np.sqrt(height_figure/22)
    txt_mode = {'member':'ensemble members', 'esm':'ESMs'}[axis_comparison]

    # preparing figure
    fig = plt.figure(figsize=(width_figure,height_figure))
    spec = gridspec.GridSpec(ncols=len(list_years), nrows=2+len(lat_bands), figure=fig, height_ratios=[1,1]+[1/2]*len(lat_bands), left=0.05, right=0.95, bottom=0.025, top=0.975, wspace=wspace, hspace=hspace )
    plt.suptitle( dico_indics[indic]+' of '+value_ref+' accross '+txt_mode, size=size_text, y=0.999 )

    # figure: years
    for i_yr, year in enumerate(list_years):
        [plot_mean,plot_stdd] = dic_TMP_yrs[year]
        scen = xps[1] if year > 2014 else xps[0]

        # mean
        ax = plt.subplot( spec[0,i_yr], projection=ccrs.Robinson() )
        func_map( plot_mean.values, ax, spatial_info=DATA, type_plot='default', fontsize_colorbar=size_text*0.9, vmin=vmin_mean, vmax=vmax_mean )
        plt.title( str(years_bnds[year][0])+'-'+str(years_bnds[year][1])+' ('+scen+')', size=size_text )
        if i_yr == 0:ax.text(-0.05, 0.55, s='Average over '+txt_mode, fontdict={'size':size_text}, color='k', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes)

        # std
        ax = plt.subplot( spec[1,i_yr], projection=ccrs.Robinson() )
        plt.title( str(years_bnds[year][0])+'-'+str(years_bnds[year][1])+' ('+scen+')', size=size_text )
        func_map( plot_stdd.values, ax, spatial_info=DATA, type_plot='default', fontsize_colorbar=size_text*0.9, vmin=vmin_stdd, vmax=vmax_stdd )
        if i_yr == 0:ax.text(-0.05, 0.55, s='Deviation over '+txt_mode, fontdict={'size':size_text}, color='k', va='bottom', ha='center', rotation='vertical', rotation_mode='anchor', transform=ax.transAxes)

    # figure: lat
    for i_band,band in enumerate(lat_bands):
        ax = plt.subplot( spec[2+i_band,:] )
        plot_mean = dic_TMP_lat[band].mean( (axis_comparison) )
        plot_stdd = dic_TMP_lat[band].std( (axis_comparison) )
        # plotting
        plt.plot( TMP.time, dic_TMP_lat[band].values.T, lw=1, color='gray',zorder=1 )
        plt.plot( TMP.time, np.nan * dic_TMP_lat[band].loc[{axis_comparison:TMP[axis_comparison][0]}].values, lw=1, color='gray', label=txt_mode,zorder=1 )
        plt.plot( TMP.time, plot_mean, lw=2, color='black', label='average',zorder=2 )
        plt.fill_between( TMP.time, (plot_mean-plot_stdd).values, (plot_mean+plot_stdd).values, lw=0, facecolor=cols_scen[scen], edgecolor=None, alpha=0.5, label='$\pm$ 1 standard deviation',zorder=3 )

        # improving plot
        plt.grid(zorder=0)
        plt.ylim( np.max([0,xmin-0.05*(xmax-xmin)]), xmax+0.05*(xmax-xmin) )
        plt.legend(loc=0, prop={'size':size_text*0.8})
        plt.xticks( size=size_text*0.9 )
        plt.yticks( size=size_text*0.9 )
        plt.xlim( [1850+int(window_average/2),2100-int(window_average/2)] )
        if i_band != len(lat_bands)-1:
            ax.tick_params(axis='x',label1On=False)
        plt.ylabel( band, size=size_text )

    # save
    fig.savefig(path_save_plotsFWI+'/'+name_figure)
    return fig
#============================================================================================
#============================================================================================


















#============================================================================================
# PREPARING OTHER FUNCTIONS FOR PLOTS, +/- GENERIC
#============================================================================================
# generic function to transform a list of FWI files into a matching of scenarios
def matching_scenarios( list_files ):
    # preparing matching of scenarios
    xp_avail_esms = {}
    for file_W in list_files:
        var, _, esm, xp, memb, grid = str.split( file_W[:-len('.nc')], '_' ) # no need for "time_res"
        if xp not in xp_avail_esms:
            xp_avail_esms[xp] = {}
        if esm not in xp_avail_esms[xp]:
            xp_avail_esms[xp][esm] = []
        if memb not in xp_avail_esms[xp][esm]:
            xp_avail_esms[xp][esm].append(memb)
            
    return xp_avail_esms


# generic function to plot map
def func_map( data_plot, ax, fontsize_colorbar, spatial_info, type_plot='default', vmin=None, vmax=None, n_levels=100 ):
    # preparing figure
    ax.coastlines()
    lon_mesh, lat_mesh = np.meshgrid(spatial_info.lon.values, spatial_info.lat.values)
    if type_plot == 'default':
        cmap = cm.get_cmap('Reds', n_levels)
    elif type_plot == 'symetric':
        cmap = cm.get_cmap('RdBu_r', n_levels)
    
    # preparing data
    if vmin is None:
        vmin = np.nanpercentile( data_plot, 1 )
    if vmax is None:
        vmax = np.nanpercentile( data_plot, 99 )
    # symetric colorbar?
    if type_plot == 'symetric':
        tmp = np.max( [np.abs(vmin), np.abs(vmax)] )
        vmin, vmax = -tmp, tmp
    
    # ploting
    pmesh = ax.pcolormesh(lon_mesh, lat_mesh, data_plot, transform=ccrs.PlateCarree(), rasterized=True, cmap=cmap, vmin=vmin, vmax=vmax)
    cbar = plt.colorbar(pmesh,fraction=0.0235, pad=0.04, orientation='vertical')
    cbar.ax.tick_params(labelsize=fontsize_colorbar)
    gridlines = ax.gridlines(draw_labels=False,zorder=0)
    
    
# generic function to preprocess loaded data
def func_preprocess_annual(data):
    '''
        Short function for preprocessing of annual indicators of CMIP6 FWI, meant to be used either within 'xarray.open_mfdataset' or directly after loading a file.

        arguments:
            data: xarray.Dataset
                annual indicator of the CMIP6 FWI
    '''
    # passing attributes as coordinates
    dico_match = {'experiment_id':'scen', 'variant_label':'member', 'source_id':'esm'}
    for var in dico_match:
        data = data.expand_dims( {dico_match[var]:[data.attrs[var]]} )
        del data.attrs[var]

    # transforming the type of time axis (not transformed previously according to cmip6-ng conventions)
    data.coords['time'] = data.time.dt.year
    return data




def func_load_sensitivity( limits, prop_calc_FWI_ref, name_axis, options, period_timeseries, date_maps, subset_vars=None, mask_FWI=None ):
    # output
    results = {}

    for value_axis in options:
        # modifying axis for path
        prop_calc_FWI_ref[name_axis] = value_axis
            
        # preparing the path of saved data:
        cfg = configuration_FWI_CMIP6(prop_calc_FWI_ref['type_variables'], prop_calc_FWI_ref['adjust_DryingFactor'], prop_calc_FWI_ref['adjust_DayLength'], prop_calc_FWI_ref['adjust_overwinterDC'], \
                                      limits_on_data={'xps':limits['scen'], 'members':limits['memb'], 'esms':limits['esm']}, \
                                      path_cmip6=prop_calc_FWI_ref['path_cmip6'], path_save=prop_calc_FWI_ref['path_save'], overwrite_path_saveFWI=None, \
                                      option_overwrite=False, overwrite_before='9999-01-01T00:00:00',\
                                      option_load_available=True, option_calc_size=False, option_mask_land=False, option_full_outputs=prop_calc_FWI_ref['option_full_outputs'] )
        cfg.func_prepare_files()

        # looking for the file
        total_lst_files = os.listdir( cfg.path_saveFWI  )
        limit_list_files = []
        for file_W in total_lst_files:
            det = str.split(file_W,'_')
            if (limits['esm'] is None or det[2] in limits['esm'])  and  (limits['scen'] is None or det[3] in limits['scen'])  and  (limits['memb'] is None or det[4] in limits['memb'])  and  (prop_calc_FWI_ref['option_full_outputs'] and ('full-outputs' in file_W) or (prop_calc_FWI_ref['option_full_outputs']==False)):
                limit_list_files.append( file_W )

        # loading
        if len(limit_list_files) == 0:
            raise Exception('No files to load')

        else:
            # preparing output
            results[value_axis] = {}
            for file_W in limit_list_files:
                path = os.path.join(cfg.path_saveFWI, file_W)
                print('loading '+path)
                data = xr.open_mfdataset(path)

                # map?
                if data.experiment_id == 'historical':
                    for _var in subset_vars:
                        results[value_axis]['map_'+_var] = data[_var].sel(time=date_maps).compute()

                # timeseries
                for _var in subset_vars:
                    results[value_axis]['time_map_'+_var+'_'+data.experiment_id] = data[_var].sel(time=slice(period_timeseries[data.experiment_id][0],period_timeseries[data.experiment_id][1])).compute()

                # cleaning
                data.close()
                del data

            if mask_FWI is not None:
                for _var in results[value_axis].keys():
                    results[value_axis][_var] = xr.where( mask_FWI, results[value_axis][_var], np.nan )
    return results
    
#============================================================================================
#============================================================================================









#============================================================================================
# SPATIAL INFO
#============================================================================================
def get_spatial_info( path_cmip6, esm, scen, memb, grid, list_spat_vars = ['sftlf','areacella'] ):
    out = xr.Dataset()
    
    # loading spatial variable
    for VAR in list_spat_vars:
        # identifying correct path
        path_tmp = os.path.join( path_cmip6, scen, 'fx', VAR, esm, memb , grid, VAR+'_fx_'+esm+'_'+scen+'_'+memb+'_'+grid+'.nc' )
        
        # loading
        with xr.open_dataset( path_tmp ) as tmp_file:
            tmp_file.load()
        out[VAR] = tmp_file[VAR]
    
    return out



def mask_percentage(regions, lon, lat):
    """Sample with 10 times higher resolution.

    Notes
    -----
    - assumes equally-spaced lat & lon!
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38 in
      August 2020
    - prototype of what will eventually be integrated in his regionmask package

    """

    lon_sampled = sample_coord(lon)
    lat_sampled = sample_coord(lat)

    mask = regions.mask(lon_sampled, lat_sampled)

    isnan = np.isnan(mask.values)

    numbers = np.unique(mask.values[~isnan])
    numbers = numbers.astype(int)

    mask_sampled = list()
    for num in numbers:
        # coarsen the mask again
        mask_coarse = (mask == num).coarsen(lat=10, lon=10).mean()
        mask_sampled.append(mask_coarse)

    mask_sampled = xr.concat(
        mask_sampled, dim="region", compat="override", coords="minimal"
    )

    mask_sampled = mask_sampled.assign_coords(region=("region", numbers))

    return mask_sampled


def sample_coord(coord):
    """Sample coords for the percentage overlap.

    Notes
    -----
    - copied from Mathias Hauser: https://github.com/mathause/regionmask/issues/38
      in August 2020
    -> prototype of what will eventually be integrated in his regionmask package

    """
    d_coord = coord[1] - coord[0]

    n_cells = len(coord)

    left = coord[0] - d_coord / 2 + d_coord / 20
    right = coord[-1] + d_coord / 2 - d_coord / 20

    return np.linspace(left, right, n_cells * 10)
#============================================================================================
#============================================================================================
