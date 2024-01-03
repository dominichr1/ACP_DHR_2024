import numpy as np
import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition
from shapely.geometry.polygon import LinearRing
import os
from matplotlib import cm
import calendar
from sklearn import datasets, linear_model
from matplotlib.ticker import AutoMinorLocator
import cmocean
from scipy import stats
from datetime import datetime, timedelta
from math import cos, pi, sin, atan2, asin
from math import isnan
import itertools
import matplotlib as mpl
import matplotlib.path as mpath
from matplotlib import gridspec
import cartopy as cy

ZEP_lat = 78.9067 #might need to change
ZEP_lon = 11.8891

South_Pole_lat = -ZEP_lat
South_Pole_lon = -(180-ZEP_lon)

South_grid = (South_Pole_lon,South_Pole_lat)

#PROJECTIONS
geo = ccrs.Geodetic()
rotated_pole = ccrs.RotatedPole(pole_latitude=ZEP_lat, pole_longitude=ZEP_lon)
North_Stereo = ccrs.NorthPolarStereo() #central_longitude=-90)
ortho = ccrs.Orthographic(central_longitude=ZEP_lon, central_latitude=ZEP_lat)
Nearside = ccrs.NearsidePerspective(central_longitude=ZEP_lon, central_latitude=ZEP_lat, satellite_height=35785831, false_easting=0, false_northing=0, globe=None)
PlateCarree = ccrs.PlateCarree()
rotated_pole = ccrs.RotatedPole(pole_latitude=ZEP_lat, pole_longitude=ZEP_lon)

dict_year_to_HYSPLIT_file = {}

for year in np.arange(2002,2011,1):
    dict_year_to_HYSPLIT_file[year] = "traj2002_10_hourly" 
for year in np.arange(2011,2020,1):
    dict_year_to_HYSPLIT_file[year] = "traj2011_19_hourly" #no trajs for 2020 yet
for year in [2020]:
    dict_year_to_HYSPLIT_file[year] = "2020_hourly" 
for year in [2021]:
    dict_year_to_HYSPLIT_file[year] = "2021_hourly" 
    
def remove_duplicates(df):
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")      
    print(len(duplicateRowsDF)) 
    #print(duplicateRowsDF)
    df_first = df.loc[~df.index.duplicated(keep='first')]
    df_last = df.loc[~df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first)))
    print("Length after: "+str(len(df_last)))
    return df_first, df_last
    
def significant_figures(value, sf_num):
    sf = '{0:.'+str(sf_num)+'f}'
    value_sf = sf.format(value)    
    return value_sf

def create_HYSPLIT_name_from_list(list_indexes):
    """use the list of indexes and alter it for the format of HYSPLIT"""
    list_strings_indexes = [str(x) for x in list_indexes]
    list_strings_indexes = [x.replace('-','') for x in list_strings_indexes]
    list_strings_indexes = [x.replace(':','') for x in list_strings_indexes]
    list_strings_indexes = [x[:11] for x in list_strings_indexes]
    list_strings_indexes = [x.replace(' ','_') for x in list_strings_indexes]
    return list_strings_indexes
    
def list_files(year, inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\"):
    """Find the a list of all the HYSPLIT files for a particular year"""
    #print("Year: "+str(year))    
    #inpath_raw_hysplit_data+str(dict_year_to_HYSPLIT_file[year])+"\\" #traj2002_10_hourly\\"
    #print("Path: "+str(inpath_processed_hysplit_dfs+str(year)+"\\"))
    list_of_files = glob.glob(inpath_processed_hysplit_dfs+str(year)+"\\"+'*')    
    #print("Number of HYSPLIT files for "+str(year)+": "+str(len(list_of_files)))
    return list_of_files

def load_data(outpath_files="C:\\Users\\DominicHeslinRees\\Documents\\Wet removal project\\Back_Traj_results\\processed_trajs_size_distribution\\",
              var = "RAINFALL", ML=True):
    var_list_steps = produce_var_list(var=var)
    dfs_analysed_backtrajectory_RAINFALL = concat_analysed_back_trajectories(outpath_files+'accumulated_trajs_'+str(var)+'\\', ML)
    if ML == True:
        dfs_analysed_backtrajectory_RAINFALL = dfs_analysed_backtrajectory_RAINFALL.dropna(subset=var_list_steps, how='all')
    #print(str(len(dfs_analysed_backtrajectory_RAINFALL)))
    return dfs_analysed_backtrajectory_RAINFALL
    
def find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names):
    """observation indexes we want to match with the list of HYSPLIT files"""
    matching_list_of_back_traj_files = [s for s in list_of_HYSPLIT_files if any(xs in s for xs in HYSPLIT_names)]
    #print("Matching files from observational data & HYSPLIT: "+str(len(matching_list_of_back_traj_files)))
    return matching_list_of_back_traj_files
    
def create_traj_dictionary_using_matching_files(matching_list_of_back_traj_files, select_for_mixed_layer=True):
    """an arrival time which maps to the trajectory dataset"""
    if select_for_mixed_layer == True:  
        print("mixed layer selected")
    trajs_dictionary = {}
    for traj_i in matching_list_of_back_traj_files: #each loaded traj_ds is quite long (a full year), so neetraj_ds to be separated  
        #print(traj_i)
        try:
            traj_ds = pd.read_csv(traj_i, parse_dates=['DateTime'], index_col=0) #cut the HYSPLIT using the index occurance of time_step = 0
            arrival_time_of_traj = str(traj_ds.index[0])
            #print("arrival time: "+str(arrival_time_of_traj))
            
            if select_for_mixed_layer == True:                
                traj_ds = traj_ds[traj_ds['MIXDEPTH'] >= traj_ds['altitude']] #select for mixed-layer
            if len(traj_ds) > 0:             
                trajs_dictionary[str(arrival_time_of_traj)] = traj_ds 
        except IndexError:
            print("Index Error - end")
    trajs_dictionary
    return trajs_dictionary
    
def dict_of_dfs_to_df(trajs_dictionary, index_col='arrival_time'):
    for key in trajs_dictionary.keys():
        trajs_dictionary[key][index_col] = key 
    # concatenating the DataFrames
    df = pd.concat(trajs_dictionary.values())
    return df
	
def produce_var_list(var, traj_length=241):
    var_list_steps = np.arange(0,traj_length,1) #add columns for the 241 hours of rainfall along the trajectory
    var_list_steps = [str(var)+'_'+str(x) for x in var_list_steps]   
    return var_list_steps
	
def concat_analysed_back_trajectories(inpath_analysed_trajectories, ML):
    #print(inpath_analysed_trajectories)
    appended_data = []
    dict_ML = {True:'ML', False:'all'}
    folder = glob.glob(inpath_analysed_trajectories+'*_'+str(dict_ML[ML])+'.dat')
    folder.sort()

    for file in folder: 
        #print("File: "+str(file))
        ds = pd.read_csv(file, sep=',',index_col=False) #, skiprows=1, names=columns)    
        appended_data.append(ds)        
    appended_data = pd.concat(appended_data, sort=True)  
    dfs_analysed_backtrajectory = appended_data.copy()
    dfs_analysed_backtrajectory = dfs_analysed_backtrajectory.set_index('Arrival time')
    dfs_analysed_backtrajectory.index = pd.to_datetime(dfs_analysed_backtrajectory.index) 
    #print(dfs_analysed_backtrajectory.index.dtype)

    #print("Length of appended df: "+str(len(dfs_analysed_backtrajectory)))
    return dfs_analysed_backtrajectory
    
def calculate_accumulated(dfs_analysed_backtrajectory_RAINFALL, var='RAINFALL'):
    var_list_steps=produce_var_list(var)
    dfs_analysed_backtrajectory_RAINFALL['accumulated'] = dfs_analysed_backtrajectory_RAINFALL.loc[:,var_list_steps].sum(axis=1)
    print("Mean sum "+str(var)+" for whole traj: "+str(dfs_analysed_backtrajectory_RAINFALL['accumulated'].mean()))
    return dfs_analysed_backtrajectory_RAINFALL
    
def calculate_mean(dfs_analysed_backtrajectory, var='RAINFALL'):
    var_list_steps=produce_var_list(var)
    dfs_analysed_backtrajectory['mean'] = dfs_analysed_backtrajectory.loc[:,var_list_steps].mean(axis=1)
    print("Mean mean "+str(var)+" for whole traj: "+str(dfs_analysed_backtrajectory['mean'].mean()))
    return dfs_analysed_backtrajectory
    
def take_averages(df, var, var_sum):  
    var_cols = produce_var_list(var)
    print("Divide by the number of trajectories at each hour:")
    df[var_sum] = df[var_sum]/df['Number of trajectories']
    max_accummulated_value = df[var_sum].max()
    print("Max "+str(var_sum)+": "+str(max_accummulated_value))
    mean_value = df[var_sum].mean()
    print("Mean "+str(var_sum)+": "+str(mean_value))    
    df.loc[:,var_cols] =  df.loc[:,var_cols].div(df['Number of trajectories'], axis=0)    
    return df
    
def bin_data(df, number_of_bins, var='accumulated'):
    df['bins'] = pd.cut(df[var], number_of_bins)
    df.loc[:, "bin_centres"] = df["bins"].apply(lambda x: x.mid)
    return df

def q25(x, percentile=0.25):
    return x.quantile(percentile)

def q75(x):
    return x.quantile(0.75)    

def produce_groupby_averages(df, var='abs637'):
    df_groupby = df.groupby('bin_centres')[var].agg(['mean', 'median', 'min', 'max', 'std', q25, q75]) 
    return df_groupby
    
def calculate_slope_intercept(x, y, idx):
    print("Idx :"+str(idx))
    slope, intercept = np.polyfit(x[:idx], y[:idx], 1)
    print("Slope: "+str(slope))
    return slope, intercept 

def errorbar_plot(df_groupby, title='', idx = 5):
    fig, ax = plt.subplots(figsize=(5,5))
    
    df_groupby.index = df_groupby.index.astype(int)
    df_groupby = df_groupby[~df_groupby.isin([np.inf, -np.inf]).any(1)]

    index = df_groupby.index
    median = df_groupby['median']
    quan_25 = df_groupby['q25'].values
    quan_75 = df_groupby['q75'].values
    error_label='25$^{\mathrm{th}}$ - 75$^{\mathrm{th}}$'
    ax.errorbar(index, median, yerr=[median-quan_25, quan_75-median], fmt='o', capsize=5, color='k', 
                mfc='None', ecolor='k', ms=1, label=error_label)
    ax.plot(index, median, label='median', color='k', ls=':') 
    ax.set_ylabel('$\sigma_{ap}$ [Mm$^{-1}$]')
    ax.set_xlabel('Average accumulated precipiation en route\n of 10 day back trajectory [mm]')
    #ax.set_ylim(0, quan_75.max()*1.05)
    #ax.set_xlim(0, index.max()*1.05)
    
    slope, intercept = calculate_slope_intercept(index, median, idx)
    slope3sf = significant_figures(slope, 3)
    ax.plot(index[:idx], slope*index[:idx] + intercept, 
            label="slope: "+str(slope3sf)+ 'Mm$^{-1}$/mm',
                   c='r', lw=1)
    ax.set_title(title, loc='left')
    ax.legend(frameon=False)
    plt.show()
    return fig    
    
def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def add_map_features(ax):
    ax.coastlines(resolution='50m')
    ax.add_feature(cy.feature.BORDERS);
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=1, color='gray', alpha=0.5, linestyle='--')
    
def polarCentral_set_latlim(lat_lims, ax):
    # Compute a circle in axes coordinates, which we can use as a boundary
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
def circular_plots(axes_projection=North_Stereo, 
                   title=None): 
    
    fig, ax = sp_map(1, projection=North_Stereo, figsize=(6,6))

    lat_lims = [66,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)

    ax.plot([ZEP_lon], [ZEP_lat], 'ro', ms=10, alpha=0.5, transform=geo) 
    ax.set_title(str(title), loc='left', size=20)    
    plt.show()
    return fig

def make_base_map(ZEP_lon, ZEP_lat, add_inset=True):
    extent = [ZEP_lon-2, ZEP_lon+17, ZEP_lat-3, ZEP_lat+2]
    lonmin, lonmax, latmin, latmax = extent
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.Mercator())       
    ax.coastlines(resolution='10m')
    ax.set_extent(extent, crs=ccrs.PlateCarree())
    ax.plot([ZEP_lon], [ZEP_lat], 'ro', ms=10, alpha=0.5, transform=geo) 
    ax.outline_patch.set_visible(False)
    if add_inset == True:    
        inset_x = 1 # inset location relative to main plot (ax) in normalized units
        inset_y = 1
        inset_size = 0.9

        ax2 = plt.axes([0, 0, 1, 1], projection=North_Stereo)

        lat_lims = [66,90]
        polarCentral_set_latlim(lat_lims, ax2)
        add_map_features(ax2)

        ax2.plot([ZEP_lon], [ZEP_lat], 'ro', ms=10, alpha=0.5, transform=geo) 
        ip = InsetPosition(ax, [inset_x - inset_size / 2,
                                inset_y - inset_size / 2,
                                inset_size,
                                inset_size])
        ax2.set_axes_locator(ip)

        nvert = 100
        lons = np.r_[np.linspace(lonmin, lonmin, nvert),
                     np.linspace(lonmin, lonmax, nvert),
                     np.linspace(lonmax, lonmax, nvert)].tolist()
        lats = np.r_[np.linspace(latmin, latmax, nvert),
                     np.linspace(latmax, latmax, nvert),
                     np.linspace(latmax, latmin, nvert)].tolist()

        ring = LinearRing(list(zip(lons, lats)))
        ax2.add_geometries([ring], ccrs.PlateCarree(),
                           facecolor='none', edgecolor='red', linewidth=0.75)
    plt.savefig('ZEP_MAP.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    

    
