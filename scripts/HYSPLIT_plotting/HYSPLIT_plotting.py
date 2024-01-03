#functions for plotting pre-processed HYSPLIT files (after using the HYSPLIT_processing functions)

import numpy as np
import glob
import pandas as pd
import matplotlib.pyplot as plt
import os
import matplotlib as mpl
import itertools
import cartopy as cy
import cartopy.crs as ccrs
from matplotlib import colors
import matplotlib.ticker as ticker
import gc
import matplotlib.path as mpath
from matplotlib import gridspec
import math
from matplotlib import cm
import cmocean
from scipy import stats
import re
from datetime import datetime
from mpl_toolkits.axes_grid1 import make_axes_locatable


def load_df(loadpath, extrapath=None, filename=None, formatdata=".dat", dict_dtype=None,):
    """load dataframe"""
    if extrapath is not None:
        path = loadpath+'\\'+extrapath+'\\'+filename+formatdata
        print("loading: "+str(path))
    if extrapath is None:
        path = loadpath+'\\'+filename+formatdata
        print("loading: "+str(path))
    df = pd.read_csv(path, index_col=0, parse_dates=True,
                     dtype=dict_dtype) #The low_memory option is not properly deprecated, but it should be, since it does not actually do anything differently
    return df

def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".png", dpi=300):
    folders = glob.glob(path)
    print(folders)
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
       
def save_df(df, path, name='', index=True, float_format=None, data_format='.dat'):
    if data_format == '.dat':
        print("Save as: "+str(path+'\\'+name+'.dat'))
        df.to_csv(path+'\\'+name+'.dat', index=index, float_format=float_format)
    if data_format == '.pickle':
        df.to_pickle(path+'\\'+name+'.pickle')

def show_memory_info(df):
    df.info(memory_usage="deep")
       
def slice_df(df, start_datetime=None, end_datetime=None):
    if (start_datetime is not None) & (end_datetime is not None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index) & (df.index <= pd.to_datetime(end_datetime))]
    if (start_datetime is not None) & (end_datetime is None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index)]
    if (start_datetime is None) & (end_datetime is not None):
        df = df.loc[(df.index <= pd.to_datetime(end_datetime))]
    return df
    
def remove_np_nan_and_inf(df):
    print("Length before: "+str(len(df)))
    df = df.replace([np.inf, -np.inf], np.nan) #cant cluster with np.nan
    df = df.dropna(how='any', axis=0)
    print("Length after: "+str(len(df)))
    return df
    
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)

def create_lists_of_90_repeating_360():
    l_0x90 = [1]*90 # list of 1s of length 90
    l_x32400 = []
    for n in range(360):
        l_nx90 = [x+n for x in l_0x90]
        l_x32400 = l_x32400 + l_nx90
    print(len(l_x32400))
    return l_x32400
	
def create_empty_arrays_hits():
    df_arrays_hits = pd.DataFrame(index=range(90*360), columns=['latitude', 'longitude'])
    l_90 = list(np.arange(1,91,1)) #1-90
    l_360 = l_90*360 #multiply by 360
    df_arrays_hits['latitude'] = l_360
    l_x32400 = create_lists_of_90_repeating_360()
    df_arrays_hits['longitude'] = l_x32400       
    return df_arrays_hits
	
def remove_duplicates(df):
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    df = df.loc[~df.index.duplicated(keep='first')]
    print("Length after: "+str(len(df)))
    return df	
    
def create_HYSPLIT_name_from_list(list_indexes):
    """use the list of indexes and alter it for the format of HYSPLIT"""
    list_strings_indexes = [str(x) for x in list_indexes]
    list_strings_indexes = [x.replace('-','') for x in list_strings_indexes]
    list_strings_indexes = [x.replace(':','') for x in list_strings_indexes]
    list_strings_indexes = [x[:11] for x in list_strings_indexes]
    list_strings_indexes = [x.replace(' ','_') for x in list_strings_indexes]
    return list_strings_indexes
    
def list_files(year, inpath_processed_hysplit_dfs):
    """Find the a list of all the HYSPLIT files for a particular year"""
    list_of_files = glob.glob(inpath_processed_hysplit_dfs+str(year)+"\\"+'*')    
    return list_of_files
    
def find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names):
    """observation indexes we want to match with the list of HYSPLIT files"""
    matching_list_of_back_traj_files = [s for s in list_of_HYSPLIT_files if any(xs in s for xs in HYSPLIT_names)]
    print("Matching files from observational data & HYSPLIT: "+str(len(matching_list_of_back_traj_files)))
    return matching_list_of_back_traj_files
    
def create_df_of_processed_trajs_for_obs(list_of_files, df_obs, var, time_col='start',
                                         traj_length=241, weekly=False):    
    
    if weekly == True:
        print("weekly resolution: ")
        df_obs.index = df_obs[time_col]
        df_obs.index = pd.to_datetime(df_obs.index)    
    if weekly == False:
        print("not weekly resolution")

    print("variable: "+str(var))
    appended_ds = []
    for file in list_of_files: #the list of HYSPLIT trajs to append/analyse
        print(file) #file
        ds = pd.read_csv(file) #read data 
        number_of_trajs = len(ds.index.unique())
        size_of_ds = len(ds)
        length_num_trajs = size_of_ds/number_of_trajs
        arrival_time = ds['DateTime'].iloc[0]
        arrival_time = pd.to_datetime(arrival_time)        
        ds = ds.iloc[:int(traj_length)*int(number_of_trajs),:]
        #merge with BC or OCEC
        try:     
            if weekly == False:
                obs_value = df_obs.loc[df_obs.index == arrival_time, var].values[0]
            if weekly == True:
                obs_value = df_obs.loc[df_obs.index == arrival_time, var].values[0]
                print("value: "+str(obs_value))
                obs_value = df_obs[var].iloc[df_obs.index.get_loc(arrival_time, method='nearest')]  
                print("value: "+str(obs_value))
                
            print("value: "+str(obs_value))
            ds['obs'] = obs_value
        except IndexError as error:
            print("no value")
            ds['obs'] = np.nan
        appended_ds.append(ds)
    try:
        ds_all_files = pd.concat(appended_ds)
    except ValueError:
        ds_all_files = None
        print("no files to concat")        
    return ds_all_files
    
# def produce_hourly_time_ranges(df_OCEC):
    # """using the datafram the start and stop column, create an hourly index"""
    # time_ranges = []
    # for i in range(len(df_OCEC)):
        # time_range = list(pd.date_range(df_OCEC['start'].iloc[i], df_OCEC['stop'].iloc[i], freq='H'))
        # time_ranges.append(time_range)
    # time_ranges = list(itertools.chain(*time_ranges))
    # time_ranges = [pd.to_datetime(x) for x in time_ranges]
    # print("size "+str(len(time_ranges)))
    # return time_ranges
   
def save_processed_trajs_with_obs(df_obs, outpath_summed_dfs="E:\\Data\\HYSPLIT\\processed\\OC\\dfs\\", 
                                  inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\",
                                  var='OC_mean_mug_m3', years=[], weekly=False, process=False):   
    if process == True:
        if len(years) == 0:
            years = df_obs.index.year.unique()
            print(years)
        for year in years:
            print("Year: "+str(year))        
            list_of_HYSPLIT_files = list_files(year, inpath_processed_hysplit_dfs)            

            if weekly == True:
                print("use weekly")
                OCEC_time_ranges = produce_hourly_time_ranges(df_obs)       
                HYSPLIT_names = create_HYSPLIT_name_from_list(OCEC_time_ranges)
            if weekly == False:
                HYSPLIT_names = create_HYSPLIT_name_from_list(df_obs.index)

            matching_list_of_back_traj_files = find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names)        
            ds_year = create_df_of_processed_trajs_for_obs(matching_list_of_back_traj_files, df_obs, var=var, weekly=weekly)
            print(ds_year)
            if ds_year is not None:        
                try:
                    ds_year.to_csv(outpath_summed_dfs+str(year)+'.dat')
                except FileNotFoundError:
                    os.makedirs(outpath_summed_dfs, exist_ok=True)
                    ds_year.to_csv(outpath_summed_dfs+str(year)+'.dat')            
                print("Saved: "+str(outpath_summed_dfs+str(year)+'.dat'))
    if process == False:
        print("if you need to process, change process to True")
        pass
    
def append_data(path, year=None, index_col=0, parse_dates=True, dict_dtype=None, 
                usecols=['DateTime', 'Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time'],
                years=None, select_for_mixed_layer=True, select_for_above_mixed_layer=False,
                data_format='.dat', nb_cluster=5, cluster_num=None, savepath=None, append_the_data=True):
    if cluster_num is not None:
        print("cluster number: "+str(cluster_num))
    appended_data = []
    print(str(path)+'\\'+str(year)+"*"+str(data_format))
    if years == None:
        print("no years provided, just a year")
        files = glob.glob(str(path)+'\\'+str(year)+"*"+str(data_format))
        print(files)
    if years is not None:
        print("years provided")
        append_files = []
        for year in years:
            print(str(path)+'\\'+str(year)+"*"+str(data_format))
            file = glob.glob(str(path)+'\\'+str(year)+"*"+str(data_format))            
            append_files.append(file[0])          
        files = append_files   
    #print(files)
    data = None
    for infile in files:  
        print(infile)
        if data is not None:
            print("delete data file, save memeory")
            del data
        if data_format == '.dat':
            data = pd.read_csv(infile, usecols=usecols, 
                           index_col=0, parse_dates=parse_dates, dtype=dict_dtype) 
        if data_format == '.pickle':
            data = pd.read_pickle(infile) 

            data = data[usecols] 
            data = remove_np_nan_and_inf(data)
            print([*dict_dtype.keys()])
            data = data.astype(dict_dtype)       
            
        if cluster_num is not None:
            data = data[data['clusters_'+str(nb_cluster)] == cluster_num]
               
        name_ml = '' 
        if select_for_mixed_layer == True:
            print("altitude of air mass below mixed layer height")
            data = data.loc[data['MIXDEPTH'] > data['altitude'], :]  #select for mix lay
            name_ml = 'below_ml'
            
        if select_for_above_mixed_layer == True:
            print("altitude of air mass below mixed layer height")
            data = data.loc[data['MIXDEPTH'] <= data['altitude'], :]  #select for mix lay
            name_ml = 'above_ml'
            
        data = data.drop(columns=['MIXDEPTH','altitude'])
        if savepath is not None:             
            print(savepath+'\\'+str(year)+str(name_ml)+'cluster_'+str(cluster_num)+'.pickle')
            data.to_pickle(savepath+'\\'+str(year)+str(name_ml)+'cluster_'+str(cluster_num)+'.pickle')
        
        if append_the_data == True:
            appended_data.append(data) #store DataFrame in list  
    if append_the_data == True:
        appended_data = pd.concat(appended_data)
        #print(appended_data)
        return appended_data
    
month_to_season =  { 1:'build_up',  2:'Haze', 3:'Haze',  
                    4:'Haze',  5:'Haze', 6:'Summer',  7:'Summer',  8:'Summer', 9:'Summer', 10:'build_up', 
                     11:'build_up', 12:'build_up'}
                     
def create_month_season_numbers(df):
    start_year = df.index.year[0]
    print("First year in data set: "+str(start_year))
    number_years = len(df.index.year.unique())+1
    print("Number of years + 1: "+str(number_years))
    df.loc[:,'month_num'] = df.index.month
    df.loc[:,'year_num'] = df.index.year
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    df = df.sort_index()
    return df
    
def add_time_cols(df):
    if type(df.index) != pd.core.indexes.datetimes.DatetimeIndex:
        df['DateTime'] = pd.to_datetime(df['DateTime'])
        df.index = pd.to_datetime(df['DateTime'])
    df = create_month_season_numbers(df)
    seasons = df.season_abb_year.unique()
    seasons_num = np.arange(1,len(df.season_abb_year.unique()),1)
    season_to_season_num = dict(zip(seasons, seasons_num))
    print(season_to_season_num)
    df.loc[:,'season_ordinal'] = df['season_abb_year'].map(season_to_season_num)
    return df
    
#####################################ARRAYS#########################################################################################################################
    
def groupby_ds_type_of_function(ds, df_arrays_hits, type_of_function_obs, var='obs'):
    print(str(type_of_function_obs))
    ds_groupby = ds.groupby(['grid_lat', 'grid_lon']).agg({'Traj_num': 'count', var: type_of_function_obs})
    df_counts = ds_groupby.reset_index()
    df_merged = pd.merge(df_arrays_hits, df_counts, how='left', left_on=['latitude', 'longitude'], right_on = ['grid_lat', 'grid_lon'])
    df_merged = df_merged.drop(['grid_lat', 'grid_lon'], axis=1)  
    df_merged = df_merged.rename({'Traj_num':'hits'}, axis=1)
    return df_merged
    
def make_array(fill):
    longitude_array = np.arange(-180, 180, 1)
    latitude_array = np.arange(0, 90, 1)
    array = np.indices((latitude_array.shape[0], longitude_array.shape[0]))
    array = array[0] + array[1]
    array = np.empty((90,360))
    array.fill(fill) 
    return array
    
def turn_grid_df_to_array(df, var, threshold=None): #'Traj_num', 'RELHUMID', 'SUN_FLUX', 'MIXDEPTH', 'RAINFALL'
    df_selected = df.iloc[:32400,:].copy()     
    print("mean: "+str(df_selected[var].mean()))
    print("median: "+str(df_selected[var].median()))    
    df_threshold = df_selected.copy()    
    if threshold:
        print('use threshold')
        df_threshold.loc[df['hits'] < threshold, str(var)] = np.nan #below threshold = np.nan
    else:
        df_threshold    
    print("mean after threshold applied: "+str(df_threshold[var].mean()))
    print("median after threshold applied: "+str(df_threshold[var].median()))
    df_threshold = df_threshold[[str(var), 'latitude', 'longitude']].copy()    
    df_threshold['latitude']  = df_threshold['latitude'] - 1
    df_threshold['longitude']  = df_threshold['longitude'] - 1    
    # df_threshold.pivot_table(str(var), 'latitude', 'longitude')    
    rowIDs = df_threshold['latitude'].astype(int)
    colIDs = df_threshold['longitude'].astype(int) 
    
    array = np.zeros((90,360))
    print(array.shape)    
    # array = np.zeros((int(rowIDs.max()+1),int(colIDs.max()+1)))
    # print(array.shape) #should be (90, 360)
    array[rowIDs, colIDs] = df_threshold[str(var)].values
    return array
    
def append_arrays_obs(path, folder, years, season=None, month_num=None, 
                      index_col=0, parse_dates=True, dict_dtype=None,
                      usecols=['DateTime', 'Traj_num','MIXDEPTH', 'altitude', 'grid_lat', 'grid_lon', 'obs']):    
    obs_annual_arrays = []
    for year in years:
        print("year: "+str(year)) #for all years        
        try:     
            print(str(path)+str(folder)+"\\dfs\\")
            df_obs = append_data(str(path)+str(folder)+"\\dfs\\", year=year, 
                                 index_col=index_col, parse_dates=parse_dates, dict_dtype=dict_dtype,
                                 usecols=usecols)            
            df_obs = add_time_cols(df_obs)
            if season != None:
                df_obs_season = df_obs[df_obs['season_abb_year'] == str(season)+'_'+str(year)]
                df_obs = df_obs_season.copy()
            if month_num != None:
                print(str(month_num))
                df_obs_month = df_obs[df_obs['month_num'] == month_num]
                df_obs = df_obs_month.copy()            
            df_arrays_hits = create_empty_arrays_hits()
            df_groupby_lat_lon_grids_nanmedian = groupby_ds_type_of_function(df_obs, df_arrays_hits, 
                                                                             type_of_function_obs=np.nanmedian)
            median_obs_array = turn_grid_df_to_array(df_groupby_lat_lon_grids_nanmedian, var='obs', threshold=50)
            obs_annual_arrays.append(median_obs_array)
        except ValueError: #ValueError: No objects to concatenate
            print("Value Error encountered")
            empty_array = make_array(np.nan)
            obs_annual_arrays.append(empty_array)
    return obs_annual_arrays
    
def concat_years(path, folder, years, index_col=0, parse_dates=True, dict_dtype=None,
                 usecols=['DateTime', 'Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time', 'MIXDEPTH',
                                 'altitude'], select_for_mixed_layer=True, select_for_above_mixed_layer=False,
                                 data_format='.dat'):
    df_obs_all_years = []
    for year in years:
        print(str(path)+str(folder)+"\\dfs\\")
        df_obs = append_data(str(path)+str(folder)+"\\dfs\\", year,
        index_col=index_col, parse_dates=parse_dates, dict_dtype=dict_dtype, 
        usecols=usecols, select_for_mixed_layer=select_for_mixed_layer, 
        select_for_above_mixed_layer=select_for_above_mixed_layer, data_format=data_format)
        
        show_memory_info(df_obs)
        print("size of df: "+str(len(df_obs)))
        df_obs_all_years.append(df_obs)            
    df_obs_concatted = pd.concat(df_obs_all_years, axis=0)
    return df_obs_concatted
    
def all_years_array_obs(path, folder, years, season=None, month_num=None,
                      var='obs', threshold=50, index_col=0, parse_dates=True, dict_dtype=None):   
    try: 
        df_obs = concat_years(path, folder, years, index_col=index_col, parse_dates=parse_dates, dict_dtype=dict_dtype)        
        print("size of df: "+str(len(df_obs)))

        if season != None:
            df_obs = add_time_cols(df_obs)
            df_obs_season = df_obs[df_obs['season_abb_year'] == str(season)+'_'+str(year)]
            df_obs = df_obs_season.copy()
        if month_num != None:
            df_obs = add_time_cols(df_obs)
            print(str(month_num))
            df_obs_month = df_obs[df_obs['month_num'] == month_num]
            df_obs = df_obs_month.copy()   
            
        df_arrays_hits = create_empty_arrays_hits()
        df_groupby_lat_lon_grids_nanmedian = groupby_ds_type_of_function(df_obs, df_arrays_hits, 
                                                                         type_of_function_obs=np.nanmedian)
        median_obs_array = turn_grid_df_to_array(df_groupby_lat_lon_grids_nanmedian, var=var, threshold=threshold)            
    except ValueError: #ValueError: No objects to concatenate
            empty_array = make_array(np.nan)            
    return median_obs_array

def make_array(fill):
    longitude_array = np.arange(-180, 180, 1)
    latitude_array = np.arange(0, 90, 1)
    array = np.indices((latitude_array.shape[0], longitude_array.shape[0]))
    array = array[0] + array[1]
    array = np.empty((90,360))
    array.fill(fill) 
    return array
    
def circular_plots_array(array, vmax, cmap, orientation, colourbar_label, colourbar_labelsize=20, colourbar_tick_fontsize=12, 
                   scientific_notation=True, decimal_places_colourbar=0, axes_projection='North_Stereo', 
                   array_projection='rotated_pole', title=None, ZEP_lat=78.906,ZEP_lon=11.888, test_data=None,
                   figsize_x=6,fig_size_y=6, vmin=0, extend='max', central_longitude=0,
                   lat_min=30, df_cluster=None, colour_cluster='k', s=6, alpha=0.5, plot_log=False,
                   vminlog=10**(-5), vmaxlog=10**(0)): 
    
    dict_projections = create_projections_dict(ZEP_lat=78.906,ZEP_lon=11.888, central_longitude=central_longitude)                   
    projection = dict_projections[axes_projection] 
    array_projection = dict_projections[array_projection] 
    geo = dict_projections['geo']    
    
    fig, ax = sp_map(1, projection=projection, figsize=(figsize_x,fig_size_y))

    lat_lims = [lat_min,90]
    polarCentral_set_latlim(lat_lims, ax)                
    lons = np.arange(-178.5, 181.5, 1) #lons = np.arange(-180.5, 179.5, 1)    
    lats = np.arange(1.5, 91.5, 1) #lats = np.arange(-0.5, 89.5, 1)    
    if plot_log == False:
        cs = ax.pcolormesh(lons, lats, array, shading='auto', transform=array_projection, vmin=vmin, vmax=vmax, cmap=cmap) 
    
    if plot_log == True:
        norm=mpl.colors.LogNorm(vmin=vminlog, vmax=vmaxlog)
        cs = ax.pcolormesh(lons, lats, array, shading='auto', transform=array_projection, norm=norm, cmap=cmap) 
        
    ax.plot([ZEP_lon], [ZEP_lat], 'bo', ms=5, alpha=0.5, transform=geo) 
    
    if test_data is not None:
        latitudes = list(test_data['latitude'].values)
        longitudes = list(test_data['longitude'].values)
        ax.plot(longitudes, latitudes, 'bo', ms=5, alpha=0.2, transform=geo) 
    
    if orientation =='horizontal':
        cax = fig.add_axes([0.15, .08, 0.72, 0.03]) # position of colorbar [left, bottom, width, height]
    if orientation =='vertical':
        cax = fig.add_axes([0.95, .2, 0.02, 0.6]) # position of colorbar [left, bottom, width, height]
    
    if scientific_notation == True:
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
        cbar = fig.colorbar(cs, extend=extend, orientation='vertical', cax=cax, format=fmt) #, **kwargs)
        cbar.set_label(colourbar_label, size=colourbar_labelsize)
        cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
        cbar.ax.yaxis.set_offset_position('left')  
        #cbar.ax.yaxis.get_offset_text().set_position((2,2))            
        cbar.update_ticks()

    else:
        kwargs = {'format': '%.'+str(decimal_places_colourbar)+'f'}
        cbar = fig.colorbar(cs, extend=extend, orientation='vertical', cax=cax, **kwargs)        
        cbar.set_label(colourbar_label, size=colourbar_labelsize)
        cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
    
    ax.set_title(str(title), loc='left', size=15)  
    add_map_features(ax) 

    #dict_cluster_to_colors[cluster]
    if df_cluster is not None:
        for i in range(len(df_cluster)):           
            lon_traj = df_cluster['lon'].values
            lat_traj = df_cluster['lat'].values            
            ax.scatter(lon_traj, lat_traj, marker="o", 
                       color=colour_cluster, transform=array_projection,
                       s=s, alpha=alpha)
    
    #plt.savefig('BC_1200.png', format='png', dpi=1200, bbox_inches='tight')
    #plt.savefig('OC_300.svg', format='svg', dpi=300, bbox_inches='tight')
    plt.show()
    return fig
    
def circular_subplots(array, vmax, vmin=0, cmap=None, orientation='horizontal', colourbar_label='', colourbar_labelsize=20, 
                      colourbar_tick_fontsize=12, 
                      scientific_notation=True, decimal_places_colourbar=0, 
                      array_projection='rotated_pole', title='', ZEP_lat=78.906,ZEP_lon=11.888, central_longitude=0, 
                      show_colourbar=False, extend=None,
                      cb_pos_x=.95, cb_width=0.02, lat_min=30, df_cluster=None, colour_mean='k', s=6, alpha=0.5,
                      plot_log=False, vminlog=10**(-5), vmaxlog=10**(0), ax=None, fig=None):
                   
    dict_projections = create_projections_dict(ZEP_lat, ZEP_lon, central_longitude)   
    array_projection = dict_projections[array_projection]    
    geo = dict_projections['geo'] 
                   
    lat_lims = [lat_min,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)            
    lons = np.arange(-178.5, 181.5, 1)  #lons = np.arange(-180.5, 179.5, 1)    
    lats = np.arange(1.5, 91.5, 1) #    lats = np.arange(-0.5, 89.5, 1)'
    
    empty_array = make_array(1)
    cs = ax.pcolormesh(lons, lats, empty_array, hatch='/', alpha=0., transform=array_projection, edgecolor='black')
    cs = ax.pcolormesh(lons, lats, array, transform=array_projection, color='k', alpha=.05)   
    
    if plot_log == False:
        #cs = ax.pcolormesh(lons, lats, array, shading='auto', transform=array_projection, vmin=vmin, vmax=vmax, cmap=cmap) 
        cs = ax.pcolormesh(lons, lats, array, transform=array_projection, vmin=vmin, vmax=vmax, cmap=cmap)
    if plot_log == True:
        norm=mpl.colors.LogNorm(vmin=vminlog, vmax=vmaxlog)
        cs = ax.pcolormesh(lons, lats, array, shading='auto', transform=array_projection, norm=norm, cmap=cmap) 
    
    #cs = ax.pcolormesh(lons, lats, empty_array, hatch='/', alpha=0., transform=array_projection,edgecolor='black')
    #cs = ax.pcolormesh(lons, lats, array, transform=array_projection, color='k', alpha=.05)            
    #cs = ax.pcolormesh(lons, lats, array, transform=array_projection, vmin=vmin, vmax=vmax, cmap=cmap) 
    
    ax.plot([ZEP_lon], [ZEP_lat], 'bo', ms=10, alpha=0.1, transform=geo)             
    ax.set_title(str(title), loc='left', size=20)    
    ax.coastlines(resolution='50m') 

       
    if df_cluster is not None:
        for endpoint in range(len(df_cluster)):           
            lon_traj = df_cluster['lon'].values
            lat_traj = df_cluster['lat'].values   
            #print(lon_traj)
            ax.scatter(lon_traj, lat_traj, marker="o", 
                       color=colour_mean, transform=array_projection,
                       s=s, alpha=alpha)        
                           
    
    if show_colourbar == True:
        if orientation =='horizontal':
            cax = fig.add_axes([0.15, .08, 0.72, 0.03]) # position of colorbar [left, bottom, width, height]
        if orientation =='vertical':
            cax = fig.add_axes([cb_pos_x, .2, cb_width, 0.6]) # position of colorbar [left, bottom, width, height]
        if scientific_notation == True:
            print("use scientific notation:")
            fmt = ticker.ScalarFormatter(useMathText=True)
            fmt.set_powerlimits((0, 0))
            cbar = fig.colorbar(cs, extend=extend, orientation='vertical', cax=cax, format=fmt) #, **kwargs)
            cbar.set_label(colourbar_label, size=colourbar_labelsize)
            cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
            cbar.ax.yaxis.set_offset_position('left')            
            cbar.update_ticks()
        else:
            kwargs = {'format': '%.'+str(decimal_places_colourbar)+'f'}
            cbar = fig.colorbar(cs, extend=extend, orientation='vertical', cax=cax, **kwargs)        
            cbar.set_label(colourbar_label, size=colourbar_labelsize)
            cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
        
    return ax
    
def round_up(n, decimals=0):
    multiplier = 10 ** decimals
    return math.ceil(n * multiplier) / multiplier
    
def array_subplots(list_of_arrays, years, nrows=3, vmin=0, vmax=1, start_year=2002,
                   colourbar_label=r"Median $\sigma_{\mathrm{ap}}$ per grid box traversed [Mm$^{-1}$]",
                   suptitle='Organic carbon', axes_projection='North_Stereo', 
                   array_projection='rotated_pole', delete_axes=None, dict_titles=None, cmap=None,
                   extend='max', central_longitude=0, cb_height=0.01, scientific_notation=True,
                   decimal_places_colourbar=0, dict_cluster_to_df_mean=None, dict_cluster_to_colors=None,
                   s=6, alpha=0.5, plot_log=False, vminlog=10**(0), vmaxlog=10**(6)): 
            
    dict_projections = create_projections_dict(ZEP_lat=78.906, ZEP_lon=11.888, central_longitude=central_longitude)                  
    projection = dict_projections[axes_projection]
    geo = dict_projections['geo'] 
    
    number_of_years = len(years)
    print("num years: "+str(number_of_years))    
    ncols = int(round_up(number_of_years/nrows))
    print("num cols: "+str(ncols))
    
    fig, axs = sp_map(nrows,ncols, projection=projection, figsize=(8*ncols,8*nrows))    
    print("number of rows: "+str(nrows))
    
    if cmap is None:
        #cmap=cmocean.cm.turbid
        cmap=cmocean.cm.haline
    if dict_titles is not None:
        print(dict_titles)
        dict_titles = {v: k for k, v in dict_titles.items()} #reverse dict to have num to season
    
    gs = gridspec.GridSpec(nrows, ncols,
         wspace=0.0, hspace=0.0, top=0.95, bottom=0.05, left=0.17, right=0.845) #idth_ratios=[1, 1, 1]    
    axs=axs.flatten()
    for i,array in enumerate(list_of_arrays):
        if dict_titles is None:
            title = start_year+i   
        if dict_titles is not None:
            print(i+1)
            print(dict_titles)
            title = dict_titles[i+1]
        
        df_cluster = None
        colour_mean = 'k'
        if dict_cluster_to_df_mean is not None:
            df_cluster = dict_cluster_to_df_mean[i+1]            
            if dict_cluster_to_colors is not None:
                print(i)
                colour_mean = dict_cluster_to_colors[i+1]
           
        circular_subplots(array=array, vmin=vmin, vmax=vmax, cmap=cmap, 
                       orientation='horizontal', 
                       colourbar_label=colourbar_label, 
                       colourbar_labelsize=20, colourbar_tick_fontsize=12, 
                       scientific_notation=scientific_notation, decimal_places_colourbar=decimal_places_colourbar,
                       array_projection=array_projection, title=title, df_cluster=df_cluster, colour_mean=colour_mean, 
                       s=s, alpha=alpha,plot_log=plot_log,  vminlog=vminlog, vmaxlog=vmaxlog, ax=axs[i])
                       
    plt.subplots_adjust(wspace=0.1, hspace=0)
    
    #Delete the unwanted axes
    if delete_axes is not None:
        for i in delete_axes:
            fig.delaxes(axs[i])   
    cbar_ax = fig.add_axes([0.15, .05, 0.72, cb_height]) #position of colorbar [left, bottom, width, height]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    
    fmt = None
    if scientific_notation == True:
        print("use scientific notation:")
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))    
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                   norm=norm, orientation='horizontal', extend=extend, format=fmt)
    cb.set_label(colourbar_label, fontsize=50)
    cb.ax.tick_params(labelsize=30)
    cb.ax.yaxis.set_offset_position('left')    
    cb.ax.yaxis.get_offset_text().set_fontsize(50)    
    cb.update_ticks()
    
    plt.suptitle(suptitle, y=0.9, fontsize=20)    
    plt.show()  
    return fig
    
#no observations added##############################################################################################################################################
               
def create_traj_dictionary_using_matching_files(matching_list_of_back_traj_files, select_for_mixed_layer=True, traj_length=241,
                                                format_data='.pickle'):
    """an arrival time which maps to the trajectory dataset"""
    if select_for_mixed_layer == True:  
        print("mixed layer selected")
    trajs_dictionary = {}
    for traj_i in matching_list_of_back_traj_files: #each loaded traj_ds is quite long (a full year), so neetraj_ds to be separated  
        #print(traj_i)
        try:
            if format_data == '.dat':
                traj_ds = pd.read_csv(traj_i, parse_dates=['DateTime'], index_col=0) #cut the HYSPLIT using the index occurance of time_step = 0
            if format_data == '.pickle':
                print("file: "+str(traj_i))
                traj_ds = pd.read_pickle(traj_i)
                traj_ds.index = pd.to_datetime(traj_ds.index)            
            arrival_time_of_traj = str(traj_ds.index[0])
            number_of_trajs = len(traj_ds.index.unique())          
            traj_ds = traj_ds[traj_ds['time_step'] > (traj_length)*(-1)]
            #traj_ds = traj_ds.iloc[:int(traj_length)*int(number_of_trajs),:]
            #print("arrival time: "+str(arrival_time_of_traj))            
            if select_for_mixed_layer == True:               
                traj_ds = traj_ds[traj_ds['MIXDEPTH'] >= traj_ds['altitude']] #select for mixed-layer
            if len(traj_ds) > 0:             
                trajs_dictionary[str(arrival_time_of_traj)] = traj_ds 
        except IndexError:
            print("Index Error - end")
    trajs_dictionary
    return trajs_dictionary
       
def create_projections_dict(ZEP_lat, ZEP_lon, central_longitude):
    dict_projections = {}
    
    geo = ccrs.Geodetic(); dict_projections['geo'] = geo
    rotated_pole = ccrs.RotatedPole(pole_latitude=ZEP_lat, pole_longitude=ZEP_lon, central_rotated_longitude=central_longitude)
    dict_projections['rotated_pole'] = rotated_pole
    North_Stereo = ccrs.NorthPolarStereo() 
    dict_projections['North_Stereo'] = North_Stereo
    ortho = ccrs.Orthographic(central_longitude=ZEP_lon, central_latitude=ZEP_lat)
    dict_projections['ortho'] = ortho
    Nearside = ccrs.NearsidePerspective(central_longitude=ZEP_lon, central_latitude=ZEP_lat, satellite_height=35785831, false_easting=0, false_northing=0, globe=None)
    dict_projections['Nearside'] = Nearside
    PlateCarree = ccrs.PlateCarree()
    dict_projections['PlateCarree'] = PlateCarree
    rotated_pole = ccrs.RotatedPole(pole_latitude=ZEP_lat, pole_longitude=ZEP_lon)  
    dict_projections['rotated_pole'] = rotated_pole
    
    return dict_projections     
          
def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def add_map_features(ax):
    ax.coastlines(resolution='50m')
    #ax.add_feature(cy.feature.BORDERS);
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
                  linewidth=.5, color='gray', alpha=0.2, linestyle='--')
    
def polarCentral_set_latlim(lat_lims, ax):
    # Compute a circle in axes coordinates, which we can use as a boundary
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
def circular_plots(axes_projection='North_Stereo', 
                   title=None, ZEP_lat=78.906,ZEP_lon=11.888, central_longitude=0): 
    
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon, central_longitude)                   
    projection = dict_projections[axes_projection]
            
    fig, ax = sp_map(1, projection=projection, figsize=(6,6))

    lat_lims = [66,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)

    ax.plot([ZEP_lon], [ZEP_lat], 'ro', ms=10, alpha=0.5, transform=geo) 
    ax.set_title(str(title), loc='left', size=20)    
    plt.show()
    return fig
    
def make_fire_plot_traj(df_traj, c, ax=None):        
    for traj in df_traj.Traj_num.unique():               
        df_traj_num = df_traj[df_traj['Traj_num'] == traj]       
        lon_traj = df_traj_num['longitude'].values
        lat_traj = df_traj_num['latitude'].values
        ax.scatter(lon_traj, lat_traj, s=1, marker="o", facecolor="none", edgecolor=c, transform=ccrs.PlateCarree())
    return ax
    
def circular_traj_plots(trajs_dictionary, axes_projection='North_Stereo', 
                        title='', ZEP_lat=78.906,ZEP_lon=11.888, central_longitude=0):  
    
    dict_projections = create_projections_dict(ZEP_lat, ZEP_lon, central_longitude)                   
    projection = dict_projections[axes_projection]
    
    fig, ax = sp_map(1, projection=projection, figsize=(6,6))

    lat_lims = [50,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
    
    #ax.axis("off")       
    ax.coastlines(resolution='50m')
    ax.gridlines(alpha=.2, lw=1)  
    norm = mpl.colors.Normalize(vmin=0, vmax=len([*trajs_dictionary.keys()])+1)
    cmap = cmocean.cm.thermal
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    for count, arrival_time in enumerate([*trajs_dictionary.keys()]):
        make_fire_plot_traj(trajs_dictionary[str(arrival_time)], c=m.to_rgba(count), ax=ax)

    geo = dict_projections['geo']
    
    ax.plot([ZEP_lon], [ZEP_lat], 'bo', ms=5, alpha=0.5, transform=geo) 
    ax.set_title(str(title), loc='left', size=15)    
    plt.show()
    return fig
    
######################OBS WITH TRAJS############################################################################################################################################################################
    
def append_dfs_obs(inpath_processed_hysplit_dfs, folder, years, season=None, month_num=None, ML=True):    
    obs_annual_arrays = []
    for year in years:
        print("year: "+str(year)) #for all years        
        try:        
            df_obs = append_data(inpath_processed_hysplit_dfs+str(folder)+"\\dfs\\", year)
                                
            if ML == True:
                df_obs = df_obs[df_obs['MIXDEPTH'] <= df_obs['altitude']]                    
                        
            df_obs = add_time_cols(df_obs)

            if season != None:
                df_obs_season = df_obs[df_obs['season_abb_year'] == str(season)+'_'+str(year)]
                df_obs = df_obs_season.copy()
            if month_num != None:
                print(str(month_num))
                df_obs_month = df_obs[df_obs['month_num'] == month_num]
                df_obs = df_obs_month.copy()

            df_groupby_lat_lon_grids_nanmedian = groupby_ds_type_of_function(df_obs, df_arrays_hits, 
                                                                             type_of_function_obs=np.nanmedian)
                                                                             
            median_obs_array = turn_grid_df_to_array(df_groupby_lat_lon_grids_nanmedian, var='obs', threshold=50)
            obs_annual_arrays.append(median_obs_array)
        except ValueError: #ValueError: No objects to concatenate
            empty_array = make_array(np.nan)
            obs_annual_arrays.append(empty_array)
    return obs_annual_arrays
    
def load_appended_yearly_traj_data(inpath_processed_hysplit_dfs, year, folder = 'abs', cols=['latitude', 'obs'], ML=True):
    path = inpath_processed_hysplit_dfs+str(folder)+"\\dfs\\"
    print(path+str(year)+"*.dat")
    appended_data = []
    files = glob.glob(path+str(year)+"*.dat")
    print(files)
    for infile in files:
        print(infile)
        data = pd.read_csv(infile, usecols=cols)
        print(data)
        if ML == True:
            data = data[data['MIXDEPTH'] <= data['altitude']]                  
    return data
    
def bin_data(df, number_of_bins, var='accumulated'):
    df['bins'] = pd.cut(df[var], number_of_bins)
    df.loc[:, "bin_centres"] = df["bins"].apply(lambda x: x.mid)
    return df
    
def produce_year_dicts_with_groupby(inpath_processed_hysplit_dfs, cols=['latitude', 'grid_lat', 'RAINFALL'], col_to_average='RAINFALL', 
                                    col_to_bin='latitude', bins=np.arange(0, 90, 1), ML=True, years=None):
    dict_year_to_ds_groupby = {}
    if years is None:
        years = np.arange(2002, 2021, 1)
        print(years)
    for year in years:    
        print("year: "+str(year))
        data = load_appended_yearly_traj_data(inpath_processed_hysplit_dfs, year, folder='abs', cols=cols, ML=ML)
        data = bin_data(data, bins, var=col_to_bin)
        ds_groupby = data.groupby(['bin_centres']).agg({col_to_average: np.nanmean})
        ds_groupby = ds_groupby.sort_index(ascending=False)

        dict_year_to_ds_groupby[year] = ds_groupby
    return dict_year_to_ds_groupby
    
def lat_plot(ds_groupby, title, c, var, ax=None):
    ax.plot(ds_groupby.index, ds_groupby[var], 'o-', mfc='None', label=title, c=c, ms=5)
    ax.set_ylabel('Average rainfall experience by 1$\degree$x1$\degree$ grid [mm]')
    ax.set_xlabel('Gridded latitude w.r.t to North Pole [$\degree$]')
    ax.legend()
    return ax

def make_plot_trajs_lat(dict_year_to_ds_groupby, var):
    fig, ax = plt.subplots(figsize=(10,5))

    years = list(dict_year_to_ds_groupby.keys())
    norm = mpl.colors.Normalize(vmin=1, vmax=len(years))
    cmap = cmocean.cm.thermal
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    for year, ds_groupby in dict_year_to_ds_groupby.items():
        lat_plot(ds_groupby, title=str(year), c=m.to_rgba(int(year-years[0])), var=var, ax=ax)
    ax.invert_xaxis()
    trans = ax.get_xaxis_transform()
    x = 79
    ax.axvline(x, ls=':', c='k')
    plt.text(x, .01, 'ZEP = 78.906$\degree$N', transform=trans)
    plt.ylim(0, 0.2)
    plt.legend(fontsize=10, markerscale=1., frameon=False, bbox_to_anchor=(1.01,1), borderaxespad=0)
    plt.show()
    return fig
    
###################CLUSTERS###############################

def merge_with_obs(df_coord, df_obs):
    #df_obs = remove_np_nan_and_inf(df_obs)
    #df_obs = convert_index_to_datetime(df_obs)
    #df_obs = remove_duplicates(df_obs)
    
    #df_coord = remove_np_nan_and_inf(df_coord)
    #single index 
    if len(df_coord.index.names) == 1:   
        print("single index")
        #df_coord = convert_index_to_datetime(df_coord)
    #multiindex
    if len(df_coord.index.names) == 2:   
        print("multi index")
        print(df_coord.index.names)
        df_coord.index.names = ['arrival_time', 'traj_num']
        df_coord = df_coord.reset_index(level=1)
        df_coord.index = pd.to_datetime(df_coord.index)

    df_coord = df_coord.merge(df_obs, left_index=True, right_index=True)    
    return df_coord
    
def get_list_files(df_obs, traj_length = 241, year = 2002, 
                   inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\"):
    list_of_HYSPLIT_files = list_files(year, inpath_processed_hysplit_dfs) 
    HYSPLIT_names = create_HYSPLIT_name_from_list(df_obs.index)
    matching_list_of_back_traj_files = find_matching_files(list_of_HYSPLIT_files,
                                                                  HYSPLIT_names)
    return matching_list_of_back_traj_files
    
def groupby_lat_lon(ds_all_files, type_of_function_obs=np.count_nonzero,
                      threshold=50):
    ds_groupby = ds_all_files.groupby(['grid_lat', 'grid_lon']).agg({'Traj_num': type_of_function_obs})
    df_counts = ds_groupby.reset_index()
    df_threshold = df_counts[df_counts['Traj_num'] > threshold]
    return df_threshold
    
def create_count_array(df_threshold):
    df_threshold = df_threshold.copy()    
    df_threshold['grid_lat']  = df_threshold.loc[:,'grid_lat'] - 1
    df_threshold['grid_lon']  = df_threshold.loc[:,'grid_lon'] - 1    
    rowIDs = df_threshold['grid_lat'].astype(int)
    colIDs = df_threshold['grid_lon'].astype(int) 
    array = np.zeros((90,360))
    print(array.shape)
    array[rowIDs, colIDs] = df_threshold['Traj_num'].values
    count_array = array
    return count_array
    
def plot_clusters(ds_all_files, cluster, produce_absolute=False, produce_norm=True,
                 year='', title=None, cmap=None, vmax=None, savepath=None, dict_cluster_to_df_mean=None,
                 threshold=0, scientific_notation=True, colourbar_label=None, s=6, alpha=0.5, dict_cluster_to_colors=None, 
                 plot_log=False, vminlog=10**(-5), vmaxlog=10**(0)):
                 
    df_cluster = ds_all_files[ds_all_files['clusters_5'] == cluster].copy()    
    df_threshold = groupby_lat_lon(df_cluster, type_of_function_obs=np.count_nonzero,
                                        threshold=threshold)
    print(df_threshold)
    count_array = create_count_array(df_threshold)  
    
    if title is None:
        title=str(year)+' - cluster: '+str(cluster)
    if cmap is None:
        cmap = cmocean.cm.turbid
    if vmax is None:
        vmax = count_array.max()     
        
    if produce_absolute == True:
        fig = circular_plots_array(count_array, vmax=vmax, cmap=cmap, 
                   orientation='vertical', colourbar_label=r"Count per grid box traversed [-]", 
                   colourbar_labelsize=20, colourbar_tick_fontsize=12, decimal_places_colourbar=0, 
                   scientific_notation=False, axes_projection='North_Stereo', array_projection='rotated_pole', 
                                      title=title, plot_log=plot_log, vminlog=vminlog, vmaxlog=vmaxlog)

    if produce_norm == True:
        max_value = np.nanmax(count_array)
        sum_value = np.sum(count_array)
        print("normalise by sum value: "+str(sum_value))
        norm_count_array = count_array/sum_value
        norm_count_array[norm_count_array == 0] = np.nan
        
        if colourbar_label is None:
            colourbar_label=r"Norm. count per grid box traversed [-]"
        
        if savepath is not None:
            #save_norm_array
            if sum_value > 0:
                np.savetxt(savepath+str(cluster)+'.txt', norm_count_array)

        df_cluster=dict_cluster_to_df_mean[cluster]
        colour_cluster = 'k'
        if dict_cluster_to_colors is not None:
            colour_cluster=dict_cluster_to_colors[cluster]
        fig = circular_plots_array(norm_count_array, vmax=vmax, cmap=cmap, 
                       orientation='vertical', colourbar_label=colourbar_label, 
                       colourbar_labelsize=20, colourbar_tick_fontsize=12, decimal_places_colourbar=2, 
                       scientific_notation=scientific_notation, axes_projection='North_Stereo', array_projection='rotated_pole', 
                                          title=title, df_cluster=df_cluster, colour_cluster=colour_cluster, s=s, alpha=alpha, plot_log=plot_log,
                                          vminlog=vminlog, vmaxlog=vmaxlog)
    return fig
    
####CWT##########################################################################################################################

def get_full_season_abb_years(start_year, number_years, first_season='SBU'):
    """loops through the years and the seasons to get a full mapping for the season and year and it's respective order"""   
    season_list=['AHZ','SUM','SBU']
    season_abb_years = []    
    if first_season == 'SBU':
        print(first_season)
        number_years = number_years + 1
        print(start_year)
    for year in np.arange(start_year, start_year+number_years+1, 1):
        for season_abb in season_list: #correct order
            season_abb_year = str(season_abb) + '_' + str(year)            
            season_abb_years.append(season_abb_year)
            if season_abb == 'SBU':
                break
    index = [idx for idx, s in enumerate(season_abb_years) if str(first_season) in s][0]
    season_abb_years = season_abb_years[index:]
    seasons_num = np.arange(1,len(season_abb_years)+1,1)
    season_to_season_num = dict(zip(season_abb_years, seasons_num))
    return season_to_season_num
    
def add_season(df, season_to_season_num):
    month_to_season =  { 1:'SBU',  2:'AHZ', 3:'AHZ',  
                     4:'AHZ',  5:'AHZ', 6:'SUM',  7:'SUM',  8:'SUM', 9:'SUM', 10:'SBU', 
                     11:'SBU', 12:'SBU'}     
    df['arrival_time'] = pd.to_datetime(df['arrival_time'])
    df.loc[:,'month_num'] = df['arrival_time'].dt.month
    df.loc[:,'year'] = df['arrival_time'].dt.year  
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df['arrival_time'].dt.year.astype(str)
    #jan is part of SBU, move jan & SBU to pervious year - SBU AND Month = 1, alter year - 1.
    df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    df.loc[:,'season_ordinal'] = df['season_abb_year'].map(season_to_season_num)   
    return df    

def add_residence_times_per_grid_per_traj(df_trajs):
    DFs = []
    for i in range(len(df_trajs)):
        df_traj = df_trajs[i]
        ds_groupby = df_traj.groupby(['grid_lat', 'grid_lon']).agg({'Traj_num': 'count'})
        df_counts = ds_groupby.reset_index()
        df_counts = df_counts.rename({'Traj_num':'hits'}, axis=1)
        df_merged = pd.merge(df_traj, df_counts, how='left', left_on=['grid_lat', 'grid_lon'], 
                         right_on = ['grid_lat', 'grid_lon'])
        DFs.append(df_merged)
    df_residence_times = pd.concat(DFs)
    return df_residence_times
    
def produce_CWT_array(df_res, var='obs', threshold=None):
    print("threshold: "+str(threshold))
    df_res['CRT'] = df_res[var]*df_res['hits'] #observation per specific arrival time x residence time per grid cel for specific arrival time 
    df_res_groupby = df_res.groupby(['grid_lat', 'grid_lon']).agg({'CRT': 'sum'}) #sum all 
    sum_grids = df_res_groupby.reset_index()    
    #TOTAL RESIDENCE TIME 
    ds_groupby = df_res.groupby(['grid_lat', 'grid_lon']).agg({'hits': 'sum'})
    ds_groupy_total = ds_groupby.reset_index()
    ds_groupy_total = ds_groupy_total.rename({'hits':'hits_total'}, axis=1)        
    df_sum_grids = pd.merge(sum_grids, ds_groupy_total, how='left', left_on=['grid_lat', 'grid_lon'], 
                             right_on = ['grid_lat', 'grid_lon'])
    df_sum_grids['CRT_per_hits_total'] = df_sum_grids['CRT']/df_sum_grids['hits_total']
    df_arrays_hits = create_empty_arrays_hits() #empty array
    df_merged_array_CWT = pd.merge(df_arrays_hits, df_sum_grids, how='left', left_on=['latitude', 'longitude'], right_on = ['grid_lat', 'grid_lon'])
    df_merged_array_CWT = df_merged_array_CWT.drop(['grid_lat', 'grid_lon'], axis=1)  
    array = turn_grid_df_to_array(df_merged_array_CWT, 'CRT_per_hits_total', threshold=threshold)
    return array
    
def load_and_create_CWT_array(year, folder='abs', data_format='.dat', path="E:\\Data\\HYSPLIT\\processed\\"):
    dict_dtype = dict(zip(['Traj_num', 'grid_lon', 'grid_lat', 'obs', 'arrival_time', 'MIXDEPTH', 'altitude'],
                      [np.int8, np.int16, np.int8, np.float16, object, np.float16, np.float16]))
    
    df = concat_years(path, folder=folder, years=np.arange(year, year+1, 1),
                             index_col=0, parse_dates=True, dict_dtype=dict_dtype,
                             usecols=['DateTime', 'Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time', 'MIXDEPTH',
                                 'altitude'], data_format=data_format)
    #select for mix layer        
    df = df.loc[df['MIXDEPTH'] > df['altitude'], :]    
    df_trajs = [x for _, x in df.groupby('arrival_time')]
    df_res = add_residence_times_per_grid_per_traj(df_trajs)
    array = produce_CWT_array(df_res)
    return array
    
def create_CWT_array_from_df(df):
    #select for mix layer 
    print("create CWT array from df")
    df = df.loc[df['MIXDEPTH'] > df['altitude'], :]    
    df_trajs = [x for _, x in df.groupby('arrival_time')]
    df_res = add_residence_times_per_grid_per_traj(df_trajs)
    array = produce_CWT_array(df_res)
    return array
        
def append_dfs_residence_times(years, season_to_season_num, path="E:\\Data\\HYSPLIT\\processed\\", folder='abs',
                               select_for_mixed_layer=True, data_format='.dat', 
                               usecols=['Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time', 'MIXDEPTH',
                                         'altitude', 'RAINFALL']):
    dict_dtype = dict(zip(['Traj_num', 'grid_lon', 'grid_lat', 'obs', 'arrival_time', 'MIXDEPTH', 'altitude'], 
                          [np.int8, np.int16, np.int8, np.float16, object ,np.float16, np.float16]))
    DFs = []
    for year in years:    
        print(path)
        df = concat_years(path=path, folder=folder, years=np.arange(year, year+1, 1),
                                 index_col=0, parse_dates=True, dict_dtype=dict_dtype,
                                 usecols=usecols, data_format=data_format, select_for_mixed_layer=select_for_mixed_layer)
        # if select_for_mixed_layer == True:
            # df = df.loc[df['MIXDEPTH'] > df['altitude'], :]  #select for mix layer altitude of endpoint is below mix layer height
        df_trajs = [x for _, x in df.groupby('arrival_time')] #separate the individual trajs from the big data set
        df_res = add_residence_times_per_grid_per_traj(df_trajs) #calculate the res times for each traj and for each grid cell     
        df_res = add_season(df_res, season_to_season_num) #add season columns (more datetimes)
        DFs.append(df_res)
    df_res_concat = pd.concat(DFs)
    return df_res_concat
    
def save_seasonal_dfs(df_res_concat, savepath='E:\\Data\\HYSPLIT\\processed\\abs\\dfs\\seasons',
                      select_for_mixed_layer=True, data_format='.dat'):
    """this will be important for displaying indivdual seasons of interest and the trend analysis of the grid cell"""
    season_ordinals = df_res_concat['season_ordinal'].unique()
    print("Unique season numbers: "+str(season_ordinals))
    for season in season_ordinals:        
        df_season = df_res_concat[df_res_concat['season_ordinal'] == season]
        if select_for_mixed_layer == True:
            ml = 'ML'
        save_df(df_season, savepath, name='season_'+str(season)+'_'+str(ml), 
                index=False, float_format=None, data_format=data_format)
                       
def save_append_CWT_arrays(years, path="E:\\Data\\HYSPLIT\\processed\\", folder='abs',
                      savepath='E:\\Data\\HYSPLIT\\processed\\abs\\arrays\\',
                      data_format='.dat', var='abs637',
                      usecols=['DateTime', 'Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time', 'MIXDEPTH','altitude'],
                      select_for_mixed_layer=True, select_for_above_mixed_layer=False):
    dict_dtype = dict(zip(['Traj_num', 'grid_lon', 'grid_lat', 'obs', 'arrival_time'], [np.int8, np.int16, np.int8, np.float16]))
    arrays_CWT = []
    for year in years:    
        df = concat_years(path=path, folder=folder, years=np.arange(year, year+1, 1),
                                 index_col=0, parse_dates=True, dict_dtype=dict_dtype, data_format=data_format,
                                 usecols=usecols, select_for_mixed_layer=True, select_for_above_mixed_layer=False)
        df_trajs = [x for _, x in df.groupby('arrival_time')] #sepaerate into the different trajs, one df for every traj
        df_res = add_residence_times_per_grid_per_traj(df_trajs) #calculate their respective residence times      
        array = produce_CWT_array(df_res, var=var) #this involves 
        np.savetxt(savepath+str(year)+'.txt', array)
        arrays_CWT.append(array)
    return arrays_CWT
    
def CWT_example_year(year, path="E:\\Data\\HYSPLIT\\processed\\", folder='abs', 
                     savepath='E:\\Data\\HYSPLIT\\processed\\abs\\arrays\\'):
    dict_dtype = dict(zip(['Traj_num', 'grid_lon', 'grid_lat', 'obs', 'arrival_time'], [np.int8, np.int16, np.int8, np.float16]))
    df_year = concat_years(path=path, folder=folder, years=np.arange(year, year+1, 1),
                                 index_col=0, parse_dates=True, dict_dtype=dict_dtype,
                                 usecols=['DateTime', 'Traj_num', 'grid_lat', 'grid_lon', 'obs', 'arrival_time', 'MIXDEPTH','altitude'])
    #select for mix layer
    df_year = df_year.loc[df_year['MIXDEPTH'] > df_year['altitude'], :]
    df_year_trajs = [x for _, x in df_year.groupby('arrival_time')]
    df_res_year = add_residence_times_per_grid_per_traj(df_year_trajs)

    season_to_season_num = get_full_season_abb_years(start_year=year, number_years=19, first_season='AHZ') 
    df_res_year = add_season(df_res_year, season_to_season_num) #add season columns
    array_year = produce_CWT_array(df_res_year) #produce array   
    np.savetxt(savepath+str(year)+'.txt', array_year) #save array

    fig = circular_plots_array(array_year, vmax=2, cmap=cmocean.cm.turbid, 
                   orientation='vertical', colourbar_label=r"CWT $\sigma_{\mathrm{ap}}$ per grid box traversed [Mm$^{-1}$]", 
                   colourbar_labelsize=25, colourbar_tick_fontsize=12, decimal_places_colourbar=3,
                   scientific_notation=False, axes_projection='North_Stereo', array_projection='rotated_pole', title=year)
    return fig
    
####trends############################################################################################################################################

def convert_season_add_year_to_datetime(season_abb_year):
    year = str(season_abb_year)[-4:]
    season_abb = str(season_abb_year)[:3] 
    if season_abb == 'AHZ':
        start = year+'-02-'+'01'
        stop = year+'-05-'+'31'
    if season_abb == 'SUM':
        start = year+'-06-'+'01'
        stop = year+'-09-'+'30'        
    if season_abb == 'SBU':
        start = year+'-10-'+'01'
        stop = str(int(year)+1)+'-01-'+'31'
    start = pd.to_datetime(start)
    stop = pd.to_datetime(stop)
    return start, stop
    
def add_mid_datetime_using_dictionary(df, season_num_to_season):
    df['season_abb_year'] = df.index.map(season_num_to_season)
    df['start'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[0])
    df['stop'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[1])
    df['mid_datetime'] = df.apply(lambda x: mid_datetime_function(x.start, x.stop), axis=1)
    return df
    
def calculate_midpoint(year):
    a = datetime(year, 1, 1)
    b = datetime(year, 12 ,31)
    midpoint = a + (b - a)/2
    return midpoint

def theil_sen(x,y):   
    x = np.array(x)
    y = np.array(y)
    idx = np.where(~np.isnan(y))    
    x = x[idx]
    y = y[idx]        
    if len(y)<=1:
        theil_slope = np.nan        
    else:
        res = stats.theilslopes(y, x, 0.95) #returns medslope, medintecept, lo_slope, up_slope
        theil_slope = res[0]
    return theil_slope

def create_df_with_trend(df_arrays_hits, x, trend_name='trend', freq=1):
    df_hits_trends = df_arrays_hits.loc[:, ~df_arrays_hits.columns.isin(['latitude', 'longitude'])]
    for index, row in df_hits_trends.iterrows():      
        y = row.values #hits for this i,j      
        theil_slope = theil_sen(x,y) #freq=24*365) #hours
        print("Trend: "+str(theil_slope)) #trend 
        df_hits_trends.loc[index, trend_name] = theil_slope*freq
    df_hits_trends['latitude'] = df_arrays_hits.iloc[:,0] #add them back the first columns
    df_hits_trends['longitude'] = df_arrays_hits.iloc[:,1]
    df_hits_trends['count'] = len(df_hits_trends.columns) - df_hits_trends.isnull().sum(axis = 1)  #df_arrays_hits.iloc[:,-1]
    return df_hits_trends
    
def produce_ordinals(df, format_datetime='%Y-%m-%d %H:%M:%S'):
    x_datetimes = df.loc[:, ~df.columns.isin(['latitude', 'longitude'])].columns
    x_datetimes = [pd.to_datetime(x, format=format_datetime) for x in x_datetimes]    
    x_days = [datetime.datetime.strptime(str(e), format_datetime).date() for e in x_datetimes] 
    x_days = [e.toordinal() for e in x_days]
    first_day = x_days[0]
    x_days = [(e-first_day) for e in x_days]
    x_days = list(set(x_days))
    x_days.sort()
    return x_days
    
def season_hits(season_arrays_CWT_list, go_back=0, season_to_season_num=None):
    df_arrays_hits = create_empty_arrays_hits()    
    if go_back > 0:
        season_arrays_CWT_list_reduced = season_arrays_CWT_list[len(season_arrays_CWT_list)-go_back:]    
    if go_back == 0:
        season_arrays_CWT_list_reduced = season_arrays_CWT_list[:] #not reduced
    season_num_to_season = {v: k for k, v in season_to_season_num.items()} #reverse dict to have num to season
    for i in range(len(season_arrays_CWT_list_reduced)):
        if go_back > 0:
            season_year = season_num_to_season[i+1+len(season_arrays_CWT_list)-go_back]   
        if go_back == 0:  
            season_year = season_num_to_season[i+1]
        start, stop = convert_season_add_year_to_datetime(season_year)
        midpoint = start + (stop - start)/2
        array = season_arrays_CWT_list_reduced[i]
        df_array = pd.DataFrame(array)
        df_array["index"] = list(np.arange(0,90,1))
        l = list(np.arange(0,len(array[0,:]),1))
        df_melt = pd.melt(df_array, id_vars=['index'], value_vars=l)
        df_melt = df_melt.rename(columns={'index': 'latitude', 'variable':'longitude', 'value':'var'})
        df_arrays_hits[str(midpoint)] = df_melt['var']
    return df_arrays_hits

def take_arrays_create_df_columns_for_time(arrays_CWT, start_year = 2002):
    """requires a list of arrays to form the df"""
    df_arrays_hits = create_empty_arrays_hits() #empty array
    for i in range(len(arrays_CWT)):
        year = start_year + i  
        print("year: "+str(year))
        array = arrays_CWT[i]
        df_array = pd.DataFrame(array)
        df_array["index"] = list(np.arange(0,90,1))
        l = list(np.arange(0,len(array[0,:]),1))
        df_melt = pd.melt(df_array, id_vars=['index'], value_vars=l)
        df_melt = df_melt.rename(columns={'index': 'latitude', 'variable':'longitude', 'value':'var'})
        midpoint = calculate_midpoint(year)
        df_arrays_hits[str(midpoint)] = df_melt['var']
    return df_arrays_hits
    
def create_df_with_trend(df_arrays_hits, x, trend_name='trend', freq=1):
    df_hits_trends = df_arrays_hits.loc[:, ~df_arrays_hits.columns.isin(['latitude', 'longitude'])]
    for index, row in df_hits_trends.iterrows():      
        y = row.values #hits for this i,j      
        theil_slope = theil_sen(x,y) #freq=24*365) #hours
        print("Trend: "+str(theil_slope)) #trend 
        df_hits_trends.loc[index, trend_name] = theil_slope*freq
    df_hits_trends['latitude'] = df_arrays_hits.iloc[:,0] #add them back the first columns
    df_hits_trends['longitude'] = df_arrays_hits.iloc[:,1]
    df_hits_trends['count'] = len(df_hits_trends.columns) - df_hits_trends.isnull().sum(axis = 1)  #df_arrays_hits.iloc[:,-1]
    return df_hits_trends
    
def produce_ordinals(df, format_datetime='%Y-%m-%d %H:%M:%S'):
    x_datetimes = df.loc[:, ~df.columns.isin(['latitude', 'longitude'])].columns
    x_datetimes = [pd.to_datetime(x, format=format_datetime) for x in x_datetimes]    
    x_days = [datetime.datetime.strptime(str(e), format_datetime).date() for e in x_datetimes] 
    x_days = [e.toordinal() for e in x_days]
    first_day = x_days[0]
    x_days = [(e-first_day) for e in x_days]
    x_days = list(set(x_days))
    x_days.sort()
    return x_days
    
def load_append_CWT_array_seasons(path='E:\\Data\\HYSPLIT\\processed\\abs\\dfs\\seasons\\',
                                  data_format='.dat'):
    season_arrays_CWT = {}
    list_of_season_files = glob.glob(path+'*'+str(data_format)) #files saved
    dict_dtype = dict(zip(['Traj_num', 'grid_lon', 'grid_lat', 'obs', 'arrival_time', 'MIXDEPTH', 'altitude'],
                          [np.int8, np.int16, np.int8, np.float16, object, np.float16, np.float16]))
    for file in list_of_season_files:
        season = re.findall(r'\d+', file)[0]  
        if data_format == '.dat':
            df_season = pd.read_csv(file, index_col=0, parse_dates=True, dtype=dict_dtype) 
        if data_format == '.pickle':
            df_season = pd.read_pickle(file)
        array_season = produce_CWT_array(df_season)
        season_arrays_CWT[season] = array_season
    return season_arrays_CWT
    
def load_year_arrays(years, path='E:\\Data\\HYSPLIT\\processed\\abs\\arrays\\'):
    arrays_CWT = [] 
    for year in years: 
        print(year)
        array = np.loadtxt(path+str(year)+'.txt') 
        arrays_CWT.append(array)
    return arrays_CWT
    
