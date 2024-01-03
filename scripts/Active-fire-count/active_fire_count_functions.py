import pandas as pd
import glob
import calendar
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cmocean
from datetime import datetime, timedelta
import numpy as np
import matplotlib.path as mpath
import matplotlib.ticker as ticker
import cartopy as cy
import os
import matplotlib.dates as mdates 
import matplotlib.lines as mlines

def load_df(loadpath, extrapath=None, filename=None, formatdata=".dat"):
    """load dataframe"""
    if extrapath is not None:
        print("loading: "+str(loadpath+'\\'+extrapath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+extrapath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False)
    if extrapath is None:
        print("loading: "+str(loadpath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False)        
    return df
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', format_of_plot=".jpeg", dpi=300):
    folders = glob.glob(path)
    if folder not in folders:
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(format_of_plot), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(format_of_plot)))
    
def save_df(df, path, name='', float_format=None):
    print("Save as: "+str(path+'\\'+name+'.dat'))
    df.to_csv(path+'\\'+name+'.dat', index=True, float_format=float_format)
    
def to_datetime(x):
    return datetime.strptime(x, '%Y-%m-%d %H:%M:%S')
    
def sf(sf_num):
    sf = '{0:.'+str(sf_num)+'f}'
    return sf
	
def load_fire_data_for_year(firepath, year):
    path = firepath+'*'+str(year)
    files = glob.glob(path+'*\\*.csv')
    print("files: "+str(len(files)))
    df_fire = pd.read_csv(files[0], sep=',', parse_dates=['acq_date']) 
    df_fire.set_index('acq_date', inplace=True)
    df_fire['month'] = df_fire.index.month
    df_fire['month_abbr'] = df_fire['month'].apply(lambda x: calendar.month_abbr[x])
    #print(df_fire['month_abbr'].unique())
    df_fire = df_fire[(df_fire['latitude'] > 0)] #just northern hemisphere
    #print(len(df_fire))
    return df_fire
    
def create_projections_dict(ZEP_lat, ZEP_lon):
    dict_projections = {}
    
    geo = ccrs.Geodetic(); dict_projections['geo'] = geo
    rotated_pole = ccrs.RotatedPole(pole_latitude=ZEP_lat, pole_longitude=ZEP_lon)
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
    
def make_fire_plot(df_fire, axes_projection='ortho', array_projection='PlateCarree',
                   title=None, ZEP_lat=78.906,ZEP_lon=11.888):
    fig, ax = plt.subplots(figsize=(8,8))
    ax.axis("off")    
    
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon)                   
    projection = dict_projections[axes_projection]
    array_projection = dict_projections[array_projection]
    
    ax = plt.axes(projection=projection)        
    ax.set_global()    
    ax.coastlines(resolution='50m')
    ax.gridlines()
    
    fire_lats = df_fire['latitude'].values
    fire_lons = df_fire['longitude'].values
    brightness = df_fire['bright_t31'].values

    cb_fires = ax.scatter(fire_lons, fire_lats, 
             marker='*', s = 10, cmap=cmocean.cm.thermal, vmin=200, vmax=370, c=brightness, #maybe use for fires
             transform=array_projection, label='Active fires (VIIRS)')

    CB = plt.colorbar(cb_fires, orientation='vertical', shrink=0.8)
    CB.ax.set_ylabel('Brightness temperature of VIIRS [K]', labelpad = 20, rotation=270, fontsize=15)
    
    ax.legend(frameon=False, fontsize='large', markerscale=0.8, loc=1)
    plt.title(title, loc='left')
    plt.show()
    return fig
    
def plot_slice_dates(df_fire, start_date, end_date):
    """slice the dataframe with fires"""
    df_fire_dates = df_fire[(start_date <= df_fire.index) & (df_fire.index <= end_date)]
    fig = make_fire_plot(df_fire_dates, title=str(start_date)+' - '+str(end_date))
    return fig
    
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
    print("Year: "+str(year))    
    #inpath_raw_hysplit_data+str(dict_year_to_HYSPLIT_file[year])+"\\" #traj2002_10_hourly\\"
    print("Path: "+str(inpath_processed_hysplit_dfs+str(year)+"\\"))
    list_of_files = glob.glob(inpath_processed_hysplit_dfs+str(year)+"\\"+'*')    
    print("Number of HYSPLIT files for "+str(year)+": "+str(len(list_of_files)))
    return list_of_files
    
def find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names):
    """observation indexes we want to match with the list of HYSPLIT files"""
    matching_list_of_back_traj_files = [s for s in list_of_HYSPLIT_files if any(xs in s for xs in HYSPLIT_names)]
    print("Matching files from observational data & HYSPLIT: "+str(len(matching_list_of_back_traj_files)))
    return matching_list_of_back_traj_files
    
def create_traj_dictionary_using_matching_files(matching_list_of_back_traj_files, select_for_mixed_layer=True, number_of_days=10,
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
                traj_ds = pd.read_pickle(traj_i)
                traj_ds.index = pd.to_datetime(traj_ds.index)
                
            #print("cut the traj by :"+str(number_of_days)+" days")
            #traj_ds = traj_ds[traj_ds['time_step'] > -(number_of_days)*24]            
            #define arrival time before 
            arrival_time_of_traj = str(traj_ds.index[0])
            
            print("arrival time: "+str(arrival_time_of_traj))            
            if select_for_mixed_layer == True:                
                traj_ds = traj_ds[traj_ds['MIXDEPTH'] >= traj_ds['altitude']] #select for mixed-layer
            if len(traj_ds) > 0:             
                trajs_dictionary[str(arrival_time_of_traj)] = traj_ds 
        except IndexError:
            print("Index Error - end")
    trajs_dictionary
    return trajs_dictionary
    
def create_array(df):   
    empty_array = make_empty_array()
    array = empty_array
    
    for i in range(len(df)):
        latitude = df['latitude'].iloc[i]
        longitude = df['longitude'].iloc[i]

        if longitude >= 0:
            grid_lon = int(abs(longitude+180))
        if longitude < 0:
            grid_lon = int(180-abs(longitude))

        grid_lon = int(grid_lon+1)       
        grid_lat = int(latitude+1)
        
        try:        
            array[grid_lat,grid_lon] += 1
        except IndexError:
            print("Index Error")
            #print(latitude, longitude)
            #print(grid_lat, grid_lon)
    return array
    
def make_empty_array():
    longitude_array = np.arange(-180, 180, 1)
    latitude_array = np.arange(0, 90, 1)
    array = np.indices((latitude_array.shape[0], longitude_array.shape[0]))
    array = array[0] + array[1]
    array = np.empty((90,360))
    array.fill(0) 
    return array
    
def create_fire_freq_array(df_fire, trajs_dictionary, arrival_time):
    df_traj = trajs_dictionary[str(arrival_time)]
    start_traj_time = to_datetime(str(df_traj.index[0]))
    end_traj_time = to_datetime(str(df_traj.index[-1]))
    df_fire_sliced_timewise = df_fire[(end_traj_time <= df_fire.index) & (df_fire.index <= start_traj_time)]
    fire_freq_array = create_array(df_fire_sliced_timewise)
    return fire_freq_array
    
def plot_fire_array(array, df_traj, df_fire, fire_count_for_traj='', axes_projection='ortho', array_projection='PlateCarree',
                    title=None, ZEP_lat=78.906,ZEP_lon=11.888, vmax=10, plot_fire=True, extend=False):
    
    fig, ax = plt.subplots(figsize=(6,6))
    ax.axis("off")  
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon)                   
    projection = dict_projections[axes_projection]
    array_projection = dict_projections[array_projection]
    
    ax = plt.axes(projection=projection)        
    ax.set_global()    
    ax.coastlines(resolution='10m')
    ax.gridlines()    
   
    lons = np.arange(-180.5, 179.5, 1)  
    lats = np.arange(-0.5, 89.5, 1) 
    cs = ax.pcolormesh(lons, lats, array, transform=array_projection, vmin=0, vmax=vmax, cmap='cmo.thermal')  
    
    cbar = fig.colorbar(cs, extend='both', orientation='vertical')
    cbar.set_label(r"Number of active fires in grid box Fredquency [-]", size=12)
    cbar.ax.tick_params(labelsize=12)
    
    lon_traj = df_traj['longitude'].values
    lat_traj = df_traj['latitude'].values
    
    ax.scatter(lon_traj, lat_traj, s=5, marker="o", transform=ccrs.PlateCarree(), label='Traj', alpha=0.5,
               facecolors='none', edgecolors='r')
    
    ###CHECK
    if plot_fire == True:
        print("Length of df_fire: "+str(len(df_fire)))
        fire_lats = df_fire['latitude'].values
        fire_lons = df_fire['longitude'].values
        ax.scatter(fire_lons, fire_lats, 
                marker='*', s = 10, alpha=0.2, transform=array_projection)

    if extend:
        ax.set_extent([-10, 20, 35, 70], ccrs.PlateCarree())
    
    number_of_grids_with_active_fires = np.count_nonzero(~np.isnan(array)) 
    sfs = sf(3)    
    percentage_of_grids = sfs.format(number_of_grids_with_active_fires/(360*90))
    number_of_active = np.nansum(array)
    
    ax.set_title("Number of grids with active fires: "+str(number_of_grids_with_active_fires)+' ('+str(percentage_of_grids)+'%)'+
                 '\nNumber of active fires: '+str(number_of_active)+
                 '\nTime interval for traj: '+str(df_traj.index[0])+' - '+str(df_traj.index[-1])+
                 '\nTime interval for fire: '+str(df_fire.index[0])[:10]+' - '+str(df_fire.index[-1])[:10]+
                 '\nFire count: '+str(fire_count_for_traj), loc='left', fontsize=12)
    
    ax.legend(markerscale=4, frameon=False, fontsize=15)
    plt.show()
    return fig
    
def produce_df_fire_trajs_dictionary(df_obs, year, fire_path='C:\\Users\\DominicHeslinRees\\Documents\\Data\\NASA_FIRE\\MODIS\\',
                                    inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\", obs_indexes=None, number_of_days=10,
                                    select_for_mixed_layer=True):
    df_fire = load_fire_data_for_year(fire_path, year)
    if df_obs is not None:
        obs_indexes = list(df_obs.index)
    if df_obs is None:
        print("use list")        
    HYSPLIT_names_obs = create_HYSPLIT_name_from_list(obs_indexes)
    list_of_HYSPLIT_files = list_files(year, inpath_processed_hysplit_dfs)
    matching_list_of_back_traj_files = find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names_obs)
    trajs_dictionary = create_traj_dictionary_using_matching_files(matching_list_of_back_traj_files, 
                       select_for_mixed_layer=select_for_mixed_layer, number_of_days=number_of_days)
    return df_fire, trajs_dictionary
    
def slice_df_fire(df_fire, trajs_dictionary, arrival_time):
    df_traj = trajs_dictionary[str(arrival_time)]
    start_traj_time = to_datetime(str(df_traj.index[0]))
    end_traj_time = to_datetime(str(df_traj.index[-1]))   
    start_traj_date = to_datetime(str(df_traj.index[0])).date()
    end_traj_date = to_datetime(str(df_traj.index[-1])).date() 
    print("start date: "+str(start_traj_date))
    print("end date: "+str(end_traj_date))
    df_fire_sliced_timewise = df_fire[(df_fire.index >= datetime.strptime(str(end_traj_date), '%Y-%m-%d')) & (df_fire.index <= datetime.strptime(str(start_traj_date), '%Y-%m-%d'))]
    return df_fire_sliced_timewise
    
def example_plots(df_fire, trajs_dictionary, arrival_time):
    year = int(arrival_time[:4])
    fire_freq_array = create_fire_freq_array(df_fire, trajs_dictionary, arrival_time)
    fire_freq_array[fire_freq_array == 0] = np.nan
    df_fire_sliced_timewise = slice_df_fire(df_fire, trajs_dictionary, arrival_time)
    fig1 = plot_fire_array(fire_freq_array, df_fire_sliced_timewise, trajs_dictionary[arrival_time], 
                    fire_count_for_traj='', array_projection='PlateCarree', plot_fire=True, extend=[-10, 20, 35, 70])
    plt.show()
    fig2 = plot_fire_array(fire_freq_array, df_fire_sliced_timewise, trajs_dictionary[arrival_time],
                    fire_count_for_traj='', array_projection='PlateCarree', plot_fire=False)
    plt.show()
    return fig1, fig2
    
def create_fire_dataframe(df_traj, array, arrival_time):    
    df_traj = df_traj[df_traj['latitude'] > 0]    
    df_traj_fire_count = pd.DataFrame(columns=['arrival_time', 'fire_count'])
    #df_traj_fire_count = pd.DataFrame(columns=['arrival_time', 'fire_count', 'unique_grid_count', 'total_grid_count'])
    fire_count = 0
    count = 0
    grids = []
    for i in range(len(df_traj)):         
        lon_traj = df_traj['longitude'].iloc[i]
        lat_traj = df_traj['latitude'].iloc[i]

        if lon_traj >= 0:
            grid_lon = int(abs(lon_traj+180))
        if lon_traj < 0:
            grid_lon = int(180-abs(lon_traj))

        grid_lon = int(grid_lon+1)       
        grid_lat = int(lat_traj+1)
       
        try:
            value = array[grid_lat,grid_lon]
            #print("value: "+str(value))
            #grids.append((grid_lat, grid_lon)) if need unique grids
            count = count + 1
        except IndexError:
            value = 0  
        fire_count = fire_count + value

    #unique = list(dict.fromkeys(grids))
    #number_of_unique = len(unique)   if need unique grids
    df_traj_fire_count = df_traj_fire_count.append({'arrival_time': arrival_time, 'fire_count': fire_count},  ignore_index=True)
    #df_traj_fire_count = df_traj_fire_count.append({'arrival_time': arrival_time, 'fire_count': fire_count, 'unique_grid_count':number_of_unique, 'total_grid_count':count}, ignore_index=True)
    return df_traj_fire_count
    
def append_fire_dfs(df_fire, trajs_dictionary, traj_arrival_times, year):    
    #print(year)
    dfs = []
    for arrival_time in traj_arrival_times:
        print('arrival_time: '+str(arrival_time))        
        df_traj=trajs_dictionary[arrival_time]
      
        if (len(df_traj) > 1):
            start_traj_time = to_datetime(str(df_traj.index[0]))
            end_traj_time = to_datetime(str(df_traj.index[-1]))

            start_traj_date = to_datetime(str(df_traj.index[0])).date()
            end_traj_date = to_datetime(str(df_traj.index[-1])).date()
            
            df_fire_sliced_timewise = df_fire[(df_fire.index >= datetime.strptime(str(end_traj_date), '%Y-%m-%d')) & (df_fire.index <= datetime.strptime(str(start_traj_date), '%Y-%m-%d'))]
            print("size of df fire "+str(len(df_fire_sliced_timewise)))
            
            if len(df_fire_sliced_timewise) > 0:
                #print("Grid size = "+str(90*360))
                fire_freq_array = create_array(df_fire_sliced_timewise) #create an array from the df fire
                fire_freq_array[np.isnan(fire_freq_array)] = 0  #if np.nan = 0, for fire     
                #create fire count
                df_traj_fire_count = create_fire_dataframe(df_traj=trajs_dictionary[arrival_time], 
                                                           array=fire_freq_array, arrival_time=arrival_time)
                dfs.append(df_traj_fire_count)
            else:
                pass
        else:
            pass
    res = pd.concat(dfs, ignore_index=True)
    return res
    
def save_year_df_fire_count(fire_path, year, df_obs, list_datatimes=None,
                            inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\"):
    df_fire = load_fire_data_for_year(fire_path, year=year) #fire dataset from NASA
    list_of_HYSPLIT_files = list_files(year, inpath_processed_hysplit_dfs=inpath_processed_hysplit_dfs)
    
    if list_datatimes == None:      
        print("index of df used:")
        obs_indexes = list(df_obs.index)
    if list_datatimes is not None:
        print("list is used:")
        list_datatimes_year = [x for x in list_datatimes if x.year==int(year)]
        obs_indexes = list_datatimes_year.copy()
        print("obs index:"+str(obs_indexes))
        
    HYSPLIT_names = create_HYSPLIT_name_from_list(obs_indexes)
    matching_list_of_back_traj_files = find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names)
    trajs_dictionary = create_traj_dictionary_using_matching_files(matching_list_of_back_traj_files)
    
    value_at_index = list(trajs_dictionary.values())[0]
    
    traj_arrival_times = [*trajs_dictionary.keys()] #dictionay keys are the arrival times
    df_fire_appended = append_fire_dfs(df_fire=df_fire, trajs_dictionary=trajs_dictionary, 
                                       traj_arrival_times=traj_arrival_times, year=year)
    return df_fire_appended
    
def process_data(fire_path, df_obs=None, outpath_datafiles = 'C:\\Users\\DominicHeslinRees\\Documents\\Data\\fire_dataset_abs\\',
                 process=False, list_datatimes=None, years=None, inpath_processed_hysplit_dfs="E:\\Data\\HYSPLIT\\processed\\"):
    if process == True:
        if years is None:  
            print("no years given:")
            if list_datatimes is not None:
                years = np.arange(list_datatimes[0].year, list_datatimes[-1].year + 1, 1)
                #print(years)
            if list_datatimes == None:
                print("must give dataframe instead of list of datetimes.")
                years = np.arange(df_obs.index[0].year, df_obs.index[-1].year + 1, 1) #
                #print(years)
            
        for year in years:    
            year = int(year)
            df_fire_appended = save_year_df_fire_count(fire_path, year, df_obs, list_datatimes,
                                                       inpath_processed_hysplit_dfs=inpath_processed_hysplit_dfs)
            df_fire_appended.to_csv(outpath_datafiles+str(year)+'.dat', index=False)
            print("saved as: "+str(outpath_datafiles)+str(year)+'.dat')
    if process == False:
        print("to process - select process == True")
        pass
        
def append_dfs(inpath_datafiles, index_col='arrival_time'):
    appended_data = []
    for file in glob.glob(inpath_datafiles+'*.dat'):
        print("file: "+str(file))
        df_fire_year = pd.read_csv(file, parse_dates=True, index_col=0)
        appended_data.append(df_fire_year)
    df_fire_count = pd.concat(appended_data)
    df_fire_count.index = pd.to_datetime(df_fire_count.index)
    return df_fire_count
        
def create_basis_plot(df_traj_fire_count, var='fire_count', ymax=200000, N=10, log=False, label='Whole Back Traj.'):    
    fig, ax = plt.subplots(figsize=(12,4))
    
    ax.plot(df_traj_fire_count.index, df_traj_fire_count[var], 'o', ms=1, label=label, c='k')

    ax.set_ylabel('Fire count: number of 1$\degree$x1$\degree$ grids \n containing active fires \nback trajectory pass through')
    ax.set_xlabel('arrival time of back trajectory')
    #ax.set_title('hourly resolution of back trajectories', loc='left')
    
    xmin, xmax = ax.get_xlim()
    ax.set_xticks(np.round(np.linspace(xmin, xmax, N), 2))
    
    loc = mdates.MonthLocator(interval=12)
    ax.xaxis.set_major_locator(loc)
    fmt = mdates.DateFormatter('%b\n%Y')
    ax.xaxis.set_major_formatter(fmt)
    #ax.ticklabel_format(style='sci',scilimits=(1,2),axis='y')
    
    ax.hlines(y=41565, xmin=pd.to_datetime(df_traj_fire_count.index[0]), xmax=pd.to_datetime(df_traj_fire_count.index[-1]),
    ls=':', color='r')
    ax.hlines(y=29342, xmin=pd.to_datetime(df_traj_fire_count.index[0]), xmax=pd.to_datetime(df_traj_fire_count.index[-1]),
    ls=':', color='r')
    ax.hlines(y=4626, xmin=pd.to_datetime(df_traj_fire_count.index[0]), xmax=pd.to_datetime(df_traj_fire_count.index[-1]),
    ls=':', color='r')
    
    #ax.legend(frameon=False)
    ax.set_ylim(-1, ymax)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    
    if log == True:
        ax.set_yscale('log')
        ax.set_ylim(10**-1, ymax)
    
    for label in ax.get_xticklabels():
        label.set_ha("right")
        label.set_rotation(90)    
    return fig
    
####################################CIRCULAR########################################################################################################################

def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def add_map_features(ax):
    ax.coastlines(resolution='50m')
    #ax.add_feature(cy.feature.BORDERS);
    gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
              linewidth=.5, color='gray', alpha=0.2, linestyle='--')
    #gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=True,
    #              linewidth=1, color='gray', alpha=0.5, linestyle='--')
             
    
def polarCentral_set_latlim(lat_lims, ax):
    # Compute a circle in axes coordinates, which we can use as a boundary
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
       
def circular_plots_array(array, df_traj, df_fire, arrival_time, fire_count_for_traj='', vmax=10, cmap=cmocean.cm.turbid, orientation='vertical', 
                         colourbar_label=r"Number of active fires in grid box Fredquency [-]", colourbar_labelsize=20, 
                         colourbar_tick_fontsize=12, scientific_notation=True, decimal_places_colourbar=0, 
                         axes_projection='North_Stereo', array_projection='PlateCarree', traj_projection='PlateCarree', 
                         title='', ZEP_lat=78.906,ZEP_lon=11.888, plot_fire=True, extend=False, count_fires=True): 
    
    dict_projections = create_projections_dict(ZEP_lat=78.906,ZEP_lon=11.888)                   
    projection = dict_projections[axes_projection] 
    array_projection = dict_projections[array_projection] 
    traj_projection = dict_projections[traj_projection] 
    geo = dict_projections['geo']    
    
    fig, ax = sp_map(1, projection=projection, figsize=(8,8))

    lat_lims = [40,90]
    polarCentral_set_latlim(lat_lims, ax)
    #add_map_features(ax)           
    #lons = np.arange(-178.5, 181.5, 1) #lons = np.arange(-180.5, 179.5, 1)    
    #lats = np.arange(1.5, 91.5, 1) #lats = np.arange(-0.5, 89.5, 1)
    
    lons = np.arange(-180.5, 179.5, 1)  
    lats = np.arange(-0.5, 89.5, 1) 
    
    if count_fires == True:
        df_traj_fire_count = create_fire_dataframe(df_traj, array, arrival_time)
        print("count: "+str(df_traj_fire_count))
        fire_count_for_traj = df_traj_fire_count.loc[df_traj_fire_count['arrival_time'] == arrival_time, 'fire_count'].values[0]
        print("count: "+str(fire_count_for_traj))
        
    array[array == 0] = np.nan #don't want grid filled with colour 
    cs = ax.pcolormesh(lons, lats, array, shading='auto', 
                       transform=array_projection, vmin=0, vmax=vmax, cmap=cmap) 
    #cs = ax.pcolormesh(lons, lats, array, transform=array_projection, vmin=0, vmax=vmax, cmap='cmo.thermal')  
        
    ###CHECK
    if plot_fire == True:
        print("Length of df_fire: "+str(len(df_fire)))
        fire_lats = df_fire['latitude'].values
        fire_lons = df_fire['longitude'].values
        ax.scatter(fire_lons, fire_lats, 
                   marker='^', s = 2, alpha=0.2, color='darkorange',
                   transform=array_projection, label='active fires')

    if extend:
        ax.set_extent([-10, 20, 35, 70], ccrs.PlateCarree())
    
    number_of_grids_with_active_fires = np.count_nonzero(~np.isnan(array)) 
    sfs = sf(3)    
    percentage_of_grids = sfs.format(number_of_grids_with_active_fires/(360*90))
    number_of_active = np.nansum(array)  
        
    #ax.set_title(str(title), loc='left', size=15)
    # ax.set_title("Number of grids with active fires: "+str(number_of_grids_with_active_fires)+' ('+str(percentage_of_grids)+'%)'+
                 # '\nNumber of active fires: '+str(number_of_active)+
                 # '\nTime interval for traj: '+str(df_traj.index[0])+' - '+str(df_traj.index[-1])+
                 # '\nTime interval for fire: '+str(df_fire.index[0])[:10]+' - '+str(df_fire.index[-1])[:10]+
                 # '\nFire count: '+str(fire_count_for_traj), loc='left', fontsize=12)
                 
    ax.set_title('\nTime interval for back traj. (within ML): '+str(df_traj.index[0])+' - '+str(df_traj.index[-1])+
             '\nTime interval for MODIS fire data: '+str(df_fire.index[0])[:10]+' - '+str(df_fire.index[-1])[:10]+
             '\nEnsemble of Trajs pass through: '+str(fire_count_for_traj)+' active fires in 24 hrs.', loc='left', fontsize=12)

    ax.plot([ZEP_lon], [ZEP_lat], '*', ms=10, alpha=1, color='g', transform=geo, label='ZEP') 
    
    lon_traj = df_traj['longitude'].values
    lat_traj = df_traj['latitude'].values
    
    ax.scatter(lon_traj, lat_traj, s=3, marker="o", transform=traj_projection, label='Traj', alpha=0.1,
               facecolors='none', edgecolors='b')
    
    if orientation =='horizontal':
        cax = fig.add_axes([0.15, .08, 0.72, 0.03]) # position of colorbar [left, bottom, width, height]
    if orientation =='vertical':
        cax = fig.add_axes([0.95, .2, 0.04, 0.6]) # position of colorbar [left, bottom, width, height]
    
    if scientific_notation == True:
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))
            
        cbar = fig.colorbar(cs, extend='max', orientation=orientation, cax=cax, format=fmt)
        cbar.set_label(colourbar_label, size=12)
        cbar.ax.tick_params(labelsize=12)

    else:
        kwargs = {'format': '%.'+str(decimal_places_colourbar)+'f'}
        cbar = fig.colorbar(cs, extend='max', orientation=orientation, cax=cax, **kwargs)        
        cbar.set_label(colourbar_label, size=colourbar_labelsize)
        cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
      
    add_map_features(ax) 
    
    black_star = mlines.Line2D([], [], color='g', marker='*', linestyle='None',
                              markersize=10, label='ZEP')
    blue_circle = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                              markersize=10, label='BT')                              
    orange_triangle = mlines.Line2D([], [], color='darkorange', marker='^', linestyle='None',
                              markersize=10, label='Fires')  
                              
    ax.legend(handles=[black_star, blue_circle,], fontsize=15, frameon=False,
               bbox_to_anchor=(-0.01, 0.99), loc="upper left")
    #ax.legend(markerscale=2, frameon=True, fontsize=15, loc=2)
    plt.show()
    return fig
    
def create_legend():
    black_star = mlines.Line2D([], [], color='k', marker='*', linestyle='None',
                              markersize=10, label='ZEP')
    blue_circle = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                              markersize=10, label='BT')
    orange_triangle = mlines.Line2D([], [], color='darkorange', marker='^', linestyle='None',
                              markersize=10, label='Fires')
    handles=[black_star, blue_circle, orange_triangle]
    return handles
    
    
def example_circular_plot(df_fire, trajs_dictionary, arrival_time):
    fire_freq_array = create_fire_freq_array(df_fire, trajs_dictionary, arrival_time)
        
    df_fire_sliced = slice_df_fire(df_fire, trajs_dictionary, arrival_time)
    fig = circular_plots_array(fire_freq_array, trajs_dictionary[arrival_time], df_fire_sliced,
                                       arrival_time)
    return fig
    
def circular_plots_array_ax(array, df_traj, df_fire, arrival_time, fire_count_for_traj='', vmax=10, cmap=cmocean.cm.turbid, orientation='vertical', 
                         colourbar_label=r"Number of active fires in grid box Fredquency [-]", colourbar_labelsize=20, 
                         colourbar_tick_fontsize=12, scientific_notation=True, decimal_places_colourbar=0, 
                         axes_projection='North_Stereo', array_projection='PlateCarree', traj_projection='PlateCarree', 
                         title='', ZEP_lat=78.906,ZEP_lon=11.888, plot_fire=True, extend=False, count_fires=True, 
                         plot_fire_array=True, plot_taj=True, show_legend=True, ax=None): 
    
    dict_projections = create_projections_dict(ZEP_lat=78.906,ZEP_lon=11.888)                   
    projection = dict_projections[axes_projection] 
    array_projection = dict_projections[array_projection] 
    traj_projection = dict_projections[traj_projection] 
    geo = dict_projections['geo']    
    
    solo_plot = False
    if ax == None:
        fig, ax = sp_map(1, projection=projection, figsize=(8,8))
        solo_plot == True

    lat_lims = [40,90]
    polarCentral_set_latlim(lat_lims, ax)
    
    lons = np.arange(-180.5, 179.5, 1)  
    lats = np.arange(-0.5, 89.5, 1) 
    
    if count_fires == True:
        df_traj_fire_count = create_fire_dataframe(df_traj, array, arrival_time)
        print("count: "+str(df_traj_fire_count))
        fire_count_for_traj = df_traj_fire_count.loc[df_traj_fire_count['arrival_time'] == arrival_time, 'fire_count'].values[0]
        print("count: "+str(fire_count_for_traj))
       
    if plot_fire_array == True:
        array[array == 0] = np.nan #don't want grid filled with colour 
        cs = ax.pcolormesh(lons, lats, array, shading='auto', 
                           transform=array_projection, vmin=0, vmax=vmax, cmap=cmap) 

    ###CHECK
    if plot_fire == True:
        print("Length of df_fire: "+str(len(df_fire)))
        fire_lats = df_fire['latitude'].values
        fire_lons = df_fire['longitude'].values
        ax.scatter(fire_lons, fire_lats, 
                   marker='^', s = 2, alpha=0.2, color='darkorange',
                   transform=array_projection, label='active fires')

    if extend:
        ax.set_extent([-10, 20, 35, 70], ccrs.PlateCarree())
    
    number_of_grids_with_active_fires = np.count_nonzero(~np.isnan(array)) 
    sfs = sf(3)    
    percentage_of_grids = sfs.format(number_of_grids_with_active_fires/(360*90))
    number_of_active = np.nansum(array)  

    # ax.set_title('\nTime interval for back traj. (within ML): '+str(df_traj.index[0])+' - '+str(df_traj.index[-1])+
             # '\nTime interval for MODIS fire data: '+str(df_fire.index[0])[:10]+' - '+str(df_fire.index[-1])[:10]+
             # '\nEnsemble of Trajs pass through: '+str(fire_count_for_traj)+' active fires in 24 hrs.', loc='left', fontsize=12)

    ax.plot([ZEP_lon], [ZEP_lat], '*', ms=10, alpha=1, color='g', transform=geo, label='ZEP') 
    
    if plot_taj == True:
        lon_traj = df_traj['longitude'].values
        lat_traj = df_traj['latitude'].values
        
        ax.scatter(lon_traj, lat_traj, s=3, marker="o", transform=traj_projection, label='Traj', alpha=0.1,
                   facecolors='none', edgecolors='b')
    
    # if orientation =='horizontal':
        # cax = fig.add_axes([0.15, .08, 0.72, 0.03]) # position of colorbar [left, bottom, width, height]
    # if orientation =='vertical':
        # cax = fig.add_axes([0.95, .2, 0.04, 0.6]) # position of colorbar [left, bottom, width, height]
    
    #if plot_fire_array == True:
    # if scientific_notation == True:
        # fmt = ticker.ScalarFormatter(useMathText=True)
        # fmt.set_powerlimits((0, 0))
            
        # cbar = fig.colorbar(cs, extend='max', orientation=orientation, cax=cax, format=fmt)
        # cbar.set_label(colourbar_label, size=12)
        # cbar.ax.tick_params(labelsize=12)

    # else:
        # kwargs = {'format': '%.'+str(decimal_places_colourbar)+'f'}
        # cbar = fig.colorbar(cs, extend='max', orientation=orientation, cax=cax, **kwargs)        
        # cbar.set_label(colourbar_label, size=colourbar_labelsize)
        # cbar.ax.tick_params(labelsize=colourbar_tick_fontsize)
      
    add_map_features(ax) 
    
    black_star = mlines.Line2D([], [], color='g', marker='*', linestyle='None',
                              markersize=10, label='ZEP')
    blue_circle = mlines.Line2D([], [], color='b', marker='o', linestyle='None',
                              markersize=10, label='BT')                              
    orange_triangle = mlines.Line2D([], [], color='darkorange', marker='^', linestyle='None',
                              markersize=10, label='Fires')  
    if show_legend == True:                        
        ax.legend(handles=[black_star, blue_circle,], fontsize=15, frameon=False,
                   bbox_to_anchor=(-0.01, 0.99), loc="upper left")
               
    if solo_plot == True:   
        print("solo plot")
        plt.show()
        return fig
    if solo_plot == False:
        print("no solo plot")
        return ax
    
####################################REMAINING############################################################################################################################################################
    
def produce_hourly_time_ranges(df_OCEC):
    """using the datafram the start and stop column, create an hourly index"""
    time_ranges = []
    for i in range(len(df_OCEC)):
        time_range = list(pd.date_range(df_OCEC['start'].iloc[i], df_OCEC['stop'].iloc[i], freq='H'))
        time_ranges.append(time_range)
    time_ranges = list(itertools.chain(*time_ranges))
    time_ranges = [pd.to_datetime(x) for x in time_ranges]
    print("size "+str(len(time_ranges)))
    return time_ranges

def dates_left_to_analyse(OCEC_time_ranges, df_fire_count):
    dates_to_analyse = []
    for date in OCEC_time_ranges:
        if date not in list(df_fire_count.index):
            print("date: "+str(date))
            dates_to_analyse.append(date)
    return dates_to_analyse
    
def find_difference(list1, list2, difference=True, same=False):
    print("list 1: "+str(len(list1)))
    print("list 2: "+str(len(list2)))
    if difference == True:
        set_difference = set(list1) - set(list2)
        list4 = list(set_difference)     
    if same == True:
        list3 = set(list1)&set(list2)
        list4 = sorted(list3, key = lambda k : list1.index(k)) #
    list4 = sorted(list4)
    return list4