import numpy as np
import glob
import pandas as pd
import xarray as xr
import datetime
import os
from math import cos, pi, sin, atan2, asin

#trajectory number
#meteorological grid number
#time of point:  year month day hour minute
#forecast hour at point
#age of the trajectory in hours 
#position latitude and longitude 
#position height in meters above ground 
    
Trajectory_Endpoints = ['Traj_num', 'meteorological_grid_num', 'Year', 'Month', 'Day', 'Hour', 'Minute',
                        'FORECAST_hour', 'time_step', 'latitude', 'longitude', 'altitude']

#n diagnostic output variables (1st  output always pressure) "surface_level_parameter_descrition and others"

# Potential Temperature in degrees Kelvin TM_TPOT (0|1)
# Ambient Temperature in degrees Kelvin TM_TAMB (0|1)
# Precipitation rainfall in mm per hour TM_RAIN (0|1)
# Mixing Depth in meters TM_MIXD (0|1)
# Relative Humidity in percent TM_RELH (0|1)
# Note: If relative humidity needs to be calculated by HYSPLIT from specific humidity, it may differ from relative humidity shown for other data sets as it may be calculated with respect to water, not ice at low temperatures, such is the case for the 0.5 degree GDAS data set.
# Specific Humidity in g/(kg air) TM_SPHU (0|1)
# Water Vapor Mixing Ratio in g/(kg dry-air) TM_MIXR (0|1)
# Solar Radiation downward solar radiation flux in watts per square meter TM_DSWF (0|1)
# Terrain Height in meters required for the trajectory plot to show underlying terrain TM_TERR (0|1)

diagnostic_output_variables = ['PRESSURE', 'THETA', 'AIR_TEMP', 'RAINFALL', 'MIXDEPTH',
       'RELHUMID', 'SPCHUMID', 'H2OMIXRA', 'TERR_MSL', 'SUN_FLUX']

back_traj_col_names = Trajectory_Endpoints + diagnostic_output_variables

def get_South_grid(ZEP_lat, ZEP_lon):
    South_Pole_lat = -ZEP_lat
    South_Pole_lon = -(180-ZEP_lon)
    South_grid = (South_Pole_lon,South_Pole_lat)
    return South_grid

def lla_to_xyz(lat, lon, alt):    
    a = 6378137 
    f = 1/298.257224
    e = np.sqrt(2*f-f**2)
         
    lat = np.radians(lat)
    lon = np.radians(lon)       
   
    C = 1/(np.sqrt( (np.cos(lat)**2 + (1-f)**2 * (np.sin(lat))**2) ))        
    S = (1-f)**2 * C 
    x = (a*C+alt)*np.cos(lat)*np.cos(lon)
    y = (a*C+alt)*np.cos(lat)*np.sin(lon)
    z = (a*S+alt)*np.sin(lat) 
    return x, y, z
    
def xyz_to_lla(x,y,z):
    #WGS84 ellipsoid constants:
    a = 6378137 
    f = 1/298.257224
    e = np.sqrt(2*f-f**2)    
    b   = np.sqrt(a**2*(1-e**2));
    ep  = np.sqrt((a**2-b**2)/b**2);
    p   = np.sqrt(x**2+y**2);
    th  = math.atan2(a*z,b*p);
    lon = math.atan2(y,x);
    lat = math.atan2((z+ep**2*b*math.sin(th)**3),(p-e**2*a*math.cos(th)**3));
    N   = a/np.sqrt(1-e**2*math.sin(lat)**2);
    alt = p/math.cos(lat)-N;    
    lon = math.degrees(lon)
    lat = math.degrees(lat)        
    return lat, lon, alt
    
def cal_mean_coords(df):            
    x = 0.0
    y = 0.0
    z = 0.0    
    df = df[['latitude','longitude','altitude']]
    df = df.dropna(how='all')
    for i, coord in df.iterrows():    
        lat = coord.latitude
        lon = coord.longitude
        alt = coord.altitude        
        x += lla_to_xyz(lat, lon, alt)[0]
        y += lla_to_xyz(lat, lon, alt)[1]
        z += lla_to_xyz(lat, lon, alt)[2]             
    total = len(df) 
    x_ = x / total
    y_ = y / total
    z_ = z / total       
    lat, lon, alt = xyz_to_lla(x_,y_,z_)   
    return lat, lon, alt
    
def calculate_distance(lat, lon, alt, lat_pre, lon_pre, alt_pre):        
    coords_1 = (lat, lon, alt)
    coords_2 = (lat_pre, lon_pre, alt_pre)
    x_1,y_1,z_1 = lla_to_xyz(coords_1[0],coords_1[1],coords_1[2])
    x_2,y_2,z_2 = lla_to_xyz(coords_2[0],coords_2[1],coords_2[2])
    distance = np.sqrt((x_2-x_1)**2 + (y_2-y_1)**2 + (z_2-z_1)**2)       
    return distance
    
def make_empty_array():
    longitude_array = np.arange(-180, 180, 1)
    latitude_array = np.arange(0, 90, 1)

    array = np.indices((latitude_array.shape[0], longitude_array.shape[0]))
    array = array[0] + array[1]
    array = np.empty((90,360))
    array.fill(0) 
    return array
    
def rotated_grid_transform(grid_in, option, SP_coor):
    lon = grid_in[0]
    lat = grid_in[1]

    lon = (lon*pi)/180 # Convert degrees to radians
    lat = (lat*pi)/180
    
    SP_lon = SP_coor[0]
    SP_lat = SP_coor[1]

    theta = 90+SP_lat # Rotation around y-axis
    phi = SP_lon # Rotation around z-axis

    theta = (theta*pi)/180
    phi = (phi*pi)/180 # Convert degrees to radians

    x = cos(lon)*cos(lat) # Convert from spherical to cartesian coordinates
    y = sin(lon)*cos(lat)
    z = sin(lat)

    if option == 1: # Regular -> Rotated
        x_new = cos(theta)*cos(phi)*x + cos(theta)*sin(phi)*y + sin(theta)*z
        y_new = -sin(phi)*x + cos(phi)*y
        z_new = -sin(theta)*cos(phi)*x - sin(theta)*sin(phi)*y + cos(theta)*z
    else:  # Rotated -> Regular
        phi = -phi;
        theta = -theta;

        x_new = cos(theta)*cos(phi)*x + sin(phi)*y + sin(theta)*cos(phi)*z
        y_new = -cos(theta)*sin(phi)*x + cos(phi)*y - sin(theta)*sin(phi)*z
        z_new = -sin(theta)*x + cos(theta)*z
        
    lon_new = atan2(y_new,x_new) # Convert cartesian back to spherical coordinates
    lat_new = asin(z_new)
    lon_new = (lon_new*180)/pi # Convert radians back to degrees
    lat_new = (lat_new*180)/pi
    return lon_new,lat_new
    
def transform_to_latitude_longitude_to_grid_lon(lat, lon, South_grid):
    """for the df_traj of the trajectory the latitude and longtitude of all the points 
    along the trajectory are used to fill in the grid"""   
    array = make_empty_array()        
    shifted_coord = rotated_grid_transform((lon,lat),1,South_grid)
    lon = shifted_coord[0]
    lat = shifted_coord[1]
    if lon >= 0:
        grid_lon = int(abs(lon+180))
    if lon < 0:
        grid_lon = int(180-abs(lon))
    grid_lon = int(grid_lon)     
    return grid_lon
    
def transform_to_latitude_longitude_to_grid_lat(lat, lon, South_grid): 
    """for the df_traj of the trajectory the latitude and longtitude of all the points 
    along the trajectory are used to fill in the grid"""  
    array = make_empty_array()        
    shifted_coord = rotated_grid_transform((lon,lat),1,South_grid)
    lat = shifted_coord[1]    
    grid_lat = int(lat)    
    return grid_lat
    
def find_indexs(file, start_string):
    """when to begin read in - function reads in each line and looks for when start string is in line"""
    file_info = open(file, 'r')
    lines = file_info.readlines()
    for i, line in enumerate(lines):         
        if start_string in line:
            start = i + 1   
    return start
    
def find_HYSPLIT_files_per_year(year, inpath, dict_year_to_HYSPLIT_folder, prefix, 
                                inputyr_2digits=False, month='', day=''):
    print("Year: "+str(year))
    folder = str(dict_year_to_HYSPLIT_folder[year])
    inpath = inpath+folder+"\\" 
    print("Path: "+str(inpath))    
    
    #list_of_files = glob.glob(inpath+str(prefix)+str(year)+'*')
    list_of_files = glob.glob(inpath+str(prefix)+'*'+str(year)+'*')    

    if len(list_of_files) == 0:
        print(year)
        print("year has a length of zero: ")  
    if len(list_of_files) != 0:  
        print(year)
        print("number of files: "+str(len(list_of_files)))
        
        if inputyr_2digits==True:
            print("inputyr_2digits=True i.e. str(year)[2:]")
            list_of_files = glob.glob(inpath+str(prefix)+'*'+str(year)[2:]+str(month)+str(day)+'*') #last 2 digits 
        if inputyr_2digits==False:
            print("inputyr_2digits=False i.e. str(year)")
            print("added day and month now*")
            print(inpath+str(prefix)+'*'+str(year)+str(month)+str(day)+'*')
            list_of_files = glob.glob(inpath+str(prefix)+'*'+str(year)+str(month)+str(day)+'*') #last 2 digits #0324 should be here - just testing
            list_of_files = [x for x in list_of_files if len(x.split(str(year))[-1]) == 6]
            
    print("Number of HYSPLIT files for "+str(year)+": "+str(len(list_of_files)))
    return list_of_files
    
def process_HYSPLIT_and_save(year, list_files, South_grid, outpath_processed_trajectories, cut_traj=False, save=True,
                             csv=False, pickle=True, parquet=False):
    for idx in range(len(list_files)):
        try:
            file = list_files[idx]
            print("read file: "+str(file))
            try:
                start = find_indexs(file=file, start_string='PRESSURE')
                print("starting index: "+str(start))
            except:
                start = 'None' 
            if start == 'None':
                ds = pd.DataFrame()
            if start != 'None':
                ds = pd.read_csv(file, sep='\s+', skiprows=start, names=back_traj_col_names, index_col=False) #, nrows=(traj_length)*27) #, dtype = dtype_dict)
            if len(ds) > 1: #only analysed trajectories which are larger than 1                
                try:
                    ds['Year'] = ds['Year'].apply(lambda x: x + 2000) # if len(str(x)) <= 1 else np.nan) #only a problem if in 90s
                    ds['DateTime'] = ds[['Year', 'Month', 'Day', 'Hour', 'Minute']].apply(lambda s : datetime.datetime(*s),axis = 1)
                    ds = ds.drop(['Year', 'Month', 'Day', 'Hour', 'Minute'], axis=1)
                    ds['DateTime'] = ds['DateTime'].apply(lambda x: x.replace(minute=0, second=0))

                    arrival_time = ds['DateTime'].iloc[0] #first row is the arrival time for the traj
                    ds['arrival_time'] = arrival_time #new
                    
                    arrival_time = str(arrival_time)[:-6]        
                    arrival_time = arrival_time.replace(' ','_')
                    arrival_time = arrival_time.replace('-','')

                    number_of_trajs = len(ds['Traj_num'].unique()) #should be 27 trajectories per 1 
                    length_num_trajs = len(ds)/number_of_trajs #print("Length divided by number of trajectories (should be 241 hr):" + str(length_num_trajs))

                    print("number of trajs per unit time: "+str(number_of_trajs))
                    print("length_num_trajs: "+str(length_num_trajs))
                    
                    if cut_traj == True:
                        ds = ds.iloc[:int(length_num_trajs)*int(number_of_trajs),:]  
                    
                    #no need to deselect columns
                    #ds = ds[['Traj_num', 'latitude', 'longitude', 'altitude', 'RAINFALL', 'MIXDEPTH', 'RELHUMID', 
                    #         'SUN_FLUX', 'DateTime']]
                    ds = ds.set_index('DateTime')

                    ds['latitude_previous'] = ds['latitude'].shift(-int(number_of_trajs)) #27 trajectories
                    ds['longitude_previous'] = ds['longitude'].shift(-int(number_of_trajs)) 
                    ds['altitude_previous'] = ds['altitude'].shift(-int(number_of_trajs))

                    ds['distance'] = ds.apply(lambda x: calculate_distance(x['latitude'], x['longitude'], x['altitude'], 
                                      x['latitude_previous'], x['longitude_previous'], x['altitude_previous']), axis=1)
                    ds['grid_lon'] = ds.apply(lambda x: transform_to_latitude_longitude_to_grid_lon(x['latitude'], x['longitude'], South_grid), axis=1)
                    ds['grid_lat'] = ds.apply(lambda x: transform_to_latitude_longitude_to_grid_lat(x['latitude'], x['longitude'], South_grid), axis=1)
                    if save == True:                        
                        folders = glob.glob(outpath_processed_trajectories)
                        if year not in folders:
                            print("make folder: "+str(outpath_processed_trajectories+"\\"+str(year)))
                            os.makedirs(outpath_processed_trajectories+"\\"+str(year), exist_ok=True)
                        #ds.to_csv(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.dat', index=True) 
                        #ds.to_feather(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.feather')
                        
                        if csv == True:                    
                            ds.to_csv(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.dat', index=True)
                            print(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.dat')
                        if parquet == True: #small memory, save quick, open normal
                            ds.to_parquet(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.parquet')
                            print(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.parquet') 
                        if pickle == True: #normal memory, save very quick, open very quick 
                            ds.to_pickle(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.pickle')
                            print(outpath_processed_trajectories+str(year)+'\\'+str(arrival_time)+'.pickle') 
                        
                    if save == False:
                        print("do not save")
                        print(ds.head(2))
                except ValueError:
                    print("datetime wrong - ")
                    print(file)
            else:
                ("ds is empty")
                pass
        except FileNotFoundError:
            print("File not found")
            print(file)
            pass              

def find_traj_file(last_processed_file, list_files):
    """find the index of the files"""
    file = [s for s in list_files if last_processed_file in s]
    idx_in_list = list_files.index(file[0])
    return file, idx_in_list
    
def find_index_of_last_processed_file(year, outpath_processed_trajectories):
    """find the index of the last processed file to help speed up the process and start where we left off"""
    print(outpath_processed_trajectories+str(year)+'\\')
    list_of_processed_traj_files = glob.glob(outpath_processed_trajectories+str(year)+'\\*')
    if len(list_of_processed_traj_files) > 0:
        last_file = list_of_processed_traj_files[-1]
        last_processed_file = last_file.replace(outpath_processed_trajectories+str(year)+'\\', '')
        last_processed_file = last_processed_file[:-4]
        print("last file processed: "+str(last_processed_file))        
        file, idx_in_list = find_traj_file(last_processed_file, list_of_processed_traj_files)
    else:
        last_processed_file = 0
        idx_in_list = 0
    return idx_in_list
    
def create_processed_data(outpath_processed_trajectories, inpath, dict_year_to_HYSPLIT_folder, years, process_data=False, 
                          ZEP_lat=78.906,ZEP_lon=11.888, prefix='t',  use_last_file_processed=True, cut_traj=False, save=False,
                          inputyr_2digits=False, month='', day=''):    
    South_grid = get_South_grid(ZEP_lat, ZEP_lon)    
    print("save processed file to: "+str(outpath_processed_trajectories))
    if process_data == True:
        print("Years to process: "+str(years))
        for year in years: 
            print("year: "+str(year))   
            list_of_HYSPLIT_files_per_year = find_HYSPLIT_files_per_year(year, inpath, dict_year_to_HYSPLIT_folder, prefix, inputyr_2digits,
            month, day) #list of HYSPLIT files
            list_files = list_of_HYSPLIT_files_per_year.copy() #copy for later
            if use_last_file_processed == True:
                idx_in_list = find_index_of_last_processed_file(year, outpath_processed_trajectories) #find the index 
                print("index of last processed file: "+str(idx_in_list))
            if use_last_file_processed == False:
                print("don't use info about what has previously been processed:")
                idx_in_list = 0
            process_HYSPLIT_and_save(year, list_files[idx_in_list:], South_grid, outpath_processed_trajectories, cut_traj, save)
    else:
        pass
        
####################ADD OBSERVATIONS##########################################################################################################################################################################

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

def create_HYSPLIT_name_from_list(list_indexes):
    """use the list of indexes and alter it for the format of HYSPLIT"""
    list_strings_indexes = [str(x) for x in list_indexes]
    list_strings_indexes = [x.replace('-','') for x in list_strings_indexes]
    list_strings_indexes = [x.replace(':','') for x in list_strings_indexes]
    list_strings_indexes = [x[:11] for x in list_strings_indexes]
    list_strings_indexes = [x.replace(' ','_') for x in list_strings_indexes]
    return list_strings_indexes
    
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
    
def find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names):
    """observation indexes we want to match with the list of HYSPLIT files"""
    matching_list_of_back_traj_files = [s for s in list_of_HYSPLIT_files if any(xs in s for xs in HYSPLIT_names)]
    print("Matching files from observational data & HYSPLIT: "+str(len(matching_list_of_back_traj_files)))
    return matching_list_of_back_traj_files
    
def list_files(year, inpath_processed_hysplit_dfs):
    """Find the a list of all the HYSPLIT files for a particular year"""
    print("Year: "+str(year))    
    print("Path: "+str(inpath_processed_hysplit_dfs+str(year)+"\\"))
    list_of_files = glob.glob(inpath_processed_hysplit_dfs+str(year)+"\\"+'*')    
    print("Number of HYSPLIT files for "+str(year)+": "+str(len(list_of_files)))
    return list_of_files

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
        #ds = ds.iloc[:int(traj_length)*int(number_of_trajs),:]
        #merge with BC or OCEC
        try:     
            if weekly == False:
                obs_value = df_obs.loc[df_obs.index == arrival_time, var].values[0]
            if weekly == True:
                obs_value = df_obs['OC_mean_mug_m3'].iloc[df_obs.index.get_loc(arrival_time, method='nearest')]            
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
    
def save_processed_trajs_with_obs(df_obs, outpath_summed_dfs, 
                                  inpath_processed_hysplit_dfs,
                                  var, years=[], weekly=False, process=False,
                                  save=False):   
    if process == True:
        if len(years) == 0:
            years = df_obs.index.year.unique()
            print(years)
        for year in years:
            print("Year: "+str(year))        
            list_of_HYSPLIT_files = list_files(year, inpath_processed_hysplit_dfs)            
            if weekly == True:
                OCEC_time_ranges = produce_hourly_time_ranges(df_OCEC)       
                HYSPLIT_names = create_HYSPLIT_name_from_list(OCEC_time_ranges)
            if weekly == False:
                HYSPLIT_names = create_HYSPLIT_name_from_list(df_obs.index)
            matching_list_of_back_traj_files = find_matching_files(list_of_HYSPLIT_files, HYSPLIT_names)        
            ds_year = create_df_of_processed_trajs_for_obs(matching_list_of_back_traj_files, df_obs, var, weekly)            
            if save == True:
                if len(ds_year) > 0:        
                    try:
                        ds_year.to_csv(outpath_summed_dfs+str(year)+'.dat')
                    except FileNotFoundError:
                        os.makedirs(outpath_summed_dfs, exist_ok=True)
                        ds_year.to_csv(outpath_summed_dfs+str(year)+'.dat')            
                    print("Saved: "+str(outpath_summed_dfs+str(year)+'.dat'))
            if save == False:
                pass                
    if process == False:
        print("if you need to process, change process to True")
        pass