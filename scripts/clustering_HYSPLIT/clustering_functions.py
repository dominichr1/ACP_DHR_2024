import sys
import pandas as pd
import glob
import numpy as np
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics.pairwise import pairwise_distances
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
from matplotlib import cm
from matplotlib.lines import Line2D
from math import cos, pi, sin, atan2, asin
import math
import matplotlib.path as mpath
import cartopy as cy
import matplotlib.dates as mdates
import statsmodels.api as sm
import os
from pandas.tseries.offsets import MonthEnd
from scipy import stats
from sklearn.cluster import MiniBatchKMeans
import re
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
from datetime import timedelta
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

def fancy(ax, fontsize=20, spines=['top','bottom','left','right'], alpha=0.5):    
    # thickning the axes spines
    for axis in spines:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('k')        
    # set the fontsize for all your ticks    
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)        
    # properties of the ticks
    ax.tick_params(direction='out', length=8, width=2, pad=10, bottom=True, top=False, left=True, right=False, color='k')    
    # add a grid to the plot
    ax.grid(True, alpha=alpha)  
    

def generate_xyz_col_names(elements_in_col_name=['x_', 'y_', 'z_'], length=241):
    list_of_list_elements = []
    for element in elements_in_col_name:
        print(element)
        list_elements = [str(element)+str(x) for x in np.arange(0, length, 1)]
        list_of_list_elements = list_of_list_elements + list_elements
    return list_of_list_elements
    
def generate_dict_dtype(list_of_list_elements, float_type=np.float32):
    max_value = np.finfo(float_type).max
    print("max for dtype: "+str(max_value))
    dict_dtype = {}
    for element in list_of_list_elements:
        dict_dtype[element] = float_type
    return dict_dtype

#load df
def load_df(loadpath, extrapath=None, filename=None, formatdata=".dat", dict_dtype=None):
    if extrapath is not None:
        print("loading: "+str(loadpath+'\\'+extrapath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+extrapath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False, dtype=dict_dtype)
    if extrapath is None:
        print("loading: "+str(loadpath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False, dtype=dict_dtype)        
    return df

def save_df(df, path, name='', index=True, float_format=None):
    print("Save as: "+str(path+'\\'+name+'.dat'))
    df.to_csv(path+'\\'+name+'.dat', index=index, float_format=float_format)
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formated=".jpeg", dpi=300):
    folders = glob.glob(path)
    print(folders)
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formated), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formated)))

#find the size of loaded variables
def sizeof_fmt(num, suffix='B'):
    ''' by Fred Cirera,  https://stackoverflow.com/a/1094933/1870254, modified'''
    for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
        if abs(num) < 1024.0:
            return "%3.1f %s%s" % (num, unit, suffix)
        num /= 1024.0
    return "%.1f %s%s" % (num, 'Yi', suffix)
    
def list_sizes_of_loaded_variables():
    print("find sizes:")
    for name, size in sorted(((name, sys.getsizeof(value)) for name, value in locals().items()), key= lambda x: -x[1])[:10]:
        print(name)
        print(size)
        print("{:>30}: {:>8}".format(name, sizeof_fmt(size)))
    
def find_matching_files(year, df_obs, inpath_processed_hysplit_dfs):
    """this function takes the df of observations (e.g. absorption) the path where the processed hysplit files are and compares them. It finds the matching files,
    this does it for a year at a time"""
    print("Year: "+str(year))
    inpath = inpath_processed_hysplit_dfs+str(year)+"\\" 
    print("Path: "+str(inpath))
    list_of_files = glob.glob(inpath+'*')    
    print("Number of HYSPLIT files for "+str(year)+": "+str(len(list_of_files)))
    df_obs_year = df_obs[df_obs.index.year == int(year)]
    number_of_obs = len(df_obs_year)
    print("Number of observations for the year "+str(year)+': '+str(number_of_obs))
    list_indexes = df_obs_year.index.to_list()    
    #make df obs index like the files names for HYSPLIT    
    list_strings_indexes = [str(x) for x in list_indexes]
    list_strings_indexes = [x.replace('-','') for x in list_strings_indexes]
    list_strings_indexes = [x.replace(':','') for x in list_strings_indexes]
    list_strings_indexes = [x[:11] for x in list_strings_indexes]
    list_strings_indexes = [x.replace(' ','_') for x in list_strings_indexes]
    matchers = list_strings_indexes #observation indexes we want to match with the list of HYSPLIT files
    matching_list_of_back_traj_files = [s for s in list_of_files if any(xs in s for xs in matchers)]
    print("Matching files from observational data & HYSPLIT: "+str(len(matching_list_of_back_traj_files)))
    return matching_list_of_back_traj_files
    
def all_matching_HYSPLIT_obs_files(df_obs, years, inpath_processed_hysplit_dfs):
    """this concats the lists of matching files for each year to find the total list for all years"""
    list_of_files_all_years = []
    for year in years:
        print("Year: "+str(year))
        list_of_files = find_matching_files(year, df_obs, inpath_processed_hysplit_dfs)
        print("Number of files: "+str(len(list_of_files)))
        list_of_files_all_years.append(list_of_files)
    list_of_files_all_years_flat = [item for sublist in list_of_files_all_years for item in sublist]
    print(len(list_of_files_all_years_flat))
    return list_of_files_all_years_flat
    
def find_index_of_last_row(df_clustering, list_of_files_all_years_flat, inpath_processed_hysplit_dfs):
    """this fucntion finds the index in the list of files of the last datetime entry of the most recent df for clustering"""
    last_index = df_clustering.index.values[-1]
    print("last datetime in df: "+str(last_index))
    last_index = pd.to_datetime(last_index)
    last_index = str(last_index)[:-6]
    last_index = last_index.replace('-','').replace(' ','_')
    last_file = inpath_processed_hysplit_dfs+str(last_index)[:4]+'\\'+str(last_index)+'.dat'
    print("last datetime as HYSPLIT file: "+str(last_file))
    index_of_list_file = list_of_files_all_years_flat.index(last_file)
    print("place in list of files: "+str(index_of_list_file))
    return index_of_list_file
    
def get_South_grid(ZEP_lat, ZEP_lon):
    """south grid is calculated - it is the antipole of the given lat and lon and is needed for the rotation"""
    South_Pole_lat = -ZEP_lat
    South_Pole_lon = -(180-ZEP_lon)
    South_grid = (South_Pole_lat, South_Pole_lon)
    return South_grid
    
def rotated_grid_transform(lat, lon, alt, ZEP_lat = 78.906, ZEP_lon = 11.888): 
    """this function provides xyz using the lat, lon and alt of the coordinates. the xyz are roatated such that ZEP is now the north pole"""
    South_grid = get_South_grid(ZEP_lat, ZEP_lon)    
    South_Pole_lat = South_grid[0]
    South_Pole_lon = South_grid[1]
    
    theta = 90+South_Pole_lat # Rotation around y-axis
    phi = South_Pole_lon # Rotation around z-axis

    theta = (theta*np.pi)/180
    phi = (phi*np.pi)/180 # Convert degrees to radians
    
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

    # Regular -> Rotated
    x_new = np.cos(theta)*np.cos(phi)*x + np.cos(theta)*np.sin(phi)*y + np.sin(theta)*z
    y_new = -np.sin(phi)*x + np.cos(phi)*y
    z_new = -np.sin(theta)*np.cos(phi)*x - np.sin(theta)*np.sin(phi)*y + np.cos(theta)*z
    return x_new, y_new, z_new
    
def create_row_xyz_from_df(df, col1, col2, col3, rotate=False):
    #i dont think there is a need to rotate!
    """this function applies the function to turn lat, lon, alt to xyz for each row"""
    if rotate == True:
        print("rotation of grid space")
        df['xyz'] = [rotated_grid_transform(*a) for a in tuple(zip(df[col1], df[col2], df[col3]))]
    if rotate == False:
        print("no rotation of grid space")
        df['xyz'] = [lla_to_xyz(*a) for a in tuple(zip(df[col1], df[col2], df[col3]))]
        
    df['x'], df['y'], df['z'] = zip(*df.xyz)       
    arrival_time = df['DateTime'].iloc[0] #find arrival time
    print("arrival time: "+str(arrival_time))

    df_T = df[['x','y','z']].T #transpose the x, y,z and the variable you need
    df = df_T.stack().to_frame().T #transpose and stack - creates multiplex columns 
    df.columns = ['{}_{}'.format(*c) for c in df.columns] #rename columns and create one with x0,x1,x2
    df = df.reindex(sorted(df.columns, key=lambda x: float(x[2:])), axis=1) #reordewr s.t. x0,y0,z0 ...  
    df_new = df.rename(index={0: arrival_time}) #rename row with arrival time
    return df_new
    
def create_row_with_lat_lon_alt_from_df(df, cols):
    df = df.rename(columns={"latitude":"lat", "longitude":"lon","altitude":"alt"})    
    arrival_time = df['DateTime'].iloc[0] #find arrival time
    df_T = df[cols].T #transpose the x, y,z and the variable you need
    df = df_T.stack().to_frame().T #transpose and stack - creates multiplex columns 
    df.columns = ['{}_{}'.format(*c) for c in df.columns] #rename columns and create one with x0,x1,x2
    df = df.reindex(sorted(df.columns, key=lambda x: float(x[4:])), axis=1) #reordewr s.t. x0,y0,z0 ...    
    df_new = df.rename(index={0: arrival_time}) #rename row with arrival time
    return df_new
    
def process_trajs_old(list_of_files, df_clustering_loaded=None): #load old one
    print("get a df ready for clustering: ")
    if df_clustering_loaded is not None:
        print("loaded df clustering is provided: find the last time point analysed")
        last_index = find_index_of_last_row(df_clustering_loaded, list_of_files) #see where you left off
        
    if df_clustering_loaded is None:
        print("no file given: start from zero analysed")
        last_index = 0 
        
    df_clustering = pd.DataFrame([])   
    for file in list_of_files[last_index:]:
        print("file: "+str(file))
        df = pd.read_csv(file)
        
        row = create_row_xyz_from_df(df, col1="latitude", col2="longitude", col3="altitude")
                
        df_clustering = df_clustering.append(row)
    return df_clustering
    
def process_trajs(list_of_files, df_clustering_loaded=None, rotate=False,
                  data_format='.dat'): #load old one
    print("get a df ready for clustering: ")
    if df_clustering_loaded is not None:
        print("loaded df clustering is provided: find the last time point analysed")
        last_index = find_index_of_last_row(df_clustering_loaded, list_of_files) #see where you left off
        
    if df_clustering_loaded is None:
        print("no file given: start from zero analysed")
        last_index = 0 
        
    #col_names = produce_xyz_ensemble_col_names()
    df_clustering = pd.DataFrame([])   
    for file in list_of_files[last_index:]:
        print("file: "+str(file))
        
        if data_format == '.dat':
            df = pd.read_csv(file)
        if data_format == '.pickle':
            df = pd.read_pickle(file)
            #print(df)
            df = df.reset_index()
            
        traj_nums = sorted(df['Traj_num'].unique()) #1-27 list
        DFs = [y for x, y in df.groupby('Traj_num', as_index=False)] #creats a list of dfs for each traj in ensemble
        
        errors = []
        for i, df_traj in enumerate(DFs):  
            #print(df_traj)
            row = create_row_xyz_from_df(df_traj, col1="latitude", col2="longitude", col3="altitude", rotate=rotate)
            #rename the columns such that it goes x1,y1, z1 to x241, y241
            #print(row.columns)
            reduced_cols = [str(x[:2])+str(int(x[2:])%27) for x in row.columns] #27 is the number of ensembles
            #print(reduced_cols)
            initial_value = int(reduced_cols[0][2:])
            reduced_cols = [str(x[:2])+str(int(x[2:])+int(i/3)) for i, x in enumerate(reduced_cols)]
            reduced_cols = [str(x[:2])+str(int(x[2:])-initial_value) for x in reduced_cols]
            row.columns = reduced_cols #rename columns           
            row = row.reset_index()
            row['traj_num'] = initial_value + 1
            row = row.set_index(['index', 'traj_num'])  
            try:
                df_clustering = df_clustering.append(row)
            except:
                errors.append(i)
                print(row)
                print(df_clustering.tail())
    if len(errors) > 0:
        print(errors)
    return df_clustering
    
# def process_trajs_lat_lon_alt_old(list_of_files, df_loaded=None): #load old one
    # print("get a df ready for clustering: ")
    # if df_loaded is not None:
        # print("loaded df clustering is provided: find the last time point analysed")
        # last_index = find_index_of_last_row(df_clustering_loaded, list_of_files) #see where you left off
        
    # if df_loaded is None:
        # print("no file given: start from zero analysed")
        # last_index = 0 
        
    # df_lat_lon_alt = pd.DataFrame([])   
    # for file in list_of_files[last_index:]:
        # print("file: "+str(file))
        # df = pd.read_csv(file)
        # row = create_row_with_lat_lon_alt_from_df(df, cols=["lat", "lon","alt"])
        # df_lat_lon_alt = df_lat_lon_alt.append(row)
    # return df_lat_lon_alt
    
def process_trajs_lat_lon_alt(list_of_files, df_loaded=None, read_csv=True, read_pickle=False): #load old one
    print("get a df ready for clustering: ")
    if df_loaded is not None:
        print("loaded df clustering is provided: find the last time point analysed")
        last_index = find_index_of_last_row(df_clustering_loaded, list_of_files) #see where you left off        
    if df_loaded is None:
        print("no file given: start from zero analysed")
        last_index = 0         
    df_lat_lon_alt = pd.DataFrame([])   
    for file in list_of_files[last_index:]:
        print("file: "+str(file))
        if read_csv == True:
            df = pd.read_csv(file)   
        if read_pickle == True:
            df = pd.read_pickle(file)           
        traj_nums = sorted(df['Traj_num'].unique()) #1-27 list
        DFs = [y for x, y in df.groupby('Traj_num', as_index=False)] #creats a list of dfs for each traj in ensemble
        errors = []
        for i, df_traj in enumerate(DFs):        
            row = create_row_with_lat_lon_alt_from_df(df_traj, cols=["lat", "lon","alt"])
            #rename the columns such that it goes x1,y1, z1 to x241, y241
            reduced_cols = [str(x[:4])+str(int(x[4:])%27) for x in row.columns] #27 is the number of ensembles
            initial_value = int(reduced_cols[0][4:])
            reduced_cols = [str(x[:4])+str(int(x[4:])+int(i/3)) for i, x in enumerate(reduced_cols)]
            reduced_cols = [str(x[:4])+str(int(x[4:])-initial_value) for x in reduced_cols]
            row.columns = reduced_cols #rename columns           
            row = row.reset_index()
            row['traj_num'] = initial_value + 1
            row = row.set_index(['index', 'traj_num'])  
            try:
                df_lat_lon_alt = df_lat_lon_alt.append(row)
            except:
                errors.append(i)
                print(row)
                print(df_lat_lon_alt.tail())
    if len(errors) > 0:
        print(errors)
    return df_lat_lon_alt
    
def remove_np_nan_and_inf(df):
    print("Length before: "+str(len(df)))
    df.replace([np.inf, -np.inf], np.nan, inplace=True) #cant cluster with np.nan
    df = df.dropna(how='any', axis=0)
    print("Length after: "+str(len(df)))
    return df
    
def create_silhouettes_elbow(df_clustering, sample_size, perform_silhouette=True, 
                             perform_elbow=False, number_to_perform=20):
    if perform_silhouette == True:
        silhouette = np.zeros((number_to_perform))
    if perform_elbow == True:
        elbow = np.zeros((number_to_perform))
        
    print("performing silhouettes or elbow")      
    if 'traj_num' in df_clustering.columns:
        df_clustering = df_clustering.reset_index()
        df_clustering = df_clustering.set_index(['arrival_time', 'traj_num'])        
                   
    df_clustering = remove_np_nan_and_inf(df_clustering)
    df_clustering = remove_duplicates(df_clustering)
        
    for n_clusters in np.arange(2,number_to_perform,1): #loop through a range of different possible clusters 
        name_nbcluster = 'nbclusters_' + str(n_clusters)
        print(name_nbcluster)
        kmeans = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0)
        
        df_kmean = df_clustering.copy()
        kmeans.fit(df_kmean)
        centroids = kmeans.cluster_centers_        
        exec(name_nbcluster + '=' + str(centroids.tolist()))
        df_kmean['cluster'] = kmeans.predict(df_kmean)
        
        if perform_silhouette == True:
            silhouette[n_clusters] = metrics.silhouette_score(df_kmean, df_kmean['cluster'], sample_size=sample_size)
        if perform_elbow == True:
            elbow[n_clusters] = kmeans.inertia_
    if perform_silhouette == True:
        return silhouette 
    if perform_elbow == True:
        return elbow
    
def plot_silhouette_elbow(silhouette, number_to_perform=20):
    fig, ax = plt.subplots(num=None, figsize=(17,5))
    ax.plot(np.arange(2, number_to_perform, 1), silhouette[2:], c='k')
    ax.set_xlabel('Number of clusters', fontsize=14)
    ax.set_ylabel('Silhouette score', fontsize=14)
    ax.set_xticks(np.arange(2, number_to_perform, 1))
    ax.set_xticklabels(np.arange(2, number_to_perform, 1), fontsize=14)
    plt.show()
    return fig
    
def cluster(df_kmean, nb_clusters):
    kmeans = KMeans(init="k-means++", n_clusters=nb_clusters, n_init=4, random_state=0)
    kmeans.fit(df_kmean)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.predict(df_kmean)
    return clusters
    
def cluster_minibatch(df_kmean, nb_clusters):
    print("mini batch")
    kmeans = MiniBatchKMeans(init="k-means++", n_clusters=nb_clusters).fit(df_kmean)
    centroids = kmeans.cluster_centers_
    clusters = kmeans.predict(df_kmean)
    return clusters
    
def convert_index_to_datetime(df):
    print(df.index.dtype)
    df.index = pd.to_datetime(df.index)
    print(df.index.dtype)
    return df
    
def remove_duplicates(df):    
    return df[~df.index.duplicated(keep='first')]
    
def clustering(df_clustering, nb_clusters=5, mini_batch=False, just_clusters=False, 
               traj_num='traj_num'):
    print("performing kmeans clustering")      
    if str(traj_num) in df_clustering.columns:
        df_clustering = df_clustering.reset_index()
        df_clustering = df_clustering.set_index(['arrival_time', traj_num])        
                   
    df_clustering = remove_np_nan_and_inf(df_clustering)
    df_clustering = remove_duplicates(df_clustering)
    
    if mini_batch == False:
        df_kmean5_clusters = cluster(df_clustering, nb_clusters)
    if mini_batch == True: 
        df_kmean5_clusters = cluster_minibatch(df_clustering, nb_clusters)
        
    if just_clusters == True:
        return df_kmean5_clusters
    
    print("add to df:")
    df_clustering['clusters_'+str(nb_clusters)] = df_kmean5_clusters
    print("unique clusters: "+str(df_clustering['clusters_'+str(nb_clusters)].unique()))
    df_clustering['clusters_'+str(nb_clusters)] = df_clustering['clusters_'+str(nb_clusters)] + 1
    print("unique clusters (add 1): "+str(df_clustering['clusters_'+str(nb_clusters)].unique()))
    return df_clustering
       
def read_in_or_process(filename='_.dat', read_in=True, process=False, clustering=True,
                       lat_lon_alt=False, number_to_process=None, inpath=None,
                       list_of_all_files=None, traj_num=None, dict_dtype=None, rotate=False,
                       data_format='.dat'):                           
    if clustering == True:
        print("for clustering file")
        if read_in == True:
            print("reading in: "+str(inpath+filename))
            df_clustering = pd.read_csv(inpath+filename, index_col=0, dtype=dict_dtype)            
            df_clustering.index = pd.to_datetime(df_clustering.index)
            df_clustering = df_clustering.sort_index()
        if process==True:
            print("processing:")
            if number_to_process is not None:
                df_clustering = process_trajs(list_of_all_files[:int(number_to_process)], rotate=rotate,
                                              data_format=data_format)
            if number_to_process is None:
                df_clustering = process_trajs(list_of_all_files[:], rotate=rotate, data_format=data_format)
        df = df_clustering.copy()
        
    if lat_lon_alt==True:
        print("for lat lon alt file")
        if read_in == True:
            print("reading in: "+str(inpath+filename))
            df_lat_lon_alt = pd.read_csv(inpath+filename, index_col=0)
            df_lat_lon_alt.index = pd.to_datetime(df_lat_lon_alt.index)
            df_lat_lon_alt = df_lat_lon_alt.sort_index()
        if process == True:
            print("processing:")
            if number_to_process is not None:
                df_lat_lon_alt = process_trajs_lat_lon_alt(list_of_all_files[:int(number_to_process)])
            if number_to_process is None:
                df_lat_lon_alt = process_trajs_lat_lon_alt(list_of_all_files[:])
        df = df_lat_lon_alt.copy()
        
    if len(df.index.names) == 1:   
        print("single index")
        df = convert_index_to_datetime(df)    
    #multiindex
    if len(df.index.names) == 2:   
        print("multi index")
        print(df.index.names)
        df.index.names = ['arrival_time', 'traj_num']    
    return df
    
def process_and_save_years(df_abs637, inpath = "C:\\Users\\DominicHeslinRees\\Documents\\Analysis\\Clustering\\",
                           HYSPLIT_inpath_files="E:\\Data\\HYSPLIT\\processed\\", start_year=2002, end_year=2021,
                           read_in=False, process=True, clustering=True, lat_lon_alt=False, number_to_process=None,
                           rotate=False):      
                                
    list_of_all_files = all_matching_HYSPLIT_obs_files(df_abs637, np.arange(start_year, end_year+1), HYSPLIT_inpath_files)
    df_clustering = read_in_or_process(read_in=read_in, process=process, clustering=clustering, 
                                                  lat_lon_alt=lat_lon_alt, inpath=inpath,
                                                  list_of_all_files=list_of_all_files,
                                                  number_to_process=number_to_process, rotate=rotate)
    if clustering == True: 
        print("saving: ")
        save_df(df_clustering, inpath, name='df_for_clustering_'+str(start_year))
    if lat_lon_alt == True:
        print("saving: ")
        save_df(df_clustering, inpath, name='df_for_lat_lon_alt_'+str(start_year))
    return list_of_all_files, df_clustering
    
def concat_all_from_path(inpath, format_of_data='.dat', traj_num=None,
                         dict_dtype=None):
    print("concat all in :"+str(inpath))
    paths = glob.glob(inpath+'\*'+str(format_of_data))
    DFs = []    
    paths = sorted(paths, key=lambda x: int(get_digits(x))) 
    for path in paths:
        print("path: "+str(path))
        file = path.replace(inpath, '')
        print("file: "+str(file))
        df_clustering = read_in_or_process(filename=file, inpath=inpath, dict_dtype=dict_dtype)
        if traj_num is not None:
            df_clustering = df_clustering[df_clustering['traj_num']==traj_num]
        DFs.append(df_clustering)
    df_merge = pd.concat(DFs)   
    return df_merge
    
def sort_remove_index(df, convert_to_array=False):
    df.index = pd.to_datetime(df.index)
    df = df.sort_index()
    df = df.reset_index()
    if 'traj_num' in df.columns:
        df = df.drop(['arrival_time','traj_num'], axis=1)
    if convert_to_array == True:
        df = df.to_numpy()
    return df
    
def sort_index(df):
    #single index 
    if len(df.index.names) == 1:   
        print("single index")
        df = df.sort_values(by=['arrival_time'])
    #multiindex
    if len(df.index.names) == 2:   
        print("multi index")
        print(df.index.names)
        df.index.names = ['arrival_time', 'traj_num'] #cheack
        df = df.sort_values(by=['arrival_time', 'traj_num'])
    df = df.reset_index()
    return df
    
def concat_dfs(df_clustering1=None, df_clustering2=None, 
               df_filename1=None, df_filename2=None, inpath='', save_mergefilename='defaultname'):    
    if (df_filename1 is not None) & (df_filename2 is not None):
        print("read in: "+str(inpath+df_filename1))
        df_clustering1 = pd.read_csv(inpath+df_filename1+'.dat', index_col=0)
        print("read in: "+str(inpath+df_filename2))
        df_clustering2 = pd.read_csv(inpath+df_filename2+'.dat', index_col=0)    
        df_merge = pd.concat([df_clustering1, df_clustering2])        
    if (df_clustering1 is not None) & (df_clustering2 is not None):
        print("files given: ")
        df_merge = pd.concat([df_clustering1, df_clustering2])      
    df_merge.index = pd.to_datetime(df_merge.index)
    save_df(df_merge, inpath, name=save_mergefilename)
    return df_merge
    
def add_missing_clusters(df, nb_clusters=5):
    list_1 = list(df['cluster'].unique())
    list_2 = list(np.arange(1, nb_clusters+1, 1))
    missing_clusters = list(set(list_2) - set(list_1))
    for missing_cluster in missing_clusters:
        df.loc[missing_cluster] = [missing_cluster, np.nan, np.nan, np.nan, np.nan] #len(df.columns)*np.nan #row 
        #print(missing_cluster)
        #df.loc[(df['cluster'] == missing_cluster), ['percentage', 0.25, 0.5, 0.75]] = len(['percentage', 0.25, 0.5, 0.75])*np.nan #add row
        #print(df.loc[(df['cluster'] == missing_cluster), ['percentage', 0.25, 0.5, 0.75]])
    return df
    
def find_missing_clusters(df, nb_clusters=5):
    list_1 = list(df['clusters_'+str(nb_clusters)].unique())
    print("list1: "+str(list_1))
    list_2 = list(np.arange(1, nb_clusters+1, 1))
    print("list2: "+str(list_2))
    missing_clusters = list(set(list_2) - set(list_1))
    print(missing_clusters)
    return missing_clusters 
    
def merge_latlonalt_with_clusters(df_lat_lon_alt, df_clustering, nb_clusters=5):
    if (len(df_lat_lon_alt.index.names) == 2) &  (len(df_clustering.index.names) == 2):
        df_lat_lon_alt = df_lat_lon_alt.merge(df_clustering[['clusters_'+str(nb_clusters)]], 
                                left_index=True, right_index=True)
    else:
        print("make sure both dfs are multiindexed i.e. arrival time and enemble")
    return df_lat_lon_alt
    
def get_xyz_cols(df, elements_in_col_name=['x_', 'y_', 'z_']):
    cols = df.columns.to_list()
    if elements_in_col_name!=['x_', 'y_', 'z_']:
        xyz_cols = []
        for element in elements_in_col_name:
            xyz_cols = [x for x in cols if (element in x)]
            xyz_cols = xyz_cols.append(xyz_cols)
    if elements_in_col_name==['x_', 'y_', 'z_']:
        xyz_cols = [x for x in cols if ('x_' in x) or ('y_' in x) or ('z_' in x)]
    return xyz_cols
    
def calculate_percentage_of_clusters_and_averages(df_obs_clusters, nb_clusters, var):
    size = len(df_obs_clusters)
    print("size of the clustering data frame: "+str(size))
    df_values = (df_obs_clusters['clusters_'+str(nb_clusters)].value_counts().to_frame()/size)*100
    df_values = df_values.sort_index() 
    df_values = df_values.reset_index()
    df_values.columns = ['cluster', 'percentage']

    #df_obs_clusters_groupby = df_obs_clusters.groupby('clusters_'+str(nb_clusters))[var].median() #group by clusters     
    df_obs_clusters_groupby = df_obs_clusters.groupby('clusters_'+str(nb_clusters))[var].quantile([0.25, 0.50, 0.75]).unstack() #group by clusters         
    df_obs_clusters_groupby = df_obs_clusters_groupby.reset_index()
    df_obs_clusters_groupby.columns = ['cluster', 0.25, 0.5, 0.75]
    
    #df_values = pd.concat([df_values, df_obs_clusters_groupby], axis=1, join="inner")
    df_values = pd.merge(df_values, df_obs_clusters_groupby, on='cluster')
    df_values  = df_values.round(3) 
    print(df_values)
    df_values = add_missing_clusters(df_values, nb_clusters)    
    return df_values
    
def merge_with_obs(df_coord, df_obs):
    df_obs = remove_np_nan_and_inf(df_obs)
    df_obs = convert_index_to_datetime(df_obs)
    df_obs = remove_duplicates(df_obs)
    
    df_coord = remove_np_nan_and_inf(df_coord)
    #single index 
    if len(df_coord.index.names) == 1:   
        print("single index")
        df_coord = convert_index_to_datetime(df_coord)
    #multiindex
    if len(df_coord.index.names) == 2:   
        print("multi index")
        print(df_coord.index.names)
        df_coord.index.names = ['arrival_time', 'traj_num']
        df_coord = df_coord.reset_index(level=1)
        df_coord.index = pd.to_datetime(df_coord.index)

    df_coord = df_coord.merge(df_obs, left_index=True, right_index=True)    
    return df_coord
    
def merge_xyz_latlonalt(df_obs, df_lat_lon_alt, df_clustering, var_obs, nb_clusters=5):  
    """Merge the observations with the xyz and the lat lon alt coordinates"""
    
    for df in [df_obs, df_clustering, df_lat_lon_alt]:
        df = remove_np_nan_and_inf(df)
        df = convert_index_to_datetime(df)
        df = remove_duplicates(df)
    
    df_clustering = pd.concat([df_obs, df_clustering], axis=1, join="inner")
    df_obs_clusters_wihtout_xyz = df_clustering[[var_obs,'clusters_'+str(nb_clusters)]]            
    df_lat_lon_alt = pd.concat([df_obs_clusters_wihtout_xyz, df_lat_lon_alt], axis=1, join="inner")    
    return df_clustering, df_obs_clusters_wihtout_xyz,  df_lat_lon_alt
    
########################################################################################################################################
    
##################################PLOTTTING#############################################################################################

def sf(sf_num):
    sf = '{0:.'+str(sf_num)+'f}'
    return sf
    
def thickax(ax):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.3)
        ax.spines[axis].set_color('k')
    plt.rc('axes', linewidth=0.2)
    fontsize = 12
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.tick_params(direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', length=4, color='k', width=1.3)

def create_legend(nb_clusters, df_values, dict_cluster_to_colors=None):  
    legend_elements = []    
    if dict_cluster_to_colors is None:
        colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))  
    clusters = df_values['cluster'].unique()
    for cluster in clusters: #np.arange(1, nb_clusters+1, 1):
        if dict_cluster_to_colors is None:
            color=colors[cluster]
        if dict_cluster_to_colors is not None:
            color = dict_cluster_to_colors[cluster]            
        #percentage = df_values.loc[cluster,'clusters_'+str(nb_clusters)]
        percentage = df_values.loc[df_values['cluster'] == cluster, 'percentage'].values[0]
        #BC_med = df_values.loc[cluster,0.5]
        #BC_25 = df_values.loc[cluster,0.25]
        #BC_75 = df_values.loc[cluster,0.75]      
        BC_med = df_values.loc[df_values['cluster'] == cluster, 0.5].values[0]
        BC_25 = df_values.loc[df_values['cluster'] == cluster, 0.25].values[0]
        BC_75 = df_values.loc[df_values['cluster'] == cluster, 0.75].values[0]
        sfs = sf(3)                  
        legend_element = Line2D([0], [0], marker='o', color='w', label=str(cluster)+': ' #+str(percentage)+'$\,$%'+
                                +str(sfs.format(BC_med))+' ('+str(sfs.format(BC_25))+' - '+str(sfs.format(BC_75))+')',
                                  markerfacecolor=color, markersize=15)
        legend_elements.append(legend_element)
    return legend_elements
    
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
    
def plot_all_trajs(df_traj, nb_clusters, axes_projection, scatter_projection, 
                   df_values, extend=None,
                   ZEP_lat=78.906,ZEP_lon=11.888, dict_cluster_to_colors=None):
    """uses lat and lon """
    
    print("Length of trajectroy: "+str(len(df_traj)))  
    
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon)                   
    projection = dict_projections[axes_projection]       
    scatter_projection = dict_projections[scatter_projection]  
    geo = dict_projections['geo']

    fig = plt.figure(figsize=(8,8))
    
    ax = plt.axes(projection=projection)  
    
    #The projection argument is used when creating plots and determines the projection of the resulting plot (i.e. what the plot looks like).    
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
    
    cols = df_traj.columns #lat0, lon0, alt0, ... , lat241, lon241, alt241
    lats = [x for x in cols if 'lat_' in x]
    lons = [x for x in cols if 'lon_' in x]
    
    if dict_cluster_to_colors is None:
        clusters = sorted(df_traj['clusters_'+str(nb_clusters)].unique())    
        colors=cm.rainbow(np.linspace(0,1,int(len(clusters))))
        dict_cluster_to_colors = dict(zip(clusters, colors))
    
    for cluster in clusters:        
        df_cluster = df_traj[df_traj['clusters_'+str(nb_clusters)] == cluster]
        if df_cluster is not None:             
            for i in range(len(df_cluster)): #loop through all trajs
                lon_traj = df_cluster.loc[:,lons].iloc[i].values                
                lat_traj = df_cluster.loc[:,lats].iloc[i].values
                                               
                ax.scatter(lon_traj, lat_traj, marker="o", 
                           color=dict_cluster_to_colors[cluster],
                           transform=scatter_projection,
                           s=2, alpha=0.5)    
    
    if axes_projection == 'North_Stereo':
        extent = [-180, 180, 45, 90]
    if axes_projection == 'ortho':
        ax.set_global()
    if extend is not None:        
        east_lon = -90
        west_lon = 90
        south_lat = 45
        north_lat = 90        
        print("extent: east_lon "+str(east_lon)+", west_lon: "+str(west_lon)+", south_lat: "+str(south_lat)+", north_lat: "+str(north_lat))
        extent = [east_lon, west_lon, south_lat, north_lat]
   
    ax.plot([ZEP_lon], [ZEP_lat], 'ro', transform=geo)    
    ax.gridlines()
    ax.coastlines(resolution='50m')    
    start_date = df_traj.index[0]
    end_date = df_traj.index[-1]    
    plt.title(str(start_date)+' - '+str(end_date), loc='left')    
    legend_elements = create_legend(nb_clusters, df_values, dict_cluster_to_colors=dict_cluster_to_colors)
    plt.legend(title="Cluster: ", handles=legend_elements, loc='upper right', frameon=False)
    plt.show()
    return fig
    
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
    
def cal_mean_coords(df, cols):            
    x = 0.0
    y = 0.0
    z = 0.0    
    df = df[cols]
    df = df.dropna(how='all')
    for i, coord in df.iterrows():    
        lat = coord[cols[0]]
        lon = coord[cols[1]]
        alt = coord[cols[2]]        
        x += lla_to_xyz(lat, lon, alt)[0]
        y += lla_to_xyz(lat, lon, alt)[1]
        z += lla_to_xyz(lat, lon, alt)[2]             
    total = len(df) 
    x_ = x / total
    y_ = y / total
    z_ = z / total     
    lat, lon, alt = fu.xyz_to_lla(x_,y_,z_)
    return lat, lon, alt
    
def find_mean_traj_old(df_obs_clusters, nb_clusters):
    """just takes the averages of the xyz and then converts to lat lon, alt afterwards"""
    cols = df_obs_clusters.columns.to_list()
    xyz_cols = [x for x in cols if ('x_' in x) or ('y_' in x) or ('z_' in x)]
        
    dict_cluster_to_df_average = {}
    clusters = df_obs_clusters['clusters_'+str(nb_clusters)].unique()  

    for cluster in clusters:
        print("cluster: "+str(cluster))        
        df_cluster = df_obs_clusters.loc[df_obs_clusters['clusters_'+str(nb_clusters)] == cluster, :].copy()        

        sum_column = df_cluster.loc[:,xyz_cols].sum(axis=0)
        df_cluster.at[-1, xyz_cols] = sum_column #add row at the end (-1) 
        df_cluster.iloc[-1, [df_cluster.columns.get_loc(c) for c in xyz_cols]] = df_cluster.iloc[-1, [df_cluster.columns.get_loc(c) for c in xyz_cols]]/len(df_obs_clusters)
        df_cluster = df_cluster.rename(index={-1: 'Time_average'})

        time_averages = df_cluster.loc['Time_average', xyz_cols].values

        x_time_averages_27 = np.add.reduceat(time_averages[::3], np.arange(0, len(time_averages[::3]), 27))/27  
        y_time_averages_27 = np.add.reduceat(time_averages[1::3], np.arange(0, len(time_averages[1::3]), 27))/27  
        z_time_averages_27 = np.add.reduceat(time_averages[2::3], np.arange(0, len(time_averages[2::3]), 27))/27  

        df_mean_coord = pd.DataFrame(columns=['lat', 'lon', 'alt'])
        for i in range(int(len(time_averages)/3)):
            try:
                x = x_time_averages_27[i]
                y = y_time_averages_27[i]
                z = z_time_averages_27[i]
                lat, lon, alt = xyz_to_lla(x,y,z)
                df_mean_coord = df_mean_coord.append({'lat': lat, 'lon':lon, 'alt':alt}, ignore_index=True)
                dict_cluster_to_df_average[cluster] = df_mean_coord
            except IndexError:
                pass
                
    missing_clusters = find_missing_clusters(df_obs_clusters, nb_clusters=5)
    
    for missing_cluster in missing_clusters:
        dict_cluster_to_df_average[missing_cluster] = None
    
    return dict_cluster_to_df_average
    
def get_df_mean_coord(dict_xyz_means):
    df_mean_coord = pd.DataFrame(columns=['lat', 'lon', 'alt'])
    
    for i in range(len(dict_xyz_means['x'])):
        x = dict_xyz_means['x'][i]
        y = dict_xyz_means['y'][i]
        z = dict_xyz_means['z'][i]
        lat, lon, alt = xyz_to_lla(x,y,z)
        df_mean_coord = df_mean_coord.append({'lat': lat, 'lon':lon, 'alt':alt}, ignore_index=True)
    return df_mean_coord 

def create_df_xyz_averages(df_clustering, cluster, nb_clusters=5, mean=True, median=False):
    cols = df_clustering.columns.to_list()
    x_cols = [x for x in cols if ('x_' in x)]
    y_cols = [x for x in cols if ('y_' in x)]  
    z_cols = [x for x in cols if ('z_' in x)]  
    df_cluster = df_clustering.loc[df_clustering['clusters_'+str(nb_clusters)] == cluster, :].copy()      
    dict_xyz_averages = {}
    for cols, col_names in zip([x_cols, y_cols, z_cols], ['x','y','z']):
        if mean == True:
            print("mean")
            average_of_columns = df_cluster.loc[:, cols].mean(axis=0)   
        if median == True:
            print("median")
            average_of_columns = df_cluster.loc[:, cols].median(axis=0)    
        dict_xyz_averages[col_names] = average_of_columns.values    
    df_average_coord = get_df_mean_coord(dict_xyz_averages)   
    return df_average_coord

def create_dict_cluster_to_df_average(df_clustering, nb_clusters=5, mean=True, median=False):
    dict_cluster_to_df_average = {}
    clusters = df_clustering['clusters_'+str(nb_clusters)].unique()
    for cluster in clusters:
        df_average_coord = create_df_xyz_averages(df_clustering, cluster, nb_clusters, mean, median)
        dict_cluster_to_df_average[cluster] = df_average_coord       
    missing_clusters = find_missing_clusters(df_clustering, nb_clusters)    
    for missing_cluster in missing_clusters:
        dict_cluster_to_df_average[missing_cluster] = None    
    return dict_cluster_to_df_average
      
def plot_trajs_clusters(df_obs_clusters, 
                        dict_cluster_to_df_average, nb_clusters, axes_projection, scatter_projection,
                        df_values, ZEP_lat=78.906,ZEP_lon=11.888, dict_cluster_to_colors=None):
    """plots the mean trajectories of the clusters. df obs clusters is only used for the start and stop times"""
    print("plot the mean trajectory")
    
    fig = plt.figure(figsize=(8,8))
    
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon)                   
    projection = dict_projections[axes_projection]       
    scatter_projection = dict_projections[scatter_projection]  
    geo = dict_projections['geo']
    
    ax = plt.axes(projection=projection)  
    #The projection argument is used when creating plots and determines the projection of the resulting plot (i.e. what the plot looks like).    
    if dict_cluster_to_colors == None:
        print("create colour map dict")
        colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
        clusters = sorted([*dict_cluster_to_df_average.keys()])
        dict_cluster_to_colors = dict(zip(clusters, colors))
    
    for cluster in clusters:        
        df_cluster = dict_cluster_to_df_average[cluster]
        if df_cluster is not None:            
            for i in range(len(df_cluster)):           
                lon_traj = df_cluster['lon'].values
                lat_traj = df_cluster['lat'].values
                ax.scatter(lon_traj, lat_traj, marker="o", color=dict_cluster_to_colors[cluster], 
                           transform=scatter_projection,
                           s=2, alpha=0.5)    

    if axes_projection == 'North_Stereo':
        extent = [-180, 180, 45, 90]
        ax.set_extent(extent, crs=ccrs.PlateCarree())         
    if axes_projection == 'ortho':
        extent = [-180, 180, 55, 90]
        ax.set_extent(extent, crs=ccrs.PlateCarree())         
        
    else:
        east_lon = -90
        west_lon = 90
        south_lat = 45
        north_lat = 90        
        extent = [east_lon, west_lon, south_lat, north_lat]
   
    ax.plot([ZEP_lon], [ZEP_lat], 'ro', transform=geo) 
    ax.gridlines()
    ax.coastlines(resolution='50m')    
    start_date = df_obs_clusters.index[0]
    end_date = df_obs_clusters.index[-1]    
    
    plt.title(str(start_date)+' - '+str(end_date), loc='left')  

    legend_elements = create_legend(nb_clusters, df_values, dict_cluster_to_colors=dict_cluster_to_colors)    
    plt.legend(title="Cluster: obs ", handles=legend_elements,  
               fontsize=12, title_fontsize=15, loc='upper left', shadow=True)   
    plt.show()
    return fig
    
def sp_map(*nrs, projection = ccrs.PlateCarree(), **kwargs):
    return plt.subplots(*nrs, subplot_kw={'projection':projection}, **kwargs)

def add_map_features(ax):
    ax.coastlines()
    gl = ax.gridlines()
    ax.add_feature(cy.feature.BORDERS);
    gl = ax.gridlines(draw_labels=True, linewidth=1)    
    #gl.xformatter = LONGITUDE_FORMATTER
    #gl.yformatter = LATITUDE_FORMATTER
    gl.top_labels = True
    gl.right_labels = True
    
def polarCentral_set_latlim(lat_lims, ax):
    ax.set_extent([-180, 180, lat_lims[0], lat_lims[1]], ccrs.PlateCarree())
    # Compute a circle in axes coordinates, which we can use as a boundary
    # for the map. We can pan/zoom as much as we like - the boundary will be
    # permanently circular.
    theta = np.linspace(0, 2*np.pi, 100)
    center, radius = [0.5, 0.5], 0.5
    verts = np.vstack([np.sin(theta), np.cos(theta)]).T
    circle = mpath.Path(verts * radius + center)
    ax.set_boundary(circle, transform=ax.transAxes)
    
def circular_plots(nb_clusters, dict_cluster_to_df_average, df_values=None, title="Cluster: BC [ngm$^{-3}$] ",
                   ZEP_lat=78.906,ZEP_lon=11.888, x_coord_legend=1, dict_cluster_to_colors=None,
                   scatter_projection='rotated_pole'):
    
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon)                   
    North_Stereo = dict_projections['North_Stereo']  
    scatter_projection = dict_projections[scatter_projection]    
    geo = dict_projections['geo']
 
    fig, ax = sp_map(1, projection=North_Stereo, figsize=(10,10))
       
    lat_lims = [60,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
    
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
    clusters = sorted([*dict_cluster_to_df_average.keys()])
    
    if dict_cluster_to_colors is None:        
        dict_cluster_to_colors = dict(zip(clusters, colors))
    
    for cluster in clusters:
        df_cluster = dict_cluster_to_df_average[cluster]
        if df_cluster is not None:
            for i in range(len(df_cluster)):           
                lon_traj = df_cluster['lon'].values
                lat_traj = df_cluster['lat'].values
                ax.scatter(lon_traj, lat_traj, marker="o", 
                           color=dict_cluster_to_colors[cluster], transform=scatter_projection,
                           s=6, alpha=0.5)                              
    if df_values is not None: 
        legend_elements = create_legend(nb_clusters, df_values, dict_cluster_to_colors=dict_cluster_to_colors)
        plt.legend(title=title, handles=legend_elements, fontsize=18, title_fontsize=25, 
                   loc='upper left', shadow=False, bbox_to_anchor=(x_coord_legend, 1.0), frameon=False)
    plt.show()
    return fig
    
def circular_plots_all_trajs(df_traj, scatter_projection, df_values,
                             title='', nb_clusters=5, ZEP_lat=78.906,ZEP_lon=11.888,
                             dict_cluster_to_colors=None):
                             
    dict_projections = create_projections_dict(ZEP_lat,ZEP_lon) 
    
    North_Stereo = dict_projections['North_Stereo']  
    rotated_pole = dict_projections['rotated_pole']    
    scatter_projection = dict_projections[scatter_projection]  
    geo = dict_projections['geo']
 
    fig, ax = sp_map(1, projection=North_Stereo, figsize=(10,10))

    lat_lims = [60,90]
    polarCentral_set_latlim(lat_lims, ax)
    add_map_features(ax)
    
    cols = df_traj.columns #lat0, lon0, alt0, ... , lat241, lon241, alt241
    lats = [x for x in cols if 'lat_' in x]
    lons = [x for x in cols if 'lon_' in x]
    
    clusters = sorted(df_traj['clusters_'+str(nb_clusters)].unique())   
    if dict_cluster_to_colors is not None:
        colors=cm.rainbow(np.linspace(0,1,int(len(clusters))))
        dict_cluster_to_colors = dict(zip(clusters, colors))
    
    for cluster in clusters:        
        df_cluster = df_traj[df_traj['clusters_'+str(nb_clusters)] == cluster]
        if df_cluster is not None:               
            for i in range(len(df_cluster)): #loop through all trajs
                lon_traj = df_cluster.loc[:,lons].iloc[i].values                
                lat_traj = df_cluster.loc[:,lats].iloc[i].values
                              
                ax.scatter(lon_traj, lat_traj, marker="o", 
                           color=dict_cluster_to_colors[cluster],
                           transform=scatter_projection,
                           s=2, alpha=0.5) 
      
    legend_elements = create_legend(nb_clusters, df_values, dict_cluster_to_colors=dict_cluster_to_colors)
    plt.legend(title=title, handles=legend_elements, fontsize=18, title_fontsize=25, 
               loc='upper left', shadow=True, bbox_to_anchor=(1.05, 0.8))
    plt.show()
    return fig
    
########################################################################################################################################
    
####################TIME SERIES AND FREQUENCY PLOTS#####################################################################################

def get_first_season(df):
    first_month = df.index.month[0]
    if first_month in [2,3,4,5]: #FMAM
        first_season = 'AHZ' #Arctic HAZE
    if first_month in [6,7,8,9]: #JJAS
        first_season = 'SUM' #SUMMER
    if first_month in [10,11,12,1]:
        first_season = 'SBU' 
    return first_season

def get_full_season_abb_years(start_year, number_years, first_season):
    """loops through the years and the seasons to get a full mapping for the season and year and it's respective order"""   
    season_list=['AHZ','SUM','SBU']
    season_abb_years = []
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

def reverse_dict(dictionary):
    inv_dict = {v: k for k, v in dictionary.items()}
    return inv_dict

def convert_season_add_year_to_datetime(season_abb_year):
    year = season_abb_year[-4:]
    season_abb = season_abb_year[:3]    
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

def q25(x, percentile=0.25):
    return x.quantile(percentile)

def q75(x):
    return x.quantile(0.75)    

def produce_averages_groupby(df, groupby_var, variable):
    idx = df.columns.get_loc(str(variable))
    df_groupby = df.groupby(by=groupby_var)[variable].agg(['mean', 'median', 'min', 'max', 'std', 'count', q25, q75]) 
    return df_groupby

def mid_datetime_function(a, b):
    return a + (b - a)/2

def add_mid_datetime_using_dictionary(df, season_num_to_season):
    df['season_abb_year'] = df.index.map(season_num_to_season)
    df['start'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[0])
    df['stop'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[1])
    df['mid_datetime'] = df.apply(lambda x: mid_datetime_function(x.start, x.stop), axis=1)
    return df

def normalise_index(df):
    df.index = df.index - df.index[0] + 1
    return df

def seasonal_averages(df, variable, season_num_to_season, 
                      groupby_var='season_ordinal', season_abb=None,
                      season_abb_year_col = 'season_abb_year'):       
    df_seasonal_averages = produce_averages_groupby(df, groupby_var=groupby_var, variable=variable)   
    
    df_seasonal_averages = add_mid_datetime_using_dictionary(df_seasonal_averages, season_num_to_season)
    df_seasonal_averages['season_abb'] = df_seasonal_averages[season_abb_year_col].apply(lambda x: x[:3] if isinstance(x, str) else x)

    if season_abb is not None:       
        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]
        df_seasonal_averages = df_season.copy() 
        
    df_seasonal_averages = normalise_index(df_seasonal_averages)
    return df_seasonal_averages
    
def month_years_clusters(df_clustering, nb_clusters):
    df_clustering.index = pd.to_datetime(df_clustering.index)
    df_months_clusters = pd.DataFrame([])
    print(nb_clusters)
    for cluster in np.arange(1, nb_clusters+1, 1):
        print("cluster: "+str(cluster))
        count = df_clustering[df_clustering['clusters_'+str(nb_clusters)] == cluster].resample('M')['clusters_'+str(nb_clusters)].count()
        df_months_clusters['cluster_'+str(cluster)] = count

    cluster_cols = df_months_clusters.columns
    df_months_clusters['month_sum'] = df_months_clusters.sum(axis=1).values
    df_months_clusters[cluster_cols] = df_months_clusters[cluster_cols].div(df_months_clusters['month_sum'].values,axis=0)
    
    cluster_cols = df_months_clusters.columns    
    cluster_cols = [x for x in cluster_cols if 'cluster' in x]    
    cluster_cols = [x.replace('_',' ') for x in cluster_cols]
    return df_months_clusters
    
def create_ticks_months_years(df_months_clusters):
    years = np.arange(df_months_clusters.index.year[0], df_months_clusters.index.year[-1] + 1, 1)
    months = ['J','F','M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    month_year = []
    for year in years:
        for month in months:
            month_year.append(month+'_'+str(year))
    start_month = df_months_clusters.index.month[0]
    month_year = month_year[start_month-1:] #starts from that month
    return month_year

def plot_all_month_and_clusters(df_month_years_clusters, nb_clusters, ymax=1,
                                ylabel='Frequency of clusters [-]', dict_cluster_to_colors=None,
                                days_to_subtract_xlim=None):
    fig, ax = plt.subplots(figsize=(25,8))
    
    df_month_years_clusters['duration'] = df_month_years_clusters.index.day

    X = list(df_month_years_clusters.index)

    cluster_cols = list(df_month_years_clusters.columns)
    cluster_cols = [x for x in cluster_cols if 'cluster' in x]
    if dict_cluster_to_colors is None:        
        colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
        dict_cluster_to_colors = dict(zip(cluster_cols, colors))

    print(dict_cluster_to_colors)
    bottom_append = [0]*len(df_month_years_clusters)    
    
    for count, col in enumerate(cluster_cols):
    
        if dict_cluster_to_colors is not None:      
            cluster_digit = int(get_digits(col))
        
        y = df_month_years_clusters[col].values        
        s=np.isnan(y)
        y[s]=0.0
        width = df_month_years_clusters['duration'].values
            
        plt.bar(X, y, width=width, bottom=bottom_append, color=dict_cluster_to_colors[cluster_digit], 
               align='center', edgecolor = 'white',
               label=str(col).replace('_',' '))
        
        bottom_append = bottom_append + y

    ax.set_title("",color='black')
    ax.legend(frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize=25)    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    

    thickax(ax)    
    ax.set_ylim(0,ymax)
           
    if days_to_subtract_xlim is not None:       
        d_min = min(X) - timedelta(days=days_to_subtract_xlim) 
        d_max = max(X) + timedelta(days=days_to_subtract_xlim) 
        ax.set_xlim(d_min,d_max)
    
    ax.tick_params(labelsize=25, axis='both', which='major')
    ax.tick_params(labelsize=25, axis='both', which='minor')    
    ax.set_ylabel(ylabel, fontsize=30) 
    ax.set_xlabel('', fontsize=20)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25) 
        tick.label.set_rotation('horizontal')   
  
    plt.show()
    return fig
    
def month_means_for_clusters(df_clustering, var='abs637'):
    cols = df_clustering.columns.to_list()
    xyz_cols = [x for x in cols if ('x_' in x) or ('y_' in x) or ('z_' in x)]
    df_clustering_without_xyz = df_clustering.drop(xyz_cols, axis=1)
    
    df_clustering_without_xyz["year_month"] = df_clustering_without_xyz["year"].astype(str) + '-' + df_clustering_without_xyz["month"].astype(str)
    dict_month_ordinal_to_year_month = dict(zip(df_clustering_without_xyz["month_ordinal"], df_clustering_without_xyz["year_month"]))
        
    df_month_mean = df_clustering_without_xyz[[var, 'month_ordinal']].groupby('month_ordinal').mean()

    start_index = df_month_mean.index[0]
    end_index = df_month_mean.index[-1]
    full_index = np.arange(start_index, end_index+1, 1)

    df_month_mean = df_month_mean.reindex(full_index, fill_value=np.nan)

    df_month_means = pd.DataFrame(columns=['cluster_1','cluster_2','cluster_3','cluster_4', 'cluster_5','mean'])
    for cluster in  df_clustering_without_xyz['clusters_5'].unique():
        print(cluster)
        df_cluster = df_clustering_without_xyz[df_clustering_without_xyz['clusters_5'] == cluster].copy()    
        df_month_clusters_mean = df_cluster[[var, 'month_ordinal']].groupby('month_ordinal').median()
        df_month_means['cluster_'+str(cluster)] = df_month_clusters_mean
        df_month_means = df_month_means.reindex(full_index, fill_value=0)
        
        
    df_month_means['mean'] = df_month_mean[var].values 
    df_month_means['month_ordinal'] = df_month_means.index
    df_month_means['year_month'] = df_month_means.index.to_series().map(dict_month_ordinal_to_year_month)    
    df_month_means = df_month_means.set_index('year_month') 
    df_month_means.index = pd.to_datetime(df_month_means.index) + MonthEnd(1)
    df_month_means = df_month_means.loc[df_month_means.index.notnull()]
    return df_month_means
    
def monthly_mean_weighted_by_occurance(df_clustering):
    cols = df_clustering.columns.to_list()
    xyz_cols = [x for x in cols if ('x_' in x) or ('y_' in x) or ('z_' in x)]
    df_clustering_without_xyz = df_clustering.drop(xyz_cols, axis=1)
    
    df_clustering_without_xyz["year_month"] = df_clustering_without_xyz["year"].astype(str) + '-' + df_clustering_without_xyz["month"].astype(str)
    dict_month_ordinal_to_year_month = dict(zip(df_clustering_without_xyz["month_ordinal"], df_clustering_without_xyz["year_month"]))

    df_month_sum = df_clustering_without_xyz[['abs637', 'month_ordinal']].groupby('month_ordinal').sum()
    df_month_count = df_clustering_without_xyz[['abs637', 'month_ordinal']].groupby('month_ordinal').count()
    df_month_mean = df_clustering_without_xyz[['abs637', 'month_ordinal']].groupby('month_ordinal').mean()

    start_index = df_month_sum.index[0]
    end_index = df_month_sum.index[-1]
    full_index = np.arange(start_index, end_index+1, 1)

    df_month_sum = df_month_sum.reindex(full_index, fill_value=np.nan)
    df_month_count = df_month_count.reindex(full_index, fill_value=np.nan)
    df_month_mean = df_month_mean.reindex(full_index, fill_value=np.nan)

    dict_cluster_to_weighted_amount = {}
    df_weighted_month_means = pd.DataFrame(columns=['cluster_1','cluster_2','cluster_3','cluster_4', 'cluster_5','mean'])
    for cluster in  df_clustering_without_xyz['clusters_5'].unique():
        df_cluster = df_clustering_without_xyz[df_clustering_without_xyz['clusters_5'] == cluster].copy()    
        df_month_clusters_value_counts = df_cluster[['clusters_5', 'month_ordinal']].groupby('month_ordinal').count()
        df_month_clusters_value_counts = df_month_clusters_value_counts.reindex(full_index, fill_value=0)

        df_month_cluster_weight = df_month_clusters_value_counts['clusters_5'].div(df_month_count['abs637'])
        df_month_cluster_weight = df_month_cluster_weight*df_month_mean['abs637']

        df_month_cluster_weight = df_month_cluster_weight.to_frame()

        df_month_cluster_weight['year_month'] = df_month_cluster_weight.index.to_series().map(dict_month_ordinal_to_year_month)    
        df_month_cluster_weight = df_month_cluster_weight.set_index('year_month')    

        dict_cluster_to_weighted_amount['cluster_'+str(cluster)] = df_month_cluster_weight 

        df_weighted_month_means['cluster_'+str(cluster)] = df_month_cluster_weight
    df_weighted_month_means['mean'] = df_month_mean['abs637'].values   
    df_weighted_month_means = df_weighted_month_means.loc[df_weighted_month_means.index.notnull()]
    df_weighted_month_means.index = pd.to_datetime(df_weighted_month_means.index) + MonthEnd(1)
    return df_weighted_month_means
    
def slice_df(df, start_datetime=None, end_datetime=None):
    if (start_datetime is not None) & (end_datetime is not None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index) & (df.index <= pd.to_datetime(end_datetime))]
    if (start_datetime is not None) & (end_datetime is None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index)]
    if (start_datetime is None) & (end_datetime is not None):
        df = df.loc[(df.index <= pd.to_datetime(end_datetime))]
    return df
    
def remove_spines(ax):
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 

def plot_all_month_and_clusters_unstacked_ax(df_weighted_month_means, cluster_num, nb_clusters=5, 
                                             ylabel='', ymax=1, fs_legend = 20, mscale = 15, start_datetime='2005-12-31',
                                             slope=True, freq=12, ax=None):
    if ax is None:
        print("ax is None")
        fig, ax = plt.subplots(figsize=(25,8))
    
    df_weighted_month_means['duration'] = df_weighted_month_means.index.day

    X = list(df_weighted_month_means.index)

    cluster_cols = list(df_weighted_month_means.columns)
    cluster_cols = [x for x in cluster_cols if 'cluster' in x]
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
    columns_and_colors = dict(zip(cluster_cols, colors))
    print(columns_and_colors)

    bottom_append = [0]*len(df_weighted_month_means)        
    cluster_cols = [cluster_num]    
    for count, col in enumerate(cluster_cols):
        y = df_weighted_month_means[col].values        
        s=np.isnan(y)
        y[s]=0.0
        width = df_weighted_month_means['duration'].values            
        plt.bar(X, y, width=width, bottom=bottom_append, color=columns_and_colors[col], 
               align='center', edgecolor = 'white') 
        
        ax.set_xticks([])
        ax.set_xticklabels([])
                
        ax.set_ylim(0,ymax[col])
        
        if slope == True:
            #df_weighted_month_means_slice = slice_df(df_weighted_month_means, start_datetime=start_datetime, end_datetime=None)        
            ordinal =  df_weighted_month_means['month_ordinal'].values             
            y = df_weighted_month_means[col].values        
            s=np.isnan(y)
            y[s]=0.0            
            #ordinal = ordinal - ordinal[0]
            ax2 = ax.twiny()
            ax2.set_xticks([])                       
            
            p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(ordinal, y, freq=freq, dp=3)            
            ax2.plot(ordinal, ordinal*m+p,'-',linewidth= 3,color=columns_and_colors[col],
                        label=str(col).replace('_',' ')+' - LMS: y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f))      
            #THEIL SEN SLOPE   
            res = stats.theilslopes(y, ordinal, 0.90)
            Theil_slope = (res[1] + res[0] * ordinal)
            lo_slope = (res[1] + res[2] * ordinal)
            up_slope = (res[1] + res[3] * ordinal)

            theil_m=res[0]
            theil_m = float(theil_m)*(freq)
            lo_m = float(res[2])*(freq)
            up_m = float(res[3])*(freq)
            
            #significant figures
            sfs = sf(3)           
            intecept=sfs.format(res[1])
            theil_m=sfs.format(theil_m)
            lo_m=sfs.format(lo_m)
            up_m=sfs.format(up_m) 

            ax2.plot(ordinal, Theil_slope, ls='--', lw=1, c=columns_and_colors[col],
                    label=str('TS: y = '+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept)))
            ax2.fill_between(ordinal, up_slope, lo_slope, alpha=0.15, color=columns_and_colors[col])                        
            ax2.legend(loc='upper right', bbox_to_anchor=(1.01, 1.01), fontsize=fs_legend, markerscale=mscale, 
            frameon=False, ncol=4)
            remove_spines(ax2)

    ax.set_title("",color='black')
    #ax.legend(frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize=25)    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    

    thickax(ax)  

    ax.tick_params(labelsize=25, axis='both', which='major')
    ax.tick_params(labelsize=25, axis='both', which='minor')    
    ax.set_ylabel(ylabel, fontsize=30) 
    ax.set_xlabel('', fontsize=20)
    remove_spines(ax)
    
    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25) 
        tick.label.set_rotation('horizontal')   
  
    if ax is None:
        return fig
    if ax is not None:
        return ax
        
def plot_month_mean_subplots(df_month_means, ymax=3, nrows = 5, ncols = 1, nb_clusters = 5):
    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)

    for cluster in np.arange(1, nb_clusters+1, 1):
        ax = plt.subplot(nrows, ncols, int(cluster))        
        plot_all_month_and_clusters_unstacked_ax(df_month_means, 'cluster_'+str(cluster), nb_clusters=5, ymax=ymax,
                                          ylabel='', ax=ax)
        if cluster <= 4:
            ax.set_xticklabels([])
                                          
    fig.text(0.5, 0.04, '', ha='center')
    fig.text(0.06, 0.5, 'Monthly mean $\sigma_{\mathrm{ap}}$ by cluster [Mm$^{-1}$]', va='center', rotation='vertical', 
            fontsize=25)
    plt.show()
    return fig
    
def groupby_count_clusters(df_clustering, nb_clusters, groupby_var='month'):    
    df_clustering.index = pd.to_datetime(df_clustering.index)
    df_clustering['month'] = df_clustering.index.month
    df_months_clusters = pd.DataFrame([])
    for cluster in np.arange(1, nb_clusters+1, 1):
        print("cluster: "+str(cluster))
        count = df_clustering[df_clustering['clusters_'+str(nb_clusters)] == cluster].groupby(groupby_var)['clusters_'+str(nb_clusters)].count()
        #sum_count = count.sum(axis)
        #print(sum_count)
        df_months_clusters['cluster_'+str(cluster)] = count

    cluster_cols = df_months_clusters.columns
    #print(cluster_cols)
    df_months_clusters['month_sum'] = df_months_clusters.sum(axis=1).values
    #print(df_months_clusters['month_sum'])
    df_months_clusters[cluster_cols] = df_months_clusters[cluster_cols].div(df_months_clusters['month_sum'].values,axis=0)
    return df_months_clusters
    
def plot_annual_clusters(df_months_clusters, nb_clusters, dict_cluster_to_colors):
    df_months_clusters['index'] = df_months_clusters.index

    cluster_cols = df_months_clusters.columns    
    cluster_cols = [x for x in cluster_cols if 'cluster' in x]
    print(cluster_cols)
    
    cluster_cols = [x.replace('_',' ')[:9] for x in cluster_cols]
    df_months_clusters.columns = cluster_cols + ['month_sum', 'check_percentage_sum', 'index']
    
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))   
    print(colors)
    #columns_and_colors = zip(cluster_cols, colors)
    colors = [*dict_cluster_to_colors.values()]
    ax = df_months_clusters.plot(x='index', y=cluster_cols, kind='bar', stacked=True, figsize=(10,7), 
                                color=colors)
    
    ax.set_title("",color='black')
    ax.legend(['North Atlantic', 'Greenland', 'Arctic Ocean', 'Siberia', 'Eurasia'],
               frameon=False, bbox_to_anchor=(1.0, 1.0), fontsize=20)
    
    months = df_months_clusters.index
    months_letters = ['J','F','M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    months_integers = np.arange(1,13, 1)

    dict_months_to_letter = dict(zip(months_integers, months_letters))
    print(dict_months_to_letter)
    letters = [dict_months_to_letter[x] for x in months]
    
    ax.set_xticklabels(letters)
    
    thickax(ax)
    ax.set_ylim(0,1)
    ax.tick_params(labelsize=25, axis='both', which='major')
    ax.tick_params(labelsize=25, axis='both', which='minor')
    
    ax.set_xlabel('Months', fontsize=30)

    for tick in ax.xaxis.get_major_ticks():
        tick.label.set_fontsize(25) 
        tick.label.set_rotation('horizontal')

    ax.set_ylabel('Frequency of clusters [-]', fontsize=30)  
    plt.show()
    fig = ax.get_figure()
    return fig
    
def columns_and_clusters(nb_clusters=5):
    clusters = list(np.arange(1, 6, 1))
    print(clusters)    
    cluster_cols = ['cluster_'+str(x) for x in clusters]    
    print(cluster_cols)
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
    dict_clusters_to_colors = dict(zip(cluster_cols, colors))   
    return dict_clusters_to_colors

def timeseries_clusters(df_obs_clusters, var, nb_clusters=5, ms=2,
                        ylabel='eBC [ngm$^{-3}$]', clusters=[1,2,3,4,5],
                        alpha=.5, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(25,6))
    
    dict_clusters_to_colors = columns_and_clusters(nb_clusters)
    print(dict_clusters_to_colors)

    ax.plot(df_obs_clusters.index, df_obs_clusters[var], 'o', c='k',
    alpha=alpha)
        

    for cluster in clusters:
        df_cluster = df_obs_clusters[df_obs_clusters['clusters_'+str(nb_clusters)] == cluster] 
        median = df_cluster['abs637'].median()
        ax.plot(df_cluster.index, df_cluster[var], 'o',  c=dict_clusters_to_colors['cluster_'+str(cluster)],
                ms=ms, label='Cluster '+str(cluster)+': $\sigma_{\mathrm{ap, med.}}$ = '+str(median)+' [Mm$^{-1}$]')

    thickax(ax)
    
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=30) 


    ax.tick_params(axis='both', which='major', labelsize=25)
    ax.tick_params(axis='both', which='minor', labelsize=25)
    plt.legend(frameon=False, fontsize=25, loc=1)
    
    if ax is None:
        plt.show()
        return fig
    if ax is not None:
        return ax
    
def add_month_ordinal(df):
    df['month_ordinal'] = df.index.month + 12*(df.index.year - df.index.year[0])
    return df

def vars_for_best_fit(x,y,freq,dp):
    x = np.array(x)
    x = sm.add_constant(x)
    model = sm.OLS(y, x, missing='drop')
    results = model.fit()
    p = results.params[0]
    m = results.params[1]    
    se = results.bse[1]      
    m_year = m*freq    
    dp = '{0:.'+str(dp)+'f}'    
    m3f = dp.format(m*(freq))
    p3f = dp.format(p) 
    m_year3f = dp.format(m_year)   
    return p,m,m3f,p3f,m_year,m_year3f

def add_axis_options(ax,ymax):
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.set_title('')
    ax.set_ylabel(variable_to_yaxis[variable],fontsize=fs_label)
    ax.set_xlim(first_season-2, last_season+2)
    ticks=np.arange(1,74,8)   
    ax.set_ylim(0, ymax)    
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[::2])
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks)
    
def create_trend_plot(df_month_years_clusters, nb_clusters=5,
                      fs_ticks = 20, fs_legend = 20, mscale = 15):
    
    df_month_years_clusters = add_month_ordinal(df_month_years_clusters)
    df_month_years_clusters.loc[df_month_years_clusters.month_sum == 0] = np.nan
    
    fig, ax = plt.subplots(figsize=(25,8))

    start_year = int(df_month_years_clusters.index.year[0])
    end_year = int(df_month_years_clusters.index.year[-1])
    year_labels = [str(x) for x in range(start_year,end_year+2)]    
    cols = df_month_years_clusters.columns
    cluster_cols = [x for x in cols if 'cluster' in x]    
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))

    for cluster_col in cluster_cols:        
        cluster = int(cluster_col[-1])        
        months = df_month_years_clusters['month_ordinal'].values
        cluster_freq = df_month_years_clusters[cluster_col].values
        ax.plot(months, cluster_freq,'o')
        p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(months, cluster_freq,freq=12, dp=3)
        ax.plot(months, months*m+p,'-',linewidth= 3,color=np.array([colors[cluster-1]]),
                    label=str(cluster_col).replace('_',' ')+': y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f))
        ax.legend(loc='upper right',bbox_to_anchor=(1.01, 1.25), fontsize=fs_legend,markerscale=mscale, frameon=False, ncol=4)
        
    ax.set_ylim(0,0.35)
    start_tick = df_month_years_clusters['month_ordinal'].values[0]
    end_tick = df_month_years_clusters['month_ordinal'].values[-1]

    thickax(ax)
    ticks=np.arange(1,(end_year-start_year+2)*12,12)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[:])
    #ax.grid(True)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks)
    ax.set_ylabel('Frequency of clusters [-]', fontsize=25)   
    
    year_labels = [str(x) for x in range(2002,2021)]    
    ticks=np.arange(1,(2019-2002+2)*12,12)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[:])

    ax.set_ylim(0,0.35)
    ax.tick_params(labelsize=25, axis='both', which='major')
    ax.tick_params(labelsize=25, axis='both', which='minor')    
    ax.set_ylabel('Frequency of clusters [-]', fontsize=25)  
    ax.set_xlim(0,213) 
    plt.show()
    return fig
    
def remove_spines(ax):
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    
def get_digits(string_with_digits):
    digit = re.findall(r'\d+', string_with_digits)[0]
    return digit
    
def trend_cluster(df_clustering_count, freq=3, nb_clusters=5,
                  fs_legend=15, fs_ticks=20, mscale=15, ncol=5,
                  ylabel='Cluster contribution [-]', fs_label=15, 
                  dict_cluster_to_colors=None):
                  
    fig, ax = plt.subplots(figsize=(15,4))
    cols = df_clustering_count.columns
        
    cluster_cols = [x for x in cols if 'cluster' in x] 
    print(cluster_cols)     
    
    if dict_cluster_to_colors is None:
        colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))
        dict_cluster_to_colors = dict(zip(cluster_cols, colors))
    
    print(dict_cluster_to_colors)
    
    for cluster in cluster_cols:
        ax.plot(df_clustering_count['mid_datetime'], 
                df_clustering_count[cluster], 'o')
    remove_spines(ax)
    ax2 = ax.twiny()
    ax2.set_xticks([]) #dont want these ticks
    remove_spines(ax2)
    for cluster in cluster_cols:
        
        if dict_cluster_to_colors is not None:    
            cluster_digit = int(get_digits(cluster))
        
        ax2.plot(df_clustering_count.index, df_clustering_count[cluster], 
                'o-', c=dict_cluster_to_colors[cluster_digit])
        time = df_clustering_count.index
        cluster_freq = df_clustering_count[cluster].values    
        ax2.plot(time, cluster_freq,'o',color=dict_cluster_to_colors[cluster_digit])
        p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(time, cluster_freq,freq=freq, dp=3)
        ax2.plot(time, time*m+p,'-',linewidth= 3,color=dict_cluster_to_colors[cluster_digit],
                    label=str(cluster).replace('_',' ')+': y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f))
        ax2.legend(loc=1,bbox_to_anchor=(1.4, 1.), fontsize=fs_legend,markerscale=mscale, frameon=False, 
        ncol=ncol)
    ax.set_ylabel(ylabel, fontsize=fs_label)
    thickax(ax)
    plt.ylim(0,0.65)
    plt.show()
    return fig
    
def month_timeseries_for_cluster(df_month_years_clusters, cluster_col='cluster_3', 
                                 nb_clusters=5, fs_ticks=15):
    fig, ax = plt.subplots(figsize=(25,8))

    start_year = int(df_month_years_clusters.index.year[0])
    end_year = int(df_month_years_clusters.index.year[-1])
    year_labels = [str(x) for x in range(start_year,end_year+2)]    
    cols = df_month_years_clusters.columns
    cluster_cols = [x for x in cols if 'cluster' in x]    
    colors=cm.rainbow(np.linspace(0,1,int(nb_clusters)))

    cluster = int(cluster_col[-1])        
    months = df_month_years_clusters['month_ordinal'].values
    cluster_freq = df_month_years_clusters[cluster_col].values
    ax.plot(months, cluster_freq,'o-')

    start_tick = df_month_years_clusters['month_ordinal'].values[0]
    end_tick = df_month_years_clusters['month_ordinal'].values[-1]

    #thickax(ax)
    ticks=np.arange(1,(end_year-start_year+2)*12,12)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[:])
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks)
    ax.set_ylabel('Frequency of clusters [-]', fontsize=25)   

    year_labels = [str(x) for x in range(2002,2021)]    
    ticks=np.arange(1,(2019-2002+2)*12,12)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[:])

    plt.show()
    return fig
    
####SELECTING CLUSTERS#####################################################################################################

def calculate_mode_cluster_for_ensembles(df_clusters, groupby_var='arrival_time', cluster_col='clusters_5'):
    df_clusters.index.name = 'arrival_time'
    df_clusters = df_clusters.reset_index()   
    clusters_mode = df_clusters.groupby(groupby_var)[cluster_col].agg(pd.Series.mode)    
    df_clusters_mode = clusters_mode.to_frame()
    return df_clusters_mode
    
def count_value_find_percentage(df_clusters, nb_clusters=5):
    df_clusters.index.name = 'arrival_time'
    df_clusters = df_clusters.reset_index()    
    df_groupby = df_clusters.groupby('arrival_time')['clusters_'+str(nb_clusters)].value_counts()
    df_groupby = df_groupby.to_frame()
    df_groupby = df_groupby.rename(columns={'clusters_'+str(nb_clusters):'count'})
    df_groupby = df_groupby.reset_index()
    df_groupby = df_groupby.set_index('arrival_time')
    df_groupby['count'] = df_groupby['count']/27
    return df_groupby
    
def plot_distribution_of_cluster_counts(df_value_count):
    df_max_proportion = df_value_count.groupby('arrival_time').agg({'count':'max'})
    x = df_max_proportion.values
    for majority_per in np.arange(.5, 1., .1):
        percentage = len(x[x > majority_per])/len(x)
    fig = sns.displot(x, alpha=.5, legend=False, height=5)
    plt.axvline(x=0.5, c='r', alpha=.5)
    
    plt.xlabel('Max fraction of the contributing clusters')
    plt.show()
    return fig
    
def find_the_max_count(df_value_count, percentage_threshold=0.5):
    idx = df_value_count.groupby(['arrival_time'])['count'].transform(max) == df_value_count['count']
    df_max_count = df_value_count[idx]
    df_max_count = df_max_count[df_max_count['count'] >= percentage_threshold]
    return df_max_count

##VIEW COLOURBAR##########################################################################################################

def create_dict_cluster_to_colours(nb_clusters = 5, cmap='Spectral'):
    clusters = np.arange(1, nb_clusters+1, 1) #sorted(df_clusters['clusters_'+str(nb_clusters)].unique())
    
    colors=plt.get_cmap(cmap)(np.linspace(0,1,int(len(clusters))))
    dict_cluster_to_colors = dict(zip(clusters, colors))
    print(dict_cluster_to_colors)
    return dict_cluster_to_colors
    
def grayscale_cmap(cmap):
    """Return a grayscale version of the given colormap"""
    cmap = plt.cm.get_cmap(cmap)
    colors = cmap(np.arange(cmap.N))
    
    # convert RGBA to perceived grayscale luminance
    # cf. http://alienryderflex.com/hsp.html
    RGB_weight = [0.299, 0.587, 0.114]
    luminance = np.sqrt(np.dot(colors[:, :3] ** 2, RGB_weight))
    colors[:, :3] = luminance[:, np.newaxis]
        
    return LinearSegmentedColormap.from_list(cmap.name + "_gray", colors, cmap.N)

def view_colormap(cmap, number_of_colours=None):
    """Plot a colormap with its grayscale equivalent"""
    if number_of_colours is None:
        cmap = plt.cm.get_cmap(cmap)
        print(cmap)
        colors = cmap(np.arange(cmap.N))    
    if number_of_colours is not None:
        #cmap = plt.cm.get_cmap(cmap)
        colors=plt.get_cmap(cmap)(np.linspace(0,1, int(number_of_colours)))
    
    cmap = grayscale_cmap(cmap)
    grayscale = cmap(np.arange(cmap.N))
    
    if number_of_colours is None:
        fig, ax = plt.subplots(2, figsize=(6, 2),
                               subplot_kw=dict(xticks=[], yticks=[]))    
        ax[0].imshow([colors], extent=[0, 10, 0, 1])    
        ax[1].imshow([grayscale], extent=[0, 10, 0, 1])
        plt.show()   
        
    if number_of_colours is not None:
        fig, ax = plt.subplots(1, figsize=(6, 2),)    
        ax.imshow([colors], extent=[0, number_of_colours*2, 0, 1]) 
        ax.set_yticklabels([])
        ax.set_xticks(np.arange(1, number_of_colours*2+1, 2))
        ax.set_xticklabels(np.arange(1, number_of_colours+1, 1))
        plt.show()   
        
######FREQUENCY VS SOURCE REDUCTIONS#################################################################################################################

def create_df_abs_year_mean_clusters(df_max_count_abs637, mean=True, median=False):
    df_abs_year_mean_clusters = pd.DataFrame([])
    for cluster in [1,2,3,4,5]:
        if mean == True:
            annual_abs_for_cluster = df_max_count_abs637.loc[(df_max_count_abs637['clusters_5'] == cluster), 'abs637'].resample('Y').mean()
            df_abs_year_mean_clusters['cluster_'+str(cluster)+'_mean'] = annual_abs_for_cluster 
        if median == True:
            annual_abs_for_cluster = df_max_count_abs637.loc[(df_max_count_abs637['clusters_5'] == cluster), 'abs637'].resample('Y').median()
            df_abs_year_mean_clusters['cluster_'+str(cluster)+'_mean'] = annual_abs_for_cluster
    cluster_cols = df_abs_year_mean_clusters.columns
    annual_abs = df_max_count_abs637['abs637'].resample('Y').mean()
    df_abs_year_mean_clusters['abs637_mean'] = annual_abs
    df_abs_year_mean_clusters['abs637_cluster_sum'] = df_abs_year_mean_clusters.sum(axis=1).values
    return df_abs_year_mean_clusters
    
def groupby_count_clusters(df_clustering, nb_clusters, groupby_var='month'):    
    df_clustering.index = pd.to_datetime(df_clustering.index)    
    if groupby_var == 'year':
        df_clustering['year'] = df_clustering.index.year
    if groupby_var == 'month':
        df_clustering['month'] = df_clustering.index.month    
    df_time_clusters = pd.DataFrame([])    
    for cluster in np.arange(1, nb_clusters+1, 1):
        print("cluster: "+str(cluster))
        count = df_clustering[df_clustering['clusters_'+str(nb_clusters)] == cluster].groupby(groupby_var)['clusters_'+str(nb_clusters)].count()
        df_time_clusters['cluster_'+str(cluster)] = count
    cluster_cols = df_time_clusters.columns
    df_time_clusters[str(groupby_var)+'_sum'] = df_time_clusters.sum(axis=1).values
    df_time_clusters[cluster_cols] = df_time_clusters[cluster_cols].div(df_time_clusters[str(groupby_var)+'_sum'].values,axis=0)
   
    cluster_cols = ['cluster_1', 'cluster_2', 'cluster_3', 'cluster_4', 'cluster_5']
    df_time_clusters['check_percentage_sum'] = df_time_clusters[cluster_cols].sum(axis=1)  
    for cluster_col in cluster_cols:
        df_time_clusters = df_time_clusters.rename(columns={cluster_col:cluster_col+'_freq'})
    return df_time_clusters
    
def calculate_weighted_means(df_year_clusters, df_max_count_abs637, mean=True, median=False):
    cluster_cols = ['cluster_1_freq', 'cluster_2_freq', 'cluster_3_freq', 'cluster_4_freq', 'cluster_5_freq']
    if mean == True:
        annual_abs = df_max_count_abs637['abs637'].resample('Y').mean().values
    if median == True:
        annual_abs = df_max_count_abs637['abs637'].resample('Y').median().values
    df_year_clusters['abs637_annual_mean'] = annual_abs
    print(df_year_clusters[cluster_cols])
    for cluster_col in cluster_cols:
        df_year_clusters[cluster_col+'_weighted'] = df_year_clusters[cluster_col]*df_year_clusters['abs637_annual_mean']
    return df_year_clusters
    
def get_first_3_years(df_abs_year_mean_clusters):
    dict_first_3_years = {}
    for cluster in [1,2,3,4,5]:
        first_3_year_mean = df_abs_year_mean_clusters['cluster_'+str(cluster)+'_mean'].iloc[:3].mean()
        dict_first_3_years[cluster] = first_3_year_mean
    print(dict_first_3_years)
    return dict_first_3_years
    
def calculate_weight_fixed(df_year_clusters, dict_first_3_years):
    cluster_cols = ['cluster_1_freq', 'cluster_2_freq', 'cluster_3_freq', 'cluster_4_freq', 'cluster_5_freq']
    for cluster_col in cluster_cols:
        print(cluster_col)
        cluster = get_digits(cluster_col)
        print(cluster)
        first_3_year_mean = dict_first_3_years[int(cluster)] 
        df_year_clusters[cluster_col+'_weighted_fixed'] = df_year_clusters[cluster_col]*first_3_year_mean    
    weighted_fixed_clusters = ['cluster_1_freq_weighted_fixed','cluster_2_freq_weighted_fixed', 
                                'cluster_3_freq_weighted_fixed', 'cluster_4_freq_weighted_fixed', 
                                'cluster_5_freq_weighted_fixed']    
    df_year_clusters['fixed_weighted_mean'] = df_year_clusters[weighted_fixed_clusters].sum(axis=1)
    return df_year_clusters
    
def weighted_fixed_trend_subplots(df_year_clusters, ylabel = '$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]',
                                  dict_cluster_to_colors=None, dict_cluster_to_name=None, dp=3):
    fig, (ax2, ax1) = plt.subplots(2,1, figsize=(10*2,6*2), sharex=True,)

    x = np.array(df_year_clusters.index)
    print(x)
    ordinal = np.array([int(i - x[0] + 1) for i in x])

    bottom_append = [0]*len(x)
    for cluster in [1,2,3,4,5]:   
        y_abs = df_year_clusters['cluster_'+str(cluster)+'_freq_weighted'].values
        ax1.bar(ordinal, y_abs, width=1, bottom=bottom_append, color=dict_cluster_to_colors[cluster], 
                   align='center', edgecolor = 'k',
                   label=str(cluster).replace('_',' '))  
        bottom_append = bottom_append + y_abs

        #frequency 
        y_freq = df_year_clusters['cluster_'+str(cluster)+'_freq'].values
        p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(ordinal, y_freq, freq=1, dp=dp)   
        
        name = ''
        if dict_cluster_to_name is not None:
            name = dict_cluster_to_name[cluster] 
        
        ax2.plot(ordinal, ordinal*m+p,'-',linewidth= 3,color=dict_cluster_to_colors[cluster],
                 label=str(cluster).replace('_',' ')+': '+str(name)+': y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f)) 
                 
        ax2.plot(ordinal, y_freq, 'o', c=dict_cluster_to_colors[cluster], mfc='none', ls='-', ms=10) 
        ax1.set_xticks([])
        ax2.set_xticks([])
        ax1.set_xticklabels([])
        ax2.set_xticklabels([])

    ax2.legend(frameon=False, loc=1, ncol=2, fontsize=15, bbox_to_anchor=(1.,1.1))
    y_sum = df_year_clusters['abs637_annual_mean'].values
    ax1.plot(ordinal, y_sum, 'o-', c='k')
    y_sum = df_year_clusters['fixed_weighted_mean'].values
    
    ax1.plot(ordinal, y_sum, '^-', c='k')
    ax1.text(-0.1, 0.6, 'b)', fontsize=25)

    #trends
    ax11 = ax1.twiny()
    ax11.set_xticks([])   
    y_fixed = df_year_clusters['fixed_weighted_mean'].values
    
    p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(ordinal, y_fixed, freq=1, dp=dp)   
    ax11.plot(ordinal, ordinal*m+p,':',linewidth= 3,color='k', 
                 label='Fixed $\sigma_{\mathrm{ap}}$: y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f))
    y_mean = df_year_clusters['abs637_annual_mean'].values
    p,m,m3f,p3f,m_year,m_year3f = vars_for_best_fit(ordinal, y_mean, freq=1, dp=dp)   
    ax11.plot(ordinal, ordinal*m+p,'-',linewidth= 3,color='k',
             label='Annual mean: y = '+str(m_year3f)+'$\,\mathdefault{x}$ +'+str(p3f))
    ax11.legend(frameon=False , loc=1, fontsize=15)

    for ax in [ax1, ax2, ax11]:
        ax.tick_params(labelsize=16, axis='both', which='major')
        ax.tick_params(labelsize=16, axis='both', which='minor')        
        ax.set_xlabel('', fontsize=20)
        thickax(ax) 
        remove_spines(ax)
        
    ax1.set_ylabel('$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]', fontsize=30) 
    ax2.set_ylabel('Normalised \n occurrence [-]', fontsize=30) 
    ax2.set_xticks(np.arange(1, len(x)+1, 2))
    ax2.set_xticklabels(x[::2], fontsize=20)
    
    ax2.text(-0.1, 0.5, 'a)', fontsize=25)
    fancy(ax1)
    fancy(ax2)
    
    #plt.rc('axes', linewidth=20,)
    
    plt.show()
    return fig