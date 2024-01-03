import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

lambda_MAAP = 637
PSAP_calibration_wavelength = 574

dict_abs_labels = {'absorption':'$\sigma_{\mathrm{ap}}$ (MAAP, '+str(lambda_MAAP)+' nm) (Mm$^{-1}$)',
                   'abs_neph':'$\sigma_{\mathrm{ap}}$ New PSAP (Heslin-Rees & Bond) [Mm$^{-1}$]', 
                   'Abs New m-1':'$\sigma_{\mathrm{ap}}$ New PSAP (Tunved & Bond) [Mm$^{-1}$]', 
                   'Abs Old m-1':'$\sigma_{\mathrm{ap}}$ Old PSAP (Tunved & Bond) [Mm$^{-1}$]',
                   'converted_abs_neph_to_574':'$\sigma_{\mathrm{ap}}$ Bond (converted to '+str(PSAP_calibration_wavelength)+' nm) (Mm$^{-1}$)',
                   'abs_new':'$\sigma_{\mathrm{ap}}$ (Tunved & Bond) (Mm$^{-1}$)', 
                   'converted_converted_abs_neph_to_574_to_637':'$\sigma_{\mathrm{ap}}$ Bond (converted to '+str(lambda_MAAP)+' nm) (Mm$^{-1}$)',
                   'converted_abs_neph_to_637':'$\sigma_{\mathrm{ap}}$ Bond (converted to '+str(lambda_MAAP)+' nm) (Mm$^{-1}$)',
                   'abs_neph_virkkula':'$\sigma_{\mathrm{ap}}$ Virkkula (Mm$^{-1}$)',
                   'converted_abs_neph_virkkula_to_637':'$\sigma_{\mathrm{ap}}$ Virkkula (converted to '+str(lambda_MAAP)+' nm) (Mm$^{-1}$)',
                   'Tr':'Transmission',
                   'Tr_x':'Transmission virkkula',
                   'Tr_y':'Transmisson Bond',
                   'Tr_old':'Transmission Old PSAP',
                   'Tr_new':'Transmisson New PSAP',
                   'Difference':'Relative Diff.',
                   'abs_new':'$\sigma_{\mathrm{ap}}$ New PSAP (Bond) [Mm$^{-1}$]',
                   'abs_old':'$\sigma_{\mathrm{ap}}$ Old PSAP (Bond) [Mm$^{-1}$]'}
                   
def load_df(loadpath, extrapath=None, filename=None, formatdata=".dat", index_col=0):
    if extrapath is not None:
        print("loading: "+str(loadpath+'\\'+extrapath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+extrapath+'\\'+filename+formatdata, index_col=index_col, parse_dates=True,
                         low_memory=False)
    if extrapath is None:
        print("loading: "+str(loadpath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+filename+formatdata, index_col=index_col, parse_dates=True,
                         low_memory=False)        
    return df
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg"):
    folders = glob.glob(path)
    if folder not in folders:
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight')
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
                   
def print_packages():
    print('\n'.join(f'{m.__name__}=={m.__version__}' for m in globals().values() if getattr(m, '__version__', None)))
                   
def save_df(df, name, path, float_format=None):
    print("Save to: "+str(path))
    df.to_csv(path+'\\'+name+'.dat',index=True, float_format=float_format)
    
def append_soot_raw(path, years, col_names, name):
    appended_data = []   
    for year in years:      
        print("year: "+str(year))
        list_RAW_PSAP_files = glob.glob(path+str(year)+'\\'+str(name)+' *')
        print('length: '+str(len(list_RAW_PSAP_files)))
        for infile in glob.glob(path+str(year)+'\\'+str(name)+' *'):
            rows_skip = []            
            with open(infile) as f:
                lines = f.readlines()
                for value, line in enumerate(lines):
                    if (len(line) <= 36) or (len(line) >= len(col_names)*11) or (line[0] == 'D') or (line[0] == np.nan):
                        rows_skip.append(value)
            try:
                df = pd.read_csv(infile, sep='\t', header=None, skiprows = rows_skip)                
                if len(df.columns) == len(col_names):                
                    df.columns = col_names
                    df["Time"] = df["Time"].astype(float)
                    df["Time"] = df["Time"].apply(lambda x: round(x,3))            
                    df["Time"] = df["Time"].astype(int)            
                    df["Time"] = df["Time"].astype(str)
                    df["Time"] = df.apply(lambda row : ((6-len(str(row['Time'])))*'0'+str(row['Time'])), axis = 1) 
                    df["Date"] = df["Date"].astype(str)            
                    df["DateTime"] = pd.to_datetime(df['Date'] + df['Time'])            
                    df["DateTime"] = pd.to_datetime(df.loc[:,"DateTime"], format='%Y%m%d%H%M%S.%fff')
                    df = df.set_index('DateTime')                                   
                    df = df.drop(['Date','Time'], axis=1)                    
                else:                    
                    df = pd.DataFrame(columns=col_names[2:], index=range(1))
            except pd.errors.EmptyDataError:                
                pass   
            appended_data.append(df)
    RAW_PSAP = pd.concat(appended_data) 
    return RAW_PSAP
    
def concat_two_dfs(df1,df2):
    frames = [df1, df2]
    df = pd.concat(frames)
    return df
    
def remove_meaningless_values(df):    
    df.dropna(subset=['Io', 'I', 'qobs'], how='all', inplace=True)
    df.index = pd.to_datetime(df.index)
    df.loc[df['Io'].values <= 0] = np.nan #do not accept values of Io less than 0 = meaningless
    df.loc[df['I'].values <= 0] = np.nan
    #df.loc[df['Io'].values < df['I'].values] = np.nan #when I is above Io = shouldn't be the case
    df.loc[df['qobs'].values <= 0] = np.nan
    df.dropna(subset=['Io', 'I', 'qobs'], how='all', inplace=True)
    df = df.sort_index()
    return df
    
def add_jumps(df):   
    df['diff_I'] = df['I'].diff().abs() 
    df['diff_Io'] = df['Io'].diff().abs() 
    print(df['diff_I'].mean())
    print(df['diff_Io'].mean())
    return df
    
def fancy(ax):    
    # thickning the axes spines
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2)
        ax.spines[axis].set_color('k')        
    # set the fontsize for all your ticks
    fontsize = 20
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)        
    # properties of the ticks
    ax.tick_params(direction='out', length=8, width=2, pad=10, bottom=True, top=False, left=True, right=False, color='k')    
    # add a grid to the plot
    ax.grid(True, alpha=0.5)    
    # mask top and right spines
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)  
      
def filter_plot(df, dt_min, dt_max):    
    fig, ax = plt.subplots(figsize=(30,8))    
    df = df[(df.index > dt_min) & (df.index < dt_max)]
    Lnr_minus = df[df['Lnr'] < -0.1] #some arbitrary value 
    plt.plot(Lnr_minus.index, Lnr_minus['Lnr'], 'o', c='red', label='Filter changes')     
    plt.plot(df.index, df['Lnr'], 'o', label='Lnr')    
    plt.plot(df.index, df['I'], 'o', lw=0.5, label='I')
    plt.plot(df.index, df['Io'], 'o', lw=0.5, label='Io')    
    times_of_filter_changes = df[df['Lnr'] < -0.1].index #mulitple points    
    for idx in times_of_filter_changes:
        plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
    plt.title(str(dt_min[:4]), fontsize=20)        
    plt.legend(fontsize=15, loc=7)
    fancy(ax)
    plt.show()
    return times_of_filter_changes
    
def find_time_difference(df):
    df['deltaT_forward'] = df.index.to_series().diff().dt.seconds.div(60, fill_value=0)
    df['deltaT_behind'] = df['deltaT_forward'].shift(-1)
    print(str(df['deltaT_forward'].mean())+' minutes')
    return df
    
def filter_plot_extralines(df, dt_min, dt_max):   
    year = pd.to_datetime(dt_min).year
    
    fig, ax = plt.subplots(figsize=(30,8))    
    df = df[(df.index > dt_min) & (df.index < dt_max)]
    Lnr_minus = df[df['Lnr'] < -0.1] #some arbitrary value 
    plt.plot(Lnr_minus.index, Lnr_minus['Lnr'], 'o', c='red', label='Filter changes')     
    plt.plot(df.index, df['Lnr'], 'o', label='Lnr')    
    plt.plot(df.index, df['I'], 'o', lw=0.5, label='I')
    plt.plot(df.index, df['Io'], 'o', lw=0.5, label='Io')    
    times_of_filter_changes = df[df['Lnr'] < -0.1].index #mulitple points
    plt.title(str(year)+" Filter lines (black): "+str(len(times_of_filter_changes)), fontsize=20)    
    for idx in times_of_filter_changes:
        plt.axvline(idx, linewidth=4, color='black', lw=1)        
    for idx in extra_filter_lines:
        if idx.year == year:
            plt.axvline(idx, linewidth=4, color='blue', lw=1)    
    plt.legend(fontsize=20, loc=1)    
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.tick_params(axis='both', which='minor', labelsize=12)
    plt.show()
    return times_of_filter_changes
    
def filter_change_indexs(df):   
    df.loc[df['deltaT_forward'] >= 10, 'Filter_change'] = 1 #more than 10 minutes difference assume a change of filter
    df.loc[df['deltaT_behind'] >= 10, 'Filter_change'] = 1 #more than 10 minutes
    df.loc[df['diff_I'] > 0.05, 'Filter_change'] = 1   #difference greater than 0.1
    df.loc[df['diff_Io'] > 0.05, 'Filter_change'] = 1   #difference greater than 0.1
    df.loc[df['Lnr'] < -0.1, 'Filter_change'] = 1
    filter_change_time = df['Filter_change'].eq(1) #find values equal to 1
    df['FChanged_cum'] = filter_change_time.ne(filter_change_time.shift()).cumsum()
    indexs = [x.index[0] for i, x in df.groupby('FChanged_cum')]  
    print("Maximum number of filters: "+str(df['FChanged_cum'].max()))
    return indexs
    
def create_dfs_for_filters(df):    
    dfs = [x.iloc[1:,:] for i, x in df.groupby('Filter_cum')] #remove initail data point
    return dfs
       
def remove_ends(df):
    """cuts the filter to remove potential noisy end"""
    length_before = len(df) 
    if length_before > 0:
        try:
            cut_index = df[df.Io.diff().abs() > 0.05].index[0]
        except IndexError:
            cut_index = df.index[-1]        
        df = df[df.index < cut_index]
    else:
        df = pd.DataFrame()
    return df
 
def minimum_size_of_dfs(dfs, minimum_length=400):
    """requires the filters to be of a minimum length"""
    dfs_minimum_size = []
    for df in dfs:
        length = len(df)  
        if length >= minimum_length:
            dfs_minimum_size.append(df)
        if length < minimum_length:
            pass
    return dfs_minimum_size 
       
def duplicates(df):
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")      
    print(len(duplicateRowsDF))        
    df_first_duplicate = df.loc[df.index.duplicated(keep='first')]
    df_last_duplicate = df.loc[df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first_duplicate)))
    print("Length after: "+str(len(df_last_duplicate)))
    return df_first_duplicate, df_last_duplicate
    
def plot_each_filter(df, i, ylabel='I/Io', ms=3):      
    fig, ax = plt.subplots(figsize=(20,5))    
    ax.plot(df.index, df['I'], 'o', c='b', ms=ms, label='I')
    ax.plot(df.index, df['Io'], 'o', c='k', ms=ms, label='Io')
    ax.set_ylim(-0.1, 10)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.legend(fontsize=20, frameon=False, loc=1)    
    ax2 = ax.twinx()
    ax2.plot(df.index, df['qobs'], 'o', c='g', ms=ms, label='qobs')
    ax2.set_ylim(-0.1, 10)
    ax2.set_ylabel('Q$_{obs}$', fontsize=20)
    ax2.legend(fontsize=20, loc=2, frameon=False)    
    plt.title("Filter: "+str(i)+" start: "+str(df.index[0].date())+' - '+str(df.index[-1].date())+' '+str(len(df)), fontsize=20)       
    plt.grid(True)
    #plt.savefig(out_path_plots+'filters\\'+str(i)+'.jpeg')
    plt.show()
    return fig
    
def create_info_table(dfs):
    filter_changes_old = pd.DataFrame(columns=['Datetime', 'Size of filter'])
    for df in dfs:
        Datetime = df.index[0]
        size = len(df)
        filter_changes_old = filter_changes_old.append({'Datetime': Datetime, 'Size of filter': size}, ignore_index=True) 
    filter_changes_old.sort_values(by=['Datetime'], inplace=True)
    filter_changes_old = filter_changes_old.set_index('Datetime')
    return filter_changes_old
    
def bond(dfs, time_step=15, window=3, min_periods=1, center=True, area = 8.04*10**(-6)):
    print("Bond correction \n")
    print("Avergaing over the time step: "+str(time_step)+" mins")
    print("Area: "+str(area))
    df_averaging_times = []
    datasets = []
    for i, df in enumerate(dfs):
        inital_filter = df.I[0] #df.I.max()#will be used to calculate tau (filter transmission)

        #assume a resolution of 1 minute
        df = df.copy()
        #df.index = df.index.round("min") #round to closest minute        
        #df = df.resample('min').mean() #resample to mean values
        #idx = pd.date_range(min(df.index), max(df.index), freq='60s') #create index for every          
        #df = df.reindex(idx, fill_value=np.nan) #fill dataframe will missing values      
        
        #do not df.iloc[1:,:]
        #df = df.rolling(window, min_periods, center, closed='both').mean() #apply a rolling mean

        filter_length = len(df) #length of filter
        
        if filter_length > time_step: #filter must be larger than time step (i.e. only works with sufficent filter lengths)
            df = df.rolling(window, min_periods, center, closed='both').mean() #apply a rolling mean
            
            times = df.index.to_numpy()
            Is = df.I.to_numpy()
            Ios = df.Io.to_numpy()
            Qs = df.qobs.to_numpy()

            #produce a arrays of length of filter minus the last time step (e.g. 15 mins)
            times = times[:-time_step]
            filter_num = np.ones(len(times))*int(i+1) #array for the filter number (i.e. same number for each data point in filter)
            ln_tr = np.log((Is[:-time_step] / Is[time_step:])) #shifted I by time step (i.e. I - deltaI/I) 
            Q = np.convolve(Qs, np.ones(time_step)/time_step, mode='valid')/1000 #rolling average
            volume = Q*time_step  #average flow x duration i.e. 15 minutes
            diff = len(volume) - len(times) #difference should be 1
            volume = volume[int(diff):] #volumne you did the averaging over (e.g. 15mins)
            #volume = volume[:-int(diff)]
            absorption_ln = ((area/volume)*(ln_tr))
            Qstd = Qs[:-time_step].std()
            
            #tau should be less than 1
            tau = Is[:-time_step]/inital_filter #the first measured transmission as reference divided by a decreasing Is
            R = 2*(0.355 + 0.5398*tau) #correction
            #append arrays
            dataset = pd.DataFrame({'Datetime': times, 'filter_num':filter_num, 'volume': volume, 
                                    'ln_tr': ln_tr, 'abs_ln': absorption_ln, 'Tr':tau, 'R':R, 'Qstd':Qstd})
            datasets.append(dataset)
        else:
            pass

    df_averaging_times = pd.concat(datasets)
    df_averaging_times = df_averaging_times.set_index('Datetime')
    df_averaging_times['abs_ln'] = df_averaging_times['abs_ln']*10**6 #change units Mm-1
    return df_averaging_times
    
def plot_transmission(df_averaging_times):
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(df_averaging_times.index, df_averaging_times.Tr, 'o', ms=1)
    ax.set_ylabel('Tr (Transmission/initial Transmission)')
    ax.set_ylim(0,1.25)
    ax.set_title("Transmission of Old PSAP filters")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.show()
    return fig
    
def select_for_transmission(df, Tr_col='Tr', Tr_min=0.5, Tr_max=1):
    df = df[(Tr_min < df[Tr_col]) & (df[Tr_col] < Tr_max)]
    return df
          
def inital_plot(df, var, ymax=100):  
    idx = df.columns.get_loc(str(var))
    fig, ax = plt.subplots(figsize=(15,5))
    plt.plot(df.index, df.iloc[:,idx].values, 'o', ms=0.5)
    plt.ylabel('absorb ln', fontsize=15)
    plt.ylim(-5, ymax)
    plt.grid(True)
    plt.show()
    return fig
    
def select_year(df, year = 2009):   
    df = df[(pd.to_datetime(str(year)+'-01-01 00:00:00') < df.index) & (df.index < pd.to_datetime(str(year)+'-07-24 14:00:00'))] 
    return df
    
def remove_duplicates(df):
    """find the df of the index which appears first"""
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")
    print(len(duplicateRowsDF))    
    df_first = df.loc[~df.index.duplicated(keep='first')]
    df_last = df.loc[~df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first)))
    print("Length after: "+str(len(df_last)))
    return df_first, df_last
    
def remove_neph(df, abs_var='abs_ln', scat_var='scat550', K1 = 0.02, K2 = 1.22): #K1 = 0.02,  K2 = 1.22
    df['abs_neph'] = (df[abs_var]/df['R'] - K1*df[scat_var])/K2 
    return df  
       
       
# def make_plot(df, start_datetime='2012-11-19 08:46:49', end_datetime='2016-10-14 08:05:25'):
    # fig, ax = plt.subplots(figsize=(20,5))
    # plt.plot(df.index, df['Io'], 'o')
    # plt.plot(df.index, df['I'], 'o')

    # times_of_filter_changes = df[df['FChanged_diff'] == 1].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
        
    # times_of_filter_changes =  df.loc[df['Lnr'] < -0.1].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
     
    # times_of_filter_changes =  df.loc[df['deltaT_forward'] >= 10].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
        
    # times_of_filter_changes =  df.loc[df['deltaT_behind'] >= 10].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
        
    # times_of_filter_changes =  df.loc[df['diff_I'] > 0.05].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
           
    # times_of_filter_changes =  df.loc[df['diff_Io'] > 0.05].index
    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes

    # plt.xlim(pd.to_datetime(start_datetime), pd.to_datetime(end_datetime))
    # plt.ylim(2,4)
    # plt.show()
    # return fig
    
# def filter_plot(df, dt_min, dt_max):    
    # plt.figure(figsize=(30,8))
    
    # df = df[(df.index > dt_min) & (df.index < dt_max)]
    
    # plt.plot(df.index, df['ChangeF'], 'o', label='Lnr')    
    # plt.plot(df.index, df['I'], 'o', lw=0.5, label='I')
    # plt.plot(df.index, df['Io'], 'o', lw=0.5, label='Io')
    
    # times_of_filter_changes = df[df['FChanged_diff'] == 1].index

    # for idx in times_of_filter_changes: #mulitple points
        # plt.axvline(idx, linewidth=4, color='r', lw=1) #vertical lines distinguishing the filter changes
   
    # plt.legend(fontsize=15, loc=7)
    # plt.show()
    # return times_of_filter_changes
    
# def filter_plot(df, dt_min, dt_max, filter_lines=None, limits=False):   
    # year = pd.to_datetime(dt_min).year
    
    # plt.figure(figsize=(30,8))    
    # df = df[(df.index > dt_min) & (df.index < dt_max)]    
    # plt.plot(df.index, df['ChangeF'], 'o', label='Lnr')    
    # plt.plot(df.index, df['I'], 'o', lw=0.5, label='I')
    # plt.plot(df.index, df['Io'], 'o', lw=0.5, label='Io')    
    # plt.title(str(dt_min[:4]), fontsize=20)
    
    # if filter_lines is None:
        # times_of_filter_changes = df[df['FChanged_diff'] == 1].index
        # for idx in times_of_filter_changes: 
            # plt.axvline(idx, linewidth=4, color='r', lw=1) 
        
    # if filter_lines is not None:
        # for idx in filter_lines:
            # if idx.year == year:
                # plt.axvline(idx, linewidth=4, color='blue', lw=1)       
        # plt.title(str(year)+" Filter lines (blue): "+str(len(filter_lines)),
                              # fontsize=20)   
    
    # if limits == True:
        # plt.xlim(dt_min, dt_max)        
    # plt.legend(fontsize=20, loc=1)
    # plt.show()
    
# def for_each_year(RAW_NEW_PSAP, years):
    # for year in years[:]:
        # filter_plot(RAW_NEW_PSAP, dt_min=str(year)+'-01-01 00:00:00', dt_max=str(year)+'-12-31 23:59:59')
        
# def create_dfs_for_filters(df):    
    # dfs = [x.iloc[1:,:] for i, x in df.groupby('Filter_cum')]
    # return dfs
    
# def remove_ends(df):
    # length_before = len(df) 
    # if length_before > 0:
        # try:
            # cut_index = df[df.Io.diff().abs() > 0.05].index[0]
        # except IndexError:
            # cut_index = df.index[-1]        
        # df = df[df.index < cut_index]
    # else:
        # df = pd.DataFrame()
    # return df
    
# def remove_ends_for_all(dfs):
    # dfs_ends_removed = pd.Series(dfs).apply(remove_ends)
    # return dfs_ends_removed
    
# def minimum_size_of_dfs(dfs, minimum_length=400):
    # dfs_minimum_size = []
    # for df in dfs:
        # length = len(df)  
        # if length >= minimum_length:
            # dfs_minimum_size.append(df)
        # if length < minimum_length:
            # pass
    # return dfs_minimum_size
    
# def plot_each_filter(df, i, out_path_plots, save=False):      
    # plt.figure(figsize=(20,5))      
    # plt.plot(df.index, df['I'], 'o', lw=0.5, label='I')
    # plt.plot(df.index, df['Io'], 'o', lw=0.5, label='Io')
    # plt.title("Filter: "+str(i), fontsize=20)       
    # plt.legend(fontsize=20, loc=2)
    # plt.ylim(0, 5.9)
    # plt.grid(True)
    # if save==True:
        # plt.savefig(out_path_plots+'filters\\'+str(i)+'.jpeg')
    # plt.show()
    
# def average_running(df):
    # df = df.copy()
    # print(df.head())
    # df.index = df.index.round("min")        
    # df = df.resample('min').mean() 
    # idx = pd.date_range(min(df.index), max(df.index), freq='60s') #create index for every          
    # df = df.reindex(idx, fill_value=np.nan) #fill dataframe will missing values          
    # df = df.rolling(window=20,min_periods=1,center=True).mean() #apply a rolling mean
    #df = df.resample('60T').mean()
    # print(df.head())
    # return df
    
# def plot_each_filter_averages(df, i, ylabel='Filter transittance', mean=False, median=False, exp=False, rolling_median=False, rolling_mean=False, median_ffilled=False,
                             # median_bfilled=False, linear_interpolation=False, ms=3):      
    # fig, ax = plt.subplots(figsize=(20,6))  
    
    # ax.plot(df.index, df['I'], 'o', c='b', ms=ms, label='I')
    # ax.plot(df.index, df['Io'], 'o', c='k', ms=ms, label='I$_{0}$')
    # ax.set_ylim(-0.1, 10)
    # ax.set_ylabel(ylabel, fontsize=20)
        
    #mean
    # if mean == True:
        # df.index = df.index.round("min")        
        # df = df.resample('min').mean() 
        # idx = pd.date_range(min(df.index), max(df.index), freq='60s') #create index for every          
        # df = df.reindex(idx, fill_value=np.nan) #fill dataframe will missing values    
    #mean
    # if median == True:
        # df.index = df.index.round("min")        
        # df = df.resample('min').median() 
        # idx = pd.date_range(min(df.index), max(df.index), freq='60s') #create index for every          
        # df = df.reindex(idx, fill_value=np.nan) #fill dataframe will missing values      
        
    #Exponential moving average 120
    # if exp == True:        
        # df['I'].ewm(span = 60*2).mean().plot(label = 'I Exponential moving average 120')
        # df['Io'].ewm(span = 60*2).mean().plot(label = 'Io Exponential moving average 120')

    #20min rolling median
    # if rolling_median == True:
        # df_median = df.rolling('20T', min_periods=1).median() 
        # ax.plot(df_median.index, df_median['I'], 'o-', lw=2.5, ms=2, label='I (20min rolling median)', c='b')
        # ax.plot(df_median.index, df_median['Io'], 'o-', lw=2.5, ms=2, label='I$_{0}$ (20min rolling median)', c='k')    

    #20min rolling
    # if rolling_mean == True:        
        # df = df.rolling(window=20,min_periods=1,center=True).mean()    
        # ax.plot(df.index, df['I'], 'o-', lw=2.5, ms=2, label='I (#20min rolling)')
        # ax.plot(df.index, df['Io'], 'o-', lw=2.5, ms=2, label='Io (20min rolling)')    
    
    #1hr. median) + forward filled to 1min
    # if median_ffilled == True:        
        # df = df.resample('60T').median()    
        # df_ffill = df.resample('60s').ffill()    
        # ax.plot(df.index, df['I'], 'o-', ms=2, label='I (hr. median) + forward filled to 1min')    
        # ax.plot(df.index, df['Io'], 'o-', lw=2.5, ms=2, label='Io (hr. median applied on 20min rolling) + forward filled to 1min')

    #1hr. median) + backward filled to 1min
    # if median_bfilled == True:        
        # df_bfill = df.resample('60s').bfill()
        # ax.plot(df.index, df['I'], 'o-', ms=2, label='I (hr. median) + backward filled to 1min')    
        # ax.plot(df.index, df['Io'], 'o-', lw=2.5, ms=2, label='Io (hr. median applied on 20min rolling) + backward filled to 1min')

    #liner interpolation
    # if linear_interpolation == True:        
        # df_interpolate = df.resample('60s').mean().interpolate(method='linear')
        # plt.plot(df.index, df['I'], 'o-', ms=2, label='I liner interpolate')    
        # plt.plot(df.index, df['Io'], 'o-', lw=2.5, ms=2, label='I hr. mean + liner interpolate')

    # ax2 = ax.twinx()
    # flow_plot, = ax2.plot(df.index, df['qobs'], 'o', c='g', ms=ms, label='')
    # ax2.set_ylim(-0.1, 5)
    # ax2.set_ylabel('Q$_{\mathrm{obs}}$ [lpm]', fontsize=20, c=flow_plot.get_color())
    # ax.legend(fontsize=20, loc=1, frameon=False)
    
    # tkw_major = dict(which='major', size=4, width=1.5, length=4, labelsize=10, direction='in', pad=10)
    # ax2.tick_params(axis='y', colors=flow_plot.get_color(), **tkw_major)
    
    # ax.tick_params(axis='both', which='major', labelsize=15)
    # ax.tick_params(axis='both', which='minor', labelsize=15)
    # ax2.tick_params(axis='both', which='major', labelsize=15)
    # ax2.tick_params(axis='both', which='minor', labelsize=15)

    # plt.title("Filter: "+str(i)+" start: "+str(df.index[0].date())+' - '+str(df.index[-1].date())+' '+str(len(df)), 
              # fontsize=15, loc='left')       
    # plt.grid(False) 
    # plt.show()
    # return fig
    
# def plot_each_filter_for_dfs(dfs, start=None, end=None, 
                            # mean=False, exp=False, rolling_median=False, rolling_mean=False, median_ffilled=False,
                            # median_bfilled=False, linear_interpolation=False, ms=3):
    # if (start is not None) & (end is not None):
        # dfs = dfs[start:end].copy()
    # length = len(dfs)
    # for i in range(length): 
        # df = dfs[i] 
        # i = i+start+1
        # fig = plot_each_filter_averages(df, i, mean=mean, exp=exp, rolling_median=rolling_median, 
                                  # rolling_mean=rolling_mean, median_ffilled=median_ffilled,
                                  # median_bfilled=median_bfilled, linear_interpolation=linear_interpolation, ms=ms)
    # return fig
                                 
# def create_info_table(dfs):
    # filter_changes_old = pd.DataFrame(columns=['Datetime', 'Size of filter'])
    # for df in dfs:
        # Datetime = df.index[0]
        # size = len(df)
        # filter_changes_old = filter_changes_old.append({'Datetime': Datetime, 'Size of filter': size}, ignore_index=True) 
    # filter_changes_old.sort_values(by=['Datetime'], inplace=True)
    # filter_changes_old = filter_changes_old.set_index('Datetime')
    # return filter_changes_old
    
# def bond_correction_averaging(dfs, time_step=15, running_average=20, area = 8.04*10**(-6)):    
    # radius = np.sqrt((area*(1000)**2)/np.pi) 
    # print(str(radius) + ' mm')
    # print('Diameter: ' + str(radius*2) + ' mm')
    # print("Bond correction \n")
    # print("Avergaing over the time step: "+str(time_step)+" mins")
    # print("Area: "+str(area))
    
    # print("Running median: "+str(running_average))
    # print("Time step: "+str(time_step))
    
    # df_averaging_times = []
    # datasets = []
    # for i, df in enumerate(dfs):
        # inital_filter = df.I[0] #df.I.max()#will be used to calculate tau (filter transmission)
        
        # df = df.copy()
        # df.index = df.index.round("min")        
        # df = df.resample('min').mean() 
        # idx = pd.date_range(min(df.index), max(df.index), freq='60s') #create index for every          
        # df = df.reindex(idx, fill_value=np.nan) #fill dataframe will missing values   
        # df = df.rolling(window=running_average, min_periods=1, center=True).mean()

        # filter_length = len(df) 
        # if filter_length > time_step: #filter must be larger than time step
            # times = df.index.to_numpy()
            # Is = df.I.to_numpy()
            # Ios = df.Io.to_numpy()
            # Qs = df.qobs.to_numpy()

            # times = times[:-time_step]
            # filter_num = np.ones(len(times))*int(i+1)

            # ln_tr = np.log((Is[:-time_step] / Is[time_step:])) 
            # Q = np.convolve(Qs, np.ones(time_step)/time_step, mode='valid')/1000 #rolling average [m3/min]
            # volume = Q*time_step #average x duration i.e. 15 minutes [m3]
            # diff = len(volume) - len(times)
            # volume = volume[:-int(diff)]            
            # absorption_ln = ((area/volume)*(ln_tr))
            # Qstd = Qs[:-time_step].std()

            # tau = Is[:-time_step]/inital_filter #the first measured transmission as reference divided by a decreasing Is
            # R = 2*(0.355 + 0.5398*tau)   
            # dataset = pd.DataFrame({'Datetime': times, 'filter_num':filter_num, 'volume': volume, 
                                    # 'ln_tr': ln_tr, 'abs_ln': absorption_ln, 'Tr':tau, 'R':R})
            # datasets.append(dataset)
        # else:
            # pass

    # df_averaging_times = pd.concat(datasets)
    # df_averaging_times = df_averaging_times.set_index('Datetime')
    # df_averaging_times['abs_ln'] = df_averaging_times['abs_ln']*10**6       
    # return df_averaging_times
    
# def concat_dfs(dfs):
    # df_concat = pd.concat(dfs)
    # return df_concat
    
# def plot_transmission(df_averaging_times):
    # fig, ax = plt.subplots(figsize=(15,5))
    # plt.plot(df_averaging_times.index, df_averaging_times.Tr, 'o', ms=1)
    # plt.ylabel('Tr (Transmission/initial Transmission)')
    # plt.show()
    # return fig
    
# def select_for_transmission(df, Tr_col='Tr', Tr_min=0.5, Tr_max=1):
    # df = df[(Tr_min < df[Tr_col]) & (df[Tr_col] < Tr_max)]
    # return df
    
# def inital_plot(df, var, y_lim=60):  
    # idx = df.columns.get_loc(str(var))
    # fig, ax = plt.subplots(figsize=(15,5))
    # plt.plot(df.index, df.iloc[:,idx].values, 'o', ms=0.5)
    # plt.ylabel('absorb ln', fontsize=15)
    # plt.ylim(-5, y_lim)
    # plt.grid(True)
    # plt.show()
    # return fig
    
# def remove_datetimes_slice(df, dt_min, dt_max):
    # df = df.copy()
    # print(len(df))
    # mask = (df.index > dt_min) & (df.index <= dt_max)
    # df = df.loc[~mask]
    # print(len(df))
    # return df
   
# def remove_extreme_values(df, var):
    # print("Mean :" + str(df[var].mean()))
    # df = df[np.abs(df[var]-df[var].mean()) <= (3*df[var].std())]
    # print("Mean (extremes removed) :" + str(df[var].mean()))
    # return df
    
# def remove_duplicates(df):
    # print("Length before: "+str(len(df)))
    # duplicateRowsDF = df.index[df.index.duplicated()]
    # print("Duplicate Rows except first occurrence based on all columns are :")
    # print(len(duplicateRowsDF))    
    # df_first = df.loc[~df.index.duplicated(keep='first')]
    # df_last = df.loc[~df.index.duplicated(keep='last')]
    # print("Length after: "+str(len(df_first)))
    # print("Length after: "+str(len(df_last)))
    # return df_first, df_last
    
# def merge_with_tolerance(df_abs_without_scat, df_scat):
    # df_scat = df_scat.sort_index()
    # df_abs_without_scat = df_abs_without_scat.sort_index()
    # df_abs_scat = pd.merge_asof(df_abs_without_scat, df_scat, left_index=True, right_index=True, 
                            # tolerance=pd.Timedelta("1H"), direction='nearest')
    # return df_abs_scat
    
# def remove_neph(df, abs_var='abs_ln', scat_var='scat550', K1 = 0.02, K2 = 1.22): #K1 = 0.02,  K2 = 1.22
    # df = df.copy()
    # df['abs_neph'] = (df[abs_var]/df['R'] - K1*df[scat_var])/K2 
    # return df  
    
# def plot_corrected_and_uncorrected(df_abs_neph):
    # fig, ax = plt.subplots(figsize=(15,5))
    # ax.plot(df_abs_neph.index, df_abs_neph.abs_ln, 'o', ms=2, ls=':', lw=1, mec='blue', c='blue',
            # label = r'ln($\frac{\mathrm{I}_{t-\Delta t}}{\mathrm{I}_{t}}$)')
    # ax.plot(df_abs_neph.index, df_abs_neph.abs_neph, 'o', ms=2, ls=':', lw=1, mec='black', c='black', 
            # label = '$\sigma_{ap}$')
    # ax.set_ylim(-2,12)
    # ax.set_ylabel('$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]', fontsize=20)
    # ax.legend(fontsize=20, loc=1, frameon=False)
    # plt.show()
    # return fig
    
# def scat_abs(df_abs_neph, df_scat, abs_var='abs_neph', ms=2,startdatetime=None, enddatetime=None):
    # fig, ax = plt.subplots(figsize=(15,5))

    # abs_scatter = ax.scatter(df_abs_neph.index, df_abs_neph[abs_var], c=df_abs_neph.Tr, s=ms,
                            # vmin=0, vmax=1, label='abs')

    # ax2 = ax.twinx()   
    # ax2.plot(df_scat.loc[df_scat['Mie'] == True, 'scat550'].index, df_scat.loc[df_scat['Mie'] == True, 'scat550'], 
             # 'o', ms=ms, c='k', label='scat$_{\mathrm{mie}}$', mec='k', mfc='None')
    # ax2.plot(df_scat.loc[df_scat['Mie'] == False, 'scat550'].index, df_scat.loc[df_scat['Mie'] == False, 'scat550'], 
             # 'o', ms=ms, label='scat$_{\mathrm{obs}}$', mec='r', mfc='None')
    
    # ax2.set_ylim(-0.1,50)
    # ax2.set_ylabel('$\sigma_{\mathrm{sp}}$ [Mm$^{-1}$]',fontsize=20)
    # ax2.legend(loc=1, frameon=False, fontsize=20, markerscale=2)
    
    # ax.set_ylim(-0.3,5) 
    # if (startdatetime and enddatetime) is not None: #(x and y) is not None
        # print(startdatetime)
        # ax.set_xlim(pd.to_datetime(startdatetime), pd.to_datetime(enddatetime))
    # else:
        # ax.set_xlim(pd.to_datetime(df_abs_neph.index[0]),pd.to_datetime(df_abs_neph.index[-1]))
    
    # ax.set_ylabel('$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]',fontsize=20)
    # cax = fig.add_axes([0.98, 0.06, 0.02, 0.88]) #x,y, width,height
    # cbar = fig.colorbar(abs_scatter, cax=cax)
    # cbar.ax.set_ylabel(r'Tr', fontsize=15)    
    # cbar.ax.tick_params(axis='y', color='white', left=True, right=True,
                        # length=5, width=1.5)
    # ax.legend(loc=2, frameon=False, fontsize=20, markerscale=5)
    # plt.grid(True)
    # plt.show()
    # return fig