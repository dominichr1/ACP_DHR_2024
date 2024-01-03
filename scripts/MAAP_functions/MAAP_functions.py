import pandas as pd
import matplotlib.pyplot as plt
import glob
import numpy as np
from datetime import datetime, timedelta
import seaborn as sns
import os

lambda_MAAP = 637 #nm
MAC=6.6
keep_extreme_values = True #no clean i.e. detection limits
apply_mean = True
apply_median = False

var_to_label = {'concentration of black carbon':'BC$_{e}$ mass concentrations, MAC of 6.6 m$^{2}$g$^{-1}$ [$\mu$g m$^{-3}$]',
                'mass of black carbon':'mass of black carbon',
                'absorption':'$\sigma_{\mathrm{ap}}$ (MAAP, '+str(lambda_MAAP)+' nm) [Mm$^{-1}$]\n(mass con. [ng/m$^{3}$] * MAC [m$^{2}$/g])',
               'Aerosol light absorption coefficient (Mm⁻¹)':'$\sigma_{\mathrm{ap}}$ (MAAP, '+str(lambda_MAAP)+' nm) [Mm$^{-1}$]',
               'Aerosol light absorption coefficient calculated from the inversion model (Mm⁻¹)':'$\sigma_{\mathrm{ap}}$ (MAAP, '+str(lambda_MAAP)+' nm inversion model) [Mm$^{-1}$]'}

def append_data(MAAP_path, years=np.arange(2014,2019,1), col_names=['date', 'time', 'status', 'concentration of black carbon', 'mass of black carbon', 'air flow rate',
                'last value', 'mean values of the concentration\n of black carbon over 1h', '3h', '24h']):
    appended_data = []   
    for year in years:      
        print("year: "+str(year))
        list_MAAP_files = glob.glob(MAAP_path+str(year)+'\\MAAP*.txt')
        print('length: '+str(len(list_MAAP_files)))    
        for infile in list_MAAP_files:        
            rows_keep = []        
            with open(infile) as f:
                lines = f.readlines()
                for value, line in enumerate(lines):
                    if len(line) == 94:
                        rows_keep.append(value)
            try:
                df = pd.read_csv(infile, sep='\s+', header=None, parse_dates={'datetime': [0, 1]}, index_col=0, skiprows = lambda x: x not in rows_keep)
                df.index = pd.to_datetime(df.index, format='%Y-%m-%d %H:%M:%S')
                df.columns = col_names                
                appended_data.append(df)
            except:
                pass
    MAAP_2014_2018 = pd.concat(appended_data)  
    df_MAAP_2014_2018_original = MAAP_2014_2018.copy()
    print("Columns in MAAP dataset: "+str(MAAP_2014_2018.columns))
    return MAAP_2014_2018
	
def remove_duplicates(df):
    """see if there are any duplicates: """
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")      
    if len(duplicateRowsDF) == 0:
        print("no duplicates")
    else:
        print(len(duplicateRowsDF))        
    df_first = df.loc[~df.index.duplicated(keep='first')]
    df_last = df.loc[~df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first)))
    print("Length after: "+str(len(df_last)))
    return df_first, df_last
	
def simple_log_plot(df, var, ymin=None, ymax=None, xmin='2014-11-19 15:14:16', 
                    xmax='2018-04-20 17:13:11', ylabel=None, add_hour_lines=False):
    fig, ax =plt.subplots(figsize=(12,3))
    ax.plot(df.index, df[var], 'o', ms=1, mec='k', mfc='None')    
    number_of_hours = ((pd.Timestamp(xmax) - pd.Timestamp(xmin)).days)*24
    
    if add_hour_lines == True:
        for hour in np.arange(1,number_of_hours,1):
            hours_added = timedelta(hours=int(hour))
            time_line = pd.Timestamp(xmin) + hours_added
            plt.axvline(time_line,color='r', lw=0.2)

    plt.title(str(var), loc='left')
    if (ymin != None) & (ymax != None):
        plt.ylim(ymin,ymax)
    plt.xlim(pd.to_datetime(xmin), pd.to_datetime(xmax))
    plt.ylabel(ylabel)
    plt.yscale('log')
    plt.show()    
    return fig
	
def remove_flow_with_threshold(MAAP_2014_2018, min_flow = 900, control_for_flow=False): #set a minimum flow for the flow rate
    if control_for_flow == True:
        print("control flow")
        MAAP_2014_2018_above_min_flow = MAAP_2014_2018[MAAP_2014_2018['air flow rate'] > min_flow]
        points_removed_flow_too_low = MAAP_2014_2018[MAAP_2014_2018['air flow rate'] < min_flow]
        print("Data points removed from dataset: "+str(len(points_removed_flow_too_low)))
    if control_for_flow == False:
        MAAP_2014_2018_above_min_flow = MAAP_2014_2018.copy() 
    return MAAP_2014_2018_above_min_flow
	
def inital_plot(df, var, ymin=-5, ymax=10):  
    idx = df.columns.get_loc(str(var))
    plt.figure(figsize=(15,8))
    plt.plot(df.index, df.iloc[:,idx].values, 'o', ms=0.5)
    plt.ylabel(var_to_label[var], fontsize=15)
    plt.ylim(ymin,ymax)
    plt.show()
	
def provide_average_time_difference(df):
    """work out the average resolution"""
    df['timestep'] = df.index
    df = df.dropna(how='any', subset=['timestep'])
    df['diff'] = df['timestep'].diff()
    mean = df['diff'].mean()
    print("average: "+str(mean))
    return mean
	
def accept_flags(MAAP_2014_2018_above_min_flow, flags, flags_accepted = ['000000', 0]):
    print("keep flags: "+str(flags_accepted))
    flags_removed = set(flags) - set(flags_accepted)
    MAAP_2014_2018_flags_removed = MAAP_2014_2018_above_min_flow.loc[MAAP_2014_2018_above_min_flow['status'].isin(flags_accepted)]
    MAAP_2014_2018_flags = MAAP_2014_2018_above_min_flow.loc[MAAP_2014_2018_above_min_flow['status'].isin(flags_removed)]
    print("Data points removed from data set: "+str(len(MAAP_2014_2018_flags)/len(MAAP_2014_2018_above_min_flow)))
    return MAAP_2014_2018_flags_removed, MAAP_2014_2018_flags
	
def count_values(MAAP_2014_2018_flags_removed, var='concentration of black carbon'):
    value_counts = MAAP_2014_2018_flags_removed[var].value_counts()
    df_value_counts = value_counts.to_frame(name='count') 
    print(df_value_counts.head())
    return df_value_counts
	
def value_count_plot(df_value_counts):
    fig, ax = plt.subplots(figsize=(10,5))
    plt.plot(df_value_counts,'o',c='k',mfc='None')
    plt.xlabel('concentration of black carbon')
    plt.ylabel('Count')
    ax.set_xticks(np.arange(start=-100, stop=100, step=1)/100)
    ax.set_xticklabels(np.arange(start=-100, stop=100, step=1)/100)
    plt.xlim(-0.1,0.1)
    plt.show()
	
def create_histogram(df, var, bin_num, ymax, xmax):
    fig, ax = plt.subplots(figsize=(10,5))
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)]
    x = df[var].values
    
    weights = np.ones_like(x) / len(x)
    n, bins, patches = plt.hist(x, bins=bin_num, weights=weights, alpha=0.75)
    plt.xlabel('concentration of black carbon') #r'$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]')
    plt.ylabel('Normalised probability')
    plt.title('Histogram of absoprtion data of length: '+str(len(df)))
    plt.title(r'Bins: '+str(bin_num), loc='right')
    plt.ylim(0,ymax)
    plt.xlim(-0.5, xmax)
    plt.grid(True)
    plt.show()
    return fig
	
def inital_2plot(df1, df2, var, ymin=-5, ymax=10):   
    idx = df1.columns.get_loc(str(var))
    plt.figure(figsize=(15,8))    
    plt.plot(df1.index, df1.iloc[:,idx].values, 'o', ms=0.5, label='Size: '+str(len(df1))+' flags removed')
    plt.plot(df2.index, df2.iloc[:,idx].values, 'o', ms=0.5, label='Size: '+str(len(df2))+' just flags')
    plt.ylabel(var_to_label[var], fontsize=15)
    plt.ylim(ymin,ymax)
    plt.legend(frameon=False)  
    plt.show()

def remove_extreme_values(df, var, keep_extreme_values=True):
    print("Keep extreme values: "+str(keep_extreme_values))
    if keep_extreme_values == False:
        print("Mean :" + str(df[var].mean()))
        df = df[np.abs(df[var]-df[var].mean()) <= (3*df[var].std())]
        print("Mean (extremes removed) :" + str(df[var].mean()))
    if keep_extreme_values == True:
        pass
    return df
	
def resample_use_detection_limits(df, var, type_var, keep_extreme_values=True):
    if keep_extreme_values == False:    
        df = df.resample('30T').mean() #detection limits are based on averaging times of 30 minutes    
        print("Number of 30 mins values: " +str(len(df)))
        if type_var == 'concentration':
            df[df[var].values < 20*10**(-3)] = np.nan  #lower limit in µg/m   
            df[df[var].values >= 60] = np.nan #upper limit µg/m
            print(df[var].min())
            print(df[var].max())        
        if type_var == 'absorption':
            df[df[var].values < 0.13] = np.nan  #lower limit in Mm-1      
            df[df[var].values >= 60*6.6] = np.nan #upper limit in Mm-1
            print(df[var].min())
            print(df[var].max())
        print("Number of non null 30 mins values: " +str(len(df[pd.notnull(df[var])])))
        df.dropna(subset=[var], inplace=True)
    if keep_extreme_values == True:
        df.dropna(subset=[var], inplace=True)        
    return df
	
def convert_mass_con_to_abs(df, mass_con='concentration of black carbon', MAC=6.6):
    print("MAC of "+str(MAC)+" used")
    df = df.copy()
    df.loc[:, 'absorption'] = df[mass_con]*MAC #*10**3 #absorption (Mm-1) = #[ng/m3] * [m²/g] * 10^3 not need must have been set to ug 
    return df
	
def correct_standard_Temp_P(df, varlist, var='scat', t_var='T_int', p_var='p_int'):
    print("Correct for standard temperature and pressure")
    
    mean_temp = df[t_var].mean()
    print("Mean temperature: "+str(mean_temp)) 
    if mean_temp > 200: #kelvin
        print("in Kelvin")
        pass
    if mean_temp < 50:
        print("convert to Kelvin")
        df[t_var] = df[t_var]+273.15 #Kelvin
        
    variables = [x for x in varlist if var in x]
    stand_temperature = 273.15
    stand_pressure = 1013.25
    for var in variables:
        df[str(var)] = df[var] * (df[p_var]*stand_temperature)/(df[t_var]*stand_pressure)
    
    df.loc[:,t_var] -= 273.15   #to celcius
    return df
	
def plot_monthly_dfs(df, label, color, ax=None):
    
    index = df.iloc[:,0].index
    mean =  df.iloc[:,0].values
    std = df.iloc[:,4].values

    ax.errorbar(index, mean, yerr=std, fmt='-*', capsize=5,  mfc='green', ecolor='black', label='std')
    sns.lineplot(x= index, y= mean, label=label, color=color)    
    ax.tick_params(rotation=45, direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', length=4, color='k', width=1.3)
    ax.set_ylabel('$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]', fontsize=20)    
    ax.xaxis.set_tick_params(rotation=45) 
    ax.set_title('MAAP: monthly averages', fontsize=15)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)    
    ax.legend(ncol=2, loc=1, frameon=False, fontsize=15)    
    return ax
	
def produce_monthly_averages(df, var = 'absorption'):
    idx = df.columns.get_loc(str(var))
    df_monthly = df.iloc[:,idx].resample("M").agg(['mean', 'median', 'min', 'max', 'std']) 
    return df_monthly
	
names = ['Epoch time: seconds from 1970-01-01T00:00:00Z',
'Fractional day of year (Midnight January 1 UTC = 1.00000)',
'Status',
'Aerosol light absorption coefficient (Mm⁻¹)',
'Inversion calculated single scattering albedo',
'Sample pressure (hPa)',
'Aerosol light absorption coefficient calculated from the inversion model (Mm⁻¹)',
'Filter ID',
'Reference detector signal',
'Sample forward detector signal',
'Transmittance',
'Sample 135 degree backscatter detector signal',
'Sample 165 degree backscatter detector signal',
'Integrated sample length Qt/A (m)',
'Pressure drop from ambient to orifice face (hPa)',
'Vacuum pressure pump drop across orifice (hPa)',
'Sample flow (lpm)',
'Accumulated sample volume (m³)',
'Ambient temperature (°C)',
'Measuring head temperature (°C)',
'System temperature (°C)',
'Inversion calculated equivalent black carbon concentration (μg/m³)',
'Inversion calculated aerosol optical depth of the filter',
'List of all system parameters',
'Spot sampling parameters']

print(names)
            
code = ['EPOCH',
'DOY',
'F1_A31',
'BaR_A31',
'ZSSAR_A31',
'P_A31',
'BacR_A31',
'Ff_A31',
'IfR_A31',
'IpR_A31',
'IrR_A31',
'Is1_A31',
'Is2_A31',
'L_A31',
'Pd1_A31',
'Pd2_A31',
'Q_A31',
'Qt_A31',
'T1_A31',
'T2_A31',
'T3_A31',
'XR_A31',
'ZIrR_A31',
'ZPARAMETERS_A31',
'ZSPOT_A31']

code_to_name = dict(zip(code, names))

def find_col_names(infile, nthline, sep):
    with open(infile, encoding="utf8") as f:
        lines = f.readlines()
        for count, line in enumerate(lines):
            if count == nthline: #nthline
                columns = line                
                columns_NOAA = columns.split(sep)                
                columns_NOAA = columns_NOAA[1:]
                columns_NOAA.remove('')
                columns_NOAA.insert(2, 'status')
                columns_NOAA = columns_NOAA[:-1]
    return columns_NOAA
	
def import_NOAA_data(MAAP_path, year):
    appended_data = []         
    list_MAAP_NOAA_files = glob.glob(MAAP_path+str(year)+'_NOAA_software\\*')
    print('length: '+str(len(list_MAAP_NOAA_files)))
    for infile in glob.glob(MAAP_path+str(year)+'_NOAA_software\\*'):
        rows_skip = []
        with open(infile, encoding="utf8") as f:
            lines = f.readlines()
            for value, line in enumerate(lines):
                if line.isspace():
                    rows_skip.append(value - 3)              
        rows_skip
        df = pd.read_csv(infile, sep='\s+', header=0, skiprows=2, low_memory=False)    
        df = df.drop(rows_skip,0)
        df["Date"] = pd.to_datetime(df.iloc[:,0], format='%d/%m/%y')
        df['Time'] = [x + ':00' for x in df.iloc[:,1]]   
        df["Time"] = pd.to_timedelta(df["Time"])    
        df["DateTime"] = df["Date"] + df["Time"]
        df = df.set_index('DateTime')
        df = df.drop(['Date','Time'], axis=1)
        df = df.iloc[:,2:]
        cols = df.columns
        cols = cols.insert(0, 'EPOCH')
        cols = cols[:-1]
        df.columns = cols
        appended_data.append(df)
    MAAP_2018 = pd.concat(appended_data)  
    MAAP_2018.rename(code_to_name, inplace=True)
    MAAP_2018.columns = MAAP_2018.columns.map(code_to_name)
    return MAAP_2018
	
def append_MAAP(MAAP_path, years=[2019, 2020, 2021, 2022]):
    appended_data = []
    for year in years:
        print(year)
        print(MAAP_path+str(year)+'\\TXTformat\\ZEP*')
        for infile in glob.glob(MAAP_path+str(year)+'\\TXTformat\\ZEP*'):
            print(infile)
            columns = find_col_names(infile, 4, sep=',')
            try:
                df = pd.read_csv(infile, sep=',', header=1, parse_dates={'date': [0]}, index_col=0, skiprows = 5, low_memory=False)
                df.index = pd.to_datetime(df.index, format='%d-%m-%y %H:%M:%S')
                appended_data.append(df)
            except:
                pass
        MAAP_2019_2020 = pd.concat(appended_data)
        MAAP_2019_2020.columns = MAAP_2019_2020.columns.map(code_to_name)
    return MAAP_2019_2020
	
def concat_dfs(frames):
    df = pd.concat(frames)
    return df

def create_dictionary_var_max_values(df):    
    cols = df.columns[3:]
    print(cols)
    dict_max = {}
    for i, col in enumerate(cols):
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        max_value = df[col].max()
        print(max_value)
        dict_max[col] = []
        dict_max[str(col)].append(max_value) 
    print(dict_max)
    return dict_max
	
def replace_max_values(MAAP_2018_2020, dict_max):
    #replace the max values with np.nan
    cols = MAAP_2018_2020.columns[3:]
    print(cols)
    for i, col in enumerate(cols):
        MAAP_2018_2020 = MAAP_2018_2020.replace(dict_max[col], np.nan)
    return MAAP_2018_2020
	
def carbon_concentration(MAAP_2018_2020, mass_var='Inversion calculated equivalent black carbon concentration (μg/m³)',
                         MAC=6.6):
    MAAP_2018_2020['Carbon_conc_MAC'] = MAAP_2018_2020[mass_var]*MAC
    return MAAP_2018_2020
	
def create_subplots(df):
    size = len(df.columns[3:])    
    fig, axs = plt.subplots(size,1, figsize=(20, 7*size), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i, col in enumerate(df.columns[3:]):        
        df[col] = pd.to_numeric(df[col], errors='coerce') 
        axs[i].plot(df.index, df[col].values, 'o', ms=1)
        axs[i].set_title(str(col)+' max value: '+str(df[col].max()))        
    plt.show()
	
def flag_makes_up(df, flag):
    print(str(flag)+" makes up " + str(df.Status.value_counts(normalize=True)[flag]*100) + " % of flags")
    makes_up = str(df.Status.value_counts(normalize=True)[flag]*100)
    return makes_up
	
# def create_subplots_flags(df):
    # size = len(df.columns[3:])    
    # fig, axs = plt.subplots(size,1, figsize=(20, 7*size), facecolor='w', edgecolor='k')
    # fig.subplots_adjust(hspace = .5, wspace=.001)
    # axs = axs.ravel()
    # for i, col in enumerate(df.columns[3:]):
        # df[col] = pd.to_numeric(df[col], errors='coerce')         
        # for flag in df.status.unique():    
            # axs[i].plot(df[df.Status == flag].index, df[df.Status == flag][col].values, 'o', ms=1, label=str(flag))
            # axs[i].set_title(str(col)+' max value: '+str(df[col].max()))
            # axs[i].legend()        
    # plt.show()
	
def select_for_flow(MAAP_2018_2020_extreme_removed, MAAP_2014_2018_flags_removed, 
                    min_flow = 900, control_for_flow=False):
    print("df size before flow control: "+str(len(MAAP_2014_2018_flags_removed)))
    if control_for_flow==True:
        min_flow_1min = min_flow/60
        print(min_flow_1min)
        MAAP_2018_2020_flowlimit = MAAP_2018_2020_extreme_removed[MAAP_2018_2020_extreme_removed['Accumulated sample volume (m³)'] > min_flow_1min]
        print(len(MAAP_2018_2020_flowlimit))
    if control_for_flow == False:
        MAAP_2018_2020_flowlimit = MAAP_2018_2020_extreme_removed.copy()
    print("df size after flow control: "+str(len(MAAP_2018_2020_flowlimit)))
    return MAAP_2018_2020_flowlimit
    
def slice_df(df, start='2019-01-01 00:00:00', end='2019-04-01 00:00:00'):
    df = df[(df.index > '2019-01-01 00:00:00') & (df.index < '2019-04-01 00:00:00')]
    return df
	
def create_subplots_flags(df, status_col='status'):
    size = len(df.columns[3:])  
    fig, axs = plt.subplots(size,1, figsize=(20, 7*size), facecolor='w', edgecolor='k')
    fig.subplots_adjust(hspace = .5, wspace=.001)
    axs = axs.ravel()
    for i, col in enumerate(df.columns[3:]):        
        df = df.copy()
        df[col] = pd.to_numeric(df[col], errors='coerce')  
        print(df[status_col].unique())
        for flag in df[status_col].unique():       
            axs[i].plot(df[df[status_col] == flag].index, df[df[status_col] == flag][col].values, 'o', ms=1, label=str(flag))
            axs[i].set_title(str(col)+' max value: '+str(df[col].max()))
            axs[i].legend()        
    return fig
	
def hourly_averages(MAAP, keep_extreme_values=True):
    MAAP = MAAP.loc[:,'absorption'].to_frame('absorption')
    MAAP_hourly_mean = MAAP.resample('60T').mean()
    MAAP_hourly_median = MAAP.resample('60T').median()
    return MAAP_hourly_mean, MAAP_hourly_median
    
def plot(series, var):  
    fig, ax = plt.subplots(figsize=(15,5))
    ax.plot(series, 'o', mfc='k', mec='k', ms=1.5)
    ax.set_ylabel(var_to_label[var], fontsize=15)
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    plt.show()
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg"):
    folders = glob.glob(path)
    print(folders)
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight')
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
	
	


	
	
	