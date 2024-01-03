import pandas as pd
import numpy as np
from scipy import stats
from statsmodels.tsa import stattools
from scipy.stats import norm
import mpmath
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib.dates as mdates
import os
import glob  
import seaborn as sns
import matplotlib.gridspec as gridspec
import mannkendall as mk
from datetime import datetime, timedelta

month_to_season =  { 1:'SBU',  2:'AHZ', 3:'AHZ',  
                     4:'AHZ',  5:'AHZ', 6:'SUM',  7:'SUM',  8:'SUM', 9:'SUM', 10:'SBU', 
                     11:'SBU', 12:'SBU'}                     
abb_to_name = { 'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}
name_to_abb = {'Slow build up':'SBU','Arctic Haze': 'AHZ', 'Summer/Clean':'SUM'}

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
    
def save_df(df, path, name='', index=True, float_format=None, format_data='.dat', header=True):
    print("Save as: "+str(path+'\\'+name+str(format_data)))
    df.to_csv(path+'\\'+name+str(format_data), index=index, float_format=float_format, header=header)

def sf(sf_num):
    sf = '{0:.'+str(sf_num)+'f}' #2 digits of precision and f is used to represent floating point number.
    return sf
    
def load_neph_data(filename="df_scat_TSI_ecotech_Mie.dat", loadpath = "C:\\Users\\DominicHeslinRees\\Documents\\Analysis\\Neph\\", select_Mie=0):
    print("load from :"+str(loadpath))
    df_neph = pd.read_csv(loadpath+str(filename), index_col=0, parse_dates=True)
    df_neph = df_neph[df_neph.Mie == select_Mie].copy()
    return df_neph
    
def mergedfs(df1, df2, how='inner'):
    df_merged = pd.merge(df1, df2, how=how, left_index=True, right_index=True)
    return df_merged
    
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

def reverse_dict(dictionary):
    inv_dict = {v: k for k, v in dictionary.items()}
    return inv_dict

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

def q25(x, percentile=0.25):
    return x.quantile(percentile)

def q75(x):
    return x.quantile(0.75)    

def produce_averages_groupby(df, groupby_var, variable):
    idx = df.columns.get_loc(str(variable))
    df_groupby = df.groupby(by=str(groupby_var))[variable].agg(['mean', 'median', 'min', 'max', 'std', 'count', q25, q75]) 
    return df_groupby

def mid_datetime_function(a, b):
    return a + (b - a)/2
    
def slice_df(df, start_datetime=None, end_datetime=None):
    if (start_datetime is not None) & (end_datetime is not None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index) & (df.index <= pd.to_datetime(end_datetime))]
    if (start_datetime is not None) & (end_datetime is None):
        df = df.loc[(pd.to_datetime(start_datetime) <= df.index)]
    if (start_datetime is None) & (end_datetime is not None):
        df = df.loc[(df.index <= pd.to_datetime(end_datetime))]
    return df

def add_mid_datetime_using_dictionary(df, season_num_to_season):
    df['season_abb_year'] = df.index.map(season_num_to_season)
    df['start'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[0])
    df['stop'] = df['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[1])
    df['mid_datetime'] = df.apply(lambda x: mid_datetime_function(x.start, x.stop), axis=1)
    return df

def normalise_index(df):
    df.index = df.index - df.index[0] + 1
    return df

def seasonal_averages_select_abb(df, variable, season_abb=None, groupby_var='season_ordinal',
                      season_abb_year_col = 'season_abb_year'): 
    df_seasonal_averages = produce_averages_groupby(df, groupby_var=groupby_var, variable=variable)          
    season_num_to_season = dict(zip(df['season_ordinal'], df['season_abb_year']))  
    df_seasonal_averages = add_mid_datetime_using_dictionary(df_seasonal_averages, season_num_to_season)    
    df_seasonal_averages = normalise_index(df_seasonal_averages)  
    
    df_seasonal_averages['season_abb'] = df_seasonal_averages[season_abb_year_col].apply(lambda x: x[:3] if isinstance(x, str) else x)
    
    if season_abb is not None:       
        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]
        df_seasonal_averages = df_season.copy()      
    return df_seasonal_averages

def remove_spines(ax):
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    
def find_number_of_duplicate_indexs(df):
    duplicateRowsDF = df.index[df.index.duplicated()]
    if len(duplicateRowsDF) != 0:
        print("Duplicate Rows except first occurrence based on all columns are :")      
        print(len(duplicateRowsDF))
    return duplicateRowsDF 
    
def load_data_and_merge(abs_filename='oldpsap_newpsap_maap', abs_df_path=r'C:\Users\DominicHeslinRees\Documents\Analysis\absorption\appended',
                        fire_filename='fire_based_on_abs', fire_df_path=r'C:\Users\DominicHeslinRees\Documents\Analysis\fire',
                        rainfall_filename='df_rainfall_accumulated_extremes', rainfall_df_path=r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\extremes',
                        merge_with_fire=True, merge_with_rainfall=False, merge_with_SSA=False):
    
    df = load_df(loadpath=abs_df_path, filename=abs_filename)
    print("Length of abs dataset: "+str(len(df)))
    find_number_of_duplicate_indexs(df)

    df_fire_count = load_df(loadpath=fire_df_path, 
                            filename=fire_filename)
    print("Length of abs dataset: "+str(len(df_fire_count)))
    
    number_of_bins = np.logspace(0, 5, base=10, num=4)
    df_fire_count['bins'] = pd.cut(df_fire_count['fire_count'], number_of_bins)
    print(df_fire_count['bins'].unique())

    df_rainfall_extremes = load_df(loadpath=rainfall_df_path, filename=rainfall_filename)          
    print("Length of rainfall dataset: "+str(len(df_rainfall_extremes)))
    
    df_neph_LOD = load_neph_data()

    if merge_with_fire == True:
        df = mergedfs(df, df_fire_count)
    if merge_with_rainfall == True:
        df = mergedfs(df, df_rainfall_extremes)
    if merge_with_SSA == True:
        df = pd.merge(df, df_neph_LOD[['scat637']], left_index=True, right_index=True)
        df['w0'] = df['scat637']/(df['scat637']+df['abs637'])
        
    return df
    
def day_resample_add_ordinal(df, mean=True, median=False, start_date=None, date_col='date'):
    df[date_col] = df.index.date
    
    if start_date == None:
        start_date = df[date_col].iloc[0]
    
    print("start date inserted: "+str(start_date))
    df[date_col] = df.index.date
    if median == True:
        df_daily = df.resample('D').median()  
    if mean == True:
        df_daily = df.resample('D').mean()  
    df['timestamp'] = pd.to_datetime(df.index)
    df['ordinal'] = df['timestamp'].apply(lambda x: x.toordinal())
    df['ordinal'] = df['ordinal'] - df['ordinal'][0] + 1
    return df
    
    
def add_year_month_ordinal(df):   
    df['timestamp'] = pd.to_datetime(df.index)
    df['day'] = df['timestamp'].dt.day
    df['month'] = df['timestamp'].dt.month
    df['year'] = df['timestamp'].dt.year    
    df['year_num'] = df['year'] - df['year'].iloc[0]
    df['month_ordinal'] = df['year_num']*12 + df['month']
    df['year_num'] = df['year_num'] + 1
    return df
    
# def create_month_season_numbers(df, full_season_to_season_num=None):
    # start_year = df.index.year[0]
    # end_year = df.index.year[-1]
    # number_years = end_year - start_year #len(df.index.year.unique())+1        
    # df.loc[:,'month_num'] = df.index.month
    # df.loc[:,'year'] = df.index.year        
    # df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    # df['season_name'] = df['season_abb'].map(abb_to_name)      
    # df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    #print("Note: the slow build-up season crosses over two years as it goes from October-January, so the year corresponds to previous year")
    # df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    # seasons = df.season_abb_year.unique()
    #print("Number of unique seasons: "+str(len(seasons)))    
    # seasons_num = np.arange(1,len(seasons)+1,1)
    # season_to_season_num = dict(zip(seasons, seasons_num))
    # df.loc[:,'season_ordinal'] = df['season_abb_year'].map(season_to_season_num)    
    # df = df.sort_index()
    # return df
    
def create_month_season_numbers(df, full_season_to_season_num=None):
    start_year = df.index.year[0]
    end_year = df.index.year[-1]
    number_years = end_year - start_year + 1 #len(df.index.year.unique())+1        
    df.loc[:,'month_num'] = df.index.month
    df.loc[:,'year'] = df.index.year        
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df['season_name'] = df['season_abb'].map(abb_to_name)      
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    print("Note: the slow build-up season crosses over two years as it goes from October-January, so the year corresponds to previous year")
    df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    seasons = df.season_abb_year.unique()
    print("Number of unique seasons: "+str(len(seasons)))   
    seasons_num = np.arange(1,len(seasons)+1,1)
    season_to_season_num = dict(zip(seasons, seasons_num))
    first_season = get_first_season(df)
    if full_season_to_season_num is None:        
        full_season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
    if full_season_to_season_num is not None:
        print("full_season_to_season_num given")
        full_season_to_season_num = full_season_to_season_num.copy()    
    df.loc[:,'season_ordinal'] = df['season_abb_year'].map(full_season_to_season_num)    
    df = df.sort_index()
    return df
    
def prepare_data(df, dict_season_num_to_season=None, remove_indexes=None):    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    df = day_resample_add_ordinal(df)
    df = add_year_month_ordinal(df)
    
    first_month = df['month'].iloc[0]

    if dict_season_num_to_season is None:
        start_year = df.index.year[0]
        print("start year: "+str(start_year))
        end_year = df.index.year[-1]
        print("end year: "+str(end_year))        
        number_years = end_year - start_year #len(df.index.year.unique())
        print("number years: "+str(number_years))   
        
        if first_month in [2,3,4,5]: #FMAM
            first_season = 'AHZ' #Arctic HAZE
        if first_month in [6,7,8,9]: #JJAS
            first_season = 'SUM' #SUMMER
        if first_month in [10,11,12,1]:
            first_season = 'SBU' 

        print("first season: "+str(first_season))
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
        df = create_month_season_numbers(df, full_season_to_season_num=season_to_season_num)
        
    if dict_season_num_to_season is not None:
        #print(dict_season_num_to_season)
        season_num_to_season = dict_season_num_to_season.copy()
        season_to_season_num = reverse_dict(season_num_to_season)
        df = create_month_season_numbers(df, full_season_to_season_num=season_to_season_num)
        
        pass
    
    if remove_indexes is not None: 
        print("old length: "+str(len(df)))
        print("removing: "+str(len(remove_indexes))+" values")
        df_removed = df[~df.index.isin(remove_indexes)].copy()
        print("new length: "+str(len(df)))
        percentage_removed = len(remove_indexes)/len(df_removed)*100 
        print("removed: "+str(len(remove_indexes)/len(df_removed)*100)+' %')
        
        dict_season_to_percentage = {}
        sfs = sf(3)             

        dict_season_to_percentage['all_years'] = sfs.format(percentage_removed)
        for season_abb in df['season_abb'].unique():
            df_season = df[df['season_abb'] == season_abb].copy()
            percentage_removed_season = (len(df_season)-len(df_season[~df_season.index.isin(remove_indexes)]))/len(df_season)*100 
            dict_season_to_percentage[season_abb] = sfs.format(percentage_removed_season)
            #print(dict_season_to_percentage)
            
        df = df_removed.copy()
        
    if remove_indexes is not None:
        return df, dict_season_to_percentage
    return df
    
def last_n_years(df, nyears):
    df_last_nyrs = df[df.index.year >= int(df.index.year[-1]) - nyears]
    return df_last_nyrs
    
def resample(df, resolution, variable, mean=True, median=False):
    if median == True:
        if resolution == 'ordinal':                
            medians = df.groupby(by=resolution).median() 
            count = df.groupby('ordinal').count()
            valid_daily_index = count[count >= 6].dropna(how='all').index
            medians = medians.loc[medians.index.intersection(valid_daily_index)]
        else:
            medians = df.groupby(by=resolution).median()
        medians = medians[variable]
        medians = medians.dropna(how='all')
        averages = medians.copy()
        
    if mean == True: 
        if resolution == 'ordinal':
            means = df.groupby(by=resolution).mean()
            count = df.groupby('ordinal').count()
            valid_daily_index = count[count >= 6].dropna(how='all').index
            means = means.loc[means.index.intersection(valid_daily_index)]
        else:
            means = df.groupby(by=resolution).mean()
            
        means = means[variable]
        means = means.dropna(how='all')
        averages = means.copy()
    return averages
    
#########################TREND ANALYSIS ############################################################################################################################

def calculate_autocorr(y):
    nlags = min(int(10 * np.log10(len(y))), len(y) - 1)
    #print("Number of lags: "+str(nlags))
    autocorr = stattools.acf(y, fft=False, nlags=nlags)
    autocorr_coeff = autocorr[1]
    c = autocorr_coeff
    return c
    
def perform_TFPW(x,y):  
    res = stats.theilslopes(y, x, 0.90)
    b=res[0]    
    #print('trend: '+str(b))
    X_ = y - b*x
    r = calculate_autocorr(X_) #calculate autocorrelation of X_    
    Y_ = X_[1:] - r*X_[:-1] #the residual    
    Y = Y_ + b*x[1:] #trend bt and the residual Y_ are blended 
    return Y
    
def mk_test(x, alpha=0.05):
    n = len(x)

    #calculate S
    s = 0
    for k in range(n-1):
        for j in range(k+1, n):
            s += np.sign(x[j] - x[k])
            #print(s)

    #calculate the unique data
    unique_x = np.unique(x)
    g = len(unique_x)

    #calculate the var(s)
    if n == g:  #there is no tie
        var_s = (n*(n-1)*(2*n+5))/18
    else:  # there are some ties in data
        tp = np.zeros(unique_x.shape)
        for i in range(len(unique_x)):
            tp[i] = sum(x == unique_x[i])
        var_s = (n*(n-1)*(2*n+5) - np.sum(tp*(tp-1)*(2*tp+5)))/18

    if s > 0:
        z = (s - 1)/np.sqrt(var_s)
    elif s < 0:
        z = (s + 1)/np.sqrt(var_s)
    else: # s == 0:
        z = 0

    #calculate the p_value    
    p = 2*(1-norm.cdf(abs(z)))
    p = mpmath.mpf(p)
    
    h = abs(z) > norm.ppf(1-alpha/2)

    if (z < 0) and h:
        trend = 'decreasing'
    elif (z > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, z, s, var_s
    
def seasonal_mk_test(df_MK, alpha=0.05):

    variance_list = [] 
    for n in df_MK.length.values:
        variance = (n*(n-1)*(2*n+5))/18
        variance_list.append(variance)
    
    S_  = df_MK.s.sum()
    total_var = sum(variance_list) 
    #print("Total variance: "+str(total_var))
    
    if S_ > 0:   
        Zsk = (S_ - 1)/np.sqrt(total_var)
        #print("Zsk: "+str(Zsk))
        
    if S_ == 0:        
        Zsk = 0
        
    if S_ < 0:   
        Zsk = (S_ +1)/np.sqrt(total_var)
        #print("Zsk: "+str(Zsk))
        
    """then H_0 is rejected and H_a is accepted if 
    Z_SK > Z_1-a/2"""

    h = abs(Zsk) > norm.ppf(1-alpha/2)
    critical_Zsk = norm.ppf(1-alpha/2)
    p = 2*(1-norm.cdf(abs(Zsk)))
    p = mpmath.mpf(p)
    
    if (Zsk < 0) and h:
        trend = 'decreasing'
    elif (Zsk > 0) and h:
        trend = 'increasing'
    else:
        trend = 'no trend'

    return trend, h, p, Zsk, S_, total_var
    
def generate_MK_tables(df, variable, resolution, res_cols=['ordinal','month_ordinal','season_ordinal','season_abb'],
                       abb_to_name = {'DJF':'winter','MAM':'spring','JJA':'summer','SON':'autumn'}, 
                       season_abbs=['MAM','JJA','SON','DJF']):

    df_MK = pd.DataFrame(columns=['season','MK','s','var_s'])
    df = df[ [variable] + res_cols ]    
    df = df.dropna(how='all')

    season_list = []
    for season_abb in season_abbs:
        #print("Season: "+str(season_abb))        
        df_season =  df[df['season_abb'] == season_abb]
        #print("Median: "+str(df_season[variable].median()))
        #print("Length of season: "+str(len(df_season)))

        seaosnal_averages = resample(df_season, resolution, variable)   
        
        x = seaosnal_averages.index   
        y = seaosnal_averages.values  
        
        if resolution == 'ordinal':      
            TFPW = perform_TFPW(x,y)            
            res = stats.theilslopes(TFPW, x[1:], 0.90)
            y = TFPW

        MK = mk_test(y)
        length = len(y)
        s = MK[4]
        var_s = MK[5]
        df_MK = df_MK.append({'season':abb_to_name[season_abb], 'MK' : MK[0] , 'length' : length,'s':s,'var_s':var_s} , ignore_index=True)

    df_MK = df_MK.append({'season':'total', 'MK' : seasonal_mk_test(df_MK, alpha=0.05)[0] , 'length' : sum(df_MK.length.values),'s':seasonal_mk_test(df_MK, alpha=0.05)[4],
                         'var_s':seasonal_mk_test(df_MK, alpha=0.05)[4]} , ignore_index=True)    
    
    return df_MK
    
def produce_dict_table(df, variable='abs637', resolutions = ['ordinal','season_ordinal'],
                       outpath = r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\mk_tables',
                       save=False, save_variable_name='default'):
    dict_res_to_df_MK = {}

    for resolution in resolutions:
            #print("resolution: "+str(resolution))
            #print("Variable:"+str(variable))
            df_MK = generate_MK_tables(df, variable, resolution, 
                                       abb_to_name = {'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}, 
                                       season_abbs=['SBU', 'AHZ', 'SUM'])
            if save == True:
                print("saved as: "+str(outpath+'\\'+str(save_variable_name)+"_"+str(resolution)+".csv"))
                df_MK.to_csv(outpath+'\\'+str(save_variable_name)+"_"+str(resolution)+".csv",float_format='%.3f',index=False)
            dict_res_to_df_MK[resolution] = df_MK
    return dict_res_to_df_MK
    
#############TREND PLOTS############################################################################################################################################
  
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg", dpi=300):
    folders = glob.glob(path)
    #print(folders)
    if folder not in folders:
        #print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
    
  
def vars_for_best_fit(x,y,freq):
    x_array = np.array(x)
    y_array = np.array(y)    
    x_array = sm.add_constant(x_array)    
    model = sm.OLS(y_array, x_array, missing='drop')
    results = model.fit()

    p = results.params[0]
    m = results.params[1]
    
    se = results.bse[1]
        
    m3f = '{0:.3f}'.format(m*(freq))
    p3f = '{0:.3f}'.format(p) 
    se3f = '{0:.3f}'.format(se*(freq)) 
   
    return p,m,m3f,p3f,se3f
    
def create_season_ordinal_list(df, season_abb, season_abb_col='season_abb'):
    season_list = df.season_ordinal[df[season_abb_col] == season_abb].unique().tolist()
    return season_list
    
#remove the spines
def remove_spines(ax):
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left() 
    
def thickax(ax, fontsize=12, linewidth=4):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.rc('axes', linewidth=linewidth)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    ax.tick_params(direction='out', length=12, width=4, pad=12, bottom=True, top=False, left=True, right=False)    
    
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
    
    
def plot_subplots_for_optical_variables(df, season_num_to_season, variable='abs637', 
                                        season_to_shape_dict = {'AHZ':'o','SUM':'v','SBU':'s'},
                                        season_to_color = {'AHZ':'red','SUM':'black','SBU':'red'},
                                        ymin=-0.1, ymax = 2, xmin=None, xmax=None, tick_frequency=6, freq = 3, average='median',
                                        season_abbs=['AHZ','SUM','SBU'], letter='a)', 
                                        ylabel='$\sigma_{\mathrm{ap, 637 nm}}$',
                                        season=None, linecolour='r', xcoord=0.02, ycoord=1.15, xcoord_legend=0.90, 
                                        ycoord_legend=1.25, percentage_removed=0, 
                                        error_label=True, units='[Mm$^{-1}$]', first_season=None, last_season=None, 
                                        variable_average_label='$\sigma_{\mathrm{\overline{ap}}}$', 
                                        year_labels=None, ticks=None, alter_tick_labels=0, fs_cbarlabel = 20,
                                        fs_cbartick = 25, fs_ticks = 15, fs_annotate = 20, fs_label = 25,  lw=2,
                                        fs_legend = 20, mscale = 2, add_vertical_line=False, x_vertical='2002-04-01 12:00:00', 
                                        ytext_coord=.95, return_slope=False, restrict_trend=None, plot_LMS=True, 
                                        plot_medians=True, plot_means=False, display_average_type='median', 
                                        sigfigs=4, plot_TS=True, add_text=True, log_scale=False, ax=None):
     
    if season is not None:
        if display_average_type == 'median':
            display_average = df.loc[df['season_abb'] == season, variable].median()
        if display_average_type == 'mean':
            display_average = df.loc[df['season_abb'] == season, variable].mean()
    if season == None:
        if display_average_type == 'mean':
            display_average = df[variable].mean()
        if display_average_type == 'median':            
            display_average = df[variable].median()

    df_seasonal_averages = seasonal_averages_select_abb(df, variable, season_abb=season, 
                                             groupby_var='season_ordinal',
                                             season_abb_year_col = 'season_abb_year')
 
    if plot_means == True: 
        ax.plot(df_seasonal_averages['mid_datetime'], df_seasonal_averages['mean'], 'x', label='', ls=':', c='k', ms=5, lw=lw,
        alpha=.0) #use ax to plot datetime
    if plot_medians == True:        
        ax.plot(df_seasonal_averages['mid_datetime'], df_seasonal_averages['median'], 'x', label='', ls=':', c='k', ms=.1, lw=.1,
        alpha=0.0)
    
    ax2 = ax.twiny() #for trend i.e. integers
    ax2.set_xticks([]) #dont want these ticks    
    
    for num, season_abb in enumerate(season_abbs):      
        df_seasonal_averages = seasonal_averages(df, variable)     
        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]
       
        # if season is not None:
            # df_season = normalise_index(df_season)
            
        time_index = df_season.index
        quan_75 = df_season['q75']
        quan_25 = df_season['q25']
        median = df_season['median']
        mean = df_season['mean']
        std = df_season['std']
        
        fmt = season_to_shape_dict[season_abb]
        mfc = season_to_color[season_abb]     
        kwargs = dict(ecolor='k', capsize=6, elinewidth=2, linewidth=2, ms=10)
        
        if error_label == True:
            label = str(season_abb)
        if error_label == False:
            label = None
            
        if plot_means == True:                
            ax2.errorbar(time_index, mean,  ls='none', c='k',label=label,
                        ms=10, lw=3, fmt=fmt, mfc=mfc, markeredgewidth=1.5) #means denoted by x
            # ax2.errorbar(time_index, mean, yerr=[mean-std, std-mean], fmt=fmt, mfc=mfc, 
                        # , ls='none', c='k', markeredgewidth=1.5, **kwargs) 
                        
        if plot_medians == True:
            ax2.errorbar(time_index, median, yerr=[median-quan_25, quan_75-median], fmt=fmt, mfc=mfc, 
                        label=label, ls='none', c='k', markeredgewidth=1.5, **kwargs)  
            ax2.errorbar(time_index, mean,  ls='', c='k',label=None,
                         ms=10, lw=3, fmt='x', mfc='k') #means denoted by x
                    
    if season == None:    
        x = df_seasonal_averages[average].index
        y = df_seasonal_averages[average]
        
        ax2.plot(x, y, label='', ls=':', c='k', ms=.1, lw=2,
            alpha=0.6)
            
    if season is not None:
        season_ordinal_list = create_season_ordinal_list(df, season)
        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]
        df_season = normalise_index(df_season)        
        x = df_season.index  
        y = df_season[average] 
        
        ax2.plot(x, y, label='', ls=':', c='k', ms=.1, lw=2,
            alpha=0.6)
        
    season_to_season_num = reverse_dict(season_num_to_season)    
    if restrict_trend is not None:
        restrict_to = season_to_season_num[restrict_trend]
        x = x[:restrict_to]
        y = y[:restrict_to]
    
    #LMS
    if plot_LMS == True:        
        p,m,m3f,p3f,se3f = vars_for_best_fit(x,y, freq)
        x_fit = x
        y_fit = m*x+p
        m = m*(freq)
        ax2.plot(x_fit, y_fit, ls='-', c=linecolour, 
        label='LMS: y = '+str(m3f)+'$\,\mathdefault{x}$ +'+str(p3f),lw=2) 

    #THEIL SEN SLOPE   
    res = stats.theilslopes(y, x, 0.90)
    Theil_slope = (res[1] + res[0] * x)
    lo_slope = (res[1] + res[2] * x)
    up_slope = (res[1] + res[3] * x)

    theil_m=res[0]
    theil_m = float(theil_m)*(freq)
    lo_m = float(res[2])*(freq)
    up_m = float(res[3])*(freq)
    
    #significant figures
    sfs = sf(sigfigs)           
    intecept=sfs.format(res[1])
    theil_m=sfs.format(theil_m)
    lo_m=sfs.format(lo_m)
    up_m=sfs.format(up_m) 

    if plot_TS == True:
        ax2.plot(x, Theil_slope, ls='--', lw=2, c=linecolour,
                label=str('TS: y = '+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept)))
        ax2.fill_between(x, up_slope, lo_slope, alpha=0.15, color=linecolour)
    
    remove_spines(ax)
    remove_spines(ax2)
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator()) #ax.xaxis.set_major_locator(mdates.YearLocator(base = 1, month = 1, day = 1))    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    
    ax.set_ylabel(ylabel+' '+units, fontsize=fs_label)   
    #ax.minorticks_on()
    
    # ax.tick_params(axis='both', which='major', labelsize=fs_ticks, direction='in', length=6, width=2,
                   # grid_alpha=0.5)
    # ax.tick_params(axis='both', which='minor', labelsize=fs_ticks, direction='in', length=3, width=2,
                   # grid_alpha=0.5)

    ax2.text(0.01, 1.25, letter, transform=ax.transAxes, size=fs_annotate)    
    display_average=sfs.format(display_average)
        
    #add removed percentage
    if add_text == True:
        if percentage_removed == 0:        
            relative_trend = 100*(float(theil_m)/float(display_average)) 
            relative_trend = sfs.format(relative_trend)
            ax.text(xcoord, ycoord, 
                    str(variable_average_label)+" : "+str(display_average) +' '+ str(units)+'\nrel. trend: '+str(relative_trend)+' [%yr$^{-1}$]', 
                    transform=ax.transAxes, size=fs_annotate, weight='bold',
                    color=linecolour)
        if percentage_removed != 0:
            ax.text(xcoord, ycoord, 
                    str(variable_average_label)+" : "+str(display_average)+' '+str(units) + ' (removed '+str(percentage_removed)+'%)', 
                    transform=ax.transAxes, size=fs_annotate, weight='bold',
                    color=linecolour)

    legend = ax2.legend(numpoints = 1,loc='upper left',bbox_to_anchor=(xcoord_legend, ycoord_legend),
              frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
    legend.get_title().set_fontsize(fs_legend)
    
    if add_vertical_line == True:
        x_vertical=pd.to_datetime(x_vertical)
        trans = ax.get_xaxis_transform()
        ax.axvline(x_vertical, ls=':', c='k')
        plt.text(x_vertical, ytext_coord,'FNL/GDAS', transform=trans, fontsize=20)

    if variable=='BC_GFED':
        ax.axhline(y=10**3, ls=':', c='k', alpha=.8, lw=1)

    if log_scale == True:
        ax.set_yscale('log')
    ax.set_ylim(ymin, ymax)

    #ax.set_xlim(df_seasonal_averages['mid_datetime'].values[0], df_seasonal_averages['mid_datetime'].values[-1])
    #ax2.set_xlim(df_seasonal_averages.index[0], df_seasonal_averages.index[-1])
        
    #thickax(ax)
    fancy(ax, fontsize=20, spines=['bottom','left'], alpha=0.)
    ax.tick_params(axis='x', which='minor', direction='out', length=5, width=2)
    if return_slope == True:
        return ax, res #theil_m   
    return ax
    
def plot_subplots_for_optical_variables_v2(df, season_num_to_season, variable='abs637', 
                                        season_to_shape_dict = {'AHZ':'o','SUM':'v','SBU':'s'},
                                        season_to_color = {'AHZ':'red','SUM':'black','SBU':'red'},
                                        ymin=-0.1, ymax = 2, xmin=None, xmax=None, tick_frequency=6, freq = 3, average='median',
                                        season_abbs=['AHZ','SUM','SBU'], letter='a)', 
                                        ylabel='$\sigma_{\mathrm{ap, 637 nm}}$',
                                        season=None, linecolour='r', xcoord=0.02, ycoord=1.15, xcoord_legend=0.90, 
                                        ycoord_legend=1.25, percentage_removed=0, 
                                        error_label=True, units='[Mm$^{-1}$]', first_season=None, last_season=None, 
                                        variable_average_label='$\sigma_{\mathrm{\overline{ap}}}$', 
                                        year_labels=None, ticks=None, alter_tick_labels=0, fs_cbarlabel = 20,
                                        fs_cbartick = 25, fs_ticks = 15, fs_annotate = 20, fs_label = 25, lw=2,
                                        fs_legend = 20, mscale = 2, add_vertical_line=False, x_vertical='2002-04-01 12:00:00', 
                                        ytext_coord=.95, return_slope=False, restrict_trend=None, plot_LMS=True, 
                                        plot_medians=True, plot_means=False, display_average_type='median', ax=None):
     
    if season is not None:
        if display_average_type == 'median':
            display_average = df.loc[df['season_abb'] == season, variable].median()
        if display_average_type == 'mean':
            display_average = df.loc[df['season_abb'] == season, variable].mean()
    if season == None:
        if display_average_type == 'mean':
            display_average = df[variable].mean()
        if display_average_type == 'median':            
            display_average = df[variable].median()

    df_seasonal_averages = seasonal_averages_select_abb(df, variable, season_abb=season, 
                                             groupby_var='season_ordinal',
                                             season_abb_year_col = 'season_abb_year')

    if plot_means == True: 
        ax.plot(df_seasonal_averages['mid_datetime'], df_seasonal_averages['mean'], 'x', label='', ls=':', c='k', ms=5, lw=lw) #use ax to plot datetime
    if plot_medians == True:        
        ax.plot(df_seasonal_averages['mid_datetime'], df_seasonal_averages['median'], 'x', label='', ls=':', c='k', ms=5, lw=lw)
    ax2 = ax.twiny() #for trend i.e. integers
    ax2.set_xticks([]) #dont want these ticks

    for num, season_abb in enumerate(season_abbs):      
        df_seasonal_averages = seasonal_averages(df, variable)          
        print("season: ")

        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]                
        if season is not None:
            df_season = normalise_index(df_season)
            
        time_index = df_season.index
        quan_75 = df_season['q75']
        quan_25 = df_season['q25']
        median = df_season['median']
        mean = df_season['mean']
        
        fmt = season_to_shape_dict[season_abb]
        mfc = season_to_color[season_abb]     
        kwargs = dict(ecolor='k', capsize=5, elinewidth=0.5, linewidth=2, ms=10)
        
        if error_label == True:
            label = str(season_abb)
        if error_label == False:
            label = None

        if plot_medians == True:
            ax2.errorbar(time_index, median, yerr=[median-quan_25, quan_75-median], fmt=fmt, mfc=mfc, 
                        label=label, ls='none', c='k', **kwargs)  
            ax2.errorbar(time_index, mean,  ls='none', c='k',label=None,
                         ms=10, lw=lw, fmt='x', mfc='k') #means denoted by x
        if plot_means == True:                
            ax2.errorbar(time_index, mean,  ls='none', c='k',label=label,
                        ms=10, lw=lw, fmt=fmt, mfc=mfc) #means denoted by x
                    
    if season == None:    
        x = df_seasonal_averages[average].index
        y = df_seasonal_averages[average]
    if season is not None:
        season_ordinal_list = create_season_ordinal_list(df, season)
        df_season = df_seasonal_averages[df_seasonal_averages.season_abb == season_abb]
        df_season = normalise_index(df_season)        
        x = df_season.index  
        y = df_season[average] 
         
    ax3 = ax.twiny() #for trend i.e. integers
    
    season_to_season_num = reverse_dict(season_num_to_season)    
    if restrict_trend is not None:
        start_restrict = restrict_trend[0]
        end_restrict = restrict_trend[-1]
        start_restrict_season = season_to_season_num[start_restrict]
        end_restrict_season = season_to_season_num[end_restrict]
        x = x[start_restrict_season:end_restrict_season]
        y = y[start_restrict_season:end_restrict_season]
        x_norm = x - x[0]
    
    #LMS
    if plot_LMS == True:        
        p,m,m3f,p3f,se3f = vars_for_best_fit(x_norm, y, freq)
        y_fit = m*x_norm+p
        m = m*(freq)
        ax2.plot(x, y_fit, ls='-', c=linecolour, label='LMS: y = '+str(m3f)+'$\,\mathdefault{x}$ +'+str(p3f),lw=1) 

    #THEIL SEN SLOPE   
    res = stats.theilslopes(y, x_norm, 0.95)
    Theil_slope = (res[1] + res[0] * x_norm)
    lo_slope = (res[1] + res[2] * x_norm)
    up_slope = (res[1] + res[3] * x_norm)

    theil_m=res[0]
    theil_m = float(theil_m)*(freq)
    lo_m = float(res[2])*(freq)
    up_m = float(res[3])*(freq)
    
    #significant figures
    sfs = sf(4)           
    intecept=sfs.format(res[1])
    theil_m=sfs.format(theil_m)
    lo_m=sfs.format(lo_m)
    up_m=sfs.format(up_m) 

    ax2.plot(x, Theil_slope, ls='--', lw=1, c=linecolour,
            label=str('TS: y = '+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept)))
    ax2.fill_between(x, up_slope, lo_slope, alpha=0.15, color=linecolour)
    
    remove_spines(ax)
    remove_spines(ax2)
    
    ax.xaxis.set_major_locator(mdates.AutoDateLocator()) #ax.xaxis.set_major_locator(mdates.YearLocator(base = 1, month = 1, day = 1))    
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))    
    ax.set_ylabel(ylabel+' '+units, fontsize=fs_label)   
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks, direction='in', length=6, width=2,
                   grid_alpha=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks, direction='in', length=3, width=2,
                   grid_alpha=0.5)

    ax2.text(-0.1, 1.3, letter, transform=ax.transAxes, size=fs_annotate)    
    display_average=sfs.format(display_average)
        
    #add removed percentage
    if percentage_removed == 0:        
        relative_trend = 100*(float(theil_m)/float(display_average)) 
        relative_trend = sfs.format(relative_trend)
        ax.text(xcoord, ycoord, 
                str(variable_average_label)+" : "+str(display_average) +' '+ str(units)+' rel. trend: '+str(relative_trend)+' [%yr$^{-1}$]', 
                transform=ax.transAxes, size=fs_annotate, weight='bold',
                color=linecolour)
    if percentage_removed != 0:
        ax.text(xcoord, ycoord, 
                str(variable_average_label)+" : "+str(display_average)+' '+str(units) + ' (removed '+str(percentage_removed)+'%)', 
                transform=ax.transAxes, size=fs_annotate, weight='bold',
                color=linecolour)

    legend = ax2.legend(numpoints = 1,loc='upper left',bbox_to_anchor=(xcoord_legend, ycoord_legend),
              frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
    legend.get_title().set_fontsize(fs_legend)
    
    if add_vertical_line == True:
        x_vertical=pd.to_datetime(x_vertical)
        trans = ax.get_xaxis_transform()
        ax.axvline(x_vertical, ls=':', c='k')
        plt.text(x_vertical, ytext_coord,'FNL/GDAS', transform=trans, fontsize=20)
    
    ax.set_ylim(ymin, ymax)
    ax.axhline(y=0, ls=':', c='k', alpha=0.2, lw=1)
    thickax(ax)
    if return_slope == True:
        return ax, res #theil_m   
    return ax    

    
def plot_trend(df, season_num_to_season=None, variable='abs637', title='', ylabel='$\sigma_{\mathrm{ap, 637 nm}}$', ymin=0.65, ymax=1.,
               xmin=None, xmax=None, variable_average_label='$\sigma_{\mathrm{\overline{ap}}}$', units='[Mm$^{-1}$]',
               xcoord=0.02, ycoord=1.15, xcoord_legend=0.90, ycoord_legend=1.25,
               alter_tick_labels=0, linecolour='r', restrict_trend=None, plot_LMS=True, 
               plot_medians=True, plot_means=False, find_MK=False, average='median', display_average_type='median',
               sigfigs=4, fs_ticks=15, season_abbs=['AHZ','SBU','SUM'], season=None, log_scale=False):
    
    if find_MK == True:
        dict_res_to_df_MK =  produce_dict_table(df, variable=variable, resolutions = ['ordinal','season_ordinal'])    
    
    if season_num_to_season is None:
        start_year = df.index.year[0]
        print("start year: "+str(start_year))
        end_year = df.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))        

        first_month = df['month'].iloc[0]
        if first_month in [2,3,4,5]: #FMAM
            first_season = 'AHZ' #Arctic HAZE
        if first_month in [6,7,8,9]: #JJAS
            first_season = 'SUM' #SUMMER
        if first_month in [10,11,12,1]:
            first_season = 'SBU' 
            start_year = start_year - 1                 
            
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
        #print('seasons dict: ')
            
    nrows = 1; ncols = 1
    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)
    plt.suptitle(title, y=0.95, fontsize=20);    
    ax = plt.subplot(nrows, ncols, 1) 
    
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=season_abbs, letter='', 
                                        ylabel=ylabel, ymin=ymin, ymax = ymax, xmin=xmin, xmax=xmax, variable_average_label=variable_average_label,
                                        units=units,  xcoord=xcoord, ycoord=ycoord, linecolour=linecolour,
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, alter_tick_labels=0, 
                                        restrict_trend=restrict_trend, plot_LMS=plot_LMS, plot_medians=plot_medians, 
                                        plot_means=plot_means, average=average, display_average_type=display_average_type, 
                                        sigfigs=sigfigs, fs_ticks=fs_ticks, season=season, log_scale=log_scale,
                                        ax=ax)
                                       
    # ax.vlines(pd.to_datetime('2015-11-01'), ymin=ymin, ymax=1, ls=':', color='k')
    # ax.vlines(pd.to_datetime('2019-11-01'), ymin=ymin, ymax=1, ls=':', color='k')
    # ax.axvspan(pd.to_datetime('2015-11-01'), pd.to_datetime('2019-11-01'), alpha=0.1, color='k')
    
    # ax.vlines(pd.to_datetime('2016-01-01'), ymin=ymin, ymax=1, ls=':', color='k')
    # ax.vlines(pd.to_datetime('2022-05-31'), ymin=ymin, ymax=1, ls=':', color='k')
    # ax.axvspan(pd.to_datetime('2016-01-01'), pd.to_datetime('2022-05-31'), alpha=0.1, color='k')
    
    plt.show() 
    return fig
    
########GROWTH RATES################################################################################################################################################

#ylabel='Annual growth rate of $\sigma_{\mathrm{ap}}$ [Mm$^{-1}$$yr^{-1}$]'
def plot_growth_rate(df_abs, season_abb=None, var='abs637', fs_ticks=10, mean=True, median=False,
                     max_value=None, ylabel='Annual growth rate of $\sigma_{\mathrm{ap}}$ [Mm$^{-1}$$yr^{-1}$]'):
    fig, ax = plt.subplots(figsize=(10,5))
    
    if season_abb is not None:
        df_season = df_abs[df_abs.season_abb == season_abb]    
    if season_abb == None:
        df_season = df_abs.copy()    
       
    if median == True:
        df_growth_rate = df_season.groupby('year').median().diff()
    if mean == True:
        df_growth_rate = df_season.groupby('year').mean().diff()   
    
    if max_value is None:
        max_value = df_growth_rate[var].max()
        min_value = df_growth_rate[var].min()  
        max_value = max(abs(max_value), abs(min_value))  
        
    df_growth_rate.index = df_growth_rate.index.astype(int)    
    start_year = df_growth_rate.index[0]
    end_year = df_growth_rate.index[-1]    
    years = np.arange(start_year, end_year+1, 1)    
    df_growth_rate = df_growth_rate.reindex(years,fill_value=np.nan)   
    ax.bar(df_growth_rate.index, df_growth_rate[var], align='center', alpha=0.4)
    years = list(df_growth_rate.index)
    ax.set_xticks(years)
    
    if season_abb is not None:
        ax.set_title(str(season_abb), loc='left')
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax.tick_params(axis='y', which='major', labelsize=fs_ticks, direction='in', length=6, width=2,
                   grid_alpha=0.5)
    ax.tick_params(axis='y', which='minor', labelsize=fs_ticks, direction='in', length=3, width=2,
                   grid_alpha=0.5)
    ax.xaxis.set_tick_params(which='both', bottom=False)
    remove_spines(ax)
    ax.set_ylim(-max_value*1.1, max_value*1.1)    
    plt.tick_params(axis='x', which='minor', bottom=False, top=False, labelbottom=False)    
    plt.show()
    return fig
    
#############TREND TABLE############################################################################################################################################

def create_trend_table(df, last_year, variable='abs637', freq = 3, average='median',
                       season_abbs=['AHZ','SUM','SBU'], season=None, 
                       percentage_removed=0, first_season=None, last_season=None, 
                       ticks=None, ss_TS=.9, years_to_restrict=4, number_sf=4):
    print("Average used: "+str(average))
    df = df[df.index.year <= last_year]  
    
    print(df.index[0])
    print(df.index[-1])
    
    dict_stats = {}    
    years = sorted(list(df.index.year.unique())) 
    print(years)
    number_of_years = len(years)
    print("number of years: "+str(number_of_years))
    years = years[:-int(years_to_restrict)] #not the last 5 years as those trends do not make sense 
    for year in years:
        print("minimum"+str(year))
        df_year = df[df.index.year >= year]        
        first_year = df_year['year'].iloc[0]              
        if last_year - first_year >= int(years_to_restrict):
            last_year = df_year['year'].iloc[-1]            
            seasons = df_year.season_ordinal.unique() #seasons      
            df_var = df_year.copy() #copy
            #average for the entire dataset and seasons
            if season is not None:
                display_average = df_year.loc[df_year['season_abb'] == season, variable].mean()
            if season == None:
                display_average = df_year[variable].mean()

            count = df_var.groupby(['season_ordinal']).size()
            quan_75s = df_var.groupby('season_ordinal')[variable].quantile(0.75)
            quan_25s = df_var.groupby('season_ordinal')[variable].quantile(0.25)
            medians = df_var.groupby('season_ordinal')[variable].median()  
            means = df_var.groupby('season_ordinal')[variable].mean()   

            for num, season_abb in enumerate(season_abbs):  
                season_ordinal_list = create_season_ordinal_list(df_var, season_abb)

                quan_75 = quan_75s[season_ordinal_list]
                quan_25 = quan_25s[season_ordinal_list]
                median = medians[season_ordinal_list] 
                mean = means[season_ordinal_list]

            if average == 'mean':
                x = means.index
                y = means    
            if average == 'median':
                x = medians.index
                y = medians

            if season is not None:                
                season_ordinal_list = create_season_ordinal_list(df_var, season)
                if average == 'mean':
                    x = means[season_ordinal_list].index
                    y = means[season_ordinal_list]     
                if average == 'median':
                    x = medians[season_ordinal_list].index
                    y = medians[season_ordinal_list] 

            p,m,m3f,p3f,se3f = vars_for_best_fit(x,y, freq)
            x_fit = x
            y_fit = m*x+p
            m = m*(freq)
            res = stats.theilslopes(y, x, ss_TS)
            Theil_slope = (res[1] + res[0] * x)
            lo_slope = (res[1] + res[2] * x)
            up_slope = (res[1] + res[3] * x)

            theil_m=res[0]
            theil_m = float(theil_m)*(freq)
            lo_m = float(res[2])*(freq)
            up_m = float(res[3])*(freq)
            sfs = sf(number_sf)              
            intecept=sfs.format(res[1])
            theil_m=sfs.format(theil_m)
            lo_m=sfs.format(lo_m)
            up_m=sfs.format(up_m) 
            years_period = str(first_year)+'_'+str(last_year)
        if last_year - first_year < int(years_to_restrict):
            theil_m = np.nan            
        dict_stats[years_period] = theil_m    
    return dict_stats
    
def dict_lastyear_to_dict_stats(df_abs):
    last_years = sorted(list(df_abs.index.year.unique()))[5:]

    dict_lastyear_to_dict_stats = {}

    for last_year in last_years:
        #print(last_year)
        dict_stats = create_trend_table(df_abs, last_year, variable='abs637', freq = 3, average='mean',
                               season_abbs=['AHZ','SUM','SBU'], season=None, 
                               percentage_removed=0, first_season=None, last_season=None, 
                               ticks=None)
        #print(dict_stats)
        dict_lastyear_to_dict_stats[last_year] = dict_stats
        df_trends = pd.DataFrame.from_dict(dict_lastyear_to_dict_stats)
        df_trends = df_trends.apply(pd.to_numeric)
        return df_trends
        
def turn_dict_to_df(dict_stats, index_name='abs'):
    df_trends = pd.DataFrame([dict_stats])
    df_trends = df_trends.apply(pd.to_numeric)
    df_trends.index.name = 'parameters'
    df_trends.index = [str(index_name)]    
    return df_trends
    
def heatmap_trends(df_trends, linewidths = 2, linecolor = "white", value_max=0.015, yticklabel=r'$\sigma_{\mathrm{ap}}$',
                   label='Trend [Mm$^{-1}$yr$^{-1}$]', colourbar_y=-0.7, fmt='0.4f'):
    names = [x.replace('_','-') for x in list(df_trends.columns)]
    fig, ax = plt.subplots(figsize=(12,1))
    cbar_ax = fig.add_axes([.11, colourbar_y, .8, .3]) #x,y, length, width    
    ax = sns.heatmap(df_trends, linewidths=linewidths, linecolor=linecolor, cmap='bwr', vmin=-float(value_max), vmax=float(value_max),
                     cbar_kws={ "orientation": "horizontal", 'label': 'Trend [Mm$^{-1}$yr$^{-1}$]'}, 
                     xticklabels=names, yticklabels=[yticklabel], fmt=fmt, annot=True, ax=ax,cbar_ax=cbar_ax)
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=15)
    ax.set_xticklabels(ax.get_xmajorticklabels(), fontsize = 10)
    ax.set_yticklabels(ax.get_ymajorticklabels(), fontsize = 15, rotation=360)
    cbar.set_label(label=label,size=15) #,weight='bold')
    plt.show()    
    return fig
    
def plot_seasonal_trends(df, season_num_to_season=None, variable='abs637', title='', 
                        ylabel='$\sigma_{\mathrm{ap, 637 nm}}$', ymin=-0.1, ymax=1.76,           
                        variable_average_label='$\sigma_{\mathrm{\overline{ap}}}$', units='[Mm$^{-1}$]',
                        xcoord=0.02, ycoord=1.15, xcoord_legend=0.50, ycoord_legend=1.15,
                        alter_tick_labels=0, nrows = 4, ncols = 1, start_year=None):
    
    dict_res_to_df_MK =  produce_dict_table(df, variable=variable, resolutions = ['ordinal','season_ordinal'])    
    print("Dictionary of resolution and MK: ")
    print(dict_res_to_df_MK)
    
    if season_num_to_season is None:
        
        if start_year is None:
            start_year = df.index.year[0]        
            print("start year: "+str(start_year))

        end_year = df.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))        

        first_month = df['month'].iloc[0]
        if first_month in [2,3,4,5]: #FMAM
            first_season = 'AHZ' #Arctic HAZE
        if first_month in [6,7,8,9]: #JJAS
            first_season = 'SUM' #SUMMER
        if first_month in [10,11,12,1]:
            first_season = 'SBU' 
                   
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)

    fig, axs = plt.subplots(num=None, figsize=(40, 6*nrows), sharex=True, sharey=True)
    plt.suptitle(title, y=0.95, fontsize=20);    
    ax = plt.subplot(nrows, ncols, 1) 
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=['AHZ','SBU','SUM'], letter='a)', 
                                        ylabel=ylabel, ymin=ymin, ymax=ymax, 
                                        variable_average_label=variable_average_label,
                                        units=units,  xcoord=xcoord, ycoord=ycoord, 
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend,
                                        alter_tick_labels=alter_tick_labels, ax=ax)
    ax = plt.subplot(nrows, ncols, 2) 
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=['AHZ'], letter='b)', season='AHZ',
                                        ylabel=ylabel, ymin=ymin, ymax=ymax, 
                                        variable_average_label=variable_average_label,
                                        units=units,  xcoord=xcoord, ycoord=ycoord, 
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend,
                                        alter_tick_labels=alter_tick_labels, ax=ax)
    ax = plt.subplot(nrows, ncols, 3)
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=['SBU'], letter='c)', season='SBU', ymax=ymax, 
                                        ylabel=ylabel, ymin=ymin, 
                                        variable_average_label=variable_average_label,
                                        units=units,  xcoord=xcoord, ycoord=ycoord, 
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, 
                                        alter_tick_labels=alter_tick_labels, ax=ax)
    ax = plt.subplot(nrows, ncols, 4) 
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=['SUM'], letter='d)', season='SUM', ymax=ymax,
                                        ylabel=ylabel, ymin=ymin, 
                                        variable_average_label=variable_average_label,
                                        units=units,  xcoord=xcoord, ycoord=ycoord, 
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, 
                                        alter_tick_labels=alter_tick_labels, ax=ax)    
    plt.subplots_adjust(hspace = .5)
    plt.tight_layout()
    plt.show()
    return fig
    
def superimposed(df, df_fires_removed, high_fire_episodes, season_num_to_season=None, nrows = 4, ncols = 1, title='',
                 xcoord_legend=0.9, ycoord_legend=1.25):
            
    if season_num_to_season is None:
        start_year = df.index.year[0]
        print("start year: "+str(start_year))
        end_year = df.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))        

        first_season = get_first_season(df)        
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
        
    dict_season_to_percentage = prepare_data(df, remove_indexes=high_fire_episodes)[1]

    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)
    
    plt.suptitle(title, y=0.95, fontsize=20)

    ax = plt.subplot(nrows, ncols, 1) 
    plot_subplots_for_optical_variables(df, season_num_to_season, season_abbs=['AHZ','SBU','SUM'], letter='a)', ax=ax)    
    plot_subplots_for_optical_variables(df_fires_removed,  season_num_to_season, season_abbs=['AHZ','SBU','SUM'], letter='a)', linecolour='b', 
                                        ycoord=1.25, percentage_removed=dict_season_to_percentage['all_years'], 
                                        error_label=False, xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, ax=ax)
    ax = plt.subplot(nrows, ncols, 2) 
    plot_subplots_for_optical_variables(df, season_num_to_season, season_abbs=['AHZ'], letter='b)', season='AHZ', ax=ax)
    plot_subplots_for_optical_variables(df_fires_removed, season_num_to_season, season_abbs=['AHZ'], letter='b)',  linecolour='b', season='AHZ', 
                                        ycoord=1.25, percentage_removed=dict_season_to_percentage['AHZ'], 
                                        error_label=False, xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, ax=ax)
    ax = plt.subplot(nrows, ncols, 3)
    plot_subplots_for_optical_variables(df, season_num_to_season, season_abbs=['SBU'], letter='c)', season='SBU', ymax=1, ax=ax)
    plot_subplots_for_optical_variables(df_fires_removed, season_num_to_season, season_abbs=['SBU'], letter='c)',  linecolour='b', season='SBU', 
                                        ymax=1, ycoord=1.25, percentage_removed=dict_season_to_percentage['SBU'],
                                        error_label=False, xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, ax=ax)
    ax = plt.subplot(nrows, ncols, 4) 
    plot_subplots_for_optical_variables(df, season_num_to_season, season_abbs=['SUM'], letter='d)', season='SUM', ymax=0.5, ax=ax)
    plot_subplots_for_optical_variables(df_fires_removed, season_num_to_season, season_abbs=['SUM'], letter='d)',  linecolour='b', season='SUM', 
                                        ymax=0.5, ycoord=1.25, percentage_removed=dict_season_to_percentage['SUM'], 
                                        error_label=False, xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, ax=ax)
    plt.subplots_adjust(hspace = .5)
    #fig.tight_layout()
    plt.show()
    return fig
    
def random_test(df, fraction_removed = 1.62/100, season = 'SUM'):
    df_season = df[df['season_abb'] == season]
    print("size of df season: "+str(len(df_season)))
    fraction_of_random_rows = df_season.sample(frac=fraction_removed)
    print("size of data removed: "+str(len(fraction_of_random_rows)))
    df_random_removed = df_season[~df_season.index.isin(fraction_of_random_rows.index)]
    print("size of data after removal: "+str(len(df_random_removed)))
    print("random mean with "+str(fraction_removed)+" removed: "+str(df_random_removed['abs637'].mean())+' [Mm$^{-1}$]')
    
def get_df_trends(df, variable='abs637', index_name='abs', end_year=2020, average='median'):
    dict_stats = create_trend_table(df, end_year, variable, freq=3, average=average,
                       season_abbs=['AHZ','SUM','SBU'], season=None, 
                       percentage_removed=0, first_season=None, last_season=None, 
                       ticks=None)  
    print(dict_stats)                   
    df_trends = pd.DataFrame([dict_stats])
    df_trends = df_trends.apply(pd.to_numeric)
    df_trends.index.name = 'parameters'
    df_trends.index = [index_name]    
    if end_year != 2020: #can't end the trend here
        yrs = 2020 - int(end_year) 
        #print("yrs missing: "+str(yrs))
        for yr in np.arange(end_year-3, 2020-3, 1):  #np.arange(end_year-4+1, 2020-4+1, 1):
            print(yr)
            df_trends[str(yr)] = np.nan  
    print(df_trends)
    return df_trends
    
def heatmap_trends_ax(df_trends, yticklabel, colourbarlabel, linewidths = 2, linecolor = "white",
                  value_max=0.3, cbar_ax=None, names=None,
                  cbar_kws={ "orientation": "horizontal", 'label': 'Trend [Mm$^{-1}$yr$^{-1}$]'}, 
                  colourbar_y=-1.3, fs_ticklabel=10, savename='trend_atp', ax=None):
    if ax is None:
        fig, axs = plt.subplots(figsize=(12,1))
        if cbar_ax is None:
            cbar_ax = fig.add_axes([.11, colourbar_y, .8, .3]) #x,y, length, width
            cbar_kws=cbar_kws
    if ax is not None:        
        pass
    if names is None:
        names = [x.replace('_','-') for x in list(df_trends.columns)]

    axs = sns.heatmap(df_trends, linewidths=linewidths, linecolor=linecolor, cmap='bwr', 
                     vmin=-float(value_max),vmax=float(value_max), 
                     cbar_kws=cbar_kws, xticklabels=names, 
                     yticklabels=[yticklabel], annot=True, cbar_ax=cbar_ax, ax=ax)
                     
    axs.set_xticklabels(axs.get_xmajorticklabels(), fontsize=fs_ticklabel)
    axs.set_yticklabels(axs.get_ymajorticklabels(), fontsize=fs_ticklabel, rotation=360)    
    if cbar_ax is None:
        cbar = axs.collections[0].colorbar
        cbar.axs.tick_params(labelsize=15)
        cbar.set_label(label=colourbarlabel,size=15,weight='bold')    
    if ax is None:
        ("ax is None")        
        plt.show()          
        return fig
    if ax is not None:        
        return axs
        
        
def produce_dict_of_df_trends(df, variable, index_name, first_end_year=2006, average='median'):
    print("Average: "+str(average))
    dict_year_df = {}
    for end_year in np.arange(first_end_year, 2021, 1): #first end year 2002+5 = 2007 
        print("end year: "+str(end_year))
        df_trends_rain = get_df_trends(df, variable, index_name, end_year=end_year, average=average)
        dict_year_df[end_year] = df_trends_rain
    return dict_year_df
    
def step_of_heatmaps(dict_year_df, colourbarlabel='Trend [mmyr$^{-1}$]', value_max=0.3,
                    cbar_kws={ "orientation": "horizontal", 'label': 'Trend [Mm$^{-1}$yr$^{-1}$]'}):    
    fig, axes = plt.subplots(len(dict_year_df), 1, figsize=(12,len(dict_year_df)))
    fig.subplots_adjust(hspace=.1)
    cbar_ax = fig.add_axes([.11, -.0, .8, .05]) #x,y, length, width
    for ax, year in zip(np.ravel(axes), dict_year_df):
        df_trend = dict_year_df[year]
        if year == 2020:
            names=list(np.arange(2002, 2017, 1))
        else:
            names=''
        heatmap_trends_ax(df_trend, yticklabel=str(year), colourbarlabel=colourbarlabel, linewidths = 2, 
                   linecolor = "white", value_max=value_max, cbar_ax=cbar_ax, names=names, 
                       cbar_kws=cbar_kws, ax=ax)
    
    fig.text(0.5, 0.07, 'Start of Trend', ha='center', va='center', fontsize=20)
    fig.text(0.06, 0.5, 'End of Trend', ha='center', va='center', rotation='vertical', fontsize=20)
    plt.show()
    return fig
    
def bin_data(df, number_of_bins, var='accumulated', quantile_based=False):
    if quantile_based == False:
        df['bins'] = pd.cut(df[var], number_of_bins)
    if quantile_based == True:
        df['bins'] = pd.qcut(df[var], number_of_bins)
    df.loc[:, "bin_centres"] = df["bins"].apply(lambda x: x.mid)
    df = df.dropna()
    return df

def q25(x, percentile=0.25):
    return x.quantile(percentile)

def q75(x):
    return x.quantile(0.75)    

def produce_groupby_averages(df, var='abs637'):
    df_groupby = df.groupby('bin_centres', as_index=False)[var].agg(['mean', 'median', 'min', 'max', 'std', q25, q75, 'count']) 
    df_groupby = df_groupby.reset_index()
    return df_groupby
    
def calculate_slope_intercept(x, y, idx):
    print("Idx :"+str(idx))
    slope, intercept = np.polyfit(x[:idx], y[:idx], 1)
    print("Slope: "+str(slope))
    return slope, intercept  
    
def significant_figures(value, sf_num=3):
    sf = '{0:.'+str(sf_num)+'f}'
    value_sf = sf.format(value)    
    return value_sf
    
def errorbar_plot(df_groupby, title='', idx=None, fontsize=15,
                  xlabel='Average accumulated precipiation en route\n of 10 day back trajectory [mm]',
                  xscale=None, ymax=None, xmax=None, ylabel='$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]',
                  annotate=None, slope_units='Mm$^{-1}$/mm'):
                  
    fig, ax = plt.subplots(figsize=(5,5))
    
    df_groupby = df_groupby.set_index('bin_centres')
    df_groupby.index = df_groupby.index.astype(float) #if bins of 1 
    df_groupby = df_groupby[~df_groupby.isin([np.inf, -np.inf]).any(1)]

    index = df_groupby.index
    median = df_groupby['median']
    mean = df_groupby['mean']
    count = df_groupby['count']
    
    quan_25 = df_groupby['q25'].values
    quan_75 = df_groupby['q75'].values
    
    error_label='25$^{\mathrm{th}}$ - 75$^{\mathrm{th}}$'
    ax.errorbar(index, median, yerr=[median-quan_25, quan_75-median], fmt='o', capsize=5, color='k', 
                mfc='None', ecolor='k', ms=1, label=error_label)
    ax.plot(index, median, label='median', marker='o', mec='k', mfc='none', c='k', ls='-') 
    ax.plot(index, mean, label='mean', marker='x', color='k', ls=':') 
        
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    
    if ymax is not None:
        ax.set_ylim(0, ymax)
    if xmax is not None:
        ax.set_xlim(0, xmax)
    #ax.set_xlim(0, index.max()*1.05)
        
    if idx is not None:        
        slope, intercept = calculate_slope_intercept(index, median, idx)
        slope3sf = significant_figures(slope, 3)
        ax.plot(index[:idx], slope*index[:idx] + intercept, 
                label="slope: "+str(slope3sf)+str(slope_units),
                       c='r', lw=1)
    ax.set_title(title, loc='left')
 
    ax.legend(frameon=False)
    ax.tick_params(axis='y', labelcolor='k', labelsize=15,which='major', direction='in', length=4, width=1.3, pad=10, 
                   bottom=False, top=False, left=True, right=False, color='k')
    ax.tick_params(axis='x', labelcolor='k', labelsize=15,which='major', direction='in', length=8, width=1.3, pad=10, 
                   bottom=False, top=False, left=True, right=False, color='k')                  
    if xscale=='log':
        ax.set_xscale('log')
    
    fancy(ax)
    ax2 = ax.twinx()
    ax2.plot(index, count, label='count', marker='x', color='b', alpha=.2)
    ax2.set_ylabel('Number of data points in bins [-]', fontsize=fontsize, rotation=270, labelpad=15, c='b')
    ax2.set_yscale('log')
    ax2.set_ylim(10**(0), 10**(4))

    if annotate is not None:
        count = count[count > 0]        
        ax2.annotate(count.iloc[annotate[0]], (count.index[annotate[0]],1), c='b')
        ax2.annotate(count.iloc[annotate[2]], (count.index[annotate[1]],1), c='b')
        ax2.annotate(count.iloc[annotate[2]], (count.index[annotate[2]],1), c='b')
    ax2.tick_params(axis='y', labelcolor='b', labelsize=15,which='major', direction='in', length=4, width=1.3, pad=10, 
               bottom=False, top=False, left=False, right=True, color='b')
    plt.show()
    return fig
    
def for_seasons(df, number_of_bins=np.arange(0, 50, 1), var='accumulated',
                xlabel='Average accumulated precipiation en route\n of 10 day back trajectory [mm]',
                normalise_by_dry_conditions=False, xscale=None, idx=None, ylabel='$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]',
                ymax=1.1, slope_units='Mm$^{-1}$/mm', save=None): 
                
    for season_abb in df.season_abb.unique():
        df_season = df[df.season_abb == season_abb].copy()        
        if normalise_by_dry_conditions == True:
            df_season.loc[:,'abs637'] = df_season.loc[:,'abs637']/df_season.loc[(df_season['accumulated'] == 0), 'abs637'].mean()        
        df_binned = bin_data(df_season, number_of_bins=number_of_bins, var=var)
        df_groupby = produce_groupby_averages(df_binned)
        fig = errorbar_plot(df_groupby, title=str(season_abb), xlabel=xlabel, xscale=xscale, idx=idx,
                            ylabel=ylabel, ymax=ymax, slope_units=slope_units) 
        if save is not None:
            save_plot(fig, name=str(save)+str(season_abb))
    
def two_subplots(df1, df2, var1, var2, var1_label, var2_label, variable_average_label2='rainfall', 
                units2='[mm]', ymax1=1, nrows = 2, ncols = 1, xcoord_legend=0.9, ycoord_legend=1.25,
                hspace=0.5, wspace=0.1, different_plot=False, season_num_to_season=None):    
                
    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)
    fig.subplots_adjust(wspace=wspace, hspace=hspace)
    
    plt.suptitle('', y=0.95, x=0.2, fontsize=20);
    
    if season_num_to_season is None:
        start_year = df1.index.year[0]
        print("start year: "+str(start_year))
        end_year = df1.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))       
        first_season = get_first_season(df1)        
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
    
    ax = plt.subplot(nrows, ncols, 1) 
    plot_subplots_for_optical_variables(df1, season_num_to_season, variable=var1, season_abbs=['AHZ','SBU','SUM'], 
                                        ymax=ymax1, letter='', ylabel=var1_label, xcoord_legend=xcoord_legend, 
                                        ycoord_legend=ycoord_legend, ax=ax)
    if different_plot == True:
        ax = plt.subplot(nrows, ncols, 2)
        
    if season_num_to_season is None:
        start_year = df2.index.year[0]
        end_year = df2.index.year[-1]
        number_years = end_year - start_year 
        first_season = get_first_season(df2)        
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
        
    ax2 = plt.subplot(nrows, ncols, 2) #ax.twinx()
    plot_subplots_for_optical_variables(df2, season_num_to_season, variable=var2, season_abbs=['AHZ','SBU','SUM'], 
                                        ymin=0, ymax=15, letter='', ylabel=var2_label,  variable_average_label=variable_average_label2,
                                        linecolour='b', error_label=False, xcoord_legend=xcoord_legend, 
                                        ycoord_legend=ycoord_legend, units=units2, ax=ax2)
    plt.show()
    return fig
    
def trend_plot(df, variable='accumulated', nrows = 1, ncols = 1, x = 13, ytext_coord=.95, ymin=0, ymax=4, units='[mm]',
               add_vertical_line=True, x_vertical='2002-04-01 12:00:00', restrict_trend=None,
               xcoord_legend=0.90, ycoord_legend=1.25, average='median', display_average_type='median',
               variable_average_label='$\sigma_{\mathrm{\overline{ap}}}$', ylabel='Accumulated \n precipiation',
               fs_ticks=15, plot_TS=True, plot_LMS=False, add_text=True):
    start_year = df.index.year[0]
    end_year = df.index.year[-1]
    number_years = end_year - start_year #len(df.index.year.unique())
    first_season = get_first_season(df)        
    season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
    season_num_to_season = reverse_dict(season_to_season_num)

    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)
    plt.suptitle('Accumulated rainfall 241 hrs.', y=0.95, x=0.22, fontsize=20)
    ax = plt.subplot(nrows, ncols, 1)     
    plot_subplots_for_optical_variables(df, season_num_to_season, variable=variable, season_abbs=['AHZ','SBU','SUM'], 
                                        ymin=ymin, ymax=ymax, letter='', ylabel= ylabel, units=units,
                                        linecolour='b', error_label=False, add_vertical_line=add_vertical_line, 
                                        x_vertical=x_vertical, restrict_trend=restrict_trend, 
                                        xcoord_legend=xcoord_legend, ycoord_legend=ycoord_legend, average=average,
                                        display_average_type=display_average_type, 
                                        variable_average_label=variable_average_label, fs_ticks=fs_ticks, 
                                        plot_TS=plot_TS, plot_LMS=plot_LMS, add_text=add_text, ax=ax)
    plt.show()
    return fig
    
###############RELATIVE TREND######################################################################################################################################

resolution_to_freq = {'ordinal':365.25,'season_ordinal':4,'month_ordinal':12}
abb_to_name = { 'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}
trend_to_initial = {'increasing':'I','decreasing':'D','no trend':'-'}
resolution_to_name = {'ordinal':'Daily','season_ordinal':'Seasonal','month_ordinal':'Monthly'}

def create_df_season(df,season_abb):
    df_season = df[df.season_abb == season_abb]
    return df_season

def vars_log_best_fit(x,y,freq):    
    y = [np.log(yi + 1.0001) for yi in y]      
    x_ = np.array(x)
    y_ = np.array(y)
    x_ = sm.add_constant(x_)
    model = sm.OLS(y_, x_, missing='drop')
    results = model.fit()    
    p = results.params[0]
    m = results.params[1]
    se = results.bse[1]        
    exp_m = (np.exp(m*(freq))-1)*100  
    m3f = '{0:.3f}'.format(m*(freq))
    exp_m3f = '{0:.3f}'.format(exp_m)    
    return p,m,m3f,se,exp_m,exp_m3f

def generate_stats_table(df,resolution,variable, MK_variable=None,
                         inpath_MK=r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\mk_tables'):
    columns=['Treatment', 'Season', 'Hourly points','points','Median','TS units', 'TS relative',
             'LMS relative','Trend(MK)','Trend(LMS)']
    
    if MK_variable == None:
        MK_variable = variable
    
    df_stats = pd.DataFrame(columns=columns)

    appended_data = []   
    
    df = df[[variable,'ordinal','month_ordinal','season_ordinal','season_abb']]    
    freq = resolution_to_freq[resolution]
    df = df.dropna(how='all')

    for season in list(df['season_abb'].unique())+['total']:  
    
        if season == 'total':
            number_of_hourly_data_points = df[variable].count()              
            median_value = df[variable].median()                     
            seaosnal_medians = resample(df,resolution,variable)
            trend_points = len(seaosnal_medians)          

        else:
       
            df_season = create_df_season(df,season)   
            number_of_hourly_data_points = df_season[variable].count()        
            seaosnal_medians = resample(df_season,resolution,variable)   
            trend_points = len(seaosnal_medians)     
            print("Number of trend points: "+str(trend_points))
                                            
        x = seaosnal_medians.index   
        y = seaosnal_medians.values        

        median_value = float(np.median(y))
        print("median value: "+str(median_value))

        if resolution == 'ordinal': 
            TFPW = perform_TFPW(x,y)            
            res = stats.theilslopes(TFPW, x[1:], 0.90)
            p,m,m3f,se,exp_m,exp_m3f = vars_log_best_fit(x[1:],TFPW,freq)            
            
        if resolution == 'season_ordinal':
            res = stats.theilslopes(y, x, 0.90)
            p,m,m3f,se,exp_m,exp_m3f = vars_log_best_fit(x,y,freq)
         
        theil_m = float(res[0])*(freq)
        theil_m_3sf = '{0:.3f}'.format(theil_m)

        if (m/se > 1.96) & (m>0):
            trend_LMS = 'increasing'
        if (m/se < -1.96) & (m<0):
            trend_LMS = 'decreasing'
        if (abs(m/se) < 1.96):
            trend_LMS = 'no trend'

        trend_LMS = trend_to_initial[trend_LMS]  
        
        try:
            print(inpath_MK+'\\'+str(MK_variable)+"_"+str(resolution)+".csv")
            df_MK = pd.read_csv(inpath_MK+'\\'+str(MK_variable)+"_"+str(resolution)+".csv")    
            print(df_MK)
            if season != "total":
                season = abb_to_name[season]
            if season == "total":
                pass
            trend_MK = trend_to_initial[df_MK.MK[df_MK.season == season].values[0]]
        except:
            trend_MK = np.nan

        percentage_change = ((res[0]*(freq))/median_value)*100
        percentage_change = float(percentage_change)
                
        df_stats = df_stats.append({columns[1]:season, columns[2]:number_of_hourly_data_points,
                                columns[3]:trend_points,columns[4]:median_value, columns[5]:theil_m, 
                                columns[6]:percentage_change, columns[7]:exp_m3f, 
                                columns[8]:trend_MK, columns[9]:trend_LMS}, ignore_index=True, 
                                    sort=True)
    appended_data.append(df_stats)   

    appended_data = pd.concat(appended_data, sort=False)
    appended_data = appended_data.reindex(columns=columns)
    appended_data = appended_data.replace(np.nan, ' ', regex=True)

    blanks = (len(appended_data)-1)*[' ']
    appended_data.loc[:,'Treatment'] = [resolution_to_name[resolution]]+blanks

    return appended_data
    
def add_stats_MK_table_save(df):
    for variable, MK_variable in zip(['abs637', 'accumulated'], ['first10yrs_abs637','first10yrs_accumulated_rainfall']):
        outpath = r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\stats_tables'

        stats_table_ordinal = generate_stats_table(df,resolution='ordinal',variable=variable, MK_variable=MK_variable)
        stats_table_season_ordinal = generate_stats_table(df,resolution='season_ordinal',variable=variable,
                                                         MK_variable=MK_variable)

        stats_table = pd.concat([stats_table_ordinal, stats_table_season_ordinal])
        stats_table.to_csv(outpath+'\\'+str(MK_variable)+'.csv',float_format='%.2f',header=False,index=False)
        print(stats_table)
        
def generate_trend_subplots(df, variable, var_list, fs_legend=25, fs_ticks=25, ax=None):
    
    ss_to_patterns = {'I':'x','D':'x','-':' '}
    
    labels = ['AHZ','SUM','SBU','All seasons']
    x_pos = np.arange(len(labels))
    x_pos_2 = [x+0.5 for x in x_pos]
    tick_pos = [x+0.25 for x in x_pos]
            
    MK_trends = df.iloc[:4,6]
    LMS_trends = df.iloc[:4,7]         
    bars_MK = ax.bar(x_pos, MK_trends, color='white', edgecolor='black', width=0.4, alpha=0.3, label='TS')
    bars_LMS = ax.bar(x_pos_2, LMS_trends, color='red', edgecolor='black', width=0.4, alpha=0.5, label='LMS')
    
    if variable == var_list[0]:
        ax.legend(loc='best', frameon=False, ncol=2, fontsize=fs_legend)
        
    ss_MK = df.iloc[:4,8]
    patterns_MK = [ss_to_patterns[s] for s in ss_MK]
    
    for bar, pattern in zip(bars_MK, patterns_MK):
        bar.set_hatch(pattern)       
       
    ss_LMS = df.iloc[:4,9]
    patterns_LMS = [ss_to_patterns[s] for s in ss_LMS]     
    
    for bar, pattern in zip(bars_LMS, patterns_LMS):
        bar.set_hatch(pattern)        
    
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(labels)
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks)
    ax.yaxis.grid(False)           
    ax.set_xlim(-0.5,4)
    ax.spines["top"].set_visible(False)  
    ax.spines["right"].set_visible(False) 
    ax.spines['bottom'].set_visible(False) 
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()   
    ax.axvline(x=2.75, color='k', linestyle=':')    
    
    return ax
    
def make_subplots(nrows=2, ncols=2, statstables_inpath = r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\stats_tables',
                  var_list = ['abs637']*4, fs_label = 25, fs_annotate = 25, 
                  variable_to_yaxis={'abs637':r'$\sigma_{\: \mathrm{ap}}$'},
                  letters = ['a)','b)','c)','d)']):   
    
    if len(var_list) == 4:
        variable_to_xlabel = {var_list[0]:'',var_list[1]:'',var_list[2]:'Seasons',var_list[3]:'Seasons'}
            
    fig, axs  = plt.subplots(nrows, ncols, figsize=(20,10), sharex=True)

    for ax,variable,letter in zip(axs.reshape(-1),var_list,letters):    
        df = pd.read_csv(statstables_inpath+'\\'+str(variable)+'.csv',header=None)
        df.replace(0, np.nan, inplace=True)    
        
        ax = generate_trend_subplots(df, variable, var_list, ax=ax)   
        
        ax.set_ylim(-14,14)    
        ax.set_xlabel('Seasons',fontsize=fs_label)
        ax.set_ylabel('Slope [%yr$^{-1}$]',fontsize=40)    
        ax.set_title(str(variable_to_yaxis[variable]), fontsize=fs_label)    
        ax.axhline(0, ls='--', color='black', lw=2)    
        ax.text(0.012, 1.01, letter, transform=ax.transAxes, 
                size=fs_annotate, weight='bold')  

        if variable == 'ratio_550':
             ax.set_ylim(-4,4)
    plt.show()
    return fig
    
def create_stats_tables_and_save(df, variables=['abs637', 'accumulated'], 
                                 outpath=r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\stats_tables'):
    for variable in variables:
        stats_table_ordinal = generate_stats_table(df, resolution='ordinal',variable=variable)
        stats_table_season_ordinal = generate_stats_table(df, resolution='season_ordinal',variable=variable)
        stats_table = pd.concat([stats_table_ordinal, stats_table_season_ordinal])
        stats_table.to_csv(outpath+'\\'+str(variable)+'.csv',float_format='%.2f',header=False,index=False)
        
################################TRENDS PER BIN#####################################################################################################################

def make_plot(df_binned, title, season_num_to_season, season_abbs=['AHZ','SUM','SBU'], season=None, nrows = 1, ncols = 1):    
    fig, axs = plt.subplots(num=None, figsize=(20, 5*nrows), sharex=True, sharey=True)
    plt.suptitle(title, y=0.95, fontsize=20);    
    ax = plt.subplot(nrows, ncols, 1) 
    #print(season_num_to_season)
    ax, res = plot_subplots_for_optical_variables(df_binned, season_num_to_season, season_abbs=season_abbs, season=season, return_slope=True, ax=ax)
    print("Slope: "+str(res))
    plt.show()
    return res
    
def get_df_counts(df_binned):
    counts = df_binned.groupby('season_ordinal')['abs637'].count()   
    df_counts = counts.to_frame()    
    season_ordinal_to_season_abb_year = dict(zip(df_binned.season_ordinal, df_binned.season_abb_year))    
    df_counts['season_abb_year'] = df_counts.index.map(season_ordinal_to_season_abb_year)
    df_counts['start'] = df_counts['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[0])
    df_counts['stop'] = df_counts['season_abb_year'].apply(lambda x: convert_season_add_year_to_datetime(x)[1])
    df_counts['mid_datetime'] = df_counts.apply(lambda x: mid_datetime_function(x.start, x.stop), axis=1)
    return df_counts
    
def plot_data_counts(df_counts):
    fig, ax = plt.subplots(figsize=(6,2))
    ax.plot(df_counts.start, df_counts.abs637, 'o-', ms=1, mfc=None, lw=1)
    ax.set_ylim(0,500)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_visible(False)
    plt.show()
    return fig

def create_dictionarries_MK_TS(df_binned, seasonal_threshold=20):    
    dict_bin_col_to_Theil_slope = {}
    dict_bin_col_to_df_MK = {}
    dict_bin_col_to_percentage_above_threshold = {}
    for bin_col in sorted(df_binned.bin_centres.unique()):
        df_precip_bin = df_binned[df_binned.bin_centres == bin_col].copy()
        df_precip_bin = df_precip_bin.dropna()
        
        #print(df_precip_bin)
        
        start_year = df_precip_bin.index.year[0]
        print("start year: "+str(start_year))
        end_year = df_precip_bin.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))        
        first_season = get_first_season(df_precip_bin)        
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)
        
        #print(season_num_to_season)
          
        try:
            res = make_plot(df_precip_bin, str(bin_col)+' mm precipitation', season_num_to_season)
        except:
            res = None
            
        df_counts = get_df_counts(df_precip_bin)
        
        #print(df_counts)

        df_counts_above_threshold = df_counts[df_counts['abs637'] > seasonal_threshold]
        
        precentage_with_enough_data = 100*len(df_counts_above_threshold)/len(df_counts)
        #print(precentage_with_enough_data)
        dict_bin_col_to_percentage_above_threshold[bin_col] = precentage_with_enough_data
        
        plot_data_counts(df_counts)
        dict_bin_col_to_Theil_slope[bin_col] = res #Theil_slope
        
        #print(season_num_to_season)
        
        try:
            dict_res_to_df_MK =  produce_dict_table(df_precip_bin, variable='abs637', 
                                                       resolutions=['ordinal','season_ordinal'], 
                                                       save=False)
            dict_bin_col_to_df_MK[bin_col] = dict_res_to_df_MK
        except:
            dict_bin_col_to_df_MK[bin_col] = np.nan
    return dict_bin_col_to_Theil_slope, dict_bin_col_to_df_MK, dict_bin_col_to_percentage_above_threshold
    
def find_trend(bin_col, dict_bin_col_to_df_MK):
    try:
        table = dict_bin_col_to_df_MK[bin_col]['ordinal']
        ss_mk_result = table['MK'][table.season == 'total'].values[0]
    except:
        ss_mk_result = 'None'
    return ss_mk_result
    
def get_data_points(df_binned):
    data_points = df_binned['bin_centres'].value_counts().to_frame()
    data_points = data_points.sort_index()
    data_points.index.name = 'precip'
    data_points = data_points.reset_index()
    return data_points
    
def create_df_trends(dict_bin_col_to_Theil_slope, dict_bin_col_to_df_MK):
    df_trends = pd.DataFrame(dict_bin_col_to_Theil_slope.items(), columns=['precip', 'abs637'])    
    df_trends = df_trends.apply(pd.to_numeric)
    df_trends['trend'] = df_trends['precip'].apply(lambda x: find_trend(float(x), dict_bin_col_to_df_MK))
    return df_trends
    
def create_trend_with_uncertainity(dict_bin_col_to_Theil_slope, dict_bin_col_to_df_MK, freq=3):
    df_trends_lo_up = pd.DataFrame(columns=['medslope', 'medintercept', 'lo_slope', 'up_slope'])
    precip = []; medslope = []; medintercept = []; lo_slope = []; up_slope = []
    for key, value in dict_bin_col_to_Theil_slope.items():
        print(key)
        print(value)
        if value != None:
            precip.append(key)   
            medslope.append(value[0]*freq)
            medintercept.append(value[1]*freq)
            lo_slope.append(value[2]*freq)
            up_slope.append(value[3]*freq)
    df_trends_lo_up['precip'] = precip
    df_trends_lo_up['medslope'] = medslope
    df_trends_lo_up['medintercept'] = medintercept
    df_trends_lo_up['lo_slope'] = lo_slope
    df_trends_lo_up['up_slope'] = up_slope
    df_trends_lo_up = df_trends_lo_up.apply(pd.to_numeric)
    df_trends_lo_up['trend'] = df_trends_lo_up['precip'].apply(lambda x: find_trend(float(x), dict_bin_col_to_df_MK))
    return df_trends_lo_up
    
def trend_vs_precip(df_trends, data_points, trend_col='abs637', lo_slope_col='lo_slope', 
                    up_slope_col='up_slope', title='', ymin=0.0001, ymax=-0.010,
                    xmin=0, xmax=50, alpha=.2):
    fig, ax = plt.subplots(figsize=(10,4))
    tkw_major = dict(which='major', size=4, width=1.5, length=4, labelsize=20, direction='in', pad=10)
    ax.plot(df_trends.precip, df_trends[trend_col], 'o-', lw=.5, c='k', ms=0)
    
    med_slope = df_trends[trend_col]
    lo_slope =  df_trends[lo_slope_col]
    up_slope =  df_trends[up_slope_col]
    
    kwargs = dict(ecolor='k', capsize=5, elinewidth=0.5, linewidth=2, ms=10)
    ax.errorbar(df_trends['precip'],df_trends[trend_col], yerr=[med_slope-lo_slope, up_slope-med_slope], mfc='k', 
                label='lo & up slope', ls='none', alpha=alpha, c='k', **kwargs)  
    
    no_trend = df_trends[df_trends['trend'] == 'no trend'].copy()
    ax.plot(no_trend.precip, no_trend[trend_col], 'o', lw=1, mec='k', ms=10, mfc='None',
            label='not s.s.')
    decreasing = df_trends[df_trends['trend'] == 'decreasing'].copy()
    ax.plot(decreasing.precip, decreasing[trend_col], 'v', lw=1, c='k', ms=10,
           label='s.s.')
    npnan = df_trends[df_trends['trend'] == 'None'].copy()
    ax.plot(npnan.precip, npnan[trend_col], '*', lw=1, mec='k', ms=10, mfc='None', label='too few')
    ax.tick_params(axis='both', colors='k', **tkw_major)
    ax.invert_yaxis()
    plt.ylabel('TS Trend [Mm$^{-1}$yr$^{-1}$]', fontsize=20)
    plt.xlabel('Binned accumulated precipitation experienced by trajectories\n [mm]', fontsize=20)
    
    ax.set_ylim(ymin,ymax)
    ax.set_xlim(xmin,xmax)
    
    ax2 = ax.twinx()
    data_points_plot, = ax2.plot(data_points['precip'], data_points.bin_centres, alpha=0.3)
    #ax2.set_ylim(0, data_points['bin_centres'].max()*1.1)
    ax2.set_ylabel('Number of data points [-]', c=data_points_plot.get_color(), fontsize=20)
    ax2.tick_params(axis='y', colors=data_points_plot.get_color(), **tkw_major)
    ax2.set_yscale('log')
    
    ax.legend(frameon=False, loc=1, fontsize=15)
    plt.title(str(title), fontsize=12, loc='left')
    plt.show()
    return fig
    
#########STATS TABLE###############################################################################################################################################

def create_stats_table_extremes(DFs, names):
    dict_name_to_df = dict(zip(names, DFs))

    df_stats_table = pd.DataFrame(columns=['median', 'mean', 'relative change in mean', 'trend', 'relative trend', 'percentage removed',
                                     'AHZ', 'relative change in mean AHZ', 'trend AHZ', 'relative trend AHZ', 'percentage removed AHZ',                                      
                                     'SUM', 'relative change in mean SUM', 'trend SUM', 'relative trend SUM', 'percentage removed SUM',
                                     'SBU', 'relative change in mean SBU', 'trend SBU', 'relative trend SBU', 'percentage removed SBU'])

    for df, name in zip(DFs, names):
        print("name: "+str(name))
        
        mean = df['abs637'].mean()
        median = df['abs637'].median()

        df_stats_table.loc[name, 'mean'] = mean
        df_stats_table.loc[name, 'median'] = median

        start_year = df.index.year[0]
        print("start year: "+str(start_year))
        end_year = df.index.year[-1]
        print("end year: "+str(end_year))
        number_years = end_year - start_year 
        print("number years: "+str(number_years))        
        first_season = get_first_season(df)        
        season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
        season_num_to_season = reverse_dict(season_to_season_num)

        res = make_plot(df, '', season_num_to_season)    
        df_stats_table.loc[name, 'trend'] = res[0]*3

        df_stats_table.loc[name, 'relative trend'] = (res[0]*3/mean)*100

        for season in ['AHZ', 'SUM', 'SBU']:
            print(season)
            df_season = df.loc[df.season_abb == season]
            mean = df_season['abs637'].mean()
            df_stats_table.loc[name, season] = mean

            start_year = df.index.year[0]
            print("start year: "+str(start_year))
            end_year = df.index.year[-1]
            print("end year: "+str(end_year))
            number_years = end_year - start_year 
            print("number years: "+str(number_years))        
            first_season = get_first_season(df)        
            season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
            season_num_to_season = reverse_dict(season_to_season_num)

            res = make_plot(df, '', season_num_to_season, season_abbs=[str(season)], season=str(season))    
            
            df_stats_table.loc[name, 'trend '+str(season)] = res[0]*3
            print('trend '+str(season)+' '+str(res[0]*3))
            print(res)

            df_stats_table.loc[name, 'relative trend '+str(season)] = (res[0]*3/mean)*100

    for name in ['fires removed', 'high rain removed', 'low rain removed']:
        print(name)
        for col in ['mean', 'AHZ', 'SUM',  'SBU']:
            relative_change = (df_stats_table.loc[name, col] - df_stats_table.loc['original', col])/df_stats_table.loc['original', col]
            
            if col in  ['AHZ', 'SBU', 'SUM']:
                df_stats_table.loc[name, 'relative change in mean '+str(col)] = relative_change*100
            if col == 'mean':
                df_stats_table.loc[name, 'relative change in mean'] = relative_change*100

        removed = 100-(len(dict_name_to_df[name])/len(dict_name_to_df['original']))*100
        df_stats_table.loc[name, 'percentage removed'] = removed

        for season in ['AHZ', 'SUM', 'SBU']:
            df1 = dict_name_to_df[name]
            df0 = dict_name_to_df['original']
            df1_season = df1.loc[df1.season_abb == season]
            df0_season = df0.loc[df0.season_abb == season]

            removed = 100-(len(df1_season)/len(df0_season))*100
            df_stats_table.loc[name, 'percentage removed '+str(season)] = removed
    return df_stats_table
    
#####SEASONALITY############################################################################################################################

def plot_monthly_dfs(df, error_var='quantiles', color='k', title='', ylabel='$\sigma_{\mathrm{ap, 637 nm}}$',
                     mfc='green', mec='k', ecolor='black', linecolour='k', ms=1, label='', display_label=False,
                     xcoord_bbox=0.65, ycoord_bbox=1.15, fs_ticks=20, rotation=None, month_xticks=False, week_ordinal_xticks=False,
                     month_ordinal_xticks=False, season_ordinal_xticks=False, startyear=2006,endyear=2020, tick_frequency=6, fs_label=15, 
                     first_season_abb='SBU', fmt="o", ymin=None, ymax=None, plot_medians=True, x_marker='x-', ax=None):  
                     
    index = df.index
    mean =  df['mean'].values
    median =  df['median'].values
    std = df['std'].values
    
    if error_var=='std':
        if display_label==True:
            error_label='std'
        if display_label==False:  
            error_label='_' 
        ax.errorbar(index, mean, yerr=std, fmt='x', capsize=5, color=color, mfc=mfc, ecolor=ecolor, ms=ms, label=error_label, mec=mec)
        ax.plot(index, mean, label=label, color=color, ls=':')
    
    if error_var=='quantiles':
        quan_25 = df['q25'].values
        quan_75 = df['q75'].values
        if display_label==True:
            error_label='Med + 25$^{\mathrm{th}}$ - 75$^{\mathrm{th}}$'
        if display_label==False:  
            error_label='_'    
        if plot_medians == True:
            ax.errorbar(index, median, yerr=[median-quan_25, quan_75-median], fmt=fmt, capsize=5, color=color, 
                        mfc=mfc, ecolor=ecolor, ms=ms, label=error_label)
            ax.plot(index, median, label='', color=color, ls=':')
        ax.plot(index, mean, x_marker, label='mean', color=color, ms=ms) 
        
    
    if rotation:
        ax.xaxis.set_tick_params(rotation=45) 
    ax.set_title(title, loc='left', fontsize=15)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.legend(ncol=2, frameon=False, fontsize=15, loc='upper left', bbox_to_anchor=(xcoord_bbox, ycoord_bbox))
    if month_xticks == True:
        ax.set_xticks(np.arange(1,13,1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'], fontsize=fs_ticks)
        ax.set_xticklabels(['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D'], fontsize=fs_ticks)        
    if week_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*52,52))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))  
    if month_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*12,12))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))          
    if season_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*3,3))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))         
    
    ax.tick_params(labelsize=fs_ticks, direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', length=4, color='k', width=1.3)
    ax.set_ylabel(ylabel, fontsize=20)    
    ax.set_ylim(ymin, ymax)
    thickax(ax, fontsize=fs_ticks)
    return ax

def make_annual_plot(df_OCEC_abs,  var='abs637', ms=3, ylabel=r'$\sigma_{\: \mathrm{ap}}$ [Mm$^{-1}$]', 
                     display_label=False, error_var='quantiles',
                     black_tubing_date=None, ymin=None, ymax=None, plot_medians=True, x_marker='-x', fs_ticks=25,
                     title=None, ax=None):
    single_plot = False
    if ax is None:
        single_plot = True
        fig, ax = plt.subplots(figsize=(15, 5))
    
    if black_tubing_date is not None:
        print("Sliced: "+str(black_tubing_date))
        df_OCEC_abs = df_OCEC_abs[df_OCEC_abs.index <= pd.to_datetime(black_tubing_date)]    
    
    df_OCEC_abs_M = produce_averages_groupby(df_OCEC_abs, groupby_var='month', variable=var)
    print(df_OCEC_abs_M)
    plot_monthly_dfs(df_OCEC_abs_M, color='k', ylabel=ylabel, ecolor='k', mec='k', mfc='None',
                     ms=ms, month_xticks=True, display_label=display_label, error_var=error_var, 
                     ymin=ymin, ymax=ymax, plot_medians=plot_medians, x_marker=x_marker, fs_ticks=fs_ticks,
                     title=title, ax=ax)
    
    if single_plot == True:
        plt.show()
        return fig
    if single_plot == False:
        return ax   
    
######################CLUSTERS####################


######################MK TEST#####################

dict_season_to_color = dict(zip(['AHZ', 'SBU', 'SUM', 'all_seasons'], ["#41b6c4", "#2c7fb8", "#253494", 'k']))
dict_season_to_ylim = dict(zip(['AHZ', 'SBU', 'SUM', 'all_seasons'], [20, 20, 20, 20]))
dict_season_to_color = dict(zip(['AHZ', 'SBU', 'SUM'], ["#41b6c4", "#2c7fb8", "#253494"]))

def convert_to_datetime(x):
    x_datetime = datetime(pd.to_datetime(x).year, pd.to_datetime(x).month, pd.to_datetime(x).day)
    return x_datetime
    
def normal_mk_test(df_days, var='abs637', resolution=0.001):
    y0 = convert_to_datetime(df_days.index[0])
    multi_obs_dts = np.array([y0+timedelta(days=item) for item in range(len(df_days.index))])
    multi_obs = df_days[var].values #np.array(df_days.values)
    print(len(multi_obs_dts))
    print(len(multi_obs))
    #decimal places or accuracy => 2dp 
    out = mk.mk_temp_aggr(multi_obs_dts, multi_obs, resolution=resolution) #resolution?? 1/24 hourly/daily?? 
    print(out[0])
    return out

def get_relative_trend(out, df_days):
    return out[0]['slope']/df_days.median()*100

def get_dict_of_season_dfs(df):
    df = df.copy()
    if 'season' not in df.columns:
        df['season'] = df['season_abb_year'].apply(lambda x: str(x)[:3])
    df_AHZ = df[df['season'] == 'AHZ']
    df_SUM = df[df['season'] == 'SUM']
    df_SBU = df[df['season'] == 'SBU']
    dict_season_to_df = {'AHZ':df_AHZ, 'SUM':df_SUM, 'SBU':df_SBU}
    return dict_season_to_df

def get_seasonal_trend_output(dict_season_to_df, var='abs637', resolution=0.001, pw_method='3pw', alpha_xhomo=80):
    if var == 'tp_era5':
        resolution = 0.1 #decimal places
    
    df_AHZ, df_SUM, df_SBU = [*dict_season_to_df.values()][0], [*dict_season_to_df.values()][1], [*dict_season_to_df.values()][2]
    multi_obs = [df_AHZ[var].values, df_SUM[var].values, df_SBU[var].values]
    multi_obs_dts = [np.array([convert_to_datetime(x) for x in df_AHZ.index]),
                     np.array([convert_to_datetime(x) for x in df_SUM.index]), 
                     np.array([convert_to_datetime(x) for x in df_SBU.index])]
    # Process it
    out = mk.mk_temp_aggr(multi_obs_dts, multi_obs, resolution=resolution, pw_method=pw_method, 
                          alpha_xhomo=alpha_xhomo)
    n_season = 3
    # Print the results
    for n in range(n_season):
        print('Season {ind}:'.format(ind=n+1), out[n])
    print('Combined yearly trend:', out[n_season])
    return out

def produce_table(out):
    parameters = ['p', 'ss', 'slope', 'ucl', 'lcl']
    print(parameters)
    df_stats_table = pd.DataFrame(columns=parameters, dtype=np.float64)
    seasons = ['AHZ', 'SUM', 'SBU', 'combined'] #needs to be correct order
    for season_n, season_name in enumerate(seasons):
        print(season_name)
        for parameter in parameters:
            print(out[season_n][parameter])
            df_stats_table.loc[season_name, parameter] = out[season_n][parameter]
    return df_stats_table

def add_seasons_to_daily(df_days):
    month_to_season =  { 1:'SBU',  2:'AHZ', 3:'AHZ',  
                     4:'AHZ',  5:'AHZ', 6:'SUM',  7:'SUM',  8:'SUM', 9:'SUM', 10:'SBU', 
                     11:'SBU', 12:'SBU'}  
    df_days.loc[:,'month_num'] = df_days.index.month
    df_days.loc[:,'year'] = df_days.index.year        
    df_days.loc[:,'season'] = df_days.month_num.map(month_to_season).values
    return df_days
    
def seasonal_averages(df, var):
    df_seasons = df.groupby('season_ordinal').median()[var].to_frame() 
    season_num_to_season = dict(zip(df['season_ordinal'], df['season_abb_year'])) 
    df_seasons = add_mid_datetime_using_dictionary(df_seasons, season_num_to_season)
    df_seasons["season_abb"]=df_seasons["season_abb_year"].apply(lambda x: x[:3])
    return df_seasons

def remove_unwanted_values(df, var):
    df = df[[var]].copy()    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    return df

def add_trend_season(df_seasons, var, freq=3, c='r', loc=1, alpha=0.05, 
                     xcoord_legend=0.5, ycoord_legend=.95, mscale = 2, fs_legend=10, uncertainty=False, 
                     sigfigs=3, linecolour='r', ax=None):
    ax2 = ax.twiny()
    x = df_seasons.index #intergers  
    y = df_seasons[var].values    
    
    res = stats.theilslopes(y, x, 0.90)
    Theil_slope = (res[1] + res[0] * x)
    lo_slope = (res[1] + res[2] * x)
    up_slope = (res[1] + res[3] * x)

    theil_m=res[0]
    theil_m = float(theil_m)*(freq)
    lo_m = float(res[2])*(freq)
    up_m = float(res[3])*(freq)

    #significant figures
    sfs = sf(sigfigs)           
    intecept=sfs.format(res[1])
    theil_m=sfs.format(theil_m)
    lo_m=sfs.format(lo_m)
    up_m=sfs.format(up_m) 
    
    if uncertainty == True:
        label=str('TS$_{\,\mathrm{S}}$:$\,$y$\,$=$\,$'+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept))
    if uncertainty == False:
        label=str('TS$_{\,\mathrm{}S}$:$\,$y$\,$=$\,$'+str(theil_m)+'$\,\mathdefault{x}$$\,$+$\,$'+str(intecept))
        
    ax2.plot(x, Theil_slope, ls='--', lw=1, c=linecolour,alpha=0.5,
            label=label)
    ax2.fill_between(x, up_slope, lo_slope, alpha=alpha, color=linecolour)
    
    legend = ax2.legend(numpoints = 1,loc='upper right',
              frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
#     legend = ax2.legend(numpoints = 1,loc='upper left',bbox_to_anchor=(xcoord_legend, ycoord_legend),
#           frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
    legend.get_title().set_fontsize(fs_legend)
    
    ax2.set_xticklabels('')
    ax2.set_xticks([])
    return ax2

def daily_medians(df, var):
    df_daily_abs = df.resample('D').median()
    df_daily_abs = df_daily_abs.replace([np.inf, -np.inf], np.nan)
    df_daily_abs = df_daily_abs.dropna(how='all')
    df_daily_abs = df_daily_abs[[var]].copy()
    return df_daily_abs

def produce_seasonal_test_daily_ax(df_daily_abs, var, fs_legend=8, ms=5, freq=365.25, sigfigs=3, linecolour = 'red', 
               ylabel='$\sigma_{\mathrm{ap}}$', units='[Mm$^{-1}$]', fs_label = 12, 
               xcoord_legend=0, ycoord_legend=.98, mscale = 2, ymin=-0.1, ymax = 2, 
               alpha=0.2, uncertainty=False, add_calculated_trend=True, res_cal=None, ax=None):    
    
    ax2 = ax.twiny()    
    #turn to intergers
    df_daily_abs['day'] = df_daily_abs.index
    df_daily_abs['ordinal'] = df_daily_abs['day'].apply(lambda x: x.toordinal())
    df_daily_abs['ordinal'] = df_daily_abs['ordinal'] - df_daily_abs['ordinal'][0] + 1

    date_ints = list(df_daily_abs['ordinal'].values)
    
    x = np.array([x-date_ints[0]+1 for x in date_ints])    
    y = df_daily_abs[var].values    
    
    res = stats.theilslopes(y, x, 0.90)
    Theil_slope = (res[1] + res[0] * x)
    lo_slope = (res[1] + res[2] * x)
    up_slope = (res[1] + res[3] * x)

    theil_m=res[0]
    theil_m = float(theil_m)*(freq)
    lo_m = float(res[2])*(freq)
    up_m = float(res[3])*(freq)

    #significant figures
    sfs = sf(sigfigs)           
    intecept=sfs.format(res[1])
    theil_m=sfs.format(theil_m)
    lo_m=sfs.format(lo_m)
    up_m=sfs.format(up_m) 
    
    if uncertainty == True:
        label=str('TS$_{\,\mathrm{D}}$: y = '+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept))
    if uncertainty == False:
        label=str('TS$_{\,\mathrm{D}}$:$\,$y$\,$=$\,$'+str(theil_m)+'$\,\mathdefault{x}$$\,$+$\,$'+str(intecept))
               
    ax2.plot(x, Theil_slope, ls='-', lw=2, c=linecolour, label=label)
    ax2.fill_between(x, up_slope, lo_slope, alpha=alpha, color=linecolour)
    
    if add_calculated_trend == True:
        
        slope_cal=res_cal[0]; intercept_cal=res_cal[1]; lo_slope_cal=res_cal[2]; up_slope_cal=res_cal[3]
        intercept_lo_cal=res_cal[4]; intercept_up_cal=res_cal[5]
        
        trend_cal_freq = slope_cal/freq                
        slope_cal = (intercept_cal + trend_cal_freq * x) #$\sigma_{\mathrm{ap, cal.}}$
        lo_slope_cal = (intercept_lo_cal + lo_slope_cal/freq * x) 
        up_slope_cal = (intercept_up_cal + up_slope_cal/freq * x) 
        
        relative_trend = sfs.format(100*float(trend_cal_freq*(freq))/float(df_daily_abs[var].median()))
        
        ax2.plot(x, slope_cal, ls='-', lw=2, c='b', 
                 label='3pw$_{\mathrm{trend}}$ = '+str(sfs.format(trend_cal_freq*(freq)))+' '+str(units)[:-1]+' yr$^{-1}$]'+'\nrel. 3pw$_{\mathrm{trend}}$ = '+str(relative_trend)+' [%yr$^{-1}$]')
        ax2.fill_between(x, up_slope_cal, lo_slope_cal, alpha=alpha, color='b')        
    #ax.set_ylabel(ylabel+' '+units, fontsize=fs_label) 
    #ax.set_ylim(ymin, ymax)
    ax2.set_xticks([])    
    legend = ax2.legend(numpoints = 1, loc='upper left', 
              frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
#     legend = ax2.legend(numpoints = 1, loc='upper left', bbox_to_anchor=(xcoord_legend, ycoord_legend),
#           frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
    legend.get_title().set_fontsize(fs_legend)
    return ax
    
def make_subplot(df, df_seasons, var, season, dict_season_to_color, fs_legend=12, fs_label=12,
                 ymax=0.8, add_calculated_trend=True, alpha_xhomo=80, sigfigs=3, ax=None):
    print("season: "+str(season))
    #seasonal
    df_seasons_ = df_seasons[df_seasons["season_abb"] == str(season)].copy()
    add_trend_season(df_seasons_, var, freq=3, c='r', xcoord_legend=0., ycoord_legend=1.02, 
                     fs_legend=fs_legend, ax=ax)
    ax.plot(df_seasons_['mid_datetime'], df_seasons_[var], marker='o', ls=':', c=dict_season_to_color[season])
    #daily
    df_ = df[df['season_abb'] == str(season)].copy()
    df_daily_ = daily_medians(df_, var)
    ax.plot(df_daily_.index, df_daily_[var], 'o', c=dict_season_to_color[season], ms=1, alpha=0.4)    
    #pw_trend 
    res_cal=None
    if add_calculated_trend == True:
        df_daily_ = add_seasons_to_daily(df_daily_)
        dict_season_to_df = get_dict_of_season_dfs(df_daily_)
        out = get_seasonal_trend_output(dict_season_to_df, var=var, alpha_xhomo=alpha_xhomo)
        df_stats_table = produce_table(out)
        print(df_stats_table)
        print(season)
        print(df_stats_table.loc[season])
        slope_cal = float(df_stats_table.loc[season]['slope'])
        lo_slope_cal = float(df_stats_table.loc[season]['lcl'])
        up_slope_cal = float(df_stats_table.loc[season]['ucl'])        
        time_med = df.iloc[int(len(df_)/2):int(len(df_)/2)+1].season_ordinal.values[0]
        intercept_cal = df_daily_[var].median() - slope_cal*time_med #c = y - mx        
        intercept_lo_cal = df_daily_[var].median() - lo_slope_cal*time_med #c = y - mx
        intercept_up_cal = df_daily_[var].median() - up_slope_cal*time_med #c = y - mx        
        res_cal = (slope_cal, intercept_cal, lo_slope_cal, up_slope_cal, intercept_lo_cal, intercept_up_cal)       
    produce_seasonal_test_daily_ax(df_daily_, var, fs_legend=fs_legend, ms=5, freq=365, sigfigs=sigfigs, linecolour = 'r', 
                   ylabel='$\sigma_{\mathrm{ap}}$', units='[Mm$^{-1}$]', fs_label=fs_label, 
                   xcoord_legend=0., ycoord_legend=.8, mscale = 2, ymin=-0.1, ymax=2, 
                   add_calculated_trend=add_calculated_trend, 
                   res_cal=res_cal, ax=ax)
    ax.set_ylim(-0, ymax)
    return ax
    
def make_subplot_full(df, df_seasons, var, fs_legend=12, fs_label=15, ymax=0.8, 
                      ylabel='$\sigma_{\mathrm{ap}}$', units='[Mm$^{-1}$]', add_calculated_trend=True, 
                      alpha_xhomo=80, c='k', sigfigs=3, ax=None):
    #seasonal
    add_trend_season(df_seasons, var, freq=3, c='r', xcoord_legend=0., ycoord_legend=.98, 
                     fs_legend=fs_legend, ax=ax)
    ax.plot(df_seasons['mid_datetime'], df_seasons[var], marker='o', ls=':', c=c)
    #daily
    df_daily = daily_medians(df, var)
    ax.plot(df_daily.index, df_daily[var], 'o', c=c, ms=1, alpha=0.4)
    #pw_trend 
    res_cal=None
    if add_calculated_trend == True:
        df_daily = add_seasons_to_daily(df_daily)
        dict_season_to_df = get_dict_of_season_dfs(df_daily)
        out = get_seasonal_trend_output(dict_season_to_df, var=var, alpha_xhomo=alpha_xhomo)
        df_stats_table = produce_table(out)
        print(df_stats_table)
        slope_cal = float(df_stats_table.loc['combined']['slope'])
        lo_slope_cal = float(df_stats_table.loc['combined']['lcl'])
        up_slope_cal = float(df_stats_table.loc['combined']['ucl'])        
        time_med = df.iloc[int(len(df)/2):int(len(df)/2)+1].season_ordinal.values[0]
        intercept_cal = df_daily[var].median() - slope_cal*time_med #c = y - mx        
        intercept_lo_cal = df_daily[var].median() - lo_slope_cal*time_med #c = y - mx
        intercept_up_cal = df_daily[var].median() - up_slope_cal*time_med #c = y - mx        
        res_cal = (slope_cal, intercept_cal, lo_slope_cal, up_slope_cal, intercept_lo_cal, intercept_up_cal)       
    produce_seasonal_test_daily_ax(df_daily, var, fs_legend=fs_legend, ms=5, freq=365, sigfigs=sigfigs, linecolour = 'r', 
                   ylabel=ylabel, units=units, fs_label=fs_label, xcoord_legend=0., ycoord_legend=.89, 
                   mscale = 2, ymin=-0.1, ymax=2, add_calculated_trend=add_calculated_trend, 
                                   res_cal=res_cal, ax=ax)
    ax.set_ylim(-0, ymax)
    return ax
    
def season_plot(df, df_seasons, var, season, ymax2,  dict_season_to_color=dict_season_to_color,
               dict_season_to_ylim=dict_season_to_ylim, fs_ticks=20, fs_label=30, ylabel='', units=''):    
    fig = plt.figure(figsize=(8, 5))
    ymax = dict_season_to_ylim[season]
    gs = gridspec.GridSpec(ncols=1, nrows=3, hspace = 0.2, wspace = 0.2, top = 1,
                           bottom = 0, left = 0, right = 1)
    ax = fig.add_subplot(gs[0:1])
    df_ = df[df['season_abb'] == str(season)].copy()
    ax.plot(df_.index, df_[var], marker='o', c=dict_season_to_color[season], ms=1, alpha=0.4)
    ax.set_ylim(ymax, ymax2)
    fancy(ax, fontsize=fs_ticks, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])
    ax = fig.add_subplot(gs[1:3])
    make_subplot(df, df_seasons, var=var, season=season, dict_season_to_color=dict_season_to_color, 
                 ax=ax)
    fancy(ax, fontsize=fs_ticks, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_ylim(0, ymax)
    fig.text(-0.15, 0.5, ylabel+' '+units, ha='center', va='center', rotation='vertical', fontsize=fs_label)
    plt.show()
    return fig
   
def subplots(df, df_seasons, var, dict_season_to_color, dict_season_to_ylim, fs_label=15, fs_legend = 10, fs_tick=15, 
             fs_letter = 20, ymax2=30, name='trend_plot', alpha_xhomo=80, c='k', ylabel='$\sigma_{\mathrm{ap}}$',
             units='[Mm$^{-1}$]'):
    fig = plt.figure(figsize=(15, 8))

    gs = gridspec.GridSpec(ncols=3, nrows=8, hspace = 0.2, wspace = 0.2, top = 1,
                               bottom = 0, left = 0, right = 1)
    #AHZ
    season='AHZ'
    ax = fig.add_subplot(gs[0:1])
    df_ = df[df['season_abb'] == str(season)].copy()
    ax.plot(df_.index, df_[var], 'o', c=dict_season_to_color[season], ms=1, alpha=0.4)
    ax.set_ylim(dict_season_to_ylim[season], ymax2)
    ax.text(.01, 1.6, 'a)', ha='left', va='top', transform=ax.transAxes, fontsize=fs_letter)
    fancy(ax, fontsize=fs_tick, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])
    ax = fig.add_subplot(gs[1:3, 0:1])
    make_subplot(df, df_seasons, var, season, dict_season_to_color, fs_legend=fs_legend, fs_label=fs_label, ax=ax)
    ax.set_ylim(0, dict_season_to_ylim[season])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fancy(ax, fontsize=fs_tick, spines=['top','bottom','left','right'], alpha=0.5)    

    #SBU
    season='SBU'
    ax = fig.add_subplot(gs[1:2])
    df_ = df[df['season_abb'] == str(season)].copy()
    ax.plot(df_.index, df_[var], 'o', c=dict_season_to_color[season], ms=1, alpha=0.4)
    ax.set_ylim(dict_season_to_ylim[season], ymax2)
    thickax(ax, fontsize=fs_label, linewidth=1)
    ax.text(.01, 1.6, 'b)', ha='left', va='top', transform=ax.transAxes, fontsize=fs_letter)
    fancy(ax, fontsize=fs_label, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1:3, 1:2])
    make_subplot(df, df_seasons, var, season, dict_season_to_color, fs_legend=fs_legend, fs_label=fs_label, ax=ax)
    ax.set_ylim(0, dict_season_to_ylim[season])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fancy(ax, fontsize=fs_tick, spines=['top','bottom','left','right'], alpha=0.5)  

    #SUM
    season='SUM'
    ax = fig.add_subplot(gs[2:3])
    df_ = df[df['season_abb'] == str(season)].copy()
    ax.plot(df_.index, df_[var], 'o', c=dict_season_to_color[season], ms=1, alpha=0.4)
    ax.set_ylim(dict_season_to_ylim[season], ymax2)
    thickax(ax, fontsize=fs_label, linewidth=1)
    ax.text(.01, 1.6, 'c)', ha='left', va='top', transform=ax.transAxes, fontsize=fs_letter)
    fancy(ax, fontsize=fs_label, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[1:3, 2:3])
    make_subplot(df, df_seasons, var, season, dict_season_to_color, fs_legend=fs_legend, fs_label=fs_label, ax=ax)
    ax.set_ylim(0, dict_season_to_ylim[season])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fancy(ax, fontsize=fs_tick, spines=['top','bottom','left','right'], alpha=0.5)  

    #FULL
    season = 'all_seasons'
    ax = fig.add_subplot(gs[4:5, 0:3])
    ax.plot(df.index, df[var], 'o', c='k', ms=1, alpha=0.4)
    ax.set_ylim(dict_season_to_ylim[season], ymax2)
    thickax(ax, fontsize=fs_label, linewidth=1)
    ax.text(.01, 1.6, 'd)', ha='left', va='top', transform=ax.transAxes, fontsize=fs_letter)
    fancy(ax, fontsize=fs_label, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])

    ax = fig.add_subplot(gs[5:8, 0:3])
    ymax = dict_season_to_ylim[season]
    make_subplot_full(df, df_seasons, var, fs_legend=fs_legend*2, fs_label=fs_label, 
                      ymax=ymax, ylabel=ylabel, units=units, alpha_xhomo=alpha_xhomo, c=c, ax=ax)
    ax.set_ylim(0, dict_season_to_ylim[season])
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fancy(ax, fontsize=fs_tick, spines=['top','bottom','left','right'], alpha=0.5)  
    fig.text(-0.08, 0.5, ylabel+' '+units, ha='center', va='center', rotation='vertical', fontsize=30)
    plt.show()

    save_plot(fig, name, formate='.png')
    return fig
   
def full_plot(df, df_seasons, var, ymax2, season='all_seasons', ylabel='', 
              units='', dict_season_to_color=dict_season_to_color,
              dict_season_to_ylim=dict_season_to_ylim, fs_ticks=20, fs_legend=20, fs_label=30,
              alpha_xhomo=80, c='k'):    
    fig = plt.figure(figsize=(12, 6))
    ymax = dict_season_to_ylim[season]
    gs = gridspec.GridSpec(ncols=1, nrows=3, hspace = 0.2, wspace = 0.2, top = 1,
                           bottom = 0, left = 0, right = 1)
    ax = fig.add_subplot(gs[0:1])
    #ax.plot(df.index, df[var], 'o', c=dict_season_to_color[season], ms=1, alpha=0.4)    
    df_AHZ = df[df['season_abb'] == 'AHZ'].copy()
    ax.plot(df_AHZ.index, df_AHZ[var], 'o', c=dict_season_to_color['AHZ'], ms=1)
    df_SBU = df[df['season_abb'] == 'SBU'].copy()
    ax.plot(df_SBU.index, df_SBU[var], 'o', c=dict_season_to_color['SBU'], ms=1)
    df_SUM = df[df['season_abb'] == 'SUM'].copy()
    ax.plot(df_SUM.index, df_SUM[var], 'o', c=dict_season_to_color['SUM'], ms=1)
    
    ax.set_ylim(ymax, ymax2)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fancy(ax, fontsize=fs_ticks, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_xticklabels([])
    ax = fig.add_subplot(gs[1:3])
    make_subplot_full(df, df_seasons, var, fs_legend=fs_legend, fs_label=fs_label, 
                      ymax=ymax, ylabel=ylabel, units=units, alpha_xhomo=alpha_xhomo, c=c, ax=ax)
    fancy(ax, fontsize=fs_ticks, spines=['top','bottom','left','right'], alpha=0.5) 
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    fig.text(-0.1, 0.5, ylabel+' '+units, ha='center', va='center', rotation='vertical', fontsize=fs_label)
    plt.show()
    return fig
    
##################clustering mktest########################################################################

colors=['#a6611a','#dfc27d','#f5f5f5','#80cdc1','#018571']
colors = ['#d7191c','#fdae61','#ffffbf','#abd9e9','#2c7bb6']
colors = sns.color_palette("colorblind")[:5]

colors = ['#2c7bb6', '#abd9e9','#fee090','#fdae61','#d7191c']
clusters = np.arange(1, 6, 1)
names = ['North Atlantic', 'Greenland', 'Arctic Ocean', 'Siberia', 'Eurasia']
dict_titles = dict(zip(names, clusters))

dict_cluster_to_colors = dict(zip(np.arange(1,6,1), colors))

dict_cluster_to_name = {1:'NA', 2:'G', 3:'AO', 4:'S', 5:'E'}

def produce_cluster_subplot_trends(df_max_count_abs637, cluster, dict_season_to_ylim, dict_season_to_color, 
                                   alpha_xhomo=0):    
    print("cluster: "+str(cluster))
    df_cluster = df_max_count_abs637[df_max_count_abs637['clusters_5'] == cluster].copy()     
    c=dict_cluster_to_colors[cluster]
    dict_season_to_color={'AHZ': c, 'SBU': c, 'SUM': c, 'all_seasons': c}    
    df_cluster = slice_df(df_cluster, start_datetime='2001-12-31') 
    first_season = get_first_season(df_cluster)        
    season_to_season_num = get_full_season_abb_years(start_year=int(df_cluster.index.year.values[0]), 
                                                               number_years=2024-int(df_cluster.index.year.values[0]), first_season=first_season)
    season_num_to_season = reverse_dict(season_to_season_num)
    df_cluster = prepare_data(df_cluster, season_num_to_season)
    df_cluster = remove_unwanted_values(df_cluster, var='abs637')
    df_cluster = create_month_season_numbers(df_cluster, full_season_to_season_num=season_to_season_num)
    df_cluster_seasons = seasonal_averages(df_cluster, var='abs637')    
    fig = subplots(df_cluster, df_cluster_seasons, var='abs637', ymax2=15, dict_season_to_color=dict_season_to_color, 
                  dict_season_to_ylim=dict_season_to_ylim, name='abs_trend_plot', alpha_xhomo=alpha_xhomo,
                  c=c, ylabel='$\sigma_{\mathrm{ap}}$ ('+dict_cluster_to_name[cluster]+')', 
                   units='[Mm$^{-1}$]')         
    save_plot(fig, name='abs_trend', formate='.png')
    return fig