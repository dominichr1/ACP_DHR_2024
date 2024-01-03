#import packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
import matplotlib as mpl
from matplotlib import cm
import cmocean
from statsmodels.tsa import stattools
from scipy.stats import norm
import mpmath
import glob    
import os

def add_ordinal(df):
    df = df.resample('D').median()  
    df['timestamp'] = pd.to_datetime(df.index)
    df['ordinal'] = df['timestamp'].apply(lambda x: x.toordinal())
    df['hours'] = df['timestamp'].dt.hour
    df['hours_ordinal'] = df['ordinal'] + df['hours'] 
    df['hours_ordinal'] = df['hours_ordinal'] - df['hours_ordinal'][0]
    df['hours_ordinal']=df['hours_ordinal'].astype(int)    
    df['time_stamp']=df.apply(lambda x:(pd.Timestamp('2002-03-08')+pd.DateOffset(days = int(x['hours_ordinal']))),axis=1)
    return df
    
def simpleaxis(ax):
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()

def day_resample_add_ordinal(df, mean=True, median=False, start_date=None, date_col='date'):
    df[date_col] = df.index.date
    
    if start_date == None:
        start_date = df[date_col].iloc[0]
    
    #print("start date inserted: "+str(start_date))
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
    
month_to_season =  { 1:'SBU',  2:'AHZ', 3:'AHZ',  
                     4:'AHZ',  5:'AHZ', 6:'SUM',  7:'SUM',  8:'SUM', 9:'SUM', 10:'SBU', 
                     11:'SBU', 12:'SBU'}
                     
abb_to_name = { 'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}
name_to_abb = {'Slow build up':'SBU','Arctic Haze': 'AHZ', 'Summer/Clean':'SUM'}

def get_full_season_abb_years(start_year, number_years):
    season_abb_years = []
    #print("generate dictionary: "+str(start_year))
    for year in np.arange(start_year, start_year+number_years+1, 1):
        for season_abb in ['AHZ','SUM','SBU']: #correct order
            season_abb_year = str(season_abb) + '_' + str(year)
            season_abb_years.append(season_abb_year)
    seasons_num = np.arange(1,len(season_abb_years)+1,1)
    season_to_season_num = dict(zip(season_abb_years, seasons_num))
    #print("dictionary"+str(season_to_season_num))
    return season_to_season_num
                     
def create_month_season_numbers(df, full_season_to_season_num=None):
    start_year = df.index.year[0]
    #print("First year in data set: "+str(start_year))
    number_years = len(df.index.year.unique())+1
    #print("Number of years + 1: "+str(number_years))
    
    #if start_year == 2002:
    #    full_season_to_season_num =  get_full_season_abb_years(start_year, number_years+1)
    #    print(full_season_to_season_num)
        
    df.loc[:,'month_num'] = df.index.month
    df.loc[:,'year'] = df.index.year
        
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df['season_name'] = df['season_abb'].map(abb_to_name)  
    
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    #print("Note: the slow build-up season crosses over two years as it goes from October-January, so the year corresponds to previous year")
    df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))

    seasons = df.season_abb_year.unique()
    #print("Number of unique seasons: "+str(len(seasons)))    
    seasons_num = np.arange(1,len(seasons)+1,1)
    season_to_season_num = dict(zip(seasons, seasons_num))

    #df.loc[:,'season_ordinal'] = df['season_abb_year'].map(full_season_to_season_num)
    df.loc[:,'season_ordinal'] = df['season_abb_year'].map(season_to_season_num)
    
    df = df.sort_index()
    return df
    
def create_season_to_season_num_dict(df):
    start_year = df.index.year[0]
    number_years = len(df.index.year.unique())+1
    df.loc[:,'month_num'] = df.index.month
    df.loc[:,'year'] = df.index.year        
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df['season_name'] = df['season_abb'].map(abb_to_name)      
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    seasons = df.season_abb_year.unique()    
    seasons_num = np.arange(1,len(seasons)+1,1)
    season_to_season_num = dict(zip(seasons, seasons_num))
    return season_to_season_num

def significant_figures(value, sf_num=3):
    sf = '{0:.'+str(sf_num)+'f}'
    value_sf = sf.format(value)    
    return value_sf
    
def create_histogram(df, var, bin_num=20, xlabel=r'$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]',
    ylabel='Normalised probability',title='Histogram of absoprtion data of length',
    xmin=-0.5, xmax=20, ymin=0, ymax=0.2, show=False, log_transform=False, constant=10, ax=None):
       
    df = df[[var]].copy()
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] #remove any infs or np.nan    
    x = df[var].values
        
    if log_transform == True:
        print("log-transformation")
        x = abs(x)*constant
        x = np.log(x)
        x = x[x != -np.inf] 
        result = np.where(x == -np.inf)
        #print(result)
        #print(min(x)) 
        x = np.asarray(x)          
        n, bins, patches = ax.hist(x, bins=bin_num, alpha=0.75, histtype='step')
    
    if log_transform == False:
        x = np.asarray(x)    
        weights = np.ones_like(x) / len(x)
        n, bins, patches = ax.hist(x, bins=bin_num, weights=weights, alpha=0.75, histtype='step', 
        lw=3, color='k')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title+': '+str(len(df)), loc='left')
    ax.set_title(r'Bins: '+str(bin_num), loc='right')
    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.grid(True)
    if show == True:
        plt.show()
    return ax
    
def remove_inf_npnan(df, var, how='any'):
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[var], how=how)
    return df
    
def calculate_AE(df, colour1, colour2, AE_name, bscat=False):
    wavelengths = [450, 550, 700]
    cols_wavelengths = ['scat450', 'scat550', 'scat700']
    colour_to_wavelength = {'blue':wavelengths[0], 'green':wavelengths[1], 'red':wavelengths[2]}
    colour_to_col = {'blue':cols_wavelengths[0], 'green':cols_wavelengths[1], 'red':cols_wavelengths[2]}
        
    col1 = colour_to_col[colour1] 
    col2 = colour_to_col[colour2] 
    wavelength1 = colour_to_wavelength[colour1]
    wavelength2 = colour_to_wavelength[colour2]
    
    AE = -(np.log10(df[col1]/df[col2])/np.log10(wavelength1/wavelength2))
    print('Created: AE_'+str(colour_to_wavelength[AE_name]))
    df['AE_'+str(colour_to_wavelength[AE_name])] = AE
    
    if bscat == True:
        df[str(bscat)+'AE_'+str(colour_to_wavelength[AE_name])] = AE
    return df
    
def calculate_all_AEs(df, bscat=False):
    df = calculate_AE(df, colour1='blue', colour2='green', AE_name='blue', bscat=bscat) #Angstrom Exponent at 450nm
    df = calculate_AE(df, colour1='blue', colour2='red', AE_name='green', bscat=bscat)   #Angstrom Exponent at 550nm
    df = calculate_AE(df, colour1='green', colour2='red', AE_name='red', bscat=bscat)  #Angstrom Exponent at 700nm
    if bscat == True:
        print("bscat")
        df = calculate_AE(df, colour1='blue', colour2='green', AE_name='blue', bscat=bscat) #Angstrom Exponent at 450nm
        df = calculate_AE(df, colour1='blue', colour2='red', AE_name='green', bscat=bscat)   #Angstrom Exponent at 550nm
        df = calculate_AE(df, colour1='green', colour2='red', AE_name='red', bscat=bscat)  #Angstrom Exponent at 700nm
    return df
    
def convert_wavelength(df, lambda1, lambda2, abs_col='abs_neph', AE_col='550', use_constant=False):
    if use_constant == True: #for absoprtion
        AAE = 1.0
        print("AAE = "+str(AAE)+" is used to convert between wavelengths for "+str(abs_col))
        df = df.rename(columns={abs_col:'abs'+str(lambda1)})        
        scat_converted = df['abs'+str(lambda1)]*(lambda1/lambda2)**AAE
        df.loc[:,'abs'+str(lambda2)] = scat_converted
    if use_constant == False: #for scattering
        print("Coverting "+str('scat'+str(lambda1))+' to '+str(lambda2)+' nm')
        scat_converted = df['scat'+str(lambda1)]*(lambda1/lambda2)**df['AE_'+str(AE_col)]
        df.loc[:,'scat'+str(lambda2)] = scat_converted
    return df

from scipy import stats
    
def theilslope(x,y):
    res = stats.theilslopes(y, x, 0.90)
    mid_slope = res[0]
    med_intercept = res[1]
    return mid_slope,med_intercept
    
def thickax(ax, fontsize=12,linewidth=2):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(2.5)
    plt.rc('axes', linewidth)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    ax.tick_params(direction='in', length=7, width=2, pad=12, bottom=True, top=True, left=True, right=True)
    
def remove_duplicates(df):
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")      
    print(len(duplicateRowsDF)) 
    print(duplicateRowsDF)
    df_first = df.loc[~df.index.duplicated(keep='first')]
    df_last = df.loc[~df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first)))
    print("Length after: "+str(len(df_last)))
    return df_first, df_last
    
def find_number_of_duplicate_indexs(df):
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are :")      
    print(len(duplicateRowsDF))
    return duplicateRowsDF 
    
def day_offset_old(df, number_of_days=1, add=True):
    if add == True:
        print("Day added: "+str(number_of_days))
        df.index = df.index + pd.DateOffset(days=number_of_days) #have to manually offset by a day
    if add == False:
        print("Day substracted: "+str(number_of_days))
        df.index = df.index - pd.DateOffset(days=number_of_days) #have to manually offset by a day        
    return df
    
def day_offset(df, number_of_days=1, split_time=None, add=True, substract=False):
    """Add or subtract days from index"""            
    if split_time != None: 
        before = df.loc[(df.index < pd.to_datetime(split_time))].index
        if add == True:
            print("Day added: "+str(number_of_days)+' from '+str(split_time))
            after = df.loc[(df.index >= pd.to_datetime(split_time))].index + pd.DateOffset(days=number_of_days)
            df.index = list(before) + list(after) 
        if substract == True:
            print("Day substracted: "+str(number_of_days)+' from '+str(split_time))
            after = df.loc[(df.index >= pd.to_datetime(split_time))].index - pd.DateOffset(days=number_of_days)
            df.index = list(before) + list(after) 
            
    if split_time == None:   
        if add == True:
            print("Day added: "+str(number_of_days))
            df.index = df.index + pd.DateOffset(days=number_of_days) #have to manually offset by a day
        if substract == True:
            print("Day substracted: "+str(number_of_days))
            df.index = df.index - pd.DateOffset(days=number_of_days) #have to manually offset by a day         
    return df
    
def save_df(df, path, name=''):
    print("Save as: "+str(path+'\\'+name+'.dat'))
    df.to_csv(path+'\\'+name+'.dat', index=True, float_format='%.3f')
    
def add_ticks(ax, labelsize=20, ylabel='$\sigma_{\mathrm{ap}}$, 637 nm [Mm$^{-1}$]'): 
    ax.legend(frameon=False, loc=1, fontsize=25)
    ax.minorticks_on()
    ax.tick_params(which='major', labelsize=labelsize, direction='in', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', direction='in', length=4, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
    ax.yaxis.set_ticks_position('left')
    ax.xaxis.set_ticks_position('bottom')
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.set_ylabel(ylabel, fontsize=20) 
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg"):
    folders = glob.glob(path)
    print(folders)
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight')
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
    
def mergedfs(df1, df2, how='inner'):
    df_merged = pd.merge(df1, df2, how=how, left_index=True, right_index=True)
    return df_merged
   
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
    
def prepare_data(df, remove_indexes=None):    
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(how='all')
    df = day_resample_add_ordinal(df)
    df = add_year_month_ordinal(df)
    df = create_month_season_numbers(df)    
    
    if remove_indexes is not None:
        #merge above         
        #print("old length: "+str(len(df)))
        #print("removing: "+str(len(remove_indexes))+" values")
        df_removed = df[~df.index.isin(remove_indexes)].copy()
        #print("new length: "+str(len(df)))
        percentage_removed = len(remove_indexes)/len(df_removed)*100 
        #print("removed: "+str(len(remove_indexes)/len(df_removed)*100)+' %')
        
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
    
def sf(sf_num):
    sf = '{0:.'+str(sf_num)+'f}'
    return sf
  
def vars_for_best_fit(x,y,freq):
    x_array = np.array(x)
    y_array = np.array(y)    
    x_array = sm.add_constant(x_array)    
    model = sm.OLS(y_array, x_array, missing='drop')
    results = model.fit()

    p = results.params[0]
    m = results.params[1]
    
    se = results.bse[1]
    #print("Standard error: "+str(se))
        
    m3f = '{0:.3f}'.format(m*(freq))
    p3f = '{0:.3f}'.format(p) 
    se3f = '{0:.3f}'.format(se*(freq))    
    return p,m,m3f,p3f,se3f
    
def create_season_ordinal_list(df, season_abb, season_abb_col='season_abb'):
    season_list = df.season_ordinal[df[season_abb_col] == season_abb].unique().tolist()
    return season_list
 
def plot_subplots_for_optical_variables(df, variable='abs637', 
                                        season_to_shape_dict = {'AHZ':'o','SUM':'v','SBU':'s'},
                                        season_to_color = {'AHZ':'red','SUM':'black','SBU':'red'},
                                        ymin=-0.1, ymax = 2, tick_frequency=6, freq = 3, average='mean',
                                        season_abbs=['AHZ','SUM','SBU'], letter='a)', ylabel='$\sigma_{\mathrm{ap, 637 nm}}$',
                                        season=None, linecolour='r', xcoord=0.02, ycoord=1.15, percentage_removed=0, 
                                        error_label=True, display_text=True, first_season=None, last_season=None, xcoord_bbox=0.95, 
                                        ycoord_bbox=1.25, c='k',no_right_spine=True, sf_num=1, 
                                        return_slope=False, ax=None): 
    
    first_year = df['year'].iloc[0]
    last_year = df['year'].iloc[-1]
    #print("first_year:"+str(first_year))
    
    year_labels = [str(x) for x in range(first_year,last_year+1)]
    
    fs_cbarlabel = 20; fs_cbartick = 25; fs_ticks = 15; fs_annotate = 20; fs_label = 20; fs_legend = 20; mscale = 2
    
    seasons = df.season_ordinal.unique()    
    
    if first_season == None:    
        #first_season = seasons[0]
        first_season = df.loc[df['season_abb'] == 'AHZ', 'season_ordinal'].iloc[0]
        #print("first season: "+str(first_season))
        #print("first AHZ: "+str(first_season))
       
    if last_season == None:
        last_season = seasons[-1]
        #print("last season: "+str(last_season))
    
    ticks=np.arange(first_season, int(len(year_labels)*(freq)+first_season), tick_frequency) 
    #print(ticks)
        
    df_var = df.copy()
    
    if season is not None:
        display_average = df.loc[df['season_abb'] == season, variable].mean()
    if season == None:
        display_average = df[variable].mean()
    
    count = df_var.groupby(['season_ordinal']).size()
    quan_75s = df_var.groupby('season_ordinal')[variable].quantile(0.75)
    quan_25s = df_var.groupby('season_ordinal')[variable].quantile(0.25)
    medians = df_var.groupby('season_ordinal')[variable].median()  
    means = df_var.groupby('season_ordinal')[variable].mean()   
    #means = df_var.groupby('season_ordinal')[variable].mean()   
        
    ax.plot(medians.index, medians, label='', ls=':', c=c, ms=0, lw=1)
    labels =  ax.get_yticks()
    #print("labels: "+str(labels))

    for num, season_abb in enumerate(season_abbs):  
        #print("season abb")
        #print(df_var.columns)
        season_ordinal_list = create_season_ordinal_list(df_var, season_abb)
        #print(season_ordinal_list)

        quan_75 = quan_75s[season_ordinal_list]
        quan_25 = quan_25s[season_ordinal_list]
        median = medians[season_ordinal_list] 
        mean = means[season_ordinal_list]
        fmt = season_to_shape_dict[season_abb]
        mfc = season_to_color[season_abb]

        kwargs = dict(ecolor='k', capsize=5, elinewidth=0.5, linewidth=1, ms=10)
        
        if error_label == True:
            label = str(season_abb)
        if error_label == False:
            label = None
        ax.errorbar(median.index, median, yerr=[median-quan_25, quan_75-median], fmt=fmt, mfc=mfc, 
                    label=label, ls='none', c='k', **kwargs)
        
        ax.errorbar(mean.index, mean,  fmt='x', mfc='steelblue', ls='none', c='k',label=None,
                    ms=10,lw=1)

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

    ax.plot(x_fit, y_fit, ls='-', c=linecolour, label='LMS: y = '+str(m3f)+'$\,\mathdefault{x}$ +'+str(p3f),lw=1) 

    res = stats.theilslopes(y, x, 0.90)

    Theil_slope = (res[1] + res[0] * x)
    lo_slope = (res[1] + res[2] * x)
    up_slope = (res[1] + res[3] * x)

    theil_m=res[0]
    theil_m = float(theil_m)*(freq)
    lo_m = float(res[2])*(freq)
    up_m = float(res[3])*(freq)

    sfs = sf(3)              

    intecept=sfs.format(res[1])
    theil_m=sfs.format(theil_m)
    lo_m=sfs.format(lo_m)
    up_m=sfs.format(up_m) 

    ax.plot(x, Theil_slope, ls='--', lw=1, c=linecolour,
            label=str('TS: y = '+str(theil_m)+' ('+str(lo_m)+' to '+str(up_m)+')$\,\mathdefault{x}$ +'+str(intecept)))
    ax.fill_between(x, up_slope, lo_slope, alpha=0.15, color=linecolour)

    
    ax.spines["top"].set_visible(False)
    
    if no_right_spine == True:    
        ax.spines["right"].set_visible(False)  
        ax.get_xaxis().tick_bottom()  
        ax.get_yaxis().tick_left()  
    
    ax.set_ylabel(ylabel,fontsize=fs_label)   
    #print("limits: "+str(first_season-2)+str(last_season+2))
    ax.set_xlim(first_season-2, last_season+2)
    ax.set_xticks(ticks=ticks)
    ax.set_xticklabels(year_labels[::2], fontsize=fs_label)
    
    #if ax is not None:
    #    #print(ax.get_yticklabels())
    #    labels = ax.get_yticks()
    #    #print("ylabels"+str(labels))
    #    labels = [significant_figures(float(x), sf_num=sf_num) for x in labels]
    #    y_min, y_max = ax.get_ylim()
    #    ticks = [(tick - y_min)/(y_max - y_min) for tick in ax.get_yticks()]        
    #    ax.set_yticks(ticks=ax.get_yticks().tolist())
    #    #print(ticks)
    #    ax.set_yticklabels(labels, fontsize=fs_label)
        
    ax.minorticks_on()
    ax.tick_params(axis='both', which='major', labelsize=fs_ticks, direction='in', length=6, width=2,
                   grid_alpha=0.5)
    ax.tick_params(axis='both', which='minor', labelsize=fs_ticks, direction='in', length=3, width=2,
                   grid_alpha=0.5)

    ax.text(-0.1, 0.98, letter, transform=ax.transAxes, size=fs_annotate, weight='bold')
    
    if display_text == True:
        display_average=sfs.format(display_average)    
        ax.text(xcoord, ycoord, "$\sigma_{\mathrm{\overline{ap}}}$ :"+str(display_average) + '[Mm$^{-1}$] (removed '+str(percentage_removed)+'%)', 
                transform=ax.transAxes, size=fs_annotate, weight='bold',
               color=linecolour)
    
    #title='Based on seasonal\n resolution: '
    legend = ax.legend(numpoints = 1,loc='upper left', bbox_to_anchor=(xcoord_bbox, ycoord_bbox),
              frameon=False, markerscale=mscale, ncol=1, fontsize=fs_legend)  
    legend.get_title().set_fontsize(fs_legend)
    #ax.legend(frameon=False, fontsize=fs_legend)
    ax.set_ylim(ymin, ymax)
    if return_slope == True:
        return ax, theil_m   
    return ax
    
def bin_plot(df_bins, number_of_bins=20, ymax=0.05, fontsize=25, ax=None):

    if type(number_of_bins) is list:
        #print("list")
        number_of_bins = len(number_of_bins)
    
    norm = mpl.colors.Normalize(vmin=1, vmax=number_of_bins)
    cmap = cmocean.cm.thermal
    m = cm.ScalarMappable(norm=norm, cmap=cmap)
    
    datetimes = df_bins.columns[1:]
    datetimes = [pd.to_datetime(x) for x in datetimes]
    y =  df_bins.iloc[1, 1:].values
    
    if ax==None:
        fig, ax = plt.subplots(figsize=(20,5))
        
    for count_row in [-1]:
        number_of_data_points_in_bin = len(df_bins.iloc[count_row,1:].dropna(how='all'))
        ax.plot(datetimes, df_bins.iloc[count_row, 1:].values, label=str(df_bins.iloc[count_row, 0])+' '+str(number_of_data_points_in_bin), 
                c=m.to_rgba(count_row))
    
    plt.tick_params(labelsize=25, axis='both', which='major')
    plt.tick_params(labelsize=25, axis='both', which='minor')    
    plt.ylabel('Fraction per month \nfor each bin \n(normalised by total in df) [-]', fontsize=25)
    plt.xlim(pd.to_datetime(datetimes[0]), pd.to_datetime(datetimes[-1]))
    #print(pd.to_datetime(datetimes[0]), pd.to_datetime(datetimes[-1]))
    
    #plt.legend(title='Bins:', frameon=False, bbox_to_anchor=(1.25, 1.0), fontsize=15, 
    #          title_fontsize=18)

    #plt.title("Average size of bins: "+str(mean_size_of_bins), loc='left', fontsize=20)
    plt.grid(True)
    if ymax:
        #print("ymax: "+str(ymax))
        plt.ylim(0,ymax)
    if ax==None:
        plt.show()
        return fig
    if ax is not None:
        return ax
        
def divide_into_lastxyears(df, lastxyears=5):
    df_lastxyrs = df[df.index.year >= int(df.index.year[-1]) - int(lastxyears)]
    df_lastxyrs = prepare_data(df_lastxyrs)
    return df_lastxyrs
    
def divide_into_firstxyears(df, firstxyears=5):
    df_firstxyrs = df[df.index.year <= int(df.index.year[0]) + int(firstxyears)]
    df_firstxyrs = prepare_data(df_firstxyrs)
    return df_firstxyrs
    
def q25(x, percentile=0.25):
    return x.quantile(percentile)

def q75(x):
    return x.quantile(0.75)    

def produce_monthly_averages(df, resample_resolution="M", var = 'absorption'):
    idx = df.columns.get_loc(str(var))
    df_monthly = df.iloc[:,idx].resample(resample_resolution).agg(['mean', 'median', 'min', 'max', 'std', q25, q75]) 
    return df_monthly
    
def produce_averages_groupby(df, groupby_resolution="month", var='absorption'):
    idx = df.columns.get_loc(str(var))
    df_groupby = df.groupby(groupby_resolution)[var].agg(['mean', 'median', 'min', 'max', 'std', q25, q75]) 
    return df_groupby
    
def plot_monthly_dfs(df, error_var='quantiles', color='k', title='', ylabel='$\sigma_{\mathrm{ap, 637 nm}}$',
                     mfc='green', ecolor='black', linecolour='k', ms=1, label='', display_label=False,
                     xcoord_bbox=0.95, ycoord_bbox=1.25, fs_ticks=20, rotation=None, month_xticks=False, week_ordinal_xticks=False,
                     month_ordinal_xticks=False, season_ordinal_xticks=False, startyear=2006,endyear=2020, tick_frequency=6, fs_label=15, 
                     first_season_abb='SBU', fmt="d", ax=None):  
                     
    index = df.index
    mean =  df['mean'].values
    median =  df['median'].values
    std = df['std'].values
    
    if error_var=='std':
        if display_label==True:
            error_label='std'
        if display_label==False:  
            error_label='_' 
        ax.errorbar(index, mean, yerr=std, fmt=fmt, capsize=5, color=color, mfc=mfc, ecolor=ecolor, ms=ms, label=errorlabel)
        ax.plot(index, mean, label=label, color=color, ls=':')
    
    if error_var=='quantiles':
        quan_25 = df['q25'].values
        quan_75 = df['q75'].values
        if display_label==True:
            error_label='25$^{\mathrm{th}}$ - 75$^{\mathrm{th}}$'
        if display_label==False:  
            error_label='_'        
        ax.errorbar(index, median, yerr=[median-quan_25, quan_75-median], fmt=fmt, capsize=5, color=color, 
                    mfc=mfc, ecolor=ecolor, ms=ms, label=error_label)
        ax.plot(index, median, label=label, color=color, ls=':')  
        
    ax.tick_params(labelsize=fs_ticks, direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', length=4, color='k', width=1.3)
    ax.set_ylabel(ylabel, fontsize=20)    
    
    if rotation:
        ax.xaxis.set_tick_params(rotation=45) 
    ax.set_title(title, fontsize=15)    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False) 
    ax.legend(ncol=2, frameon=False, fontsize=15, loc='upper left', bbox_to_anchor=(xcoord_bbox, ycoord_bbox))
    if month_xticks == True:
        ax.set_xticks(np.arange(1,13,1))
        ax.set_xticklabels(['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']) 
    if week_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*52,52))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))  
    if month_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*12,12))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))          
    if season_ordinal_xticks == True:
        ax.set_xticks(np.arange(1,(endyear-startyear+1)*3,3))
        ax.set_xticklabels(np.arange(startyear,endyear+1,1))         
        #df['season_ordinal'] = df.index
        #first_season = df.loc[df['season_abb'] == first_season_abb, 'season_ordinal'].iloc[0]
        #print("first season: "+str(first_season)))
        #last_season = df['season_ordinal'].iloc[0]
        #year_labels = np.arange(startyear,endyear+1,1)
        #freq=3
        #ticks=np.arange(first_season, int(len(year_labels)*(freq)+first_season), tick_frequency)
        #ax.set_xlim(first_season-2, last_season+2)
        #ax.set_xticks(ticks=ticks)
        #ax.set_xticklabels(year_labels[::2], fontsize=fs_label)        
    return ax
    
def create_a_merge_of_thermal_and_optical_EC(df_OCEC, EC='EC_mean_mug_m3', ECOP='OC_mean_mug_m3', 
                                             black_tubing_date='2011-06-14'):
    black_tubing_date=pd.to_datetime(black_tubing_date)
    
    EC_thermal = df_OCEC[EC][df_OCEC['start'] < black_tubing_date] #good thermal data is before date
    EC_Optical = df_OCEC[ECOP][df_OCEC['start'] > black_tubing_date] #optical data is used afterwards

    frames = [EC_thermal, EC_Optical] #merge the two frames
    EC_combined = pd.concat(frames, names='EC_combined') #cooncat
    df_EC_combined = EC_combined.to_frame(name="EC_combined") #turn into dataframes
    df_OCEC = pd.merge(df_OCEC, df_EC_combined, left_index=True, right_index=True) # add column of data
    
    return df_OCEC
    
def calculate_AE(df, colour1, colour2, AE_name, bscat=False):
    wavelengths = [450, 550, 700]
    cols_wavelengths = ['scat450', 'scat550', 'scat700']
    colour_to_wavelength = {'blue':wavelengths[0], 'green':wavelengths[1], 'red':wavelengths[2]}
    colour_to_col = {'blue':cols_wavelengths[0], 'green':cols_wavelengths[1], 'red':cols_wavelengths[2]}
        
    col1 = colour_to_col[colour1] 
    col2 = colour_to_col[colour2] 
    wavelength1 = colour_to_wavelength[colour1]
    wavelength2 = colour_to_wavelength[colour2]
    
    AE = -(np.log10(df[col1]/df[col2])/np.log10(wavelength1/wavelength2))
    #print('Created: AE_'+str(colour_to_wavelength[AE_name]))
    df['AE_'+str(colour_to_wavelength[AE_name])] = AE
    
    if bscat == True:
        df[str(bscat)+'AE_'+str(colour_to_wavelength[AE_name])] = AE
    return df
    
def calculate_all_AEs(df, bscat=False):
    df = calculate_AE(df, colour1='blue', colour2='green', AE_name='blue', bscat=bscat) #Angstrom Exponent at 450nm
    df = calculate_AE(df, colour1='blue', colour2='red', AE_name='green', bscat=bscat)   #Angstrom Exponent at 550nm
    df = calculate_AE(df, colour1='green', colour2='red', AE_name='red', bscat=bscat)  #Angstrom Exponent at 700nm
    if bscat == True:
        #print("bscat")
        df = calculate_AE(df, colour1='blue', colour2='green', AE_name='blue', bscat=bscat) #Angstrom Exponent at 450nm
        df = calculate_AE(df, colour1='blue', colour2='red', AE_name='green', bscat=bscat)   #Angstrom Exponent at 550nm
        df = calculate_AE(df, colour1='green', colour2='red', AE_name='red', bscat=bscat)  #Angstrom Exponent at 700nm
    return df
    
def convert_wavelength(df, lambda1, lambda2, abs_col='abs_neph', AE_col='550', use_constant=False):
    if use_constant == True: #for absoprtion
        AAE = 1.0
        print("AAE = "+str(AAE)+" is used to convert between wavelengths for "+str(abs_col))
        df = df.rename(columns={abs_col:'abs'+str(lambda1)})        
        scat_converted = df['abs'+str(lambda1)]*(lambda1/lambda2)**AAE
        df.loc[:,'abs'+str(lambda2)] = scat_converted
    if use_constant == False: #for scattering
        print("Coverting "+str('scat'+str(lambda1))+' to '+str(lambda2)+' nm')
        scat_converted = df['scat'+str(lambda1)]*(lambda1/lambda2)**df['AE_'+str(AE_col)]
        df.loc[:,'scat'+str(lambda2)] = scat_converted
    return df
    
####TRENDS#################################################################
    
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
    
def calculate_autocorr(y):
    nlags = min(int(10 * np.log10(len(y))), len(y) - 1)
    print("Number of lags: "+str(nlags))
    autocorr = stattools.acf(y, fft=False, nlags=nlags)
    autocorr_coeff = autocorr[1]
    c = autocorr_coeff
    return c    
    
def perform_TFPW(x,y):  
    res = stats.theilslopes(y, x, 0.90)
    b=res[0]    
    print('trend: '+str(b))
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
    print("Total variance: "+str(total_var))
    
    if S_ > 0:   
        Zsk = (S_ - 1)/np.sqrt(total_var)
        print("Zsk: "+str(Zsk))
        
    if S_ == 0:        
        Zsk = 0
        
    if S_ < 0:   
        Zsk = (S_ +1)/np.sqrt(total_var)
        print("Zsk: "+str(Zsk))
        
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
        print("Season: "+str(season_abb))
        
        df_season =  df[df['season_abb'] == season_abb]
        print("Median: "+str(df_season[variable].median()))
        print("Length of season: "+str(len(df_season)))

        seaosnal_averages = resample(df_season, resolution, variable)   
        
        x = seaosnal_averages.index   
        y = seaosnal_averages.values  
                
        try:       
            #preform pre-whitening for ordinal
            if resolution == 'ordinal': 
                print("preform pre-whitening:")
                TFPW = perform_TFPW(x,y)            
                res = stats.theilslopes(TFPW, x[1:], 0.90)
                y = TFPW

            MK = mk_test(y)
            print("MK: "+str(MK))            
            MK_result = MK[0]             
            length = len(y)
            s = MK[4]
            var_s = MK[5]
        except:
            MK_result = np.nan; length = np.nan; s = np.nan; var_s = np.nan
            
        df_MK = df_MK.append({'season':abb_to_name[season_abb], 'MK' : MK_result , 'length' : length,'s':s,'var_s':var_s} , ignore_index=True)

    df_MK = df_MK.append({'season':'total', 'MK' : seasonal_mk_test(df_MK, alpha=0.05)[0] , 'length' : sum(df_MK.length.values),'s':seasonal_mk_test(df_MK, alpha=0.05)[4],
                         'var_s':seasonal_mk_test(df_MK, alpha=0.05)[4]} , ignore_index=True)    
    
    return df_MK
    
def produce_dict_table(df, variable='abs637', resolutions = ['ordinal','season_ordinal'],
                       outpath = r'C:\Users\DominicHeslinRees\Documents\Analysis\HYSPLIT\mk_tables',
                       save=False, save_variable_name='default'):
    dict_res_to_df_MK = {}

    for resolution in resolutions:
            print("resolution: "+str(resolution))
            print("Variable:"+str(variable))
            df_MK = generate_MK_tables(df, variable, resolution, 
                                       abb_to_name = {'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}, 
                                       season_abbs=['SBU', 'AHZ', 'SUM'])
            if save == True:
                print("saved as: "+str(outpath+'\\'+str(save_variable_name)+"_"+str(resolution)+".csv"))
                df_MK.to_csv(outpath+'\\'+str(save_variable_name)+"_"+str(resolution)+".csv",float_format='%.3f',index=False)
            dict_res_to_df_MK[resolution] = df_MK
    return dict_res_to_df_MK
    
def annual_precip(df_monthly_mean, var='sum_precipitation_daily', ymax=3.5, label='', c='k', ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(7,3))
    ax.plot(df_monthly_mean.index, df_monthly_mean[var], 'o-', c=c, label=label)
    ax.legend(frameon=False)
    plt.ylim(0, ymax)
    plt.xticks(np.arange(1,13,1), ['J','F','M','A','M','J','J','A','S','O','N','D'])
    #plt.show()
    simpleaxis(ax)
    if ax is None:
        return fig
    if ax is not None:
        return ax
        
def find_top_both_quantiles(df, var):
    df = df.copy()
    df[var] = df[var].replace([np.inf, -np.inf], np.nan).dropna()
    qtop = np.nanquantile(df[var].values, 0.99)
    qbottom = np.nanquantile(df[var].values, 0.01)
    return qtop, qbottom
    
def prepare_df_bins(df_bins_T):
    df_bins_T['datetime'] = df_bins_T.index
    df_bins_T['wk_num_yr'] = df_bins_T['datetime'].dt.strftime('%Y-%U') #let's use week number
    df_bins_T['wk_num'] = df_bins_T['datetime'].dt.strftime("%V") #let's use week number
    df_bins_T['year'] = df_bins_T['datetime'].dt.year
    df_bins_T['wk_ordinal'] = (df_bins_T['year'].astype(int) - df_bins_T['year'].astype(int)[0])*52 + df_bins_T['wk_num'].astype(int) #.dt.strftime("%V") #let's use week number
    print(df_bins_T.columns)
    print(df_bins_T.head(2))
    return df_bins_T
    
def updateYear(x, year):
    n= x.split("-")
    n[0]=str(year)
    return "-".join(n)

def alter_year_in_index(df, year): 
    df.index = df.index.to_series().apply(lambda x:updateYear(str(x), year))
    return df
    