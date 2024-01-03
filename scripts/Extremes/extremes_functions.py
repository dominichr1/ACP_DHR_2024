import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import cmocean
import statsmodels.api as sm
from scipy import stats
import glob
import os

def load_df(loadpath, extrapath=None, filename=None, formatdata=".dat"):
    if extrapath is not None:
        print("loading: "+str(loadpath+'\\'+extrapath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+extrapath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False)
    if extrapath is None:
        print("loading: "+str(loadpath+'\\'+filename+formatdata))
        df = pd.read_csv(loadpath+'\\'+filename+formatdata, index_col=0, parse_dates=True,
                         low_memory=False)        
    return df
    
def save_df(df, name='default_name', path=r'C:\Users\DominicHeslinRees\Documents\Analysis\absorption'):
    print("Save as: "+str(path+'\\'+name+'.dat'))
    df.to_csv(path+'\\'+name+'.dat',index=True, float_format='%.3f')
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg", dpi=300):
    folders = glob.glob(path)
    print(folders)
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
    
def merge_dfs(df1, df2):
    df_merge = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    return df_merge
    
def thickax(ax, fontsize=12):
    for axis in ['top','bottom','left','right']:
        ax.spines[axis].set_linewidth(1.5)
    plt.rc('axes', linewidth=2)
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
        tick.label1.set_fontweight('bold')
    ax.tick_params(direction='in', length=7, width=2, pad=12, bottom=True, top=False, left=True, right=False)
    
def significant_figures(value, sf_num=3):
    sf = '{0:.'+str(sf_num)+'f}'
    value_sf = sf.format(value)    
    return value_sf
    
def slice_df_by_year(df, start_year=2002, end_year=2021):
    df = df.loc[(start_year <= df.index.year) & (df.index.year <= end_year)]
    return df

def timeseries_with_outliers(df, title='', ymax=5, outlier_col='outlier', add_labels=False, 
                             outlier_to_label={1:'top',0: '', -1:'bottom'}):
    """plot the timeseries and mark the outliers"""
    
    outlier_to_color = {1:'lightcoral', 0: 'k', -1:'blue'}
    fig, ax = plt.subplots(figsize=(20,5))
    
    for outlier in df[outlier_col].unique():
        print("outlier type: "+str(outlier))
        df_outlier = df[df.outlier == outlier]
        plt.plot(df_outlier.index, df_outlier['abs637'], 'o', ms=1, mfc=outlier_to_color[outlier], mec=outlier_to_color[outlier], label=outlier_to_label[outlier])
        thickax(ax)  
   
    if add_labels==True:
        print("add labels:")
        xcoords = [df_oldpsap637.index[0], df_oldpsap637_hourly.index[-1], df_MAAP.index[0], df_MAAP.index[-1]]
        labels = ['Old PSAP EBAS', 'Old PSAP', 'New PSAP', 'MAAP']
        trans = ax.get_xaxis_transform()
        for xc, label in zip(xcoords, labels):
            plt.axvline(x=xc, ls=':', c='k', alpha=0.5) #label='line at x = {}'.format(xc)
            plt.text(xc, float(ymax)*1.05, label, rotation=0, horizontalalignment='right', alpha=0.5,  fontsize=15)
    
    plt.ylabel(r'$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]', fontsize=15) 
    plt.title(title, loc='left')
    plt.ylim(-1, ymax)
    plt.legend(loc=1, frameon=False, fontsize=15, markerscale=5)
    plt.show()
    return fig
    
month_to_season =  { 1:'SBU',  2:'AHZ', 3:'AHZ',  
                     4:'AHZ',  5:'AHZ', 6:'SUM',  7:'SUM',  8:'SUM', 9:'SUM', 10:'SBU', 
                     11:'SBU', 12:'SBU'}
                     
abb_to_name = { 'SBU':'Slow build up', 'AHZ':'Arctic Haze', 'SUM':'Summer/Clean'}
name_to_abb = {'Slow build up':'SBU','Arctic Haze': 'AHZ', 'Summer/Clean':'SUM'}
    
def add_year_month_ordinal(df):   
    df = df.copy()
    df.loc[:,'timestamp'] = pd.to_datetime(df.index)
    df.loc[:,'day'] = df['timestamp'].dt.day
    df.loc[:,'month'] = df['timestamp'].dt.month
    df.loc[:,'year'] = df['timestamp'].dt.year    
    df.loc[:,'year_num'] = df['year'] - df['year'].iloc[0]
    df.loc[:,'month_ordinal'] = df['year_num']*12 + df['month']
    df.loc[:,'year_num'] = df['year_num'] + 1
    return df
    
def add_datetime_columns_and_resample_D(df, resample=False):
    #df = df[df > 0]
    df = add_year_month_ordinal(df)
    df = create_month_season_numbers(df)
    df['ordinal'] = df['timestamp'].apply(lambda x: x.toordinal()) 
    df['ordinal'] = df['ordinal'] - df['ordinal'].iloc[0] + 1
    df['DOY'] = df.index.strftime('%j')
    df['DOY'] = df['DOY'].astype(int)
    
    if resample == True:
        df_D = df.resample('D').mean()
        df_D['DOY'] = df_D.index.strftime('%j')
        df_D['DOY'] = df_D['DOY'].astype(int)
        df_D['datetime'] = df_D.index
        return df, df_D
    return df
    
# def create_month_season_numbers(df, full_season_to_season_num=None):
    # start_year = df.index.year[0]
    #print("First year in data set: "+str(start_year))
    # number_years = len(df.index.year.unique())+1
        
    # df = df.copy()
    # df.loc[:,'month_num'] = df.index.month
    # df.loc[:,'year'] = df.index.year
    # df = df.copy()
    # df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    # df['season_name'] = df['season_abb'].map(abb_to_name)  
    
    # df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    # print("Note: the slow build-up season crosses over two years as it goes from October-January, so the year corresponds to previous year")
    # df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))

    # seasons = df.season_abb_year.unique()
    # print("Number of unique seasons: "+str(len(seasons)))    
    # seasons_num = np.arange(1,len(seasons)+1,1)
    # season_to_season_num = dict(zip(seasons, seasons_num))

    # df.loc[:,'season_ordinal'] = df['season_abb_year'].map(season_to_season_num)
    
    # df = df.sort_index()
    #return df
    
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
    
def create_month_season_numbers(df, full_season_to_season_num=None):
    start_year = df.index.year[0]
    end_year = df.index.year[-1]
    number_years = end_year - start_year + 1 #len(df.index.year.unique())+1        
    df.loc[:,'month_num'] = df.index.month
    df.loc[:,'year'] = df.index.year        
    df.loc[:,'season_abb'] = df.month_num.map(month_to_season).values
    df['season_name'] = df['season_abb'].map(abb_to_name)      
    df.loc[:, "season_abb_year"] = df["season_abb"].astype(str) + '_' +df.index.year.astype(str)
    #print("Note: the slow build-up season crosses over two years as it goes from October-January, so the year corresponds to previous year")
    df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"] = df.loc[(df['season_abb'] == 'SBU') & (df['month_num'] == 1),  "season_abb_year"].apply(lambda x: x[:-4]+str(int(x[-4:])-1))
    seasons = df.season_abb_year.unique()
    #print("Number of unique seasons: "+str(len(seasons)))    
    #seasons_num = np.arange(1,len(seasons)+1,1)
    #season_to_season_num = dict(zip(seasons, seasons_num))
    first_season = get_first_season(df)
    if full_season_to_season_num is None:        
        full_season_to_season_num = get_full_season_abb_years(start_year, number_years, first_season)
    df.loc[:,'season_ordinal'] = df['season_abb_year'].map(full_season_to_season_num)    
    df = df.sort_index()
    return df
    
    
def calculate_rolling_quantile(df, var, quantile, window, average_type='median'):
    idx = pd.date_range(min(df.index), max(df.index), freq='1H') #create index for every          
    df = df.reindex(idx, fill_value=np.nan)    
    df_rolling = df.rolling(window, min_periods=4, center=True).quantile(quantile, interpolation='midpoint')    
    df_rolling['DOY'] = df_rolling.index.strftime('%j')  
    if average_type == 'mean':
        df_percentile_DOY = df_rolling.groupby('DOY').mean()  
    if average_type == 'median':
        df_percentile_DOY = df_rolling.groupby('DOY').median() 
    df_percentile_DOY.index = df_percentile_DOY.index.astype(int)        
    return df, df_percentile_DOY
    
def create_df_of_percentiles(df, top_percentiles=[99,98,97,96,95,90,80,70,60,50],  var='abs637'):
    df_percentiles = pd.DataFrame()
    bottom_percentiles = [100-x for x in top_percentiles]
    percentiles = top_percentiles + bottom_percentiles 
    for percentile in percentiles:
        df_rolling, df_percentile_DOY = calculate_rolling_quantile(df, var, quantile=percentile/100, window=24*15)
        df_percentiles['percentile_'+str(percentile)] = df_percentile_DOY[var]
    return df_percentiles

def add_percentiles(df, df_percentiles, var, percentile):
    print(df_percentiles.columns)   
    
    df_DOY_top = df_percentiles[['percentile_'+str(percentile)]].copy()
    df_DOY_top = df_DOY_top.rename(columns={'percentile_'+str(percentile):str(var)+'_DOY_quantile_top'})
    
    df_DOY_bottom = df_percentiles[['percentile_'+str(100-percentile)]].copy()
    df_DOY_bottom = df_DOY_bottom.rename(columns={'percentile_'+str(100-percentile):str(var)+'_DOY_quantile_bottom'})
    
    df = pd.merge(df, df_DOY_top[[str(var)+'_DOY_quantile_top']], on='DOY')    
    df = pd.merge(df, df_DOY_bottom[[str(var)+'_DOY_quantile_bottom']], on='DOY')
    return df
    
def produce_differeent_rolling_quantiles(df, var='abs637'):
    df_DOY_50 = calculate_rolling_quantile(df, var=var, quantile=.5, window=24*15)[1]
    df_percentile_DOY_95, df_DOY_95 = calculate_rolling_quantile(df, var=var, quantile=.95, window=24*15)
    df_DOY_99 = calculate_rolling_quantile(df, var=var, quantile=.99, window=24*15)[1]
    df_DOY_01 = calculate_rolling_quantile(df, var=var, quantile=.01, window=24*15)[1]
    return df_DOY_50, df_percentile_DOY_95, df_DOY_95, df_DOY_99, df_DOY_01
    
def plot_percentiles(df_DOY_01 ,df_DOY_50, df_DOY_95, df_DOY_99, var='abs637', 
                     ylabel='$\sigma_{\mathrm{ap}}$ [Mm$^{-1}$]', 
                     ms=1, ymin=-0.1, ymax=2.1):
    fig, ax = plt.subplots(figsize=(12,4))

    norm = mpl.colors.Normalize(vmin=1, vmax=5)
    cmap = cmocean.cm.thermal
    m = cm.ScalarMappable(norm=norm, cmap=cmap)

    ax.plot(df_DOY_01.index, df_DOY_01[var], 'o-', label='1st', ms=ms, c=m.to_rgba(1))
    ax.plot(df_DOY_50.index, df_DOY_50[var], 'o-', label='50th', ms=ms, c=m.to_rgba(2))
    ax.plot(df_DOY_95.index, df_DOY_95[var], 'o-', label='95th', ms=ms, c=m.to_rgba(3))
    ax.plot(df_DOY_99.index, df_DOY_99[var], 'o-', label='99th', ms=ms, c=m.to_rgba(4))

    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_title('Base percentiles: taking rolling 15-day percentiles', loc='left')

    ax.set_xticks(np.arange(1,365,31))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    ax.set_ylim(ymin,ymax)
    plt.legend(title='Percentiles:', loc=1, frameon=False, markerscale=5)
    plt.show()
    return fig
    
def plot_timeseries(df_percentile_DOY_95, var='abs637'):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_percentile_DOY_95.index, df_percentile_DOY_95[var], 'o', label='df percentile', ms=1)
    plt.legend(frameon=False)
    plt.show()
    return fig
    
def quick_plot(df_percentile_DOY_95, df_DOY_95, var, ymax=None, ymin=0):
    fig, ax = plt.subplots(figsize=(8,4))
    ax.plot(df_percentile_DOY_95['DOY'], df_percentile_DOY_95[var], 'o', label='df percentile', ms=1)
    ax.plot(df_DOY_95.index, df_DOY_95[var], 'o', label='abs', ms=1)
    ax.set_xticks(np.arange(1,365,31))
    ax.set_xticklabels(['J','F','M','A','M','J','J','A','S','O','N','D'])
    
    if ymax is None:
        ymax = df_percentile_DOY_95[var].max()                 
        
    ax.set_ylim(ymin,ymax)
    
    plt.legend(frameon=False)
    plt.show()
    return fig
    
def quantiles_df(df_, df_D, df_DOY_99,  df_DOY_01, var):
    df_DOY_top = df_DOY_99.copy()
    df_DOY_top = df_DOY_top.rename(columns={var:str(var)+'_DOY_quantile_top'})

    df_DOY_bottom = df_DOY_01.copy()
    df_DOY_bottom = df_DOY_bottom.rename(columns={var:str(var)+'_DOY_quantile_bottom'})

    df_add_DOY_quantile_D = pd.merge(df_D, df_DOY_top[[str(var)+'_DOY_quantile_top']], on='DOY')
    df_add_DOY_quantile = pd.merge(df_, df_DOY_top[[str(var)+'_DOY_quantile_top']], on='DOY')

    df_add_DOY_quantile_D = pd.merge(df_add_DOY_quantile_D, df_DOY_bottom[[str(var)+'_DOY_quantile_bottom']], on='DOY')
    df_add_DOY_quantile = pd.merge(df_add_DOY_quantile, df_DOY_bottom[[str(var)+'_DOY_quantile_bottom']], on='DOY')

    return df_add_DOY_quantile_D, df_add_DOY_quantile 
    
def add_outlier(df, var):
    quantile_top=str(var)+'_DOY_quantile_top'
    quantile_bottom=str(var)+'_DOY_quantile_bottom'
    
    df = df.copy()
    df.loc[:,'outlier'] = 0 
    
    df = df.copy()
    #df['outlier'].iloc[np.where(df[var].values > df[quantile_top].values)] = 1 
    df.loc[(df[var].values > df[quantile_top].values), 'outlier'] = 1 
    
    df = df.copy()
    #df['outlier'].iloc[np.where(df[var].values < df[quantile_bottom].values)] = -1 
    df.loc[(df[var].values < df[quantile_bottom].values), 'outlier'] = -1 
    
    print("High outliers (proportion)")
    print(len(df[df['outlier'] == 1])/len(df))
    print("Low outliers (proportion)")
    print(len(df[df['outlier'] == -1])/len(df))
    return df
    
def plot_outliers(df_add_DOY_quantile, var='abs637', datetime_col='timestamp',
                  ylabel='$\sigma_{\mathrm{ap}}$, 637 nm [Mm$^{-1}$]'):    
    fig, ax = plt.subplots(figsize=(12,4))
    outlier_to_color = {1:'lightcoral', 0: 'k', -1:'blue'}
    outlier_to_label={1:'top',0: 'non-extreme', -1:'bottom'}
    for outlier in df_add_DOY_quantile['outlier'].unique():
        df_outlier = df_add_DOY_quantile[df_add_DOY_quantile.outlier == outlier]
        size = len(df_outlier)/len(df_add_DOY_quantile)*100
        size = significant_figures(size, sf_num=3)        
        plt.plot(df_outlier[datetime_col], df_outlier[var], 'o', ms=1, mfc=outlier_to_color[outlier], 
                 mec=outlier_to_color[outlier], label=outlier_to_label[outlier]+': '+str(size)+' %')
        thickax(ax)  
    ax.set_ylabel(ylabel, fontsize=12)
    plt.legend(loc=1, frameon=False)
    plt.show()
    return fig
    
def res_percentage_outliers(df, var, resolution='season_ordinal', top_quantile="top_quantile_0.95",
                                    bottom_quantile="bottom_quantile_0.05"):  
    
    top_quantile=str(var)+'_DOY_quantile_top'
    bottom_quantile=str(var)+'_DOY_quantile_bottom'  
    
    df = df.dropna(how='all')    
    df_resolution_to_percentage_outliers = pd.DataFrame(columns=["resolution", 
                                                            "hourly_points",
                                                            "number_of_hourly_points_above_top",
                                                            "number_of_hourly_points_below_bottom",
                                                            "percentage_above_outliers",
                                                            "percentage_below_outliers"])
    for resolution_unit in df[resolution].unique():
        
        df_resolution_unit = df[df[resolution] == resolution_unit]
        number_per_resolution_unit = len(df_resolution_unit)  
        number_above = len(df_resolution_unit[df_resolution_unit[var] > df_resolution_unit[top_quantile]])      
        try:
            percentage_above_outliers = (number_above/number_per_resolution_unit)*100
        except ZeroDivisionError: 
            percentage_above_outliers = np.nan        
        number_below = len(df_resolution_unit[df_resolution_unit[var] < df_resolution_unit[bottom_quantile]])      
        try:
            percentage_below_outliers = (number_below/number_per_resolution_unit)*100
        except ZeroDivisionError: 
            percentage_below_outliers = np.nan                
        df_resolution_to_percentage_outliers = df_resolution_to_percentage_outliers.append({"resolution":resolution_unit, 
                                                                                  "hourly_points":number_per_resolution_unit,
                                                                                  "number_of_hourly_points_above_top":number_above,
                                                                                  "number_of_hourly_points_below_bottom":number_below,
                                                                                  "percentage_above_outliers":percentage_above_outliers,
                                                                                  "percentage_below_outliers":percentage_below_outliers},
                                                                                  ignore_index=True)        
    return df_resolution_to_percentage_outliers
    
def vars_for_best_fit(x,y,freq):
    x_array = np.array(x)
    y_array = np.array(y)    
    x_array = sm.add_constant(x_array)    
    model = sm.OLS(y_array, x_array, missing='drop')
    results = model.fit()
    p = results.params[0]
    m = results.params[1]        
    m3f = '{0:.3f}'.format(m*(freq))
    p3f = '{0:.3f}'.format(p) 
    return p,m,m3f,p3f
    
def sf(sf_num):
    sf = '{0:.'+str(sf_num)+'f}'
    return sf
    
def add_line(df_seasons_concated, var, freq=3, linecolour = 'k', ax=None):
    x = df_seasons_concated['resolution'].values   
    y = df_seasons_concated[var].values    
    p,m,m3f,p3f = vars_for_best_fit(x,y, freq)
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
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    return ax
    
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
    
def produce_monthly_plot(df_month_to_percentage_outliers, freq, resolution="resolution", 
                        ylabel1='Percentage of outliers per month\n (number of 99th percentile exceedances) [%]', 
                         ylabel2='Num. hourly data points [-]', title='',
                        ymax=0.25, plot_top=True, plot_bottom=True, add_line_bool=True, 
                        start_year=2002):  
                        
    fig, ax = plt.subplots(figsize=(25,7))    
    if plot_bottom == True:
        if add_line_bool == True:    
            add_line(df_month_to_percentage_outliers, var='percentage_below_outliers', 
                     freq=freq, linecolour = 'k', ax=ax)
        ax.plot(df_month_to_percentage_outliers[resolution], df_month_to_percentage_outliers['percentage_below_outliers'], 
                'o-', label='% below', color='b')
    if plot_top == True:
        if add_line_bool == True:    
            add_line(df_month_to_percentage_outliers, var='percentage_above_outliers',
                     freq=freq, linecolour = 'k', ax=ax)
        ax.plot(df_month_to_percentage_outliers[resolution], df_month_to_percentage_outliers['percentage_above_outliers'], 
                'o-', label='% above', color='r')    
                
    ax.set_ylim(0, ymax)
    add_ticks(ax, ylabel=ylabel1)    
    ax2 = ax.twinx()    
    
    #'number_of_hourly_points_below_bottom'
    if plot_bottom == True:
        ax2.bar(df_month_to_percentage_outliers[resolution], 
                df_month_to_percentage_outliers['number_of_hourly_points_below_bottom'], alpha=0.1, label='num. hr. below',
                color='b')
    if plot_top == True:
        ax2.bar(df_month_to_percentage_outliers[resolution], 
                df_month_to_percentage_outliers['number_of_hourly_points_above_top'], alpha=0.1, label='num. hr. after',
               color='r')
    ax2.legend(frameon=False, loc=2, fontsize=25)
    ax2.minorticks_on()
    ax2.set_ylabel(ylabel2, fontsize=20, rotation=270, labelpad=20)
    ax2.tick_params(labelsize=20)
    #ax2.tick_params(which='major', labelsize=20, direction='in', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    #ax2.tick_params(which='minor', direction='in', length=4, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
        
    ticklabels = np.arange(start_year, 2021, 1)    
        
    ticks = np.arange(1, len(ticklabels)*freq-1, freq)   
    plt.xticks(ticks, ticklabels)
    plt.title(str(title), fontsize=25, loc='left')
    plt.show()
    return fig
    
def find_fractions_plot(df, var='accumulated', resolution='month_ordinal',
                        events='rainfall', percentile=99, ymax_high=50, ymax_low=12, plot_bottom=False,
                        plot_top=False, start_year=2002): 
    if resolution == 'month_ordinal':
        freq = 12
    if resolution == 'season_ordinal':
        freq = 3
        
    print("Frequency: "+str(freq))
                            
    df_res_to_percentage_outliers = res_percentage_outliers(df, var=var, resolution=resolution)
    df_res_to_percentage_outliers = df_res_to_percentage_outliers.sort_values('resolution')
    
    fig_high = produce_monthly_plot(df_res_to_percentage_outliers, freq=freq, 
                               title='Frequency of extremely high '+str(events)+' events along trajectory: exceedances of '+str(percentile)+'$^{\mathrm{th}}$',
                               ymax=ymax_high, plot_bottom=plot_bottom, start_year=start_year)
    
    fig_low = produce_monthly_plot(df_res_to_percentage_outliers, freq=freq,
                               title='Frequency of extremely low '+str(events)+' events along trajectory: occurances below of '+str(100-percentile)+'$^{\mathrm{st}}$',
                               ymax=ymax_low, plot_top=plot_top, start_year=start_year)
    return fig_high, fig_low
    
##############proportions of extremes due to extremes ###########################################################################################################

def produce_proportion_of_extremes_related_to_extremes(df_abs_extremes, resolution = 'season_ordinal', outlier_var='rain'):
    df_proportion_res_outliers = pd.DataFrame(columns=[resolution])
    for outlier in [0, 1, -1]:
        sum_outliers = df_abs_extremes.groupby(resolution).apply(lambda x: (x['outlier_'+str(outlier_var)]==outlier).sum())
        df_proportion_res_outliers[resolution] = sum_outliers.index
        df_proportion_res_outliers[str(outlier_var)+'_outlier_'+str(outlier)] = sum_outliers.values
    col_list = [str(outlier_var)+'_outlier_0', str(outlier_var)+'_outlier_1', str(outlier_var)+'_outlier_-1']
    df_proportion_res_outliers["sum"] = df_proportion_res_outliers[col_list].sum(axis=1)
    df_proportion_res_outliers[col_list] = df_proportion_res_outliers[col_list].div(df_proportion_res_outliers['sum'], axis=0)
    df_proportion_res_outliers[col_list] = df_proportion_res_outliers[col_list]*100
    return df_proportion_res_outliers
    
def add_line_proportions(df, var, resolution="month_ordinal", 
                         freq=12, linecolour = 'k', plot_LMS=True, plot_TS=True, ax=None):   
             
    x = df[resolution].values   
    y = df[var].values    
    
    if plot_LMS == True:
        p,m,m3f,p3f = vars_for_best_fit(x,y, freq)
        x_fit = x
        y_fit = m*x+p
        m = m*(freq)    
        ax.plot(x_fit, y_fit, ls='-', c=linecolour, label='LMS: y = '+str(m3f)+'$\,\mathdefault{x}$ +'+str(p3f),lw=1) 
    
    if plot_TS == True:
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
    ax.spines["right"].set_visible(False)  
    ax.get_xaxis().tick_bottom()  
    ax.get_yaxis().tick_left()  
    return ax
    
def mean_std_var(df_abs_extremes, var='accumulated', outlier=1, outlier_var='outlier_rain', units='[mm]'):
    mean_rain = df_abs_extremes.loc[(df_abs_extremes[outlier_var] == outlier), var].mean()
    std_rain = df_abs_extremes.loc[(df_abs_extremes[outlier_var] == outlier), var].std()
    values = str(significant_figures(mean_rain, sf_num=3))+'±'+str(significant_figures(std_rain, sf_num=3))+' '+str(units)    
    return values  

def find_string_in_list_of_strings(string, list_of_strings):
    matching = [s for s in list_of_strings if string in s] 
    return matching[0]
    
def produce_res_stacked_bar_with_trends_plot(df, var, resolution="month_ordinal", freq=12,
                         col_list = ['rain_outlier_0', 'rain_outlier_1', 'rain_outlier_-1'],
                         alphas = [0.4, 0.8, 0.6], colors=['black', 'blue', 'red'], hatches=['','',''],
                         labels = ['Wet (extreme high ATP) (mean + std)', 'Dry (extreme low ATP) (mean + std)', 'non-extreme'], 
                         event_type='clean events', df_full=None, var_full='accumulated', start_year=2002,
                         title='', ymax=100,  plot_LMS=True, plot_TS=True, units='', end_year=2021, 
                         ):
                         
    fig, ax = plt.subplots(figsize=(25,7)) 
    
    add_line_proportions(df, var, resolution=resolution, freq=freq, linecolour = 'k', plot_LMS=plot_LMS, plot_TS=plot_TS, ax=ax)    
    X = list(df[resolution])
    
    columns_and_colors = dict(zip(col_list, colors))
    columns_and_alpha = dict(zip(col_list, alphas))    
    columns_and_label = dict(zip(col_list, labels))
    columns_and_hatch = dict(zip(col_list, hatches))
            
    if df_full is not None:
        dict_cols_to_means = {}
        nonevent = find_string_in_list_of_strings('0', col_list)        
        dict_cols_to_means[nonevent] = ''
        values = mean_std_var(df_full, var=var_full, outlier=1, outlier_var='outlier_'+str(col_list[0][:-10]), units=units)
        highevent = find_string_in_list_of_strings('_1', col_list)
        dict_cols_to_means[highevent] = values
        lowevent = find_string_in_list_of_strings('_-1', col_list)
        values = mean_std_var(df_full, var=var_full, outlier=-1, outlier_var='outlier_'+str(col_list[0][:-10]), units=units)
        dict_cols_to_means[lowevent] = values    
    
    print(produce_res_stacked_bar_with_trends_plot)
    
    bottom_append = [0]*len(df)        
    for count, col in enumerate(col_list):
        y = df[col].values 
        s=np.isnan(y)
        y[s]=0.0                              
        plt.bar(X, y, width=1, bottom=bottom_append, color=columns_and_colors[col], 
               align='center', linewidth=5, edgecolor = 'white', alpha=columns_and_alpha[col], 
               label=columns_and_label[col]+': '+str(dict_cols_to_means[col] ),
               hatch=columns_and_hatch[col])
        bottom_append = bottom_append + y    
    ax.set_ylim(0, ymax)
 
    ax.legend(frameon=False, loc=2, fontsize=25)
    ax.minorticks_on()
    
    if df_full is not None:
        mean = df_full.abs637.mean()
        std = df_full.abs637.std()
        values = str(significant_figures(mean, sf_num=3))+'±'+str(significant_figures(std, sf_num=3))+' [Mm$^{-1}$]'
        ylabel='Proportion of '+str(len(df_full))+' '+str(event_type)+'\n('+str(values)+') [%]'
    if df_full is None: 
        ylabel='Proportion of '+str(event_type)+' events [%]'
    
    ax.set_ylabel(ylabel, fontsize=20) 
    
    ax.tick_params(labelsize=20)
    ax.tick_params(which='major', labelsize=20, direction='in', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(which='minor', direction='in', length=4, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
    ticklabels = np.arange(start_year, end_year, 1)
    ticks = np.arange(1, len(ticklabels)*freq-1, freq)   
    plt.xticks(ticks, ticklabels)
    plt.title(str(title), fontsize=25, loc='left')
    plt.show()
    return fig