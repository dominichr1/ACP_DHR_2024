import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import datasets, linear_model
from matplotlib.ticker import AutoMinorLocator
import cmocean
from scipy import stats
import glob
import os
from matplotlib.colors import LogNorm
import matplotlib as mpl

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
    
def save_plot(fig, path=r'C:\Users\DominicHeslinRees\Pictures\black_carbon\final_plots', folder='', 
              name='default_name', formate=".jpeg", dpi=300):
    folders = glob.glob(path)    
    if folder not in folders:
        print("make folder")
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
    
def find_number_of_duplicate_indexs(df):
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are: "+str(len(duplicateRowsDF)))      
    return duplicateRowsDF 
    
def remove_duplicates(df):
    print("Length before: "+str(len(df)))
    duplicateRowsDF = df.index[df.index.duplicated()]
    print("Duplicate Rows except first occurrence based on all columns are : "+str(len(duplicateRowsDF)))      
    df_first = df.loc[~df.index.duplicated(keep='first')]
    df_last = df.loc[~df.index.duplicated(keep='last')]
    print("Length after: "+str(len(df_first)))
    print("Length after: "+str(len(df_last)))
    return df_first, df_last
    
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
    
def significant_figures(value, sf_num=3):
    sf = '{0:.'+str(sf_num)+'f}'
    value_sf = sf.format(value)    
    return value_sf

def multiIndex_into_one_Index(df):
    df.columns = [' '.join(col).strip() for col in df.columns.values]
    return df
	
def hourly_resample(df, cols=['abs_neph', 'scat550', 'Tr'], functionlist=['mean', 'median', 'std', 'var']):
    dfhourly = df[cols].resample('60T').agg(functionlist)
    dfhourly = multiIndex_into_one_Index(dfhourly)
    print(dfhourly.columns)
    return dfhourly
	
def mergedfs(df1, df2):
    df_merged = pd.merge(df1, df2, how='inner', left_index=True, right_index=True)
    #print(df_merged.columns)
    return df_merged

def limit_dfs_values(df, x_var, y_var, xmin=0, xmax=60, ymin=0, ymax=60):
    df.loc[df[x_var] < xmin, [x_var]] = np.nan
    df.loc[df[x_var] > xmax, [x_var]] = np.nan
    df.loc[df[y_var] < ymin, [y_var]] = np.nan
    df.loc[df[y_var] > ymax, [y_var]] = np.nan
    df = df.dropna(subset=[x_var, y_var], how='any')
    return df
    
def theilslope(x,y):
    res = stats.theilslopes(y, x, 0.90)
    mid_slope = res[0]
    med_intercept = res[1]
    return mid_slope,med_intercept
    
def using_hist2d(ax, x, y, bins=(10, 10)):
    ax.hist2d(x, y, bins, cmap='Blues')
    
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
    
def create_simple_regession_plot(df, x_var, y_var, set_max_value, set_min_value=-0.1, dict_abs_labels={}, title='', vmin=0, vmax=1, 
                                 xlabel='', ylabel='', xmin=0, xmax=60, ymin=0, ymax=60, stepsize=.5, show_points=True, 
                                 loc_legend='lower right', plot_annomalies=False, percentile=95, fs_label=20, legend_out_of_box=True, 
                                 loc_legend_out_box=1.4, s=10, ax=None):    
    single_plot = False
    if ax is None:  
        fig, ax = plt.subplots(figsize=(4,4)) 
        single_plot = True        
        
    df = limit_dfs_values(df, x_var, y_var, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) #limits    
    df = df.copy()    

    df.loc[:, 'datetime'] = df.index
    df = df[[x_var, y_var]].copy()    
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] #remove any infs or np.nan 
    #LMS
    model = linear_model.LinearRegression(fit_intercept=True)    
    x = np.array(df[x_var].values)
    y = np.array(df[y_var].values) 
    
    model.fit(x[:, np.newaxis], y)
    R2 = model.score(x[:, np.newaxis], y)
    R2 =  "{:.2f}".format(float(R2))
    xfit = x
    yfit = model.predict(xfit[:, np.newaxis])
    intercept = "{:.2f}".format(float(model.intercept_))
    coef = "{:.2f}".format(float(model.coef_[0]))    
    cf_LMS = float(1/(model.coef_[0]))

    #ax.tick_params(bottom=True, top=True, left=True, right=True)

    ax.plot(np.linspace(0, set_max_value, num=10), np.linspace(0, set_max_value, num=10), linestyle='--', alpha=0.5, c='k')
    
    if np.sign(float(intercept)) == 1:
        sign_str = '+'
    else:
        sign_str = ''
    if show_points == True:
        smap = ax.scatter(x, y, s=s, edgecolors='k', c='None', lw=0.5, alpha=0.6, vmin=0.5, vmax=1)
    
    ax.plot(xfit, yfit, ls=':', color='blue', linewidth=2, label='Fitted Line: \ny = '+str(coef)+'$\,$x '+str(sign_str)+str(intercept)+
           '\nR$^{2}$ = '+str(R2)+'\nCF = '+str("{:.2f}".format(cf_LMS) ))    
    ###########################################################################################
    if (x_var and y_var) in [*dict_abs_labels.keys()]: 
        ax.set_xlabel(dict_abs_labels[x_var],fontsize=fs_label)
        ax.set_ylabel(dict_abs_labels[y_var],fontsize=fs_label)        
    if (x_var and y_var) not in [*dict_abs_labels.keys()]: 
        ax.set_xlabel(xlabel,fontsize=fs_label)
        ax.set_ylabel(ylabel,fontsize=fs_label)
        
    ax.set_title(title + ' No. data points: '+str(len(df))+',\n '+str(df.index[0].strftime("%d-%m-%Y")) +' to '+str(df.index[-1].strftime("%d-%m-%Y")),
                 loc='left', fontsize=18)
    mid_slope,med_intercept = theilslope(x,y)    
    intercept_ts = "{:.2f}".format(float(med_intercept))
    coef_ts = "{:.2f}".format(float(mid_slope))
    cf_ts = float(1/float(mid_slope))    
    
    ax.plot(x, med_intercept + mid_slope * x, c='red', lw=2, ls='--', 
            label='Theil-Sen Slope: \ny = '+str(coef_ts)+'$\,$x + '+str(intercept_ts)+'\nCF$_{\mathrm{TS}}$ = '+str("{:.2f}".format(cf_ts)))   
        
    ax.tick_params(which='major', direction='in', length=6, width=1.3, pad=10, bottom=True, top=True, left=True, right=True, color='k',
                   labelsize=20)
    ax.tick_params(axis='both', which='minor', direction='in', length=3, color='k', width=1.3, 
                   bottom=True, top=True, left=True, right=True, labelsize=20) 
                   
    #ax.set_yticks(np.arange(0, set_max_value, stepsize))
    #ax.set_xticks(np.arange(0, set_max_value, stepsize))  
    
    ax.set_xlim(set_min_value, set_max_value)
    ax.set_ylim(set_min_value, set_max_value)
    ax.minorticks_on()    
    ax.set_yticks(np.arange(int(set_min_value), int(set_max_value)+1, 1))
    ax.set_xticks(np.arange(int(set_min_value), int(set_max_value)+1, 1))

    #using_hist2d(ax, x, y, bins=(50, 50))
    
    if plot_annomalies == True:
        df["Distance_LMS"] = abs(y -  yfit) #distance from line of best fit
        df["Distance_TS"] = abs(y -  (med_intercept + mid_slope * x)) #/y    
        Top_perecent = np.percentile(df["Distance_TS"], percentile) #biggest from line of best fit   
        df_anomalies = df[df["Distance_TS"] > Top_perecent]  
        df_anomalies['y_predict'] = med_intercept + mid_slope*df_anomalies[x_var]
        df_anomalies = df_anomalies[df_anomalies['y_predict'] > df_anomalies['abs637 mean_y']]   
        ax.scatter(df_anomalies[x_var], df_anomalies[y_var], c='red', s=35, edgecolors='r', lw=0.5, alpha=0.5,
                   label='top '+str(percentile)+'$^{\mathrm{th}}$ from TS + below') 
            
    ax.legend(frameon=False, fontsize=18, loc=loc_legend)  
    if legend_out_of_box == True:
        ax.legend(bbox_to_anchor=(loc_legend_out_box, 1), borderaxespad=0, frameon=False, fontsize=15)
    ax.grid(True, alpha=0.2)
    fancy(ax, fontsize=20)
        
    dict_stats = {}    
    dict_stats['cf_LMS'] = cf_LMS
    dict_stats['cf_TS'] = cf_ts
    dict_stats['coef'] = coef  
    dict_stats['coef_ts'] = coef_ts
    dict_stats['R2'] = R2
    dict_stats['length'] = len(df)
    dict_stats['start'] = df.index[0].strftime("%Y-%m-%d")
    dict_stats['stop'] = df.index[-1].strftime("%Y-%m-%d")   
    if single_plot == False:
        return ax
    if single_plot == True:
        plt.show()    
        return fig, dict_stats
	
def create_regession_plot(df, x_var, y_var, c_var, set_max_value, set_min_value=-0.1, dict_abs_labels={}, title='', vmin=0, vmax=1, 
                          percentile=99, clabel='', xlabel='', ylabel='', xmin=0, xmax=60, ymin=0, ymax=60, 
                          loc_legend='lower right', legend_out_of_box=True, ax=None):    
    single_plot = False
    if ax is None:  
        fig, ax = plt.subplots(figsize=(4,4)) 
        single_plot = True        
              
    df = limit_dfs_values(df, x_var, y_var, xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax) #limits    
    df_copy = df.copy()    
    df = df_copy.copy()    
    df.loc[:, 'datetime'] = df.index
    df = df[[x_var, y_var, c_var]].copy()    
    df = df[~df.isin([np.nan, np.inf, -np.inf]).any(1)] #remove any infs or np.nan  
    
    model = linear_model.LinearRegression(fit_intercept=True)
    
    x = np.array(df[x_var].values)
    y = np.array(df[y_var].values) 
    c = np.array(df[c_var].values) 
    
    model.fit(x[:, np.newaxis], y)
    R2 = model.score(x[:, np.newaxis], y)
    R2 =  "{:.2f}".format(float(R2))

    xfit = x
    yfit = model.predict(xfit[:, np.newaxis])

    intercept = "{:.2f}".format(float(model.intercept_))
    coef = "{:.2f}".format(float(model.coef_[0]))    
    cf_LMS = float(1/(model.coef_[0]))

    #ax.tick_params(bottom=True, top=True, left=True, right=True)
    #ax.xaxis.set_minor_locator(AutoMinorLocator())
    #ax.yaxis.set_minor_locator(AutoMinorLocator())
   
    ax.plot(np.linspace(0, set_max_value, num=10), np.linspace(0, set_max_value, num=10), linestyle='--', alpha=0.5)
    
    if 'Tr' in c_var:
        print("Colour "+str(c_var))
        smap = ax.scatter(x, y, c=c, s=35, edgecolors='k', lw=0.5, cmap = cmocean.cm.deep, alpha=0.6, vmin=0.5, vmax=1)
    if 'volume' in c_var:
        print("Colour "+str(c_var))
        smap = ax.scatter(x, y, c=c, s=35, edgecolors='k', lw=0.5, cmap = cmocean.cm.deep, alpha=0.6, vmin=min(c), vmax=max(c))                   
    if c_var == 'datetime': 
        smap = ax.scatter(x, y, c=c, s=35, edgecolors='k', lw=0.5, cmap = cmocean.cm.deep, alpha=0.6)

    if np.sign(float(intercept)) == 1:
        sign_str = '+'
    else:
        sign_str = ''

    ax.plot(xfit, yfit, ls=':', color='blue', linewidth=2, label='Fitted Line: \ny = '+str(coef)+'$\,$x '+str(sign_str)+str(intercept)+
           '\nR$^{2}$ = '+str(R2)+'\nCF$_{\mathrm{LMS}}$ = '+str("{:.2f}".format(cf_LMS) ))
    
    #colorbar#################################################################################
    N_TICKS = 5
    cbaxes = fig.add_axes([0.95, 0.1, 0.03, 0.8]) 
    
    if c_var == 'datetime':
        indexes = [df.index[i] for i in np.linspace(0,df.shape[0]-1,N_TICKS).astype(int)] 
        cb = fig.colorbar(smap, orientation='vertical',
                  ticks= df.loc[indexes].index.astype(int), cax = cbaxes)
        if c in [*dict_abs_labels.keys()]: 
            cb.ax.set_yticklabels([index.strftime('%d %b %Y') for index in indexes])
    if 'Tr' or 'volume' in c_var:
        cb = fig.colorbar(smap, orientation='vertical', cax = cbaxes)  
    
    #label
    if c_var in [*dict_abs_labels.keys()]: 
        cb.ax.set_ylabel(str(dict_abs_labels[c_var]), rotation=270,  labelpad=25, fontsize=25)           
        cb.ax.tick_params(labelsize=15)
        #ticklabs = cbar.ax.get_yticklabels()
        #cb.ax.set_yticklabels(ticklabs, fontsize=15)
    
    if c_var not in [*dict_abs_labels.keys()]: 
        cb.ax.tick_params(labelsize=15)
        cb.ax.set_ylabel(clabel, rotation=270,  labelpad=25, fontsize=25)   
    
    ###########################################################################################
    if (x_var and y_var) in [*dict_abs_labels.keys()]: 
        ax.set_xlabel(dict_abs_labels[x_var],fontsize=15)
        ax.set_ylabel(dict_abs_labels[y_var],fontsize=15)
        
    if (x_var and y_var) not in [*dict_abs_labels.keys()]: 
        ax.set_xlabel(xlabel,fontsize=15)
        ax.set_ylabel(ylabel,fontsize=15)

    ax.set_title(title + ' No. data points: '+str(len(df))+',\n '+str(df.index[0].strftime("%d-%m-%Y")) +' to '+str(df.index[-1].strftime("%d-%m-%Y")))
    mid_slope,med_intercept = theilslope(x,y)
    
    intercept_ts = "{:.2f}".format(float(med_intercept))
    coef_ts = "{:.2f}".format(float(mid_slope))
    cf_ts = float(1/float(mid_slope))
    
    ax.plot(x, med_intercept + mid_slope * x, c='red', lw=2, ls='--', 
            label='Theil-Sen Slope: \ny = '+str(coef_ts)+'$\,$x + '+str(intercept_ts)+'\nCF$_{\mathrm{TS}}$ = '+str("{:.2f}".format(cf_ts)))
      
    #df = pd.concat([df, df_copy], axis=1, join="inner")    
    df["Distance_LMS"] = abs(y -  yfit) #distance from line of best fit
    df["Distance_TS"] = abs(y -  (med_intercept + mid_slope * x)) #/y    
    Top_perecent = np.percentile(df["Distance_TS"], percentile) #biggest from line of best fit   
    df_anomalies = df[df["Distance_TS"] > Top_perecent]
    
    #df_anomalies = df_anomalies.apply(lambda x: np.nan if x[y_var] < (med_intercept + mid_slope*x[x_var]) else x, axis=1) 
    df_anomalies = df_anomalies.copy()
    df_anomalies['y_predict'] = med_intercept + mid_slope*df_anomalies[x_var]
    df_anomalies = df_anomalies[df_anomalies['y_predict'] > df_anomalies[y_var]]   
    ax.scatter(df_anomalies[x_var], df_anomalies[y_var], c='red', s=35, edgecolors='r', lw=0.5, alpha=0.5,
               label=str(percentile)+'th from TS') #
    
    ax.legend(frameon=False, fontsize=15, loc=loc_legend) 
    if legend_out_of_box == True:
        ax.legend(bbox_to_anchor=(1.4, 1), borderaxespad=0, frameon=False, fontsize=15)
    ax.minorticks_on()
    ax.tick_params(direction='in', length=6, width=1.3, pad=10, bottom=True, top=True, left=True, right=True, color='k',
                   labelsize=15, axis='both')
    ax.tick_params(which='minor', direction='in', length=3, color='k', width=1.3, bottom=True, top=True, left=True, right=True,
                   labelsize=15, axis='both')
    ax.set_xlim(set_min_value, set_max_value)
    ax.set_ylim(set_min_value, set_max_value)
    ax.set_yticks(np.arange(int(set_min_value), int(set_max_value)+1, 1))
    ax.set_xticks(np.arange(int(set_min_value), int(set_max_value)+1, 1))
        
    ax.grid(True, alpha=0.5)
    plt.show()    
    
    dict_stats = {}
    
    dict_stats['cf_LMS'] = cf_LMS
    dict_stats['cf_TS'] = cf_ts
    dict_stats['coef'] = coef  
    dict_stats['coef_ts'] = coef_ts
    dict_stats['R2'] = R2
    dict_stats['length'] = len(df)
    dict_stats['start'] = df.index[0].strftime("%Y-%m-%d")
    dict_stats['stop'] = df.index[-1].strftime("%Y-%m-%d")
    
    if single_plot == False:
        return ax
    if single_plot == True:
        plt.show()    
        return fig, dict_stats, df_anomalies
	    
   
def get_nearest(data, timestamp):
    idx = data.index[data.index.get_loc(timestamp, method='nearest')]
    return idx
	
def create_annomly_timeseries(df_anomalies, df_main, RAW_NEW_PSAP, var_main, var_anomalies, percentile, 
                              filter_num_col='Filter_cum', min_index=None, max_index=None, set_max_value=10, 
                              name='PSAP', plot_raw=True, ms=5, df_Mie=None, Mie_var='scat550', Miemax=40, 
                              label1='',label2='',labelMie='',  plot_ax1_only=False):  
    
    if plot_ax1_only == True:
        fig, ax1 = plt.subplots(1,1, figsize=(12,3))
    if plot_ax1_only == False:  
        fig, (ax1,ax2) = plt.subplots(2,1, figsize=(12,6))
        fig.subplots_adjust(hspace=.5)

    ax1.set_title(str(name), loc='left')
    ax1.plot(df_main.index, df_main[var_main], 'o', mfc='None', ms=ms, label='')
    print(df_anomalies[var_anomalies].head(2))    
    
    ax1.plot(df_anomalies.index, df_anomalies[var_anomalies], 'o', mec='r', mfc='r', ms=ms, 
             label=str(percentile)+'th from TS')        
    ax1.set_ylim(0, set_max_value)
    ax1.legend(frameon=False, loc=1, fontsize=15, markerscale=2) 
    ax1.set_ylabel(label1, fontsize=20)   
    
    if df_Mie is not None:
        ax12 = ax1.twinx()
        print(df_Mie[Mie_var])
        ax12.plot(df_Mie.index, df_Mie[Mie_var], 'D', mec='k', mfc='k', ms=ms, alpha=0.4,
                 label=str(Mie_var)) 
        ax12.set_ylim(0, Miemax)
        ax12.set_ylabel(labelMie, fontsize=15)   
        ax12.tick_params(labelsize=12, direction='in', length=6, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
        ax12.tick_params(labelsize=12, which='minor', direction='in', length=3, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
        fancy(ax12, fontsize=12, spines=['top','bottom','left','right'])

    if min_index is None:
        min_index = df_anomalies.index[0] - pd.DateOffset(days=5)    
    if max_index is None:
        max_index = df_anomalies.index[-1] + pd.DateOffset(days=5)
    else:
        min_index = pd.to_datetime(min_index)
        max_index = pd.to_datetime(max_index)
    
    if plot_ax1_only == False:
        ####plot filters i.e. raw data
        if plot_raw == True:        
            indexes = []
            for timestamp in df_anomalies.index: 
                index = get_nearest(RAW_NEW_PSAP, timestamp)
                indexes.append(index)

            
            RAW_NEW_PSAP_anomalises = RAW_NEW_PSAP[RAW_NEW_PSAP.index.isin(indexes)]    
            ax2.plot(RAW_NEW_PSAP.index, RAW_NEW_PSAP['Io'], 'o', mfc='None', ms=2, label='I$_{\mathrm{o}}$')
            ax2.plot(RAW_NEW_PSAP.index, RAW_NEW_PSAP['I'], 'o', mfc='None', ms=2, label='I')
            ax2.plot(RAW_NEW_PSAP.index, RAW_NEW_PSAP['qobs'], 'o', mfc='None', ms=2, label='Q$_{\mathrm{obs}}$')
            
            #print(RAW_NEW_PSAP)
            #for x_time in RAW_NEW_PSAP:
            #    print(x_time)
            #    ax2.vlines(x=pd.to_datetime(x_time), ymin=0, ymax=30, colors='r')

            ax2.plot(RAW_NEW_PSAP_anomalises.index, RAW_NEW_PSAP_anomalises['Io'], 'o', mec='r', mfc='r', ms=ms)
            ax2.plot(RAW_NEW_PSAP_anomalises.index, RAW_NEW_PSAP_anomalises['I'], 'o', mec='r', mfc='r', ms=ms)
            ax2.plot(RAW_NEW_PSAP_anomalises.index, RAW_NEW_PSAP_anomalises['qobs'], 'o', mec='r', mfc='r', ms=ms)
            ax2.set_ylabel(label2, fontsize=15)   
            
            unique_indexs = RAW_NEW_PSAP_anomalises[filter_num_col].unique()
            unique_indexs_commas = [str(x).replace("'",'')+','for x in unique_indexs]        
            
            ax2.legend(frameon=False, loc=1, fontsize=15, ncol=3, markerscale=2) 
            ax2.set_title(label='Filters with anomalies:\n'+str(unique_indexs_commas)[1:-2],  loc='left', fontsize=10)   
        else:
            indexes = []
            unique_indexs = []
            
    if plot_ax1_only == True:
        axs = [ax1]
    if plot_ax1_only == False:
        axs = [ax1, ax2]
        
    for ax in axs:       
        ax.set_xlim(min_index, max_index)
        ax.yaxis.set_ticks_position('left')
        ax.xaxis.set_ticks_position('bottom')  
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.tick_params(labelsize=12, direction='in', length=6, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
        ax.tick_params(labelsize=12, which='minor', direction='in', length=3, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
        fancy(ax, fontsize=12, spines=['top','bottom','left','right'])
        
    #plt.grid(True, alpha=0.5)
    plt.show()
    if plot_ax1_only == False:
        return fig, unique_indexs
    if plot_ax1_only == True:
        return fig

def plot_each_filter(dfraw, filter_num=1, filter_var='Filter_cum', ms=2, ylabel='I/Io'):   
    df_filter = plot_filter(dfraw, filter_num=filter_num, filter_var=filter_var)
    
    if len(df_filter) > 0:    
        fig, ax = plt.subplots(figsize=(20,6))  
        ax.plot(df_filter.index, df_filter['I'], 'o', c='b', ms=ms, label='I')
        ax.plot(df_filter.index, df_filter['Io'], 'o', c='k', ms=ms, label='Io')
        ax.set_ylim(-0.1, 10)
        ax.set_ylabel(ylabel, fontsize=20)
        ax.legend(fontsize=20, loc=1, frameon=False)
        ax2 = ax.twinx()
        ax2.plot(df_filter.index, df_filter['qobs'], 'o', c='g', ms=ms, label='qobs')
        ax2.set_ylim(-0.1, 2.1)
        ax2.set_ylabel('Q$_{obs}$', fontsize=20)
        ax2.legend(fontsize=20, loc=2, frameon=False)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.tick_params(axis='both', which='minor', labelsize=12)        
        ax2.tick_params(axis='both', which='major', labelsize=15)
        ax2.tick_params(axis='both', which='minor', labelsize=12)        
        plt.title("Filter: "+str(filter_num)+" start: "+str(df_filter.index[0].date())+' - '+str(df_filter.index[-1].date())+' '+str(len(df_filter)), fontsize=20)       
        plt.grid(True) 
        plt.show()
		
def plot_filter(df, filter_num=1, filter_var='Filter_cum'):
    df_filter = df.loc[df[filter_var] == int(filter_num)]
    return df_filter
    
def apply_correction(df, cf, mean_or_median, var='abs637'):
    var=str(var)+' '+str(mean_or_median)
    print("Applying correction factor of "+str(cf))
    df[var+'_cf'] = df[var]*float(cf)
    print("new variable: "+str(var+'_cf'))
    return df
    
def quick_timeseries_plot(df_oldpsap637_hourly, df_oldpsapEBAS637_hourly, var='abs637 mean',
                         start='2006-12-22 06:00:00', end='2006-12-30 21:00:00'):
    fig, ax = plt.subplots(1,1, figsize=(10,2))
    fig.subplots_adjust(hspace=.5)
    ax.plot(df_oldpsap637_hourly.index, df_oldpsap637_hourly[var], 'o', mfc='None', ms=1, label='abs')
    ax.plot(df_oldpsapEBAS637_hourly.index, df_oldpsapEBAS637_hourly[var], 'o', mfc='None', ms=1, label='EBAS')
    ax.set_ylim(0, 2)
    ax.set_xlim(pd.to_datetime(start), pd.to_datetime(end))
    plt.legend(frameon=False)
    plt.show()
    return fig
    
def append_all_dfs(dfoldpsapEBAS, dfoldpsap, dfnewpsap, dfmaap, oldpsapEBASabsvar='abs637 mean', 
                   oldpsapabsvar='abs637 mean_cf', newpsapabsvar='abs637 mean_cf',
                   maapabsvar='absorption'): #notice the maap var has no cf
    
    dfoldpsapEBAS = dfoldpsapEBAS.copy()
    print("old PSAP EBAS")
    print('Start time: '+str(dfoldpsapEBAS.index[0])+' End time: '+str(dfoldpsapEBAS.index[-1]))
    dfoldpsapEBAS['abs637'] = dfoldpsapEBAS[oldpsapEBASabsvar] 
    dfoldpsapEBAS = dfoldpsapEBAS[['abs637']]    
    
    dfoldpsap = dfoldpsap.copy()
    print("old PSAP")
    print('Start time: '+str(dfoldpsap.index[0])+' End time: '+str(dfoldpsap.index[-1]))
    dfoldpsap['abs637'] = dfoldpsap[oldpsapabsvar] 
    dfoldpsap = dfoldpsap[['abs637']]    
        
    dfnewpsap = dfnewpsap.copy()
    print("new PSAP")
    print('Start time: '+str(dfnewpsap.index[0])+' End time: '+str(dfnewpsap.index[-1]))
    dfnewpsap['abs637'] = dfnewpsap[newpsapabsvar] 
    dfnewpsap = dfnewpsap[['abs637']]
                          
    dfmaap = dfmaap.copy()
    print("MAAP")
    print('Start time: '+str(dfmaap.index[0])+' End time: '+str(dfmaap.index[-1]))
    dfmaap['abs637'] = dfmaap[maapabsvar] 
    dfmaap = dfmaap[['abs637']]
       
    dfoldpsapEBAS = dfoldpsapEBAS[dfoldpsapEBAS.index < dfoldpsap.index[0]]
    dfnewpsap = dfnewpsap[dfoldpsap.index[-1] < dfnewpsap.index] #newpsap begins after old psap
    dfnewpsap = dfnewpsap[dfnewpsap.index < dfmaap.index[0]] #newpsap stops when maap begins
    
    old_PSAPEBAS_min = significant_figures(dfoldpsapEBAS.min().values[0])
    print("Minimum value for EBAS: "+str(old_PSAPEBAS_min))    
    old_PSAP_min = significant_figures(dfoldpsap.min().values[0])
    print("Minimum value for EBAS: "+str(old_PSAP_min))
    new_PSAP_min = significant_figures(dfnewpsap.min().values[0])
    print("Minimum value for EBAS: "+str(new_PSAP_min))
    MAAP_min = significant_figures(dfmaap.min().values[0])
    print("Minimum value for EBAS: "+str(MAAP_min))
    
    DFs = [dfoldpsapEBAS, dfoldpsap, dfnewpsap, dfmaap] #all data
                          
    dfoldpsap_newpsap_maap = pd.concat(DFs)
    dfoldpsap_newpsap_maap = dfoldpsap_newpsap_maap.dropna()

    return dfoldpsap_newpsap_maap

def remove_outliers(df, var='abs637', top_percentile_value=99.99, keep_extreme_values=True): #resample_use_detection_limits
    df = df.copy()
    array = df[var].values
    
    print("Using the perecentile: "+str(top_percentile_value))
    top_percentile = np.percentile(array, top_percentile_value)
    
    bottom_percentile_value=100-top_percentile_value    
    bottom_percentile = np.percentile(array, bottom_percentile_value)
    
    print("Smallest value: "+str(bottom_percentile))    
    print("Largest value: "+str(top_percentile))
    
    if keep_extreme_values == False:
        df = df[df[var] >= bottom_percentile]
        df = df[df[var] <= top_percentile]
    if keep_extreme_values == True:
        pass
    return df
    
def quick_log_plot(df):
    fig, ax = plt.subplots(figsize=(10,5))
    ax.plot(df.index, df.abs637, 'o', ms=1)
    plt.ylim(10**(-4),10**(1))
    plt.yscale('log')
    plt.show()
    return fig
    
def quick_plot(df, df_oldpsap637_hourly, df_newpsap637_hourly, df_MAAP, 
               var='abs_homogenised', ylabel='$\sigma_{\mathrm{ap, 637nm}}$ ', wavelength=637, 
               units='[Mm$^{-1}$]',ymax=30, fontsize=25, add_labels=True, label_factor=1.05, ax=None):        
    if ax is None:
        print("return fig")
        fig, ax = plt.subplots(figsize=(20,6)) 
        
    #ylabel=str(ylabel)+str(wavelength)+' nm '+str(units)
    ylabel=str(ylabel)+str(units)
    ax.plot(df.index, df[var],'o', ms=1, mfc='none', mec='k', label='')
    ax.set_ylabel(ylabel, fontsize=fontsize)
    xcoords = [df_oldpsap637_hourly.index[-1], df_newpsap637_hourly.index[-1], df_MAAP.index[-1]]
    labels = ['manual PSAP', 'automatic PSAP', 'MAAP']
    trans = ax.get_xaxis_transform()    
    
    if add_labels==True:
        for xc, label in zip(xcoords, labels):
            plt.axvline(x=xc, ls=':', label='line at x = {}'.format(xc), c='k', alpha=0.5)
            plt.text(xc, float(ymax)*label_factor, label, rotation=0, horizontalalignment='right', alpha=0.5,  fontsize=12)
            
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(True)    
    
    fancy(ax, fontsize=20, spines=['bottom','left'], alpha=0)
    
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())   
    ax.minorticks_on()
    ax.tick_params(axis='y', labelsize=20, direction='in', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    ax.tick_params(axis='y', which='minor', direction='in', length=4, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
    
    ax.tick_params(axis='x', labelsize=20, direction='out', length=8, width=1.3, pad=10, bottom=True, top=False, left=True, right=False, color='k')
    #ax.tick_params(axis='x', which='minor', direction='out', length=4, color='k', width=1.3, bottom=True, top=False, left=True, right=False)
    ax.set_ylim(-2, ymax)
    if ax is not None:
        return ax
    if ax is None:
        print("return fig")
        return fig        
        
def create_lists_of_n_repeating_m(m,n):
    l_xn = [1]*n #list of 1s of length n
    l_xnxm = []
    for n in range(m):
        l_nxm = [x+n for x in l_xn]
        l_xnxm = l_xnxm + l_nxm
    return l_xnxm

def create_empty_arrays_hits(m,n):
    df_arrays = pd.DataFrame(index=range(m*n), columns=['bin_x', 'bin_y'])
    l_m = list(np.arange(1,m+1,1)) 
    l_n = l_m*n
    df_arrays['bin_y'] = l_m*n
    l_xnxm = create_lists_of_n_repeating_m(m,n)
    df_arrays['bin_x'] = l_xnxm       
    return df_arrays
        
def produce_density_plot(df_merged, x_var='absorption', y_var='abs637 mean', value_limit=3.5, 
                         step = 0.1, show_colour=True, cmap='viridis', ax=None):    
    size=int(value_limit/step)
    df_merged = df_merged[[x_var, y_var]].copy()    
    df_merged = df_merged[~df_merged.isin([np.nan, np.inf, -np.inf]).any(1)] #remove any infs or np.nan 
    
    x = np.array(df_merged[x_var])
    y = np.array(df_merged[y_var])    
    max_value = max(x.max(),y.max())    
    df_merged = df_merged[df_merged < value_limit]
    bins = np.array(np.arange(0, value_limit, step))

    inds_x = np.digitize(x, bins, right=True)
    inds_y = np.digitize(y, bins, right=True)
        
    df_merged['bin_x'] = inds_x
    df_merged['bin_y'] = inds_y
    df_groupby = df_merged.groupby(['bin_x', 'bin_y']).count().reset_index()
    df_empty = create_empty_arrays_hits(m=size,n=size)
    df_empty = df_empty.merge(df_groupby, on=['bin_x', 'bin_y'], how='outer')
    df_empty = df_empty.sort_values(['bin_x', 'bin_y'])
    
    rowIDs = df_empty['bin_x'].astype(int)
    colIDs = df_empty['bin_y'].astype(int)

    array = np.zeros((int(rowIDs.max()),int(colIDs.max())))
    array[colIDs-1, rowIDs-1] = df_empty[x_var].values
    array[array == 0] = np.nan
    
    im = ax.imshow(array, extent=[0,value_limit,0,value_limit], interpolation='none', origin='lower',
                   norm=LogNorm(vmin=1, vmax=1000), cmap=cmap)
    if show_colour == True:
        cbar = plt.colorbar(im, orientation='vertical', location='right', anchor=(-0.2, 0.3), shrink=0.7)
        cbar.set_label('# of data points', rotation=270, labelpad=10, fontsize=15)
        cbar.ax.tick_params(labelsize=15)
    return ax
    
def combination_timeseries_and_regression(df_abs637, df_newpsap637, df_oldpsap637_hourly, df_newpsap637_hourly, 
                                          df_MAAP,mean_or_median='mean', alpha=0.1):
    fig, ax = plt.subplots(figsize=(22,7)) 

    quick_plot(df_abs637, df_oldpsap637_hourly, df_newpsap637_hourly, df_MAAP, 'abs637', add_labels=True, 
                              ymax=15, ax=ax)
    plt.ylim(0, 15)
    plt.xlim(df_abs637.index[0], df_abs637.index[-1])
    face_color1 = 'b'
    face_color2 = 'g'
    ax.axvspan('2012-11-19 14:00:00', '2013-03-21 14:00:00', alpha= alpha, color=face_color1)
    ax.axvspan('2014-11-19 14:00:00', '2016-10-13 14:00:00', alpha= alpha, color=face_color2)

    a = plt.axes([0.15, 0.5, .4, .4]) # this is another inset axes over the main axes
    a.patch.set_facecolor(face_color1)
    a.patch.set_alpha(alpha)
    df_merged = mergedfs(df_newpsap637_hourly, df_oldpsap637_hourly) # merge 
    produce_density_plot(df_merged, x_var='abs637 mean_x', y_var='abs637 mean_y',
                         value_limit=3.5, step = 0.05, ax=a)
    create_simple_regession_plot(df_merged,  x_var='abs637 mean_x', y_var='abs637 mean_y',
                                set_max_value=3.5, dict_abs_labels={}, 
                                title='', vmin=0, vmax=1, 
                                xlabel='$\sigma_{\mathrm{automatic\,PSAP\,ap}}$ [Mm$^{-1}$]', 
                                ylabel='$\sigma_{\mathrm{manual\,PSAP\,ap}}$ [Mm$^{-1}$]',
                                show_points=False, loc_legend='upper left', ax=a)

    a = plt.axes([.53, .5, .4, .4]) # this is an inset axes over the main axes
    a.patch.set_facecolor(face_color2)
    a.patch.set_alpha(alpha)
    df_newpsap637_hourly = hourly_resample(df_newpsap637, cols=['filter_num', 'abs637', 'scat550', 'Tr']) 
    df_merged = mergedfs(df_MAAP, df_newpsap637_hourly) #merge 
    produce_density_plot(df_merged, x_var='absorption', y_var='abs637 mean', value_limit=3.5, 
                            step = 0.05, ax=a)
    create_simple_regession_plot(df_merged, x_var='absorption', y_var='abs637 '+str(mean_or_median), 
                                set_max_value=3.5, dict_abs_labels={}, 
                                title='', vmin=0, vmax=1, 
                                xlabel='$\sigma_{\mathrm{MAAP\,ap}}$ [Mm$^{-1}$]', 
                                ylabel='$\sigma_{\mathrm{automatic\,PSAP\,ap}}$ [Mm$^{-1}$]', 
                                show_points=False, loc_legend='lower right', ax=a)

    plt.show()
    return fig
    
import matplotlib.gridspec as gridspec

def add_colournar(fig, scientific_notation=False, vmin=1, vmax=1000, colourbar_label='# of data points', cmap='viridis',
                  extend='both'):
    cbar_ax = fig.add_axes([1.05, .2, 0.03, 0.7]) #position of colorbar [left, bottom, width, height]
    norm = mpl.colors.Normalize(vmin=vmin, vmax=vmax)
    norm = mpl.colors.LogNorm(vmin=vmin, vmax=vmax) 
    
    fmt = None
    if scientific_notation == True:
        print("use scientific notation:")
        fmt = ticker.ScalarFormatter(useMathText=True)
        fmt.set_powerlimits((0, 0))    
    cb = mpl.colorbar.ColorbarBase(cbar_ax, cmap=cmap,
                                   norm=norm, orientation='vertical', extend=extend, format=fmt)
    cb.set_label(colourbar_label, fontsize=25)
    cb.ax.tick_params(labelsize=15)
    cb.ax.yaxis.set_offset_position('left')    
    cb.ax.yaxis.get_offset_text().set_fontsize(25)    
    cb.update_ticks()
    return cb

def gridspec_subplots(df_abs637, df_newpsap637, df_oldpsap637_hourly, df_newpsap637_hourly, 
             df_MAAP,mean_or_median='mean', alpha=0.1, size=10, label_factor=1, cmap='viridis'):
    # define a figure and choose the size
    fig = plt.figure(figsize=(size, size))
    
    face_color1 = 'b'
    face_color2 = 'g'

    gs = gridspec.GridSpec(ncols=4, nrows=3, hspace = 0., wspace = 0.5, top = 1,
                           bottom = 0, left = 0, right = 1)
    # y x
    # create an ax with gs
    ax = fig.add_subplot(gs[0:2, 0:2])
    ax.text(0.05, 0.95, 'a)', fontsize=17, color='k', transform=ax.transAxes, 
            horizontalalignment='center', verticalalignment='center')
    ax.patch.set_facecolor(face_color1)
    ax.patch.set_alpha(alpha)
    df_merged = mergedfs(df_newpsap637_hourly, df_oldpsap637_hourly) # merge 
    produce_density_plot(df_merged, x_var='abs637 mean_x', y_var='abs637 mean_y',
                         value_limit=3.5, step = 0.05, show_colour=False, cmap=cmap, ax=ax)
    create_simple_regession_plot(df_merged,  x_var='abs637 mean_x', y_var='abs637 mean_y',
                                set_max_value=3.5, dict_abs_labels={}, 
                                title='', vmin=0, vmax=1, 
                                xlabel='$\sigma_{\mathrm{automatic\,PSAP\,ap}}$ [Mm$^{-1}$]', 
                                ylabel='$\sigma_{\mathrm{manual\,PSAP\,ap}}$ [Mm$^{-1}$]',
                                show_points=False, loc_legend='upper left', legend_out_of_box=False, ax=ax)       
                        

    ax = fig.add_subplot(gs[0:2, 2:4])
    ax.text(0.05, 0.95, 'b)', fontsize=17, color='k', transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center')
    ax.patch.set_facecolor(face_color2)
    ax.patch.set_alpha(alpha)
    df_newpsap637_hourly = hourly_resample(df_newpsap637, cols=['filter_num', 'abs637', 'scat550', 'Tr']) 
    df_merged = mergedfs(df_MAAP, df_newpsap637_hourly) #merge 
    produce_density_plot(df_merged, x_var='absorption', y_var='abs637 mean', value_limit=3.5, 
                            step = 0.05, show_colour=False, cmap=cmap, ax=ax)
    create_simple_regession_plot(df_merged, x_var='absorption', y_var='abs637 '+str(mean_or_median), 
                                set_max_value=3.5, dict_abs_labels={}, 
                                title='', vmin=0, vmax=1, 
                                xlabel='$\sigma_{\mathrm{MAAP\,ap}}$ [Mm$^{-1}$]', 
                                ylabel='$\sigma_{\mathrm{automatic\,PSAP\,ap}}$ [Mm$^{-1}$]', 
                                show_points=False, loc_legend='lower right', legend_out_of_box=False, ax=ax)


    ax = fig.add_subplot(gs[2, 0:4])
    ax.text(0.03, 0.95, 'c)', fontsize=17, color='k', transform=ax.transAxes,
            horizontalalignment='center', verticalalignment='center')
    quick_plot(df_abs637, df_oldpsap637_hourly, df_newpsap637_hourly, df_MAAP, 'abs637', add_labels=True, 
                              ymax=15, label_factor=label_factor, ax=ax)
    ax.set_ylim(0, 15)
    ax.set_xlim(df_abs637.index[0], df_abs637.index[-1])
    ax.axvspan('2012-11-19 14:00:00', '2013-03-21 14:00:00', alpha= alpha, color=face_color1)
    ax.axvspan('2014-11-19 14:00:00', '2016-10-13 14:00:00', alpha= alpha, color=face_color2)

    add_colournar(fig, cmap=cmap)
    plt.show()
    return fig