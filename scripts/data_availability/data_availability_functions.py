import pandas as pd
import matplotlib.pylab as plt
import numpy as np
from matplotlib import cm
import os
import glob

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
              name='default_name', formate=".jpeg", dpi=300):
    folders = glob.glob(path)
    if folder not in folders:
        os.makedirs(path+"\\"+folder, exist_ok=True)
    fig.savefig(path+"\\"+folder+"\\"+str(name)+str(formate), bbox_inches='tight', dpi=dpi)
    print("saved as: "+str(path+"\\"+folder+"\\"+str(name)+str(formate)))
    
def rename_col(df, old_col_name, new_col_name):
    df =  df.rename(columns={old_col_name:new_col_name})
    return df
    
def set_datetime_index(df, col_index):
    df[col_index]= pd.to_datetime(df[col_index])
    df = df.set_index(col_index)
    return df
    
def plot_data_availability(dfs, dict_instru_variables, dti=None, colormap='viridis',
                           df_2000_2020_Mie_OLD=None, df_2000_2020_Mie_NEW=None):
    fig, ax = plt.subplots(1, figsize=(20,5))
    instruments = ['']
    
    dict_name_order = {}
    for count, name in enumerate(dfs.keys()):
        instruments.append(name)       
        df = dfs[name]   
        df['Date'] = df.index
        df['Date'] = pd.to_datetime(df['Date'])
        df['Date'] = df['Date'].dt.strftime('%Y-%m-%d') 
        variable = dict_instru_variables[name]
        print(variable)
        df[name] = np.nan
        df.loc[~np.isnan(df[variable]), name] = count+1
        dict_name_order[name] = count + 1
        
    print(dict_name_order)
                
    if df_2000_2020_Mie_OLD is not None:
        ax = df_2000_2020_Mie_OLD['DMPS ($\sigma_{\mathrm{sp-Mie}}$)'].plot(style='s', ms=15, legend=False, c='b', alpha=0.4)
    
    if df_2000_2020_Mie_NEW is not None:
        ax = df_2000_2020_Mie_NEW['DMPS ($\sigma_{\mathrm{sp-Mie}}$)'].plot(style='s', ms=15, legend=False, c='b', alpha=0.4)
        
        
    my_cm = cm.get_cmap(colormap, len(dfs)+1)     
    for count, name in enumerate(dfs.keys()):
        df = dfs[name]
        ax = df[name].plot(style='s', ms=10, legend=False, c='k', alpha=0.4) #=my_cm.colors[count], alpha=0.5)
        ax.set_yticks(range(1+len(dfs.keys())))
        ax.set_yticklabels(instruments, fontsize=15)

    ax.yaxis.set_ticks_position('none') 
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)

    #ax.yaxis.set_ticks_position('left') # Only show ticks on the left and bottom spines
    #ax.xaxis.set_ticks_position('bottom')
        
    ax.set_xlabel(' ',fontsize=12)
    ax.tick_params(axis='both', which='major', labelsize=15)
    #ax.tick_params(axis='both', which='minor', labelsize=12)
    
    ax.tick_params(axis='x', rotation=0)
    for label in ax.get_xticklabels():
        label.set_horizontalalignment('center')

    #MAAP to NEW PSAP     
    plt.vlines(dfs['MAAP'].index[0], ymin=dict_name_order['MAAP'], ymax=dict_name_order['Automatic PSAP'], 
                linestyles ='dashed', color='k')
    plt.vlines(dfs['Automatic PSAP'].index[-1], ymin=dict_name_order['MAAP'], ymax=dict_name_order['Automatic PSAP'], 
                linestyles ='dashed', color='k')
    
    #New PSAP to Old PSAP
    plt.vlines(dfs['Automatic PSAP'].index[0], ymin=dict_name_order['Automatic PSAP'], ymax=dict_name_order['Manual PSAP'], 
                linestyles ='dashed', color='k')
    plt.vlines(dfs['Manual PSAP'].index[-1], ymin=dict_name_order['Automatic PSAP'], ymax=dict_name_order['Manual PSAP'], 
                linestyles ='dashed', color='k')

    #ecotech with TSI  
    plt.vlines('2019-04-10 20:00:00', ymin=dict_name_order['Ecotech'], 
               ymax=dict_name_order['TSI'], linestyles ='dashed', color='k')
    plt.vlines('2019-06-14 08:00:00', ymin=dict_name_order['Ecotech'], 
              ymax=dict_name_order['TSI'], linestyles ='dashed', color='k')
              
    if dti is not None:
        for time in dti:
            plt.vlines(time, ymin=0, ymax=9, 
                    linestyles ='dashed', color='k', lw=0.5, alpha=.5)   
    ax.set_ylim(0, 9)      
    plt.show()
    return fig
	