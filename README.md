# ACP_DHR_2024
Scripts used for the data anaysis for ACP paper 2024. Including function files, and jupyter notebooks.

# ACP_DHR_2023

Scripts used for the data analysis for ACP paper 2023. Including function files, and Jupyter notebooks. 

See below for figure and corresponding notebook:

|     Figures      	|     Notebook                                                 	|
|------------------	|--------------------------------------------------------------	|
|     Figure1      	|     Harmonisation_inc_figure1                                	|
|     Figure2      	|     Trend_plots_in_abs_ATP_inc_figures2_6_S8_S13             	|
|     Figure3      	|     Spatial_trends_using_CWT_arrays_inc_figure3              	|
|     Figure4      	|     Cluster_mappings_inc_figure4                             	|
|     Figure5      	|     Cluster_plots_inc_figure5_S12_S6                         	|
|     Figure6      	|     Trend_plots_in_abs_ATP_inc_figures2_6_S8_S13             	|
|     Figure7      	|     Calculating_estimated_abs_from_ATP_figure7               	|
|     Table1       	|     Collocate_back_trajs_with_GFED_inc_table1                	|
|     FigureS1     	|     Data_availability_inc_figureS1                           	|
|     FigureS2     	|     Compare_data_with_aethalomter_inc_figureS2_S3            	|
|     FigureS3     	|     Compare_data_with_aethalomter_inc_figureS2_S3            	|
|     FigureS4     	|     Read_aethalometer_cal_AAE_inc_figureS4                   	|
|     FigureS5     	|     Collocate_ERA5_and_back_trajectorys_vectorized_inc_S5    	|
|     FigureS6     	|     Cluster_plots_inc_figure5_S12_S6                         	|
|     FigureS7     	|     Active_fires_MODIS_gridding_with_hysplit_inc_figureS7    	|
|     FigureS8     	|     Trend_plots_in_abs_ATP_inc_figures2_6_S8_S13             	|
|     FigureS9     	|     Calculates_SSA_inc_S9                                    	|
|     FigureS10    	|     Concentration_weighted_plots_inc_figureS10               	|
|     FigureS11    	|     Masks_eclipse_trend_array_inc_figureS11                  	|
|     FigureS12    	|     Cluster_plots_inc_figure5_S12_S6                         	|
|     FigureS13    	|     Trend_plots_in_abs_ATP_inc_figures2_6_S8_S13             	|
|     FigureS14    	|     Seasonality_plot_inc_figureS14                           	|
|     FigureS15    	|     Extremes_values_inc_figureS15                            	|
|     FigureS16    	|     Arithmetic_trends_impact_of_fires_inc_figureS16          	|
|     FigureS17    	|     Calculating_estimated_abs_from_ATP_figure7               	|

Notebooks	Description

***Harmonisation_inc_figure1	compare instruments:***
- applying correction factors
- produce harmonised timeseries 

***Data_availability_inc_figureS1:***
- loads the data sets
- plots the data availability as a simple bar chart

***Cluster_mappings_inc_figure4:***
- load the data for the clusters
- plot all the lat and lon endpoints as frequency plots

***Concentration_weighted_plots_inc_figureS10:***
- Reads in full years’ worth of data
- generates the CWT arrays, saves them.
- Also, loads the arrays that were made for the manuscript

***Extremes_values_inc_figureS15:***
- Generates the extreme values by defining them using a rolling percentile of 15-days 99th, 95th etc...

***Active_fires_MODIS_gridding_with_hysplient_inc_figureS7:***
- Uses the MODIS Satellite data 
- Grids the number active fires
- Counts the number of fires in grids traversed

***Seasonality_plot_inc_figureS14:***
- plot for the annual cycle
- for ECLIPSE emission inventory
- Accumulated back trajectory precipitation (ATP) 

***Spatial_trends_using_CWT_arrays_inc_figure3:***
- Produces the spatial trend plots using the CWT arrays, which are loaded in

***Masks_eclipse_trend_array_inc_figureS11:***
- Uses the trend arrays to apply a mask for the eclipse array

***Trend_plots_in_abs_ATP_inc_figures2_6_S8_S13:***
- Subplot for the trends in the absorption coefficient 
- Trend for all seasons for the precipitation

***Cluster_plots_inc_figure5_S12_S6:***
- Reads in the data files containing the cluster data sets

***Compare_data_with_aethalomter_inc_figureS2_S3:***
- comparison between with Aethalometer and all the different instruments
- comparisons with 3-month intervals of the PSAPs and MAAP

***Read_aethalometer_cal_AAE_inc_figureS4:***
- read in the Aeth data and calculate the Absorbing Ångström Exponent. 
- produce the data file for the full time series of the Aethalometer data at 660 nm
- Arithmetic_trends_impact_of_fires_inc_figureS16	Removes the extreme B.B. events from the data set to see the impact

***Calculates_SSA_inc_S9:***
- Calculates SSA
- Timeseries of the absorption coefficient, scattering coefficient and single scattering albedo

***Collocate_back_trajs_with_GFED_inc_table1:***
- Here we read in the HYSPLIT data then we collocate it with the Global Fire Emission Database, which we have pre-processed and saved as .nc files
- Saves as a GFED.dat file

***Calculating_estimated_abs_from_ATP_figure7:***
- Compare absorption and precipitation 
- Map values to estimate time series

***Collocate_ERA5_and_back_trajectorys_vectorized_inc_S5:***
- Collocates the ERA data with the HYSPLIT output
- produces figureS5
  

***Processing scripts only:***	

***Converts_global_fire_emission_database_tonetcdf:***
- Converts hdf5 files to netcdf for later
- Processes_hysplit_output	takes the 'raw' HYSPLIT data i.e. what you get from the output, processes them one by one and saves them in a form with also distance calculated and also the rotated latitudes and longitudes as grid cells.

***Generate_trend_mapping_for_ECLIPSE_and_array:***
- Plots the trend array for the ECLIPSE emissions and generates the trend array .txt file

***Looks_for_missing_hysplit_output:***
- look at the folders where you have saved your runs. 
- Create a dataframe which lists the missing runs 
- Once the dataframe of the missing datetimes is created.
- loop through them and generate HYSPLIT files for them.  

***Generates_back_trajectories_using_pysplit_and_GDAS:***
- Use Pysplit to generate trajectories with metrological data from GDAS

***Generates_back_trajectories_using_pysplit_and_FNL:***
- Use Pysplit to generate trajectories with metrological data from FNL (different due to the resolution)

***Reads_in_and_processes_MAAP_data:***
- process the MAAP data

***Reads_in_ecotech_data:***
- read in and process the Ecotech nephelometer 

***Compare_TSI_Ecotech:***
- compare the TSI and Ecotech data

***Reading_in_the_automatic_PSAP_data_applying_Bond:***
- reads the raw PSAP data processes it and applies bond

***Reading_in_the_manual_PSAP_data_applying_Bond:***
- reads the raw PSAP data processes it and applies bond
