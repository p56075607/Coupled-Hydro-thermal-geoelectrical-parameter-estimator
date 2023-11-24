import sys
sys.path.append(r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC')
from read_THMC_flow_result import read_THMC_flow_result
from read_THMC_heat_result import read_THMC_heat_result
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from scipy import stats

# Load ASCII dat file
# work_dir = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\PEST\mycase\THMC\workspace_2D'
dat_file_name = os.path.join('PostP','001_H_SbFlow.dat')
Variable_all,var_10_cm,var_20_cm,var_30_cm,var_50_cm,var_100_cm = read_THMC_flow_result(dat_file_name,Var_ind=4)

# Creat datetime array
time_len = int(529)
# Creat datetime array
datetime_array = np.arange(datetime(2022,2,12,00), datetime(2022,3,6,1), timedelta(hours=1)).astype(datetime)
time_index = np.arange(0,time_len,1)
datetime_THMC = datetime_array[time_index]

# Output SWC time-series to pandas dataframe format and .csv file
df_var_20_cm = pd.DataFrame([])
df_var_20_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_20_cm['SWC_20_cm'] = pd.DataFrame(stats.zscore(var_20_cm))
df_var_20_cm['SWC_20_cm']= pd.DataFrame((var_20_cm))#/np.linalg.norm(var_20_cm))
df_var_20_cm.to_csv(os.path.join('SWC_20_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')

df_var_30_cm = pd.DataFrame([])
df_var_30_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_30_cm['SWC_30_cm'] = pd.DataFrame(stats.zscore(var_30_cm))
df_var_30_cm['SWC_30_cm']= pd.DataFrame((var_30_cm))#/np.linalg.norm(var_30_cm))
df_var_30_cm.to_csv(os.path.join('SWC_30_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')

df_var_50_cm = pd.DataFrame([])
df_var_50_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_50_cm['SWC_50_cm'] = pd.DataFrame(stats.zscore(var_50_cm))
df_var_50_cm['SWC_50_cm']= pd.DataFrame((var_50_cm))#/np.linalg.norm(var_50_cm))
df_var_50_cm.to_csv(os.path.join('SWC_50_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')

df_var_100_cm = pd.DataFrame([])
df_var_100_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_100_cm['SWC_100_cm'] = pd.DataFrame(stats.zscore(var_100_cm))
df_var_100_cm['SWC_100_cm']= pd.DataFrame((var_100_cm))#/np.linalg.norm(var_100_cm))
df_var_100_cm.to_csv(os.path.join('SWC_100_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')                    

# Load ASCII dat file
# work_dir = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\PEST\mycase\THMC\workspace_2D'
dat_file_name = os.path.join('PostP','001_T_Tmptr.dat')
Variable_all,var_0_cm,var_5_cm,var_10_cm,var_20_cm,var_50_cm,var_100_cm = read_THMC_heat_result(dat_file_name,Var_ind=2)
var_0_cm  = var_0_cm  -273
var_5_cm  = var_5_cm  -273
var_10_cm = var_10_cm -273
var_20_cm = var_20_cm -273
var_50_cm = var_50_cm -273
var_100_cm= var_100_cm-273

# Output Temp time-series to pandas dataframe format and .csv file
df_var_20_cm = pd.DataFrame([])
df_var_20_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_20_cm['Temp_20_cm'] = pd.DataFrame(stats.zscore(var_20_cm))
df_var_20_cm['Temp_20_cm'] = pd.DataFrame((var_20_cm))#/np.linalg.norm(var_20_cm))
df_var_20_cm.to_csv(os.path.join('Temp_20_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')

df_var_50_cm = pd.DataFrame([])
df_var_50_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_50_cm['Temp_50_cm'] = pd.DataFrame(stats.zscore(var_50_cm))
df_var_50_cm['Temp_50_cm']= pd.DataFrame((var_50_cm))#/np.linalg.norm(var_50_cm))
df_var_50_cm.to_csv(os.path.join('Temp_50_cm.csv')
                    ,index=False
                    , encoding = 'utf-8') 

df_var_100_cm = pd.DataFrame([])
df_var_100_cm['Time'] = pd.Series(datetime_THMC).values
# df_var_100_cm['Temp_100_cm'] = pd.DataFrame(stats.zscore(var_100_cm))
df_var_100_cm['Temp_100_cm']= pd.DataFrame((var_100_cm))#/np.linalg.norm(var_100_cm))
df_var_100_cm.to_csv(os.path.join('Temp_100_cm.csv')
                    ,index=False
                    , encoding = 'utf-8')       