# %%
import sys
import os
import shutil
import platform
import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "Consolas-with-Yahei"
import pyemu

from datetime import datetime, timedelta
# %%
# Prepare for the real observation value to calibrate
# Input hydrological data
hydro_path = 'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF'
hydro_fname = "hydrolic_one_hour.csv"
hydro_data=pd.read_csv(os.path.join(hydro_path,hydro_fname),encoding='utf-8')
hydro_data['WF_datetime'] = pd.to_datetime(hydro_data['WF_datetime'])
# hydro_data = hydro_data.set_index('WF_datetime')

time_len = int(529)
# Creat datetime array
datetime_array = np.arange(datetime(2022,2,12,00), datetime(2022,3,6,1), timedelta(hours=1)).astype(datetime)
time_index = np.arange(0,time_len,1)
datetime_THMC = datetime_array[time_index]

filterd_hydro_data = pd.DataFrame( columns=hydro_data.columns )
for i in range(time_len):     
    mask = (hydro_data['WF_datetime'] == datetime_THMC[i])
    filterd_hydro_data = pd.concat([filterd_hydro_data, hydro_data.loc[mask]])

# filterd_hydro_data['SWC_2'] = stats.zscore(filterd_hydro_data['SWC_2'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D','SWC_20_cm.csv')
                        ,columns=['WF_datetime','SWC_2']
                        ,header=['Time','SWC_20_cm']
                        ,index=False
                        ,encoding = 'utf-8')

# filterd_hydro_data['SWC_3'] = stats.zscore(filterd_hydro_data['SWC_3'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D','SWC_30_cm.csv')
                        ,columns=['WF_datetime','SWC_3']
                        ,header=['Time','SWC_30_cm']
                        ,index=False
                        ,encoding = 'utf-8')

# filterd_hydro_data['SWC_4'] = stats.zscore(filterd_hydro_data['SWC_4'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D','SWC_50_cm.csv')
                        ,columns=['WF_datetime','SWC_4']
                        ,header=['Time','SWC_50_cm']
                        ,index=False
                        ,encoding = 'utf-8')

theta_s, lamda, beta = 0.445, 0.725, 0.011
filterd_hydro_data['SWC_5_new'] = theta_s/(-beta*filterd_hydro_data['suction_2'])**lamda
# filterd_hydro_data['SWC_5_new'] = stats.zscore(filterd_hydro_data['SWC_5_new'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D','SWC_100_cm.csv')
                        ,columns=['WF_datetime','SWC_5_new']
                        ,header=['Time','SWC_100_cm']
                        ,index=False
                        ,encoding = 'utf-8')

filterd_hydro_data['thermal_1'] = filterd_hydro_data['thermal_1']#/np.linalg.norm(filterd_hydro_data['thermal_1']) #stats.zscore(filterd_hydro_data['thermal_1'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D_H','Temp_50_cm.csv')
                        ,columns=['WF_datetime','thermal_1']
                        ,header=['Time','Temp_50_cm']
                        ,index=False
                        ,encoding = 'utf-8')

filterd_hydro_data['thermal_2'] = filterd_hydro_data['thermal_2']#/np.linalg.norm(filterd_hydro_data['thermal_2']) #stats.zscore(filterd_hydro_data['thermal_2'])
filterd_hydro_data.to_csv(os.path.join('THMC','workspace_2D_H','Temp_100_cm.csv')
                        ,columns=['WF_datetime','thermal_2']
                        ,header=['Time','Temp_100_cm']
                        ,index=False
                        ,encoding = 'utf-8')

# Temperature input
filename = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\WF\G2F820_twoyears.xlsx'
two_year_data = pd.read_excel(filename)
two_year_data['0cm'] = two_year_data['0cm'].interpolate(method='linear',limit_direction='both')
two_year_data['觀測時間'] = two_year_data['觀測時間'].round('H')
two_year_data['datetime'] = two_year_data['觀測時間']
# two_year_data = two_year_data.set_index('觀測時間')
two_year_data['serial_number'] = np.arange(len(two_year_data))

filterd_two_year_data = pd.DataFrame( columns=two_year_data.columns )
for i in range(time_len):     
    mask = (two_year_data['datetime'] == datetime_THMC[i])
    filterd_two_year_data = pd.concat([filterd_two_year_data, two_year_data.loc[mask]])

filterd_two_year_data['20cm'] = filterd_two_year_data['20cm']+2.2

filterd_two_year_data['20cm'] = filterd_two_year_data['20cm']#/np.linalg.norm(filterd_two_year_data['20cm']) #stats.zscore(filterd_two_year_data['20cm'])
filterd_two_year_data.to_csv(os.path.join('THMC','workspace_2D_H','Temp_20_cm.csv')
                        ,columns=['datetime','20cm']
                        ,header=['Time','Temp_20_cm']
                        ,index=False
                        ,encoding = 'utf-8')

# Copy measured ERT data
src = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\Petrophysic\sim_time_9\test_csv'
# src_files = os.listdir(src)
# for file_name in src_files:
full_file_name = os.path.join(src, '22022209_m_E1.csv')
ERT_data = pd.read_csv(full_file_name,encoding='utf-8')
ERT_data['rhoa_022209'] = ERT_data['rhoa_022209']#np.log10()#stats.zscore()
ERT_data.to_csv(os.path.join('THMC','workspace_2D','sim_csv','22022209_m_E1.csv')
                        ,columns=['rhoa_022209','index']
                        ,header=['rhoa_022209','index']
                        ,index=False
                        ,encoding = 'utf-8')

full_file_name = os.path.join(src, '22021209_m_E1.csv')
ERT_data_read = pd.read_csv(full_file_name,encoding='utf-8')
ERT_data['rhoa_021209'] = ERT_data_read['rhoa_021209']#stats.zscore(np.log10())
ERT_data.to_csv(os.path.join('THMC','workspace_2D','sim_csv','22021209_m_E1.csv')
                        ,columns=['rhoa_021209','index']
                        ,header=['rhoa_021209','index']
                        ,index=False
                        ,encoding = 'utf-8')
# %%
# folder containing original model files
org_d = os.path.join('THMC', 'workspace_2D_H')

# a dir to hold a copy of the org model files
case = 'THMC_files'
tmp_d = os.path.join(case)

if os.path.exists(tmp_d):
    shutil.rmtree(tmp_d)
shutil.copytree(org_d,tmp_d)

# get executables
bin_path = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\PEST\pypest\GMDSI_notebooks\bin_new\win'
files = os.listdir(bin_path)
for f in files:
    if os.path.exists(os.path.join(tmp_d,f)):
        os.remove(os.path.join(tmp_d,f))
    shutil.copy2(os.path.join(bin_path,f),os.path.join(tmp_d,f))

# %%
# instantiate PstFrom
template_ws = os.path.join(case+'_opt')
pf = pyemu.utils.PstFrom(original_d=tmp_d, # where the model is stored
                            new_d=template_ws, # the PEST template folder
                            remove_existing=True, # ensures a clean start
                            longnames=True, # set False if using PEST/PEST_HP
                            zero_based=False, # does the MODEL use zero based indices? For example, MODFLOW does NOT
                            echo=False) # to stop PstFrom from writting lots of infromation to the notebook; experiment by setting it as True to see the difference; usefull for troubleshooting

# %%
# check the output csv file and 
df = pd.read_csv(os.path.join(template_ws,"SWC_50_cm.csv"),index_col=0)
df.head()
# add observation to pf
SWC_50_cm_df = pf.add_observations("SWC_50_cm.csv", # the model output file to read
                            insfile="SWC_50_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="swc") #prefix to all observation names; choose somet

df = pd.read_csv(os.path.join(template_ws,"SWC_20_cm.csv"),index_col=0)
df.head()
SWC_20_cm_df = pf.add_observations("SWC_20_cm.csv", # the model output file to read
                            insfile="SWC_20_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="swc") #prefix to all observation names; choose somet

df = pd.read_csv(os.path.join(template_ws,"SWC_30_cm.csv"),index_col=0)
df.head()
SWC_30_cm_df = pf.add_observations("SWC_30_cm.csv", # the model output file to read
                            insfile="SWC_30_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="swc") #prefix to all observation names; choose somet

df = pd.read_csv(os.path.join(template_ws,"SWC_100_cm.csv"),index_col=0)
df.head()
SWC_100_cm_df = pf.add_observations("SWC_100_cm.csv", # the model output file to read
                            insfile="SWC_100_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="swc") #prefix to all observation names; choose somet                            

df = pd.read_csv(os.path.join(template_ws,"Temp_20_cm.csv"),index_col=0)
df.head()
Temp_20_cm_df = pf.add_observations("Temp_20_cm.csv", # the model output file to read
                            insfile="Temp_20_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="temp") #prefix to all observation names; choose somet                            

df = pd.read_csv(os.path.join(template_ws,"Temp_50_cm.csv"),index_col=0)
df.head()
Temp_50_cm_df = pf.add_observations("Temp_50_cm.csv", # the model output file to read
                            insfile="Temp_50_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="temp") #prefix to all observation names; choose somet          

df = pd.read_csv(os.path.join(template_ws,"Temp_100_cm.csv"),index_col=0)
df.head()
Temp_100_cm_df = pf.add_observations("Temp_100_cm.csv", # the model output file to read
                            insfile="Temp_100_cm.csv.ins", #optional, the instruction file name
                            index_cols="Time", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="temp") #prefix to all observation names; choose somet                            
# %%
ERT_csv_ph = os.path.join(template_ws,"sim_csv")
files = os.listdir(ERT_csv_ph)
for file_name in files:
    ERT_csv_file_name = os.path.join("sim_csv",file_name)
    ERT_ins_file_name = os.path.join("sim_ins",file_name+'.ins')
    df = pd.read_csv(os.path.join(template_ws,ERT_csv_file_name),index_col=1)
    pf.add_observations(ERT_csv_file_name, # the model output file to read
                            insfile=ERT_ins_file_name, #optional, the instruction file name
                            index_cols="index", #column header to use as index; can also use column number (zero-based) instead of the header name
                            use_cols=list(df.columns.values), #names of columns that include observation values; can also use column number (zero-based) instead of the header name
                            prefix="rhoa") #prefix to all observation names; choose somet                            
# %%
# add parameters to the PEST
df_K1 = pf.add_parameters(filenames='K_1.pinp',
                            par_type="constant",
                            par_name_base='K1',
                            pargp = 'Hydro',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[0],
                            transform='log',
                            lower_bound=0.00001, upper_bound=0.05, # PEST parameter upper/lower bound
                            ult_lbound =0.00001,  ult_ubound=0.05,  # Ultimate upper/lower bound for model input
                            initial_value=0.000165
)

df_K2 = pf.add_parameters(filenames='K_2.pinp',
                            par_type="constant",
                            par_name_base='K2',
                            pargp = 'Hydro',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[1],
                            transform='log',
                            lower_bound=0.00001, upper_bound=0.05, # PEST parameter upper/lower bound
                            ult_lbound =0.00001,  ult_ubound=0.05,  # Ultimate upper/lower bound for model input
                            initial_value=0.001363
)

df_lam1 = pf.add_parameters(filenames='lam.pinp',
                            par_type="constant",
                            par_name_base='lam1',
                            pargp = 'Thermal',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[1],
                            transform='none',
                            lower_bound=0.01, upper_bound=2.7, # PEST parameter upper/lower bound
                            ult_lbound=0.01, ult_ubound=2.7,  # Ultimate upper/lower bound for model input
                            initial_value=0.96
)

df_lam2 = pf.add_parameters(filenames='lam_2.pinp',
                            par_type="constant",
                            par_name_base='lam2',
                            pargp = 'Thermal',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[1],
                            transform='none',
                            lower_bound=0.01, upper_bound=2.7, # PEST parameter upper/lower bound
                            ult_lbound=0.01, ult_ubound=2.7,  # Ultimate upper/lower bound for model input
                            initial_value=1.56
)
df_allu_n = pf.add_parameters(filenames='allu_n.pinp',
                            par_type="constant",
                            par_name_base='allu_n',
                            pargp = 'Geophy',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[1],
                            transform='none',
                            lower_bound=0.1, upper_bound=2.5, # PEST parameter upper/lower bound
                            ult_lbound =0.1, ult_ubound =2.5,  # Ultimate upper/lower bound for model input
                            initial_value=2
)
df_allu_cw = pf.add_parameters(filenames='allu_cw.pinp',
                            par_type="constant",
                            par_name_base='allu_cw',
                            pargp = 'Geophy',
                            index_cols = [0],
                            use_cols=[1],
                            # use_rows=[1],
                            transform='log',
                            lower_bound=0.01, upper_bound=1, # PEST parameter upper/lower bound
                            ult_lbound =0.01, ult_ubound =1,  # Ultimate upper/lower bound for model input
                            initial_value=0.02
)
# %%
# start by adding a command line instruction to forward_run.py using 'mod_sys_cmds'
pf.mod_sys_cmds.append("python pinp2inp.py")
pf.mod_sys_cmds.append("THMC2D")
pf.mod_sys_cmds.append("FUPP")
pf.mod_sys_cmds.append("python dat2csv.py")
pf.mod_sys_cmds.append("python dat2csv_petrophy.py")

# %%
# Testing the magnitude of phi components
# constructing the PEST control file
pst = pf.build_pst()
pst.control_data.noptmax = 0
pst.write(os.path.join(template_ws,case+".pst"))
# %%
pyemu.os_utils.run("pestpp-ies.exe {0}".format('THMC_files.pst'),cwd=template_ws)
m_d = os.path.join('THMC_files_opt')
case = 'THMC_files'
pst = pyemu.Pst(os.path.join(m_d,f"{case}.pst"))
print(pst.phi)

nnz_phi_components = {k:pst.phi_components[k] for k in pst.nnz_obs_groups}
pst.plot(kind="phi_pie")
# %%
obs = pst.observation_data

##注意:這裡所設定的權重在PEST中計算時，會以平方值帶入!
obs.loc[(obs.obgnme=='oname:swc_otype:lst_usecol:swc_100_cm'), 'weight']= np.sqrt(1/sum((filterd_hydro_data['SWC_5_new']-np.mean(filterd_hydro_data['SWC_5_new']))**2))
obs.loc[(obs.obgnme=='oname:swc_otype:lst_usecol:swc_20_cm'), 'weight'] = np.sqrt(1/sum((filterd_hydro_data['SWC_2']-np.mean(filterd_hydro_data['SWC_2']))**2))
obs.loc[(obs.obgnme=='oname:swc_otype:lst_usecol:swc_30_cm'), 'weight'] = np.sqrt(1/sum((filterd_hydro_data['SWC_3']-np.mean(filterd_hydro_data['SWC_3']))**2))
obs.loc[(obs.obgnme=='oname:swc_otype:lst_usecol:swc_50_cm'), 'weight'] = np.sqrt(1/sum((filterd_hydro_data['SWC_4']-np.mean(filterd_hydro_data['SWC_4']))**2))

obs.loc[(obs.obgnme=='oname:temp_otype:lst_usecol:temp_100_cm'), 'weight'] = np.sqrt(1/sum((filterd_hydro_data['thermal_2']-np.mean(filterd_hydro_data['thermal_2']))**2))
obs.loc[(obs.obgnme=='oname:temp_otype:lst_usecol:temp_20_cm'), 'weight'] =  np.sqrt(1/sum((filterd_two_year_data['20cm']-np.mean(filterd_two_year_data['20cm']))**2))
obs.loc[(obs.obgnme=='oname:temp_otype:lst_usecol:temp_50_cm'), 'weight'] =  np.sqrt(1/sum((filterd_hydro_data['thermal_1']-np.mean(filterd_hydro_data['thermal_1']))**2))

obs.loc[(obs.obgnme=='oname:rhoa_otype:lst_usecol:rhoa_022209'), 'weight'] = np.sqrt(1/sum((ERT_data['rhoa_022209']-np.mean(ERT_data['rhoa_022209']))**2))
obs.loc[(obs.obgnme=='oname:rhoa_otype:lst_usecol:rhoa_021209'), 'weight'] = np.sqrt(1/sum((ERT_data['rhoa_021209']-np.mean(ERT_data['rhoa_021209']))**2))

pst.plot(kind="phi_pie")
# %%
pst.control_data.noptmax = 100
# pst.control_data.phiredstp = 0.001
# pst.control_data.nphistp = 4
pst.svd_data.svdmode = 0
pst.write(os.path.join(template_ws,case+".pst"))
# _ = [print(line.rstrip()) for line in open(os.path.join(template_ws,"forward_run.py"))]

# %%
# RUN PEST 
# pyemu.os_utils.run('pestpp-glm THMC_files.pst', cwd=template_ws)
# set parallel workers
num_workers = 5 #update this according to your resources
m_d = os.path.join('master_glm_THMC')
# run glm in parallel
pyemu.os_utils.start_workers(template_ws,"pestpp-glm",f"{case}.pst",
                           num_workers=num_workers,
                           worker_root=".",
                           master_dir=m_d)

# %%
m_d = os.path.join('master_glm_THMC_hydrothermalgeophy')
pst = pyemu.Pst(os.path.join(m_d,f"{case}.pst"))
print('Total phi: ',pst.phi)

df_paru_k = pd.read_csv(os.path.join(m_d, f"{case}.par.usum.csv"),index_col=0)
# df_paru_k

for i in range(len(df_paru_k)):
    if (i > 1)&(i < 5):
        print('{:s}:  {:.2f} ({:.2f}±{:.2f}) ({:.2f}~{:.2f})'.format(df_paru_k.index[i],df_paru_k['post_mean'][i],df_paru_k['post_mean'][i],df_paru_k['post_stdev'][i],(df_paru_k['post_mean'][i]-df_paru_k['post_stdev'][i]),(df_paru_k['post_mean'][i]+df_paru_k['post_stdev'][i])))
    else:
        print('{:s}:  {:.6f} (10^[{:.2f}±{:.2f}]) ({:.6f}~{:.6f})'.format(df_paru_k.index[i],10**df_paru_k['post_mean'][i],df_paru_k['post_mean'][i],df_paru_k['post_stdev'][i],10**(df_paru_k['post_mean'][i]-df_paru_k['post_stdev'][i]),10**(df_paru_k['post_mean'][i]+df_paru_k['post_stdev'][i])))

def nse(targets,predictions):
    return 1-(np.sum((targets-predictions)**2)/np.sum((targets-np.mean(targets))**2))

def rrmse(targets,predictions):
    return np.sqrt(np.sum((targets-predictions)**2 / len(targets))/(max(targets)-min(targets)))

def rmse(targets,predictions):
    return np.sqrt(np.sum((targets-predictions)**2 / len(targets)))

i1 = 0
i2 = 275
print('rhoa 2/12 09:00 RMSE:',rmse(pst.res['measured'][i1:i2],pst.res['modelled'][i1:i2]))
i1 = 275
i2 = 275*2
print('rhoa 2/22 09:00 RMSE:',rmse(pst.res['measured'][i1:i2],pst.res['modelled'][i1:i2]))
leng = 529
res = 0
for i in range(4):
    b = i2 + i*leng
    res = res + rmse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])
print('SWC RMSE:',res)
# %%
i1 = 0
i2 = 275
fig, ax = plt.subplots()
ax.scatter(np.log10(pst.res['measured'][i1:i2]),np.log10(pst.res['modelled'][i1:i2]),s=1,label='2022/2/12 9:00 nse={:.2f}'.format(nse(pst.res['measured'][i1:i2],pst.res['modelled'][i1:i2])))
i1 = 275
i2 = 275*2
ax.scatter(np.log10(pst.res['measured'][i1:i2]),np.log10(pst.res['modelled'][i1:i2]),s=1,label='2022/2/22 9:00 nse={:.2f}'.format(nse(pst.res['measured'][i1:i2],pst.res['modelled'][i1:i2])))
ax.set_aspect('equal')
lim_max = 3.5#max([max(np.log10(pst.res['measured'][i1:i2])),max(np.log10(pst.res['modelled'][i1:i2]))])
lim_min = 2.0#min([min(np.log10(pst.res['measured'][i1:i2])),min(np.log10(pst.res['modelled'][i1:i2]))])
ax.plot([lim_min,lim_max],[lim_min,lim_max],'k-',linewidth=1)
ax.set_xlim([lim_min,lim_max])
ax.set_ylim([lim_min,lim_max])
ax.set_xlabel(r'Measured $log_{10}(\rho_a)$',fontsize=13)
ax.set_ylabel(r'Modeled $log_{10}(\rho_a)$',fontsize=13)
ax.legend()

plt.savefig(os.path.join('results','resistivity_hydrothermalgeophy.png'),dpi=300, bbox_inches='tight')
# %%
# PLot convergence curve
# Residuals
df_obj = pd.read_csv(os.path.join(m_d, f"{case}.iobj"),index_col=0)
# plot out the dataframe that was shown as a table above
fig, ax = plt.subplots(figsize=(15,3))
ax.plot(df_obj.loc[:,["total_phi"]], linestyle='-', marker='o',c='black',linewidth=3)
ax.set_ylabel('Objective func. $\Phi$',c='black',fontsize=13)
ax.set_xlabel('Iteration number',c='black',fontsize=11)
ax2 = ax.twinx()
ax2.plot(df_obj.loc[:,["model_runs_completed"]], linestyle='--', marker='o',c='blue')
ax2.set_ylabel('Numbers of model runs',c='blue',fontsize=13)
ax2.set_yticks(ax2.get_yticks()[1:-1])
ax2.set_yticklabels(ax2.get_yticks().astype(int),color='blue')
ax.set_title('Parameters optimization convergence curve & total model runs',fontsize=15)
ax.set_xticks(np.linspace(0,len(df_obj)-1,len(df_obj)))
ax.grid()

plt.savefig(os.path.join('results','phi_iter_hydrothermalgeophy.png'),dpi=300, bbox_inches='tight')

# %%
leng = 529
b = i2

fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '100 cm SWC (hydro-thermal-geophysical)'
ax[0].set_ylabel(r'SWC $\theta$',fontsize=13)
ax[0].set_ylim([0,0.45])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured SWC',fontsize=13)
ax[1].set_ylabel('Modeled SWC',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)
# %%
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '20 cm SWC (hydro-thermal-geophysical)'
ax[0].set_ylabel(r'SWC $\theta$',fontsize=13)
ax[0].set_ylim([0,0.45])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured SWC',fontsize=13)
ax[1].set_ylabel('Modeled SWC',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)
# %%
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '30 cm SWC (hydro-thermal-geophysical)'
ax[0].set_ylabel(r'SWC $\theta$',fontsize=13)
ax[0].set_ylim([0,0.45])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured SWC',fontsize=13)
ax[1].set_ylabel('Modeled SWC',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)
# %%
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '50 cm SWC (hydro-thermal-geophysical)'
ax[0].set_ylabel(r'SWC $\theta$',fontsize=13)
ax[0].set_ylim([0,0.45])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured SWC',fontsize=13)
ax[1].set_ylabel('Modeled SWC',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)

# %%
# Plot Temperature results
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '100 cm Temperature (hydro-thermal-geophysical)'
ax[0].set_ylabel('Temperature',fontsize=13)
ax[0].set_ylim([17.5,24.5])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+',NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured Temperature',fontsize=13)
ax[1].set_ylabel('Modeled Temperature',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)



# %%
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '20 cm Temperature (hydro-thermal-geophysical)'
ax[0].set_ylabel('Temperature',fontsize=13)
ax[0].set_ylim([17.5,24.5])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured Temperature',fontsize=13)
ax[1].set_ylabel('Modeled Temperature',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)
# %%
b = b+leng
fig, ax = plt.subplots(1,2,figsize=(15,5), gridspec_kw={'width_ratios': [3, 2]})
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['modelled'][b:b+leng],label='modeled')
ax[0].plot(filterd_hydro_data['WF_datetime'],pst.res['measured'][b:b+leng],label='measured')
title_str = '50 cm Temperature (hydro-thermal-geophysical)'
ax[0].set_ylabel('Temperature',fontsize=13)
ax[0].set_ylim([17.5,24.5])
ax[0].legend(fontsize=11)
ax[0].set_title(title_str+', NSE={:.2f}'.format(nse(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng])),fontsize=15)

ax[1].scatter(pst.res['measured'][b:b+leng],pst.res['modelled'][b:b+leng],s=2,c='b')
ax[1].set_aspect('equal')
lim_max = max([max(pst.res['measured'][b:b+leng]),max(pst.res['modelled'][b:b+leng])])
lim_min = min([min(pst.res['measured'][b:b+leng]),min(pst.res['modelled'][b:b+leng])])
ax[1].plot([lim_min,lim_max],[lim_min,lim_max],'k--',linewidth=1)
ax[1].set_xlim([lim_min,lim_max])
ax[1].set_ylim([lim_min,lim_max])
ax[1].set_xlabel('Measured Temperature',fontsize=13)
ax[1].set_ylabel('Modeled Temperature',fontsize=13)
ax[1].set_title(title_str+' crossplot',fontsize=15)
fig.savefig(os.path.join('results',title_str+'.png'),dpi=300)
# %%
