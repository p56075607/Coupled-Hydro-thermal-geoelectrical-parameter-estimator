# %%
# Just some plotting settings
import matplotlib.pyplot as plt
from matplotlib import ticker
plt.style.use("seaborn-notebook")
import numpy as np
import pandas as pd
import pygimli as pg # import pygimli with short name
from pygimli import meshtools as mt # import a module 
from pygimli.viewer import showMesh # import a function
from pygimli.physics.petro import transFwdArchieS as ArchieTrans
from pygimli.frameworks import PetroInversionManager
from pygimli.physics import ert  # the module
import scipy
from datetime import datetime, timedelta
from numpy import newaxis
# import matplotlib.font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
plt.rcParams["font.family"] = "Book Antiqua"
from os.path import join
import os
import pickle
import sys
sys.path.append(r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\THMC')
from read_THMC_flow_result import read_THMC_flow_result
from read_THMC_heat_result import read_THMC_heat_result
fig_save_ph = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\Petrophysic\result'
from scipy import stats
# %%
# Setting nodes
yTHMC = -np.round(pg.cat(pg.cat(pg.utils.grange(0, 0.6, n=31),
                                pg.utils.grange(0.64, 1, n=10)),

                         pg.cat(pg.utils.grange(1.1, 2, n=10),
                                pg.utils.grange(2.2, 5, n=15)))
                 ,2)[::-1]
xTHMC = np.round(pg.cat(pg.cat(pg.utils.grange(20, 24, n=11),
                        pg.utils.grange(24.2, 26, n=10)),
                pg.utils.grange(26.4, 30, n=10))
        ,1)
hydroDomain = pg.createGrid(x=xTHMC,
                                y=yTHMC)
XTHMC,YTHMC = np.meshgrid(xTHMC,yTHMC)

# %%
# Import THMC hydro-model
# Load ASCII dat file for flow
dat_file_name = join('PostP','001_H_SbFlow.dat')
Flow_all,var_10_cm,_,_,_,_ = read_THMC_flow_result(dat_file_name,Var_ind=2)
# Load ASCII dat file for heat
dat_file_name = join('PostP','001_T_Tmptr.dat')
Heat_all,_,_,_,_,_,_ = read_THMC_heat_result(dat_file_name,Var_ind=2)
# Re-interpolate the grid and plot the inverted proifle
# Creat datetime array
time_index = []
for i in range(int(len(var_10_cm))):
        time_index.append(i)
datetime_array = np.arange(datetime(2022,2,12,00), datetime(2022,3,6,1), timedelta(hours=1)).astype(datetime)
time_index = np.array(time_index)
datetime_THMC = datetime_array[time_index]

# Load Archie's parameter for each layer
# parameters_fill_n = pd.read_csv('fill_n.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )
# parameters_fill_cw = pd.read_csv('fill_cw.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )
# parameters_fill_cs = pd.read_csv('fill_cs.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )
parameters_allu_n = pd.read_csv('allu_n.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )
parameters_allu_cw = pd.read_csv('allu_cw.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )
# parameters_allu_cs = pd.read_csv('allu_cs.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )
# Archie's transform
# fill_n = parameters_fill_n['par_valu'][0]
# fill_cw = parameters_fill_cw['par_valu'][0]
# fill_cs = parameters_fill_cs['par_valu'][0]
allu_n = parameters_allu_n['par_valu'][0]
allu_cw = parameters_allu_cw['par_valu'][0]
# allu_cs = parameters_allu_cs['par_valu'][0]

# Archie's law
def Archie(S,cFluid,n,cSurface=0,phi=0.4,d=1.3,temp_correct=False,T=25):
    if temp_correct:
         sigma_b = phi**d*( cFluid*S**n + (phi**-d-1)*cSurface ) * (1+0.0183*(T-25))
    else:
         sigma_b = phi**d*( cFluid*S**n + (phi**-d-1)*cSurface )
    return sigma_b


# %%
ERT_time_index = [9,249]
# for i in range(11):
#     ERT_time_index.append(9+i*24)
for ERT_i in ERT_time_index:

    Saturation_THMC = Flow_all[:,:,ERT_i]
    Temperature_THMC = Heat_all[:,:,ERT_i]-273

    res = np.empty((len(yTHMC),len(xTHMC)))
    for x in range(len(xTHMC)):
        for y in range(len(yTHMC)):
            if yTHMC[y] >= -0.3: # Fill layer
                res[y,x] = 1/Archie(S=Saturation_THMC[y,x],cFluid=allu_cw,phi=0.302,n=allu_n,temp_correct=True,T=Temperature_THMC[y,x])
            else: # Alluvium layer
                res[y,x] = 1/Archie(S=Saturation_THMC[y,x],cFluid=allu_cw,phi=0.441,n=allu_n,temp_correct=True,T=Temperature_THMC[y,x])
                # res[y,x] = 1/Archie(S=Saturation_THMC[y,x],cFluid=fill_cw,n=fill_n,cSurface=fill_cs)

    # Interpolate grid from THMC domain to ERT domain
    yDevide = 1.0 - np.logspace(np.log10(1.0), np.log10(3.0),21 )[::-1]
    xDevide = np.linspace(start=0, stop=50, num=201)
    ERTDomain = pg.createGrid(x=xDevide,y=yDevide)
    XERT,YERT = np.meshgrid(xDevide,yDevide)
    intp_func_flow = scipy.interpolate.interp2d(xTHMC,yTHMC,res, kind='linear')
    res_ERT = intp_func_flow(xDevide,yDevide)
    # Load ERT configurations
    ERT_data_conf_ph = r'E:\VM\Win7_64bit_EarthImagerVM 1\Share_Folder\R2MS_TARI\data\external\Petrophysic\sim_time_9\shm\22021209_m_E1.shm'
    data = pg.load(ERT_data_conf_ph)
    # Simulate ERT
    meshERT = pg.meshtools.appendTriangleBoundary(ERTDomain, marker=1,
                                        xbound=20, ybound=20,
                                    #    area=10
                                        )
    ERT = pg.physics.ert.ERTManager(verbose=False, sr=False)
    res_ERT_1D = np.reshape(res_ERT,len(ERTDomain.nodes()))
    res_cell = pg.interpolate(ERTDomain,res_ERT_1D,ERTDomain.cellCenters())
    simulated_ert_data = ERT.simulate(mesh=ERTDomain, meshI=meshERT, scheme=data, res=res_cell, noiseLevel=0.01)
    isExist = os.path.exists('sim_shm')
    if not isExist:
        os.makedirs('sim_shm')
    simulated_ert_data.save(join('sim_shm',"{}{}{}{}_m_E1.shm".format(str(datetime_THMC[ERT_i])[2:4]
                                                                        ,str(datetime_THMC[ERT_i])[5:7]
                                                                        ,str(datetime_THMC[ERT_i])[8:10]
                                                                        ,str(datetime_THMC[ERT_i])[11:13]))
                                                                        , "a b m n err k r rhoa valid")

    df_rhoa = pd.DataFrame([])
    sim_data = np.array(simulated_ert_data['rhoa'])#np.log10()#stats.zscore()
    df_rhoa['rhoa_{}{}{}'.format(str(datetime_THMC[ERT_i])[5:7]
                            ,str(datetime_THMC[ERT_i])[8:10]
                            ,str(datetime_THMC[ERT_i])[11:13])] = pd.DataFrame(sim_data)
    df_rhoa['index'] = df_rhoa.index
    isExist = os.path.exists('sim_csv')
    if not isExist:
        os.makedirs('sim_csv')
    df_rhoa.to_csv(join('sim_csv',"{}{}{}{}_m_E1.csv".format(
                                                    str(datetime_THMC[ERT_i])[2:4]
                                                ,str(datetime_THMC[ERT_i])[5:7]
                                                ,str(datetime_THMC[ERT_i])[8:10]
                                                ,str(datetime_THMC[ERT_i])[11:13]))
                        ,index=False
                        , encoding = 'utf-8')
