# %%
import pandas as pd

# Read parameters in .pinp file
parameters_K = pd.read_csv('K_1.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )
parameters_K2 = pd.read_csv('K_2.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )                        
# parameters_phi = pd.read_csv('phi_1.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )  
# parameters_phi2 = pd.read_csv('phi_2.pinp'
#                         ,delimiter=' '
#                         ,header=None
#                         ,names=['par_name','par_valu']
#                         )            
parameters_lam = pd.read_csv('lam.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )
parameters_lam2 = pd.read_csv('lam_2.pinp'
                        ,delimiter=' '
                        ,header=None
                        ,names=['par_name','par_valu']
                        )
# Read original .inp file
Lines = []
with open('TARI.inp', 'r') as read_obj:
    for i,line in enumerate(read_obj):
        Lines.append(line)

# Change something about parameters ...
DATA_SET_6_pos = Lines.index('DATA SET 6: Material properties \n')

layer1_hydro_property = Lines[DATA_SET_6_pos+3].split()
layer1_hydro_property[3] = str(parameters_K['par_valu'][0]) # Kh
layer1_hydro_property[4] = str(parameters_K['par_valu'][0]) # Kv
# layer1_hydro_property[2] = str(parameters_phi['par_valu'][0]) # phi
layer1_hydro_property += '\n'
Lines[DATA_SET_6_pos+3] = ' '.join(layer1_hydro_property)

layer2_hydro_property = Lines[DATA_SET_6_pos+4].split()
layer2_hydro_property[3] = str(parameters_K2['par_valu'][0]) # Kh
layer2_hydro_property[4] = str(parameters_K2['par_valu'][0]) # Kv
# layer2_hydro_property[2] = str(parameters_phi2['par_valu'][0]) # phi
layer2_hydro_property += '\n'
Lines[DATA_SET_6_pos+4] = ' '.join(layer2_hydro_property)

layer1_thermal_property = Lines[DATA_SET_6_pos+41].split()
layer1_thermal_property[4] = '{:.2e}'.format(parameters_lam['par_valu'][0]*3600**3) # lambda_h
layer1_thermal_property[5] = '{:.2e}'.format(parameters_lam['par_valu'][0]*3600**3) # lambda_v
layer1_thermal_property += '\n'
Lines[DATA_SET_6_pos+41] = ' '.join(layer1_thermal_property)

layer2_thermal_property = Lines[DATA_SET_6_pos+42].split()
layer2_thermal_property[4] = '{:.2e}'.format(parameters_lam2['par_valu'][0]*3600**3) # lambda_h
layer2_thermal_property[5] = '{:.2e}'.format(parameters_lam2['par_valu'][0]*3600**3) # lambda_v
layer2_thermal_property += '\n'
Lines[DATA_SET_6_pos+42] = ' '.join(layer2_thermal_property)

# Write edited .inp file
with open('TARI.inp', 'w') as write_obj:
    for _,content in enumerate(Lines):
        write_obj.write('%s'%content)