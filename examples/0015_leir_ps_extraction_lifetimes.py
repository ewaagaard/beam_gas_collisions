"""
Calculate beam lifetimes and cross sections from beam-gas interactions - at extraction in LEIR and PS
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from beam_gas_collisions import IonLifetimes, DataObject

os.makedirs('plots_and_output', exist_ok=True)

# Load data 
projectile = 'Pb54'

# Instantiate classes for LEIR, PS, SPS
LEIR_rest_gas = IonLifetimes(projectile=projectile, machine='LEIR')
PS_rest_gas = IonLifetimes(projectile=projectile, machine='PS')


# Initiate empty arrays for all ion species
data = DataObject()
tau_dict = {'tau_LEIR_inj': [],
            'tau_LEIR_extr': [],
            'tau_PS_inj': [],
            'tau_PS_extr': []}

tau_values_LEIR = np.zeros(len(data.projectile_data.T.columns))
tau_values_PS = np.zeros(len(data.projectile_data.T.columns))
tau_values_LEIR_extr = np.zeros(len(data.projectile_data.T.columns))
tau_values_PS_extr = np.zeros(len(data.projectile_data.T.columns))

# Iterate over the different projectiles 
for i, projectile in enumerate(data.projectile_data.T.columns):
    
    ##### LEIR #####
    # Injection energy
    LEIR_rest_gas.projectile = projectile
    LEIR_rest_gas.set_projectile_data(data)  
    tau_dict['tau_LEIR_inj'].append(LEIR_rest_gas.calculate_total_lifetime_full_gas())
    # Extraction energy
    LEIR_rest_gas.set_projectile_data(data, at_injection=False)  
    tau_dict['tau_LEIR_extr'].append(LEIR_rest_gas.calculate_total_lifetime_full_gas())

    ##### PS #####
    # Injection energy
    PS_rest_gas.projectile = projectile
    PS_rest_gas.set_projectile_data(data)  
    tau_dict['tau_PS_inj'].append(PS_rest_gas.calculate_total_lifetime_full_gas())
    # Extraction energy
    PS_rest_gas.set_projectile_data(data, at_injection=False)  
    tau_dict['tau_PS_extr'].append(PS_rest_gas.calculate_total_lifetime_full_gas())

# Convert dictionary to pandas dataframe
df_tau = pd.DataFrame(tau_dict, index=data.projectile_data.T.columns)
df_tau.to_csv('plots_and_output/LEIR_PS_at_extr.csv')
print(df_tau)