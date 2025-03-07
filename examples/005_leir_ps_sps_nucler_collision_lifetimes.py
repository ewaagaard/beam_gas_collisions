"""
Calculate beam lifetimes and cross sections due to inelastic nuclear collisions
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re
from beam_gas_collisions import IonLifetimes, DataObject, BeamGasCollisions

os.makedirs('plots_and_output', exist_ok=True)

# Load data 
projectile = 'Pb54'

# Instantiate classes for LEIR, PS, SPS
LEIR_rest_gas = BeamGasCollisions(projectile=projectile, machine='LEIR')
PS_rest_gas = BeamGasCollisions(projectile=projectile, machine='PS')
SPS_rest_gas =  BeamGasCollisions(projectile=projectile, machine='SPS')

# Initiate empty arrays for all ion species
data = DataObject()
tau_values_LEIR = np.zeros(len(data.projectile_data.T.columns))
tau_values_PS = np.zeros(len(data.projectile_data.T.columns))
tau_values_SPS = np.zeros(len(data.projectile_data.T.columns))


# Iterate over the different projectiles 
for i, projectile in enumerate(data.projectile_data.T.columns):
    
    # Calculate lifetimes and cross sections for LEIR 
    LEIR_rest_gas.projectile = projectile
    LEIR_rest_gas.set_projectile_data(data)  
    tau_values_LEIR[i], N_dict_LEIR = LEIR_rest_gas.get_lifetimes_from_nuclear_collisions()

    # Calculate lifetimes and cross sections for PS
    PS_rest_gas.projectile = projectile
    PS_rest_gas.set_projectile_data(data)  
    tau_values_PS[i], N_dict_PS = PS_rest_gas.get_lifetimes_from_nuclear_collisions()
    

    # Calculate lifetimes and cross sections for SPS
    SPS_rest_gas.projectile = projectile
    SPS_rest_gas.set_projectile_data(data)   
    tau_values_SPS[i], N_dict_SPS = SPS_rest_gas.get_lifetimes_from_nuclear_collisions()
    

######## PLOT THE DATA ###########
x = np.arange(len(data.projectile_data.index))

# Function to convert to LaTeX-style charge states
def convert_to_charge_state(label):
    # Split the label into element and charge state using regex
    match = re.match(r"([A-Za-z]+)(\d+)", label)
    if match:
        element = match.group(1)
        charge = match.group(2)
        # Return the element and the LaTeX superscript for charge
        return f"{element}$^{{{charge}+}}$"

# Apply the function to each label in the index
latex_labels = [convert_to_charge_state(label) for label in data.projectile_data.index]

# Only plot unique mass numbers, i.e. not different charge states
unique_index = [0, 2, 6, 7, 8, 9, 10, 11, 12]
x_unique = np.arange(len(unique_index))

bar_width = 0.21
fig, ax = plt.subplots(1, 1, figsize = (9, 5.3), constrained_layout=True)
bar1 = ax.bar(x_unique - 1.15*bar_width, tau_values_LEIR[unique_index], bar_width, color='deepskyblue', label='LEIR') #
bar2 = ax.bar(x_unique, tau_values_PS[unique_index], bar_width, color='orangered', label='PS') #
bar3 = ax.bar(x_unique + 1.15*bar_width, tau_values_SPS[unique_index], bar_width, color='chartreuse', label='SPS') #
ax.bar_label(bar1, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_LEIR[unique_index]], padding=3, color='black', fontsize=11) 
ax.bar_label(bar2, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_PS[unique_index]], padding=3, color='black', fontsize=11) 
ax.bar_label(bar3, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_SPS[unique_index]], padding=3, color='black', fontsize=11) 
ax.set_yscale('log')
ax.set_xticks(x_unique)
ax.set_xticklabels(np.array(latex_labels)[unique_index])
ax.set_ylabel(r"Beam lifetime $\tau_{N}$ [s]")
ax.legend(fontsize=14)
ax.set_ylim(10., 1e10)
plt.grid(alpha=0.55)
fig.savefig('plots_and_output/LEIR_PS_SPS_nuclear_interaction_lifetimes.png', dpi=250)
plt.show()   