"""
Calculate beam lifetimes and cross sections from beam-gas interactions 
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
SPS_rest_gas =  IonLifetimes(projectile=projectile, machine='SPS')

# Initiate empty arrays for all ion species
data = DataObject()
tau_values_LEIR = np.zeros(len(data.projectile_data.T.columns))
tau_values_PS = np.zeros(len(data.projectile_data.T.columns))
tau_values_SPS = np.zeros(len(data.projectile_data.T.columns))

# Rows represent each projectile, columns each gas constituent
# third dimension stacks 3 quantities: EL cross section, EC cross section and total cross section
sigmas_LEIR = np.zeros([len(data.projectile_data.T.columns), np.count_nonzero(data.gas_fractions_all['LEIR']), 3])
sigmas_PS = np.zeros([len(data.projectile_data.T.columns), np.count_nonzero(data.gas_fractions_all['PS']), 3])
sigmas_SPS = np.zeros([len(data.projectile_data.T.columns), np.count_nonzero(data.gas_fractions_all['SPS']), 3])


# Iterate over the different projectiles 
for i, projectile in enumerate(data.projectile_data.T.columns):
    
    # Calculate lifetimes and cross sections for LEIR 
    LEIR_rest_gas.projectile = projectile
    LEIR_rest_gas.set_projectile_data(data)  
    tau_values_LEIR[i] = LEIR_rest_gas.calculate_total_lifetime_full_gas()
    sigmas_LEIR[i, :, 0] = LEIR_rest_gas.return_all_sigmas()[0] # EL cross section
    sigmas_LEIR[i, :, 1] = LEIR_rest_gas.return_all_sigmas()[1] # EC cross section
    sigmas_LEIR[i, :, 2] = LEIR_rest_gas.return_all_sigmas()[0] + LEIR_rest_gas.return_all_sigmas()[1] # total cross section

    # Calculate lifetimes and cross sections for PS
    PS_rest_gas.projectile = projectile
    PS_rest_gas.set_projectile_data(data)  
    tau_values_PS[i] = PS_rest_gas.calculate_total_lifetime_full_gas()
    sigmas_PS[i, :, 0] = PS_rest_gas.return_all_sigmas()[0] # EL cross section
    sigmas_PS[i, :, 1] = PS_rest_gas.return_all_sigmas()[1] # EC cross section
    sigmas_PS[i, :, 2] = PS_rest_gas.return_all_sigmas()[0] + PS_rest_gas.return_all_sigmas()[1] # total cross section


    # Calculate lifetimes and cross sections for SPS
    SPS_rest_gas.projectile = projectile
    SPS_rest_gas.set_projectile_data(data)   
    tau_values_SPS[i] = SPS_rest_gas.calculate_total_lifetime_full_gas()
    sigmas_SPS[i, :, 0] = SPS_rest_gas.return_all_sigmas()[0] # EL cross section
    sigmas_SPS[i, :, 1] = SPS_rest_gas.return_all_sigmas()[1] # EC cross section
    sigmas_SPS[i, :, 2] = SPS_rest_gas.return_all_sigmas()[0] + SPS_rest_gas.return_all_sigmas()[1] # total cross section
    

# Make dataframes out of cross sections 
df_sigmas_EL_LEIR = pd.DataFrame(sigmas_LEIR[:, :, 0])
df_sigmas_EC_LEIR = pd.DataFrame(sigmas_LEIR[:, :, 1])
df_sigmas_tot_LEIR = pd.DataFrame(sigmas_LEIR[:, :, 2])    

df_sigmas_EL_PS = pd.DataFrame(sigmas_PS[:, :, 0])
df_sigmas_EC_PS = pd.DataFrame(sigmas_PS[:, :, 1])
df_sigmas_tot_PS = pd.DataFrame(sigmas_PS[:, :, 2])    

df_sigmas_EL_SPS = pd.DataFrame(sigmas_SPS[:, :, 0])
df_sigmas_EC_SPS = pd.DataFrame(sigmas_SPS[:, :, 1])
df_sigmas_tot_SPS = pd.DataFrame(sigmas_SPS[:, :, 2])    

# Concatenate the dataframes and add the machine and cross section type columns
machines = ['LEIR', 'PS', 'SPS']
cross_section_types = ['EL', 'EC', 'Total']
dataframes = [
    df_sigmas_EL_LEIR, df_sigmas_EC_LEIR, df_sigmas_tot_LEIR,
    df_sigmas_EL_PS, df_sigmas_EC_PS, df_sigmas_tot_PS,
    df_sigmas_EL_SPS, df_sigmas_EC_SPS, df_sigmas_tot_SPS
]

# Add machine and cross section type 
for i, df in enumerate(dataframes):
    
    machine = machines[i // len(cross_section_types)]
    df = df.set_index(data.projectile_data.index)
    df.columns = data.gas_fractions_all[machine][data.gas_fractions_all[machine] > 0.].index # gas_fractions.index

    # Add the 'Machine' and 'Cross Section Type' columns
    df['Machine'] = machine
    df['Cross Section Type'] = cross_section_types[i % len(cross_section_types)]
    
    # Update the dataframe in the original dataframes array
    dataframes[i] = df

# Concatenate the modified dataframes and rename columns
df_sigma_all = pd.concat(dataframes)
df_sigma_all.to_csv('plots_and_output/Cross_sections_all_gases_and_machines.csv')

# Make plot with gas fractions for 
gas_fractions_data = data.gas_fractions_all.loc[(data.gas_fractions_all!=0).any(axis=1)] 


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


bar_width = 0.21
fig, ax = plt.subplots(1, 1, figsize = (13,5.8), constrained_layout=True)
bar1 = ax.bar(x - 1.15*bar_width, tau_values_LEIR, bar_width, color='cyan', label='LEIR') #
bar2 = ax.bar(x, tau_values_PS, bar_width, color='red', label='PS') #
bar3 = ax.bar(x + 1.15*bar_width, tau_values_SPS, bar_width, color='limegreen', label='SPS') #
ax.bar_label(bar1, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_LEIR], padding=3, color='black', fontsize=10) 
ax.bar_label(bar2, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_PS], padding=3, color='black', fontsize=10) 
ax.bar_label(bar3, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_SPS], padding=3, color='black', fontsize=10) 
ax.set_yscale('log')
ax.set_xticks(x)
ax.set_xticklabels(latex_labels)
ax.set_ylabel(r"Lifetime $\tau$ [s]")
ax.legend()
plt.grid(alpha=0.55)
fig.savefig('plots_and_output/LEIR_PS_SPS_full_lifetime_plot_compact.png', dpi=250)
    
    
# Plot the cross sections of each projectile on H2 (first column)
bar_width = 0.25
fig2, ax2 = plt.subplots(1, 1, figsize = (13,5.8), constrained_layout=True)
#fig2.suptitle('Projectile cross sections on H2', fontsize=18)
bar11 = ax2.bar(x - 1.15*bar_width, df_sigmas_EL_LEIR[:][0], bar_width, hatch='//', color='royalblue', label='LEIR EL') #
bar12 = ax2.bar(x - 1.15*bar_width, df_sigmas_EC_LEIR[:][0], bar_width, color='cyan', hatch='\\',  alpha=0.65, label='LEIR EC') #
bar21 = ax2.bar(x, df_sigmas_EL_PS[:][0], bar_width, color='coral', label='PS EL') #
bar22 = ax2.bar(x, df_sigmas_EC_PS[:][0], bar_width, color='maroon', alpha=0.8, label='PS EC') #
bar31 = ax2.bar(x + 1.15*bar_width, df_sigmas_EL_SPS[:][0], bar_width, color='forestgreen', label='SPS EL') #
bar32 = ax2.bar(x + 1.15*bar_width, df_sigmas_EC_SPS[:][0], bar_width, color='lime', alpha=0.8, label='SPS EC') #
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(latex_labels)
ax2.set_ylabel(r"Cross section $\sigma$ on H$_{2}$ [m$^{2}$]")
ax2.legend(fontsize=11)
plt.grid(alpha=0.55)
fig2.savefig('plots_and_output/Cross_sections_on_H2.png', dpi=250)


# Make plot of different gas fractions
x3 = np.arange(len(gas_fractions_data.index))
fig3, ax3 = plt.subplots(1, 1, figsize = (11,5), constrained_layout=True)
bar13 = ax3.bar(x3 - 1.15*bar_width, gas_fractions_data['LEIR'], bar_width, color='cyan', label='LEIR: P = {:.1e} mbar'.format(data.pressure_data['LEIR'].values[0])) #
bar23 = ax3.bar(x3, gas_fractions_data['PS'], bar_width, color='red', label='PS:     P = {:.1e} mbar'.format(data.pressure_data['PS'].values[0])) #
bar33 = ax3.bar(x3 + 1.15*bar_width, gas_fractions_data['SPS'], bar_width, color='limegreen', label='SPS:   P = {:.1e} mbar'.format(data.pressure_data['SPS'].values[0])) #
ax3.bar_label(bar13, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['LEIR']], padding=3, color='black', fontsize=9) 
ax3.bar_label(bar23, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['PS']], padding=3, color='black', fontsize=9) 
ax3.bar_label(bar33, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['SPS']], padding=3, color='black', fontsize=9) 
ax3.set_xticks(x3)
ax3.set_xticklabels(gas_fractions_data.index)
ax3.set_ylabel(r"Rest gas fraction")
ax3.legend()
#ax3.set_yscale('log')
plt.grid(alpha=0.55)
fig3.savefig('plots_and_output/LEIR_PS_SPS_gas_composition.png', dpi=250)
plt.show()
