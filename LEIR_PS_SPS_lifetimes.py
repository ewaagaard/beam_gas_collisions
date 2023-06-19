#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Calculate beam lifetimes and cross sections from beam-gas interactions 

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from beam_gas_collisions import beam_gas_collisions

# Plotting specifications
bars_in_separate_plot = False

# Load data 
projectile = 'Pb54'
gas_fractions = pd.read_csv('Data/Gas_fractions.csv', index_col=0)
pressure_data = pd.read_csv('Data/Pressure_data.csv', index_col=0).T
projectile_data = pd.read_csv('Data/Projectile_data.csv', index_col=0)

# Instantiate classes for LEIR, PS, SPS
LEIR_rest_gas =  beam_gas_collisions(pressure_data['LEIR'].values[0],
                                     gas_fractions['LEIR'].values)

PS_rest_gas =  beam_gas_collisions(pressure_data['PS'].values[0],
                                     gas_fractions['PS'].values)

SPS_rest_gas =  beam_gas_collisions(pressure_data['SPS'].values[0],
                                     gas_fractions['SPS'].values)

# Initiate empty arrays for lifetime and rest gas 
tau_values_LEIR = np.zeros(len(projectile_data.T.columns))
tau_values_PS = np.zeros(len(projectile_data.T.columns))
tau_values_SPS = np.zeros(len(projectile_data.T.columns))

# Rows represent each projectile, columns each gas constituent
# third dimension stacks 3 quantities: EL cross section, EC cross section and total cross section
sigmas_LEIR = np.zeros([len(projectile_data.T.columns), np.count_nonzero(gas_fractions['LEIR']), 3])
sigmas_PS = np.zeros([len(projectile_data.T.columns), np.count_nonzero(gas_fractions['PS']), 3])
sigmas_SPS = np.zeros([len(projectile_data.T.columns), np.count_nonzero(gas_fractions['SPS']), 3])


# Iterate over the different projectiles 
for i, projectile in enumerate(projectile_data.T.columns):
    
    # Calculate lifetimes and cross sections for LEIR 
    projectile_data_LEIR = np.array([projectile_data['Z'][projectile],
                                     projectile_data['LEIR_q'][projectile],
                                     projectile_data['LEIR_Kinj'][projectile],
                                     projectile_data['I_p'][projectile], 
                                     projectile_data['n_0'][projectile], 
                                     projectile_data['LEIR_beta'][projectile]])

    LEIR_rest_gas.set_projectile_data(projectile_data_LEIR)  
    tau_values_LEIR[i] = LEIR_rest_gas.calculate_total_lifetime_full_gas()
    sigmas_LEIR[i, :, 0] = LEIR_rest_gas.return_all_sigmas()[0] # EL cross section
    sigmas_LEIR[i, :, 1] = LEIR_rest_gas.return_all_sigmas()[1] # EC cross section
    sigmas_LEIR[i, :, 2] = LEIR_rest_gas.return_all_sigmas()[0] + LEIR_rest_gas.return_all_sigmas()[1] # total cross section

    # Calculate lifetimes and cross sections for PS
    projectile_data_PS = np.array([projectile_data['Z'][projectile],
                                     projectile_data['PS_q'][projectile],
                                     projectile_data['PS_Kinj'][projectile],
                                     projectile_data['I_p'][projectile], 
                                     projectile_data['n_0'][projectile], 
                                     projectile_data['PS_beta'][projectile]])

    PS_rest_gas.set_projectile_data(projectile_data_PS)  
    tau_values_PS[i] = PS_rest_gas.calculate_total_lifetime_full_gas()
    sigmas_PS[i, :, 0] = PS_rest_gas.return_all_sigmas()[0] # EL cross section
    sigmas_PS[i, :, 1] = PS_rest_gas.return_all_sigmas()[1] # EC cross section
    sigmas_PS[i, :, 2] = PS_rest_gas.return_all_sigmas()[0] + PS_rest_gas.return_all_sigmas()[1] # total cross section


    # Calculate lifetimes and cross sections for SPS
    projectile_data_SPS = np.array([projectile_data['Z'][projectile],
                                     projectile_data['SPS_q'][projectile],
                                     projectile_data['SPS_Kinj'][projectile],
                                     projectile_data['I_p'][projectile], 
                                     projectile_data['n_0'][projectile], 
                                     projectile_data['SPS_beta'][projectile]])

    SPS_rest_gas.set_projectile_data(projectile_data_SPS)  
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
    df = df.set_index(projectile_data.index)
    df.columns = gas_fractions[machine][gas_fractions[machine] > 0.].index # gas_fractions.index

    # Add the 'Machine' and 'Cross Section Type' columns
    df['Machine'] = machine
    df['Cross Section Type'] = cross_section_types[i % len(cross_section_types)]
    
    # Update the dataframe in the original dataframes array
    dataframes[i] = df

# Concatenate the modified dataframes and rename columns
df_sigma_all = pd.concat(dataframes)
df_sigma_all.to_csv('Output/Cross_sections_all_gases_and_machines.csv')

# Make plot with gas fractions for 
gas_fractions_data = gas_fractions.loc[(gas_fractions!=0).any(axis=1)] 


######## PLOT THE DATA ###########
SMALL_SIZE = 12
MEDIUM_SIZE = 15
BIGGER_SIZE = 20
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

x = np.arange(len(projectile_data.index))

if bars_in_separate_plot:
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (16,5))
    fig.suptitle("Total beam-gas lifetimes")
    bar1 = ax1.bar(x, tau_values_LEIR, color='cyan', label='LEIR') #
    bar2 = ax2.bar(x, tau_values_PS, color='red', label='PS') #
    bar3 = ax3.bar(x, tau_values_SPS, color='forestgreen', label='SPS') #
    ax1.set_xticks(x)
    ax2.set_xticks(x)
    ax3.set_xticks(x)
    ax1.set_xticklabels(projectile_data.index)
    ax2.set_xticklabels(projectile_data.index)
    ax3.set_xticklabels(projectile_data.index)
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax1.bar_label(bar1, labels=[f'{e:,.1e}' for e in tau_values_LEIR], padding=3, color='black', fontsize=8) 
    ax2.bar_label(bar2, labels=[f'{e:,.1e}' for e in tau_values_PS], padding=3, color='black', fontsize=8) 
    ax3.bar_label(bar3, labels=[f'{e:,.1e}' for e in tau_values_SPS], padding=3, color='black', fontsize=8) 
    ax1.set_ylabel(r"Lifetime $\tau$ [s]")
    ax1.legend()
    ax2.legend()
    ax3.legend()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig("Output/LEIR_PS_SPS_full_lifetime_plot.png", dpi=250)
else:

    bar_width = 0.25
    fig, ax = plt.subplots(1, 1, figsize = (11,5))
    bar1 = ax.bar(x - 1.15*bar_width, tau_values_LEIR, bar_width, color='cyan', label='LEIR') #
    bar2 = ax.bar(x, tau_values_PS, bar_width, color='red', label='PS') #
    bar3 = ax.bar(x + 1.15*bar_width, tau_values_SPS, bar_width, color='forestgreen', label='SPS') #
    ax.bar_label(bar1, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_LEIR], padding=3, color='black', fontsize=9) 
    ax.bar_label(bar2, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_PS], padding=3, color='black', fontsize=9) 
    ax.bar_label(bar3, labels=[f'{e:,.1e}'.replace('+0', '') for e in tau_values_SPS], padding=3, color='black', fontsize=9) 
    ax.set_yscale('log')
    ax.set_xticks(x)
    ax.set_xticklabels(projectile_data.index)
    ax.set_ylabel(r"Lifetime $\tau$ [s]")
    ax.legend()
    fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    fig.savefig('Output/LEIR_PS_SPS_full_lifetime_plot_compact.png', dpi=250)
    
    
# Plot the cross sections of each projectile on H2 (first column)
bar_width = 0.25
fig2, ax2 = plt.subplots(1, 1, figsize = (11,5))
fig2.suptitle('Projectile cross sections on H2', fontsize=18)
bar11 = ax2.bar(x - 1.15*bar_width, df_sigmas_EL_LEIR[:][0], bar_width, hatch='//', color='royalblue', label='LEIR EL') #
bar12 = ax2.bar(x - 1.15*bar_width, df_sigmas_EC_LEIR[:][0], bar_width, color='cyan', hatch='\\',  alpha=0.65, label='LEIR EC') #
bar21 = ax2.bar(x, df_sigmas_EL_PS[:][0], bar_width, color='coral', label='PS EL') #
bar22 = ax2.bar(x, df_sigmas_EC_PS[:][0], bar_width, color='maroon', alpha=0.8, label='PS EC') #
bar31 = ax2.bar(x + 1.15*bar_width, df_sigmas_EL_SPS[:][0], bar_width, color='forestgreen', label='SPS EL') #
bar32 = ax2.bar(x + 1.15*bar_width, df_sigmas_EC_SPS[:][0], bar_width, color='lime', alpha=0.8, label='SPS EC') #
ax2.set_yscale('log')
ax2.set_xticks(x)
ax2.set_xticklabels(projectile_data.index)
ax2.set_ylabel(r"Cross section $\sigma$ [m$^{2}$]")
ax2.legend()
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('Output/Cross_sections_on_H2.png', dpi=250)


# Make plot of different gas fractions
x3 = np.arange(len(gas_fractions_data.index))
fig3, ax3 = plt.subplots(1, 1, figsize = (11,5))
bar13 = ax3.bar(x3 - 1.15*bar_width, gas_fractions_data['LEIR'], bar_width, color='cyan', label='LEIR: P = {:.1e} mbar'.format(pressure_data['LEIR'].values[0])) #
bar23 = ax3.bar(x3, gas_fractions_data['PS'], bar_width, color='red', label='PS:     P = {:.1e} mbar'.format(pressure_data['PS'].values[0])) #
bar33 = ax3.bar(x3 + 1.15*bar_width, gas_fractions_data['SPS'], bar_width, color='forestgreen', label='SPS:   P = {:.1e} mbar'.format(pressure_data['SPS'].values[0])) #
ax3.bar_label(bar13, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['LEIR']], padding=3, color='black', fontsize=9) 
ax3.bar_label(bar23, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['PS']], padding=3, color='black', fontsize=9) 
ax3.bar_label(bar33, labels=[f'{e:,.1e}'.replace('+0', '') for e in gas_fractions_data['SPS']], padding=3, color='black', fontsize=9) 
ax3.set_xticks(x3)
ax3.set_xticklabels(gas_fractions_data.index)
ax3.set_ylabel(r"Rest gas fraction")
ax3.legend()
#ax3.set_yscale('log')
fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig3.savefig('Output/LEIR_PS_SPS_gas_composition.png', dpi=250)