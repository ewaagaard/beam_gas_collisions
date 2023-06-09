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

tau_values_LEIR = np.zeros(len(projectile_data.T.columns))
tau_values_PS = np.zeros(len(projectile_data.T.columns))
tau_values_SPS = np.zeros(len(projectile_data.T.columns))

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

    # Calculate lifetimes and cross sections for PS
    projectile_data_PS = np.array([projectile_data['Z'][projectile],
                                     projectile_data['PS_q'][projectile],
                                     projectile_data['PS_Kinj'][projectile],
                                     projectile_data['I_p'][projectile], 
                                     projectile_data['n_0'][projectile], 
                                     projectile_data['PS_beta'][projectile]])

    PS_rest_gas.set_projectile_data(projectile_data_PS)  
    tau_values_PS[i] = PS_rest_gas.calculate_total_lifetime_full_gas()


    # Calculate lifetimes and cross sections for SPS
    projectile_data_SPS = np.array([projectile_data['Z'][projectile],
                                     projectile_data['SPS_q'][projectile],
                                     projectile_data['SPS_Kinj'][projectile],
                                     projectile_data['I_p'][projectile], 
                                     projectile_data['n_0'][projectile], 
                                     projectile_data['SPS_beta'][projectile]])

    SPS_rest_gas.set_projectile_data(projectile_data_SPS)  
    tau_values_SPS[i] = SPS_rest_gas.calculate_total_lifetime_full_gas()
    
    
#### PLOT THE DATA #######
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