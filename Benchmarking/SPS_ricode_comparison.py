#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate lifetimes for Pb54+, Pb80+ and Pb81+ in SPS to compare with RICODE-M data
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

# Semi-empirical formula for EL not meant for 1s shell, so in reality not relevant for Pb81+
also_plot_Pb81 = False

# Define approximate relevant energy range for SPS injection 
E_kin_min = 5.0
E_kin_max = 13.0

# Load data 
m_e = 0.510998950e6  # electron mass in eV
gas_fractions = pd.read_csv('../Data/Gas_fractions.csv', index_col=0)
pressure_data = pd.read_csv('../Data/Pressure_data.csv', index_col=0).T

# Load RICODE-M data from SPS (transcribed from Hirlaender et al, 2018) 10.18429/JACOW-IPAC2018-THPMF015
xdata_Pb54 = np.array([1.22, 1.9, 3.0, 4.9, 7.8, 12.5, 19., 32., 49., 79., 120., 195.])
ydata_Pb54 = np.array([2.35, 2.35, 2.3, 2.2, 2.15, 2.1, 2.05, 2.02, 2.01, 2.0, 2.0, 2.0])
xdata_Pb80 = np.array([2.6, 4.1, 6.8, 10.5, 13.0, 20.5, 33.0, 52.0, 82.0, 130.0, 205.0, 330., 515., 820.,1320., 2125., 5000.])
ydata_Pb80 = np.array([280., 250., 225., 200., 190., 180., 170., 163., 161., 159., 157., 157., 157., 157., 157., 157., 157.])
xdata_Pb81 = np.array([2.7, 4.2, 6.9, 11.0, 13.7, 21.0, 34.0, 53.0, 84.0, 134.0, 210.0, 330., 525., 840.,1350., 2150., 5000.])
ydata_Pb81 = np.array([605., 520., 480., 420., 400., 370., 355., 340., 338., 336., 336., 335., 336., 336., 336., 336., 336.])

# Define vector for injection energy 
energies_kin_inj = np.logspace(np.log10(1.5), np.log10(6000), 15) # in GeV/u 

# Instantiate rest gas class objects for the different ions - same gas compositions for all 
SPS_Pb54 =  beam_gas_collisions(pressure_data['SPS'].values[0], gas_fractions['SPS'].values)
SPS_Pb80 =  beam_gas_collisions(pressure_data['SPS'].values[0], gas_fractions['SPS'].values)
SPS_Pb81 =  beam_gas_collisions(pressure_data['SPS'].values[0], gas_fractions['SPS'].values)

# Initiate empty arrays for lifetime and rest gas 
tau_values_SPS_Pb54 = np.zeros(len(energies_kin_inj))
tau_values_SPS_Pb80 = np.zeros(len(energies_kin_inj))
tau_values_SPS_Pb81 = np.zeros(len(energies_kin_inj))

# Set projectile data common for all Pb ions - 
Z = 82.0
A = 208.0
m_u = 931.49410242e6  # 1 Dalton in eV/c^2 -- atomic mass unit 
atomic_mass_pb208 = 207.9766519 # Pb208 atomic mass from The AME2016 atomic mass evaluation (II).
m_ion = atomic_mass_pb208*m_u  

# Set projectile data for Pb54 in the SPS 
n0_Pb54 = 3
Ip_Pb54 = 5.414 # keV
q_Pb54 = 54.0
m_Pb54 = m_ion - (Z - q_Pb54) * m_e  # subtract electron mass
m_Pb54_in_u = m_Pb54/m_u 

# Set projectile data for Pb80 in the SPS 
n0_Pb80 = 1
Ip_Pb80 = 99.49185 # keV
q_Pb80 = 80.0
m_Pb80 = m_ion - (Z - q_Pb80) * m_e
m_Pb80_in_u = m_Pb80/m_u 

# Set projectile data for Pb81 in the SPS 
n0_Pb81 = 1
Ip_Pb81 = 101.3367 # keV
q_Pb81 = 81.0
m_Pb81 = m_ion - (Z - q_Pb81) * m_e
m_Pb81_in_u = m_Pb81/m_u 

# Iterate over the different energies
for i, E_kin in enumerate(energies_kin_inj):
    
    # Define energy for Pb54 and calculate rest gas 
    E_kin_Pb54 = E_kin*1e9*m_Pb54_in_u # total kinetic energy in eV per particle at injection
    E_tot_Pb54 = m_Pb54 + E_kin_Pb54
    gamma_Pb54 = E_tot_Pb54/m_Pb54
    beta_Pb54 = np.sqrt(1 - 1/gamma_Pb54**2)

    projectile_data_Pb54 = np.array([Z,
                                     q_Pb54,  
                                     E_kin*1e3,  # in MeV/u
                                     Ip_Pb54, 
                                     n0_Pb54,
                                     beta_Pb54
                                     ])
    SPS_Pb54.set_projectile_data(projectile_data_Pb54)  
    tau_values_SPS_Pb54[i] = SPS_Pb54.calculate_total_lifetime_full_gas()

    # Define energy for Pb80 and calculate rest gas 
    E_kin_Pb80 = E_kin*1e9*m_Pb80_in_u # total kinetic energy in eV per particle at injection
    E_tot_Pb80 = m_Pb80 + E_kin_Pb80
    gamma_Pb80 = E_tot_Pb80/m_Pb80
    beta_Pb80 = np.sqrt(1 - 1/gamma_Pb80**2)

    projectile_data_Pb80 = np.array([Z,
                                     q_Pb80,  
                                     E_kin*1e3,  # in MeV/u
                                     Ip_Pb80, 
                                     n0_Pb80,
                                     beta_Pb80
                                     ])
    SPS_Pb80.set_projectile_data(projectile_data_Pb80)  
    tau_values_SPS_Pb80[i] = SPS_Pb80.calculate_total_lifetime_full_gas()

    # Define energy for Pb81 and calculate rest gas 
    E_kin_Pb81 = E_kin*1e9*m_Pb81_in_u # total kinetic energy in eV per particle at injection
    E_tot_Pb81 = m_Pb81 + E_kin_Pb81
    gamma_Pb81 = E_tot_Pb81/m_Pb81
    beta_Pb81 = np.sqrt(1 - 1/gamma_Pb81**2)

    projectile_data_Pb81 = np.array([Z,
                                     q_Pb81,  
                                     E_kin*1e3,  # in MeV/u
                                     Ip_Pb81, 
                                     n0_Pb81,
                                     beta_Pb81
                                     ])
    SPS_Pb81.set_projectile_data(projectile_data_Pb81)  
    tau_values_SPS_Pb81[i] = SPS_Pb81.calculate_total_lifetime_full_gas()
    
# Relative error for lifetime from relative error in cross section and molecular gas density
d_sigma = 0.5
d_n = 0.1 
d_tau = np.sqrt(d_sigma**2 + d_n**2)
    
######## PLOT THE DATA ###########
SMALL_SIZE = 10
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

fig, ax = plt.subplots(1, 1, figsize = (6,5))

# Pb54+
ax.plot(energies_kin_inj, tau_values_SPS_Pb54, color='purple', label='Pb54+ semi-empirical')
ax.plot(xdata_Pb54, ydata_Pb54, 'o', ms=9, markerfacecolor='magenta', color='black', label='Pb54+ RICODE-M')
#ax.plot(xdata_Pb54, ydata_Pb54 * 2, '*', ms=10, markerfacecolor='magenta', color='black', label='Pb54+ RICODE w. relativistic correction')

# Pb80+
ax.plot(energies_kin_inj, tau_values_SPS_Pb80, color='red', label='Pb80+ semi-empirical')
ax.plot(xdata_Pb80, ydata_Pb80, 'o', ms=9, markerfacecolor='brown', color='black', label='Pb80+ RICODE-M')
#ax.plot(xdata_Pb80, ydata_Pb80 * 2, '*', ms=10, markerfacecolor='brown', color='black', label='Pb80+ RICODE w. relativistic correction')

# Pb81+ in reality not relevant for this case, but possible to plot it 
if also_plot_Pb81:
    ax.plot(energies_kin_inj, tau_values_SPS_Pb81, color='gold')
    ax.plot(xdata_Pb81, ydata_Pb81, 'o', ms=9, markerfacecolor='yellow', color='black', label='Pb81+ RICODE')

ax.axvspan(E_kin_min, E_kin_max, color='green', alpha=0.2, label='Relevant SPS injection energy')
ax.set_yscale('log')
ax.set_xscale('log')
ax.set_xlim(1, 5200)
ax.set_ylim(1, 1000)
ax.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.8)
ax.grid(visible=True, which='minor', color='grey', linestyle='--', alpha=0.55) 

ax.set_xlabel('E [GeV/u]')
ax.set_ylabel(r'Lifetime $\tau$ [s]')
ax.legend()
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('Output/SPS_RICODE_benchmarking_lifetimes.png', dpi=250)