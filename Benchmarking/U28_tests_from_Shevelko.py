#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking tests to compare lifetime and cross sections of U28+ from Fig. 13 and Fig. 14 in Shevelko (2011)
https://www.sciencedirect.com/science/article/pii/S0168583X11003272
"""

import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd 
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

# 
also_generate_transparent_image = True 

# Rest gas composition of ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'He', 'O2', 'Ar']
gas_fractions = np.array([0.758, 0.049, 0.026, 0.119, 0.002, 0.034, 0.004, 0.008])
p = 1e-10
ion_beam_U28 = beam_gas_collisions(p, gas_fractions)

# Projectile_ddata
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 

# Initiate arrays
energies_kin_inj = np.logspace(np.log10(5.0), np.log10(1e4), 15) # in MeV/u 
tau_values_U28 = np.zeros(len(energies_kin_inj))

###### STEP 1: calculate lifetime of U28 in rest gas ######

# Iterate over the different energies
for i, E_kin in enumerate(energies_kin_inj):

    # Calculate lifetimes and cross sections for LEIR 
    projectile_data_U28 = np.array([Z_p,
                                    q_p,  
                                    E_kin,  # in MeV/u
                                    Ip, 
                                    n0,
                                    atomic_mass_U238_in_u
                                    ])
        
    ion_beam_U28.set_projectile_data(projectile_data_U28, provided_beta=False)  
    tau_values_U28[i] = ion_beam_U28.calculate_total_lifetime_full_gas()
    
###### STEP 2: calculate U28 EL and EC cross sections ######
    
EC_energy_range =  np.logspace(np.log10(1e-3), np.log10(10), 10) # in MeV/u 
EL_energy_range =  np.logspace(np.log10(1e-1), np.log10(1e4), 10) # in MeV/u 

# Empty arrays where rows contain U28+ on H2, N2 and Ar:
sigma_EC = np.zeros([3, len(EC_energy_range)])
sigma_EL = np.zeros([3, len(EL_energy_range)])
Z_H = 1.0
Z_N = 7.0
Z_Ar = 18.0

# Initiate beam-gas collision object
ion_beam_U28_sigmas = beam_gas_collisions()

# Iterate over the different energies
for i, E_kin_EC in enumerate(EC_energy_range):

    # Calculate sigma for electron capture in cm^2 - provide energy in keV 
    sigma_EC[0, i] = ion_beam_U28_sigmas.calculate_sigma_electron_capture(Z_H, q_p, 1e3 * EC_energy_range[i], SI_units=False)
    sigma_EC[1, i] = ion_beam_U28_sigmas.calculate_sigma_electron_capture(Z_N, q_p, 1e3 * EC_energy_range[i], SI_units=False)
    sigma_EC[2, i] = ion_beam_U28_sigmas.calculate_sigma_electron_capture(Z_Ar, q_p, 1e3 * EC_energy_range[i], SI_units=False)
    
    # Calculate sigma for electron loss in cm^2
    sigma_EL[0, i] = ion_beam_U28_sigmas.calculate_sigma_electron_loss(Z_H, Z_p, q_p, EL_energy_range[i], Ip, n0, SI_units=False)
    sigma_EL[1, i] = ion_beam_U28_sigmas.calculate_sigma_electron_loss(Z_N, Z_p, q_p, EL_energy_range[i], Ip, n0, SI_units=False)
    sigma_EL[2, i] = ion_beam_U28_sigmas.calculate_sigma_electron_loss(Z_Ar, Z_p, q_p, EL_energy_range[i], Ip, n0, SI_units=False)


######## PLOT THE DATA ###########
SMALL_SIZE = 10.5
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


##### U28+ lifetime on rest gas #####
fig, ax = plt.subplots(1, 1, figsize = (6,5))
ax.plot(energies_kin_inj, tau_values_U28, '*', color='blue', label='Semi-empirical formula')
#ax.errorbar(energies_kin_inj, tau_values_SPS_Pb54, yerr=0.5*tau_values_SPS_Pb54, fmt='*', ms=9, capsize=4, color='mediumblue', label='Pb54+ semi-empirical')
#ax.set_yscale('log')
ax.set_xscale('log')
#ax.set_xlim(1, 5200)
#ax.set_ylim(1, 1000)
ax.grid(visible=True, which='major', color='k', linestyle='-', alpha=0.8)
ax.grid(visible=True, which='minor', color='grey', linestyle='--', alpha=0.55) 
ax.set_xlabel('E [MeV/u]')
ax.set_ylabel(r'Lifetime $\tau$ [s]')
ax.legend()
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
#fig.savefig('Output/SPS_RICODE_benchmarking_lifetimes.png', dpi=250)

##### U28+ EC and EL cross sections #####
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12,5))
ax1.plot(EC_energy_range, sigma_EC[0, :], color='black', linestyle='dashed', marker='o', markersize=12, markerfacecolor='maroon', label=r'EC: H$_{2}$')
ax1.plot(EL_energy_range, sigma_EL[0, :], color='black', linestyle='dashed', marker='*', markersize=12, markerfacecolor='coral', label=r'EL: H$_{2}$')
ax2.plot(EC_energy_range, sigma_EC[1, :], color='black', linestyle='dashed', marker='o', markersize=12, markerfacecolor='forestgreen',label=r'EC: N$_{2}$')
ax2.plot(EL_energy_range, sigma_EL[1, :], color='black', linestyle='dashed', marker='*', markersize=12, markerfacecolor='lime', label=r'EL: N$_{2}$')
ax3.plot(EC_energy_range, sigma_EC[2, :], color='black', linestyle='dashed', marker='o', markersize=12, markerfacecolor='blue', label='EC: Ar')
ax3.plot(EL_energy_range, sigma_EL[2, :], color='black', linestyle='dashed', marker='*', markersize=12, markerfacecolor='cyan', label='EL: Ar')
ax1.set_ylabel(r'$\sigma$ [cm$^{2}$/atom]')
ax1.set_xlabel('E [MeV/u]')
ax2.set_xlabel('E [MeV/u]')
ax3.set_xlabel('E [MeV/u]')
ax1.set_xscale('log')
ax2.set_xscale('log')
ax3.set_xscale('log')
ax1.set_yscale('log')
ax2.set_yscale('log')
ax3.set_yscale('log')
ax1.set_ylim(1e-20, 1e-14)
ax2.set_ylim(1e-20, 1e-14)
ax3.set_ylim(1e-20, 1e-14)
ax1.legend()
ax2.legend()
ax3.legend()
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)

# Generate transparent image to compare with Shevelko cross section plot 
if also_generate_transparent_image:
    fig3, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (9,5))
    ax.tick_params(labelbottom=False, labelleft=False)
    ax1.plot(EC_energy_range, sigma_EC[0, :], color='maroon', linestyle='-', linewidth=2)
    ax1.plot(EL_energy_range, sigma_EL[0, :], color='red', linestyle='dashed', linewidth=2)
    ax2.plot(EC_energy_range, sigma_EC[1, :], color='forestgreen', linestyle='-', linewidth=2)
    ax2.plot(EL_energy_range, sigma_EL[1, :], color='lime', linestyle='dashed', linewidth=2)
    ax3.plot(EC_energy_range, sigma_EC[2, :], color='blue', linestyle='-', linewidth=2)
    ax3.plot(EL_energy_range, sigma_EL[2, :],  color='cyan', linestyle='dashed', linewidth=2)
    ax1.set_xscale('log')
    ax2.set_xscale('log')
    ax3.set_xscale('log')
    ax1.set_yscale('log')
    ax2.set_yscale('log')
    ax3.set_yscale('log')
    ax1.set_xlim(1e-3, 1e4)
    ax2.set_xlim(1e-3, 1e4)
    ax3.set_xlim(1e-3, 1e4)
    ax1.set_ylim(1e-20, 1e-14)
    ax2.set_ylim(1e-20, 1e-14)
    ax3.set_ylim(1e-20, 1e-14)
    fig3.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
    ax1.tick_params(labelbottom=False, labelleft=False)
    ax2.tick_params(labelbottom=False, labelleft=False)
    ax3.tick_params(labelbottom=False, labelleft=False)
    fig3.savefig('Output/U28_EC_EL_cross_lifetimes.png', dpi=250, transparent=True)