#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate lifetimes for U28+ as compared to this study by DuBois (2009): https://linkinghub.elsevier.com/retrieve/pii/S0168583X07006490
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

#### Rest gas composition of['H2', 'H2O', 'CO', 'CH4', 'CO2', 'He', 'O2', 'Ar'] ####
gas_fractions = np.array([0.83, 0.1, 0.0, 0.05, 0.01, 0.0, 0.0, 0.01])
p = 2e-11
ion_beam_U28 = beam_gas_collisions(p, gas_fractions)

#### Projectile_data ####
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 

#### Define vector for injection energy, and SIS18 experimental values ####
energy_array = np.linspace(10, 200, 50)
energies = np.array([10., 20., 40., 50., 80., 120., 180.])
tau_exp = np.array([3.14, 4.27, 5.31, 5.5, 6.27, 7.17, 8.16])  
tau_exp_error = np.array([0.47, 0.64, 0.8, 0.83, 0.94, 1.08, 1.22])

tau_values_U28 = np.zeros(len(energy_array))

# Iterate over the different energies
for i, E_kin in enumerate(energy_array):
    
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

fig, ax = plt.subplots(1, 1, figsize = (6,5))
ax.plot(energy_array , tau_values_U28,  color='blue', label='Semi-empirical formula')
#ax.errorbar(energies, tau_exp, yerr=tau_exp_error, fmt='o', color='black', markersize=10, markerfacecolor='brown', label='Experimental data')
ax.set_xlabel('E [MeV/u]')
ax.set_ylabel(r'Beam lifetime $\tau$ [s]')
ax.legend()
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('Output/Lifetime_U28_semiempirical_vs_SIS18.png', dpi=250)