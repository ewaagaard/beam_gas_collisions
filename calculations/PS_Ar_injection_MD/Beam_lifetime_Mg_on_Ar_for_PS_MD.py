#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Estimate on Ar gas pressure effect on Mg ion beam lifetime in the PS 
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 

from beam_gas_collisions import BeamGasCollisions, Data

# Load data 
projectile = 'Mg7'
data = Data()

# Current lifetime and pressure
tau0 = 1.5
p0 = 2.4*5e-10

# Argon target data
Z_Ar = 18.0
Z_H = 1.0
p_range = np.logspace(np.log10(1e-10), np.log10(1e-7), 15) # in mbar

# Calculate lifetimes and cross sections for PS
projectile_data_PS = np.array([data.projectile_data['Z'][projectile],
                                data.projectile_data['PS_q'][projectile],
                                data.projectile_data['PS_Kinj'][projectile],
                                data.projectile_data['I_p'][projectile], 
                                data.projectile_data['n_0'][projectile], 
                                data.projectile_data['PS_beta'][projectile]])

# Update ionization energy for Mg7, according to NIST values
#projectile_data_PS[-3] = 0.265924

# Initiate PS beam-gas interactions
PS_rest_gas =  BeamGasCollisions()
PS_rest_gas.set_projectile_data(projectile_data_PS)  

# Initiate empty lifetime array and iterate over the different pressures 
taus_Ar = np.zeros(len(p_range))
taus_H2 = np.zeros(len(p_range))
sigma_EL_Ar = np.zeros(len(p_range))
sigma_EC_Ar = np.zeros(len(p_range))
sigma_EL_H2 = np.zeros(len(p_range))
sigma_EC_H2 = np.zeros(len(p_range))

for i, p in enumerate(p_range):
    taus_Ar[i], sigma_EL_Ar[i], sigma_EC_Ar[i] = PS_rest_gas.calculate_lifetime_on_single_gas(p, Z_Ar)
    taus_H2[i], sigma_EL_H2[i], sigma_EC_H2[i] = PS_rest_gas.calculate_lifetime_on_single_gas(p, Z_H, atomicity=2)
    
# Print value of cross section on H2 vs Ar - cross sections are all identical over pressure
print('\nEC cross sections: Ar = {:.3e} m-2, H2 = {:.3e} m-2'.format(sigma_EC_Ar[0], sigma_EC_H2[0]))
print('EL cross sections: Ar = {:.3e} m-2, H2 = {:.3e} m-2'.format(sigma_EL_Ar[0], sigma_EL_H2[0]))

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

# Plot expected lifetimes as a function of pressure
fig, ax = plt.subplots(1, 1)
ax.plot(p_range, taus_Ar, ls='-', linewidth=2, color='dimgrey', label='Predicted lifetime\nMg7+ on Ar')
ax.plot(p_range, taus_H2, ls='-', linewidth=2, color='coral', label='Predicted lifetime\nMg7+ on H2')
ax.set_yscale('log')
ax.set_xscale('log')
#ax.axhline(y=tau0, color='r', linestyle='-', label='Current Mg54+ lifetime')
ax.axvline(x=p0, color='purple', linestyle='-.', label='Current total BGI pressure level')
ax.set_xlabel('Pressure [mbar]')
ax.set_ylabel(r'Lifetime $\tau$ [s]')
ax.legend()
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('Output/Beamlifetime_Mg_on_Ar_in_PS.png', dpi=250)

# Plot cross sections as a function of pressure
fig2, ax2 = plt.subplots(1, 1)
ax2.plot(p_range, sigma_EL_Ar, ls='-', linewidth=2, color='forestgreen', label='$\sigma_{EL}$ Mg7+ on Ar')
ax2.plot(p_range, sigma_EC_Ar, ls='-', linewidth=2, color='lime', label='$\sigma_{EC}$ Mg7+ on Ar')
ax2.plot(p_range, sigma_EL_H2, ls='-', linewidth=2, color='slateblue', label='$\sigma_{EL}$ Mg7+ on H2')
ax2.plot(p_range, sigma_EC_H2, ls='-', linewidth=2, color='darkviolet', label='$\sigma_{EC}$ Mg7+ on H2')
ax2.set_yscale('log')
ax2.set_xscale('log')
ax2.set_xlabel('Pressure [mbar]')
ax2.set_ylabel('Cross section $\sigma$ [m$^{2}$]')
ax2.legend()
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('Output/Sigmas_Mg_on_Ar_in_PS.png', dpi=250)
plt.show()