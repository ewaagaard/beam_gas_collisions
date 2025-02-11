"""
Scanning energy ranges for different projectiles on H2, N2 and Ar targets
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from beam_gas_collisions import IonLifetimes, DataObject
import os

# Initialize data object
data = DataObject()

# Define target data
Z_H = 1.0
Z_N = 7.0
Z_Ar = 18.0

# Generate unconfigured ion beam lifetime object
ion_beams = IonLifetimes(machine='SPS')

# Projectile_data U28+
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 

# Projectile data Xe39
Z_p_Xe = 54.
q_p_Xe = 39.
Ip_Xe = data.projectile_data['I_p']['Xe39']
n0_Xe = data.projectile_data['n_0']['Xe39']

# Projectile data Pb54
projectile = 'Pb54'
Z_p_Pb = 82.
q_p_Pb = 54.
Ip_Pb = data.projectile_data['I_p'][projectile]
n0_Pb = data.projectile_data['n_0'][projectile]

###### Calculate EL and EC cross sections ######
energy_range =  np.logspace(np.log10(1e-3), np.log10(1e4), 30) # in MeV/u 

# Empty arrays where rows contain H2, N2 and Ar:
sigma_EC_U28 = np.zeros([3, len(energy_range)])
sigma_EL_U28 = np.zeros([3, len(energy_range)])
sigma_EC_Pb54 = np.zeros([3, len(energy_range)])
sigma_EL_Pb54 = np.zeros([3, len(energy_range)])
sigma_EC_Xe39 = np.zeros([3, len(energy_range)])
sigma_EL_Xe39 = np.zeros([3, len(energy_range)])

# Iterate over the different energies
for i, E_kin_EC in enumerate(energy_range):


    ### U28 ###
    # Calculate sigma for electron capture in cm^2 - provide energy in keV 
    sigma_EC_U28[0, i] = ion_beams.calculate_sigma_electron_capture(Z_H, q_p, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_U28[1, i] = ion_beams.calculate_sigma_electron_capture(Z_N, q_p, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_U28[2, i] = ion_beams.calculate_sigma_electron_capture(Z_Ar, q_p, 1e3 * energy_range[i], SI_units=False)
    
    # Calculate sigma for electron loss in cm^2
    sigma_EL_U28[0, i] = ion_beams.calculate_sigma_electron_loss(Z_H, Z_p, q_p, energy_range[i], Ip, n0, SI_units=False)
    sigma_EL_U28[1, i] = ion_beams.calculate_sigma_electron_loss(Z_N, Z_p, q_p, energy_range[i], Ip, n0, SI_units=False)
    sigma_EL_U28[2, i] = ion_beams.calculate_sigma_electron_loss(Z_Ar, Z_p, q_p, energy_range[i], Ip, n0, SI_units=False)

    ### Pb54 ###
    sigma_EC_Pb54[0, i] = ion_beams.calculate_sigma_electron_capture(Z_H, q_p_Pb, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_Pb54[1, i] = ion_beams.calculate_sigma_electron_capture(Z_N, q_p_Pb, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_Pb54[2, i] = ion_beams.calculate_sigma_electron_capture(Z_Ar, q_p_Pb, 1e3 * energy_range[i], SI_units=False)
    
    # Calculate sigma for electron loss in cm^2
    sigma_EL_Pb54[0, i] = ion_beams.calculate_sigma_electron_loss(Z_H, Z_p_Pb, q_p_Pb, energy_range[i], Ip_Pb, n0_Pb, SI_units=False)
    sigma_EL_Pb54[1, i] = ion_beams.calculate_sigma_electron_loss(Z_N, Z_p_Pb, q_p_Pb, energy_range[i], Ip_Pb, n0_Pb, SI_units=False)
    sigma_EL_Pb54[2, i] = ion_beams.calculate_sigma_electron_loss(Z_Ar, Z_p_Pb, q_p_Pb, energy_range[i], Ip_Pb, n0_Pb, SI_units=False)

    ### Xe39 ###
    sigma_EC_Xe39[0, i] = ion_beams.calculate_sigma_electron_capture(Z_H, q_p_Xe, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_Xe39[1, i] = ion_beams.calculate_sigma_electron_capture(Z_N, q_p_Xe, 1e3 * energy_range[i], SI_units=False)
    sigma_EC_Xe39[2, i] = ion_beams.calculate_sigma_electron_capture(Z_Ar, q_p_Xe, 1e3 * energy_range[i], SI_units=False)
    
    # Calculate sigma for electron loss in cm^2
    sigma_EL_Xe39[0, i] = ion_beams.calculate_sigma_electron_loss(Z_H, Z_p_Xe, q_p_Xe, energy_range[i], Ip_Xe, n0_Xe, SI_units=False)
    sigma_EL_Xe39[1, i] = ion_beams.calculate_sigma_electron_loss(Z_N, Z_p_Xe, q_p_Xe, energy_range[i], Ip_Xe, n0_Xe, SI_units=False)
    sigma_EL_Xe39[2, i] = ion_beams.calculate_sigma_electron_loss(Z_Ar, Z_p_Xe, q_p_Xe, energy_range[i], Ip_Xe, n0_Xe, SI_units=False)


######## PLOT THE DATA ###########
SMALL_SIZE = 12
MEDIUM_SIZE = 19
BIGGER_SIZE = 26
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


##### Plot the data #####
os.makedirs('output_and_plots', exist_ok=True)

##### U28+ EC and EL cross sections #####
fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize = (12,5))

# H2
ax1.plot(energy_range, sigma_EL_U28[0, :], color='black', linestyle='--', linewidth=2, label='EL: U28') 
ax1.plot(energy_range, sigma_EC_U28[0, :], color='black', linestyle='-', linewidth=2, label='EC: U28')
ax1.plot(energy_range, sigma_EL_Pb54[0, :], color='purple', linestyle='--', linewidth=2, label='EL: Pb54') 
ax1.plot(energy_range, sigma_EC_Pb54[0, :], color='purple', linestyle='-', linewidth=2, label='EC: Pb54')
ax1.plot(energy_range, sigma_EL_Xe39[0, :], color='green', linestyle='--', linewidth=2, label='EL: Xe39') 
ax1.plot(energy_range, sigma_EC_Xe39[0, :], color='green', linestyle='-', linewidth=2, label='EC: Xe39')
ax1.text(0.08, 0.05, r'H$_{2}$', fontsize=23, transform=ax1.transAxes)

# N2
ax2.plot(energy_range, sigma_EL_U28[1, :], color='black', linestyle='--', linewidth=2, label='EL: U28')
ax2.plot(energy_range, sigma_EC_U28[1, :], color='black', linestyle='-', linewidth=2, label='EC: U28')
ax2.plot(energy_range, sigma_EL_Pb54[1, :], color='purple', linestyle='--', linewidth=2, label='EL: Pb54') 
ax2.plot(energy_range, sigma_EC_Pb54[1, :], color='purple', linestyle='-', linewidth=2, label='EC: Pb54')
ax2.plot(energy_range, sigma_EL_Xe39[1, :], color='green', linestyle='--', linewidth=2, label='EL: Xe39') 
ax2.plot(energy_range, sigma_EC_Xe39[1, :], color='green', linestyle='-', linewidth=2, label='EC: Xe39')
ax2.text(0.08, 0.05, r'N$_{2}$', fontsize=23, transform=ax2.transAxes)

# Ar
ax3.plot(energy_range, sigma_EL_U28[2, :], color='black', linestyle='--', linewidth=2, label='EL: U28')
ax3.plot(energy_range, sigma_EC_U28[2, :], color='black', linestyle='-', linewidth=2, label='EC: U28')
ax3.plot(energy_range, sigma_EL_Pb54[2, :], color='purple', linestyle='--', linewidth=2, label='EL: Pb54') 
ax3.plot(energy_range, sigma_EC_Pb54[2, :], color='purple', linestyle='-', linewidth=2, label='EC: Pb54')
ax3.plot(energy_range, sigma_EL_Xe39[2, :], color='green', linestyle='--', linewidth=2, label='EL: Xe39') 
ax3.plot(energy_range, sigma_EC_Xe39[2, :], color='green', linestyle='-', linewidth=2, label='EC: Xe39')
ax3.text(0.08, 0.05, r'Ar', fontsize=23, transform=ax3.transAxes)

ax1.set_ylabel(r'$\sigma$ [cm$^{2}$/atom]')
for ax in (ax1, ax2, ax3):
    ax.set_xlabel('E [MeV/u]')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_ylim(1e-21, 1e-14)
    ax.set_xlim(1e-2, 1e4)
    ax.grid()

ax3.legend(fontsize=10)
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('output_and_plots/002_Xe39_Pb54_U28_EC_EL_cross_sections_over_energies.png', dpi=250)
plt.show()