"""
Benchmarking tests to semi-empirical cross sections of U28+ from experiments in Fig. 13 and Fig. 14 in Shevelko (2011)
https://www.sciencedirect.com/science/article/pii/S0168583X11003272
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd 
from beam_gas_collisions import IonLifetimes
import os

# Rest gas composition of ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'He', 'O2', 'Ar']
gas_fractions = np.array([0.758, 0.049, 0.026, 0.119, 0.002, 0.034, 0.004, 0.008])
p = 1e-10 #mbar
ion_beam_U28 = IonLifetimes(machine=None, p=p, molecular_fraction_array=gas_fractions)

# Projectile_data
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 

# Initiate arrays
energies_kin_inj = np.logspace(np.log10(5.0), np.log10(1e4), 15) # in MeV/u 
tau_values_U28 = np.zeros(len(energies_kin_inj))

##### Experimental data - energy arrays in MeV/u ######
# --> from https://link.aps.org/doi/10.1103/PhysRevSTAB.18.034403

# Electron loss
E_exp_H2 = np.array([1.4, 3.5, 6.5, 10., 20., 30., 40., 50.])
sigma_EL_H2_exp = 1e-18 * np.array([2.25, 1.62, 1.14, 0.74, 0.51, 0.31, 0.28, 0.25]) # convert 1e6 barn to cm^2
H2_error_exp = np.array([0., 0.35, 0.26, 0.18, 0.13, 0.13, 0.07, 0.06])

E_exp_N2 = np.array([1.4, 3.5, 6.5, 20., 30., 50.])
sigma_EL_N2_exp = 1e-18 * np.array([32.6, 22.52, 14.69, 8.8, 6.21, 3.48]) # convert 1e6 barn to cm^2
N2_error_exp = np.array([0.0, 1.07, 0.82, 2.20, 1.56, 0.87])

E_exp_Ar = np.array([1.4, 3.5, 6.5, 30., 50.])
sigma_EL_Ar_exp = 1e-18 * np.array([47.8, 45.38, 33.15, 15.61, 11.93]) # convert 1e6 barn to cm^2
Ar_error_exp = np.array([6.7, 1.62, 1.25, 4.09, 2.40])

# Electron capture --> see points directly in plotting 
Olsen_E_H2 = np.array([3.5, ])
Olsen_EC_H2 = np.array([0.048, ])
Olsen_error_H2 = np.array([0.016])


####


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
        
    ion_beam_U28.set_projectile_data_manually(projectile_data_U28)  
    tau_values_U28[i] = ion_beam_U28.calculate_total_lifetime_full_gas()
    
###### STEP 2: calculate U28 EL and EC cross sections ######
    
EC_energy_range =  np.logspace(np.log10(1e-3), np.log10(1e4), 30) # in MeV/u 
EL_energy_range =  np.logspace(np.log10(1e-3), np.log10(1e4), 30) # in MeV/u 

# Empty arrays where rows contain U28+ on H2, N2 and Ar:
sigma_EC = np.zeros([3, len(EC_energy_range)])
sigma_EL = np.zeros([3, len(EL_energy_range)])
Z_H = 1.0
Z_N = 7.0
Z_Ar = 18.0

# Initiate beam-gas collision object
ion_beam_U28_sigmas = IonLifetimes()

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
ax1.plot(EL_energy_range, sigma_EL[0, :], color='black', linestyle='--', linewidth=2, label=r'EL: semi-empirical') # marker='*', markersize=12, markerfacecolor='coral',
ax1.plot(EC_energy_range, sigma_EC[0, :], color='black', linestyle='-', linewidth=2, label=r'EC: Schlachter formula')
ax1.plot(E_exp_H2, sigma_EL_H2_exp, 'o',  color='black', markersize=10, markerfacecolor='maroon', label='EL experimental') # (Franzke, Olsen, Weber)'
#ax1.plot(1.4, 4.0e-18, '*', markersize=12, markerfacecolor='coral', label='EC: Franzke')  # Franzke 1981 paper
ax1.errorbar([1.4, 3.5], [4.0e-18, 0.048e-18], yerr=[0.0, 0.016e-18], fmt='v', color='black', markersize=10, markerfacecolor='brown', label='EC experimental') # Franzke 1981 paper and Olsen 2004 paper
ax1.text(0.08, 0.05, r'H$_{2}$', fontsize=23, transform=ax1.transAxes)

# N2
ax2.plot(EL_energy_range, sigma_EL[1, :], color='black', linestyle='--', linewidth=2, label=r'EL: semi-empirical') #  marker='*', markersize=12, markerfacecolor='lime',
ax2.plot(EC_energy_range, sigma_EC[1, :], color='black', linestyle='-', linewidth=2, label=r'EC: Schlachter formula')
ax2.plot(E_exp_N2, sigma_EL_N2_exp, 'o', color='black', markersize=10, markerfacecolor='forestgreen', label='EL Experimental') # (Franzke, Olsen, Weber, Perumal)
#ax2.plot(1.4, 1.0e-16, '*', markersize=12, markerfacecolor='lime', label='EC: Franzke') # Franzke 1981 paper
ax2.errorbar([1.4, 3.5, 6.5], [1.0e-16, 5.03e-18, 0.33e-18], fmt='v', color='black', yerr=[0.0, 0.48e-18, 0.08e-18], markersize=10, markerfacecolor='darkgreen', label='EC experimental') # Franzke 1981 paper and Olsen 2004 paper
ax2.text(0.08, 0.05, r'N$_{2}$', fontsize=23, transform=ax2.transAxes)

# Ar
ax3.plot(EL_energy_range, sigma_EL[2, :], color='black', linestyle='--', linewidth=2, label=r'EL: semi-empirical') # marker='*', markersize=12, markerfacecolor='cyan',
ax3.plot(EC_energy_range, sigma_EC[2, :], color='black', linestyle='-', linewidth=2, label=r'EC: Schlachter formula')
ax3.plot(E_exp_Ar, sigma_EL_Ar_exp, 'o', markersize=10, color='black', markerfacecolor='royalblue', label='EL experimental') #(Erb, Olsen, Weber)
ax3.errorbar([3.5, 6.5], [11.20e-18, 0.39e-18], yerr=[0.89e-18, 0.09e-18], color='black',fmt='v', markersize=10, markerfacecolor='cyan', label='EC experimental') # Olsen 2004 paper
ax3.text(0.08, 0.05, r'Ar', fontsize=23, transform=ax3.transAxes)

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
ax1.set_xlim(1e-2, 1e4)
ax2.set_xlim(1e-2, 1e4)
ax3.set_xlim(1e-2, 1e4)
ax1.grid()
ax2.grid()
ax3.grid()
#ax1.legend()
#ax2.legend()
ax3.legend(fontsize=10)
fig2.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig2.savefig('output_and_plots/U28_EC_EL_cross_section_comparison_plot.png', dpi=250)
plt.show()