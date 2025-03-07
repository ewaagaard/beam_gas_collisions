"""
Script to compare semi-empirical formula from G. Weber with experimental data
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

from beam_gas_collisions import IonLifetimes

# Load reference data
df = pd.read_csv('benchmarking_data/electron_loss_experiment_vs_semiempirical_formattet.csv', header=0, sep=';')
df = df.dropna(how='all')

### Define projectile data - Z_p : Z of projectile, q : charge of projectile, e_kin : collision energy in MeV/u.
# I_p : first ionization potential of projectile in keV, n_0: principal quantum number

projectile_data_Fe4 = np.array([26.,
                                56.,
                                4.,  
                                0.36, # in MeV/u --> 0.36 was written in Geunter's file, but 20 MeV in the reference --> where did he get the data from?
                                1e-3 * 75.0,  # from NIST table
                                3,  # from NIST table
                                55.934935  # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])
Zt_for_Fe4 = np.logspace(np.log10(2.0), np.log10(54), 5)   #np.array([2., 7., 10., 11., 13., 15.])
ion_beam_Fe4 = IonLifetimes()

projectile_data_Xe18 = np.array([54.,
                                 132.,
                                18.,  
                                6.0,  # in MeV/u
                                0.549,  # from NIST table
                                4,  # from NIST table
                                131.904155087  # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])
Zt_for_Xe18 = np.logspace(np.log10(1.0), np.log10(50.), 5)
ion_beam_Xe18 = IonLifetimes()


projectile_data_Au52 = np.array([79.,
                                 197,
                                52.,  
                                100.0,  # in MeV/u
                                5.013,  # from NIST table
                                3,  # from NIST table
                                196.9665701  # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])
Zt_for_Au52 = np.logspace(np.log10(1.0), np.log10(8.), 5)
ion_beam_Au52 = IonLifetimes()

projectile_data_Xe45 = np.array([54.,
                                 132.,
                                45.,  
                                140.0,  # in MeV/u
                                7.889,  # from NIST table
                                2,  # from NIST table
                                131.904155087  # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])
Zt_for_Xe45 = np.logspace(np.log10(6.0), np.log10(80.), 5)
ion_beam_Xe45 = IonLifetimes()

projectile_data_U83 = np.array([92.,
                                238.,
                                83.,  
                                955.,  # in MeV/u
                                25.680,  # from NIST table
                                2, # from NIST table
                                238.050787 # most abundant isotope atomic_mass_in_u, from AME2016 Table 
                                ])
Zt_for_U83 = np.logspace(np.log10(4.0), np.log10(80.), 5)
ion_beam_U83 = IonLifetimes()

U83_experimental = np.array([])

# To contain the electron loss data for all 5 ions
sigma_EL = np.zeros([5, len(Zt_for_Fe4)])

# Iterate over the different target Z
for i, Z_t in enumerate(Zt_for_Fe4):
    ion_beam_Fe4.set_projectile_data_manually(projectile_data_Fe4)
    sigma_EL[0, i] = ion_beam_Fe4.calculate_sigma_electron_loss(Zt_for_Fe4[i], 
                                                                ion_beam_Fe4.Z_p, 
                                                                ion_beam_Fe4.q, 
                                                                ion_beam_Fe4.e_kin, 
                                                                ion_beam_Fe4.I_p, 
                                                                ion_beam_Fe4.n_0, 
                                                                SI_units=False)
    
    ion_beam_Xe18.set_projectile_data_manually(projectile_data_Xe18)
    sigma_EL[1, i] = ion_beam_Xe18.calculate_sigma_electron_loss(Zt_for_Xe18[i], 
                                                                ion_beam_Xe18.Z_p, 
                                                                ion_beam_Xe18.q, 
                                                                ion_beam_Xe18.e_kin, 
                                                                ion_beam_Xe18.I_p, 
                                                                ion_beam_Xe18.n_0, 
                                                                SI_units=False)
    
    ion_beam_Au52.set_projectile_data_manually(projectile_data_Au52)
    sigma_EL[2, i] = ion_beam_Au52.calculate_sigma_electron_loss(Zt_for_Au52[i], 
                                                                ion_beam_Au52.Z_p, 
                                                                ion_beam_Au52.q, 
                                                                ion_beam_Au52.e_kin, 
                                                                ion_beam_Au52.I_p, 
                                                                ion_beam_Au52.n_0, 
                                                                SI_units=False)
    
    ion_beam_Xe45.set_projectile_data_manually(projectile_data_Xe45)
    sigma_EL[3, i] = ion_beam_Xe45.calculate_sigma_electron_loss(Zt_for_Xe45[i], 
                                                                ion_beam_Xe45.Z_p, 
                                                                ion_beam_Xe45.q, 
                                                                ion_beam_Xe45.e_kin, 
                                                                ion_beam_Xe45.I_p, 
                                                                ion_beam_Xe45.n_0, 
                                                                SI_units=False)
    
    ion_beam_U83.set_projectile_data_manually(projectile_data_U83)
    sigma_EL[4, i] = ion_beam_U83.calculate_sigma_electron_loss(Zt_for_U83[i], 
                                                                ion_beam_U83.Z_p, 
                                                                ion_beam_U83.q, 
                                                                ion_beam_U83.e_kin, 
                                                                ion_beam_U83.I_p, 
                                                                ion_beam_U83.n_0, 
                                                                SI_units=False)
    
######## PLOT THE DATA ###########
os.makedirs('output_and_plots', exist_ok=True)

SMALL_SIZE = 11.5
MEDIUM_SIZE = 15
BIGGER_SIZE = 18
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

##### EL cross sections #####
fig, ax1 = plt.subplots(1, 1, figsize = (6,5))
ax1.plot(df['Fe4_Zt'].dropna().values, df['Fe4_exp'].dropna().values,'*', color='black', markersize=12, markerfacecolor='coral', label=r'Fe$^{4+}$')
ax1.plot(df['Xe18_Zt'].dropna().values, df['Xe18_exp'].dropna().values,'o', color='black', markersize=12, markerfacecolor='cyan', label=r'Xe$^{18+}$')
ax1.plot(df['Au52_Zt'].dropna().values, df['Au52_exp'].dropna().values, 'v', color='black', markersize=12, markerfacecolor='green', label=r'Au$^{52+}$')
ax1.plot(df['Xe45_Zt'].dropna().values, df['Xe45_exp'].dropna().values, '^', color='black', markersize=12, markerfacecolor='red', label=r'Xe$^{45+}$')
ax1.plot(df['U83_Zt'].dropna().values, df['U83_exp'].dropna().values, 's', color='black', markersize=12, markerfacecolor='magenta', label=r'U$^{83+}$')

ax1.plot(Zt_for_Fe4, sigma_EL[0, :], color='black', linestyle='dashed', label='Semi-empirical\nmodel')
ax1.plot(Zt_for_Xe18, sigma_EL[1, :], color='black', linestyle='dashed')
ax1.plot(Zt_for_Au52, sigma_EL[2, :], color='black', linestyle='dashed')
ax1.plot(Zt_for_Xe45, sigma_EL[3, :], color='black', linestyle='dashed')
ax1.plot(Zt_for_U83, sigma_EL[4, :], color='black', linestyle='dashed')

ax1.set_ylabel(r'$\sigma_{EL}$ [cm$^{2}$/atom]')
ax1.set_xlabel(r'Target atomic number Z$_{T}$')
ax1.set_xscale('log')
ax1.set_yscale('log')
ax1.set_ylim(1e-21, 5e-15)
ax1.set_xlim(7e-1, 1e2)
legend = ax1.legend(loc='upper center', bbox_to_anchor=(0.5, 1.1),
          ncol=3, fancybox=True, fontsize=11.5)
legend.get_frame().set_alpha(None)
fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('output_and_plots/Semi_empirical_vs_exp.png', dpi=250)
plt.show()