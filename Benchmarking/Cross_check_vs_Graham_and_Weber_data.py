#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Benchmarking cross sections from semi-empirical formulae with previous data

G. Weber provided some data from 
Graham et al., PHYSICAL REVIEW A VOLUME 30, NUMBER 2 AUGUST 1984 
and PHYSICAL REVIEW SPECIAL TOPICS - ACCELERATORS AND BEAMS 18, 034403 (2015) for different charge states.
"""
import numpy as np
import sys
import matplotlib.pyplot as plt
import pandas as pd 
sys.path.append("..")

from beam_gas_collisions import beam_gas_collisions

# Initiate beam-gas class object
Pb_EL_checks = beam_gas_collisions()

# Set projectile data common for all Pb ions 
m_e = 0.510998950e6  # electron mass in eV
Z = 82.0
A = 208.0
m_u = 931.49410242e6  # 1 Dalton in eV/c^2 -- atomic mass unit 
atomic_mass_pb208 = 207.9766519 # Pb208 atomic mass from The AME2016 atomic mass evaluation (II).
m_ion = atomic_mass_pb208*m_u  

# Input data as arrays
Z_ref = np.array([82., 82., 82., 82., 82., 82.])  # projectile 
q_ref = np.array([52., 54., 59., 55., 54., 54.])
e_kin_ref = np.array([4.66, 4.66, 4.66, 5.9, 4.66, 4.66])  # in MeV/u
m_ref = m_ion - (Z_ref - q_ref) * m_e  # exact projectile mass in eV
m_ref_in_u = m_ref/m_u   # exact projectile mass in dalton 
E_tot = m_ref + e_kin_ref*1e6*m_ref_in_u# total kinetic energy in eV per particle at injection
gammas = E_tot/m_ref # relativistic gamma factor
betas = np.sqrt(1 - 1/gammas**2) # relativistic beta

Ip_ref = np.array([3.2352, 5.414, 6.0252, 5.5778, 5.414, 5.414])  # ionization potential in keV
n0_ref = np.array([4., 3., 3., 3., 3., 3.])  # principal quantum number 
Z_t_ref = np.array([1., 1., 1., 7., 2., 7.]) 
element_ref = ["H2", "H2", "H2", "N2", "He", "N2"]
atoms_in_molecule = np.array([2, 2, 2, 2, 1, 2])
sigma_measured_ref = np.array([1.75e-19, 1.25e-19, 0.7e-19, 4.3E-19, 2.8E-19, 1.6E-18])

sigmas_EL = np.zeros(len(Z_ref))

# iterate over the different values
for i, Z in enumerate(Z_ref):
    
    projectile_data = np.array([Z_ref[i],
                                q_ref[i],  
                                e_kin_ref[i],  # in MeV/u
                                Ip_ref[i], 
                                n0_ref[i],
                                betas[i]
                                ])
    
    sigma_EL = atoms_in_molecule[i] * Pb_EL_checks.calculate_sigma_electron_loss(Z_t_ref[i], Z_ref[i], q_ref[i], e_kin_ref[i], Ip_ref[i], n0_ref[i], SI_units=False)
    print("Calculated vs measured sigma Pb{}+ on {}: {:.4e} cm^2   vs   {:.4e} cm^2".format(
                                                                                int(q_ref[i]), \
                                                                                element_ref[i], \
                                                                                sigma_EL, \
                                                                                sigma_measured_ref[i]))
    sigmas_EL[i] = sigma_EL


# Convert to dataframe
data = {
    'Z_ref': Z_ref,
    'q_ref': q_ref,
    'e_kin_ref': e_kin_ref,
    'm_ref_in_u': m_ref_in_u,
    'Ip_ref': Ip_ref,
    'n0_ref': n0_ref,
    'Z_t_ref': Z_t_ref,
    'element_ref': element_ref,
    'atoms_in_molecule': atoms_in_molecule,
    'sigma_measured_ref': sigma_measured_ref,
    'sigmas_EL': sigmas_EL
}

df = pd.DataFrame(data)

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

# Plot the cross sections of each projectile on H2 (first column)
fig, ax = plt.subplots(1, 1, figsize = (5,4))

# Create a dictionary to store the color of each unique element
element_colors = {}
custom_colors = ['lawngreen', 'magenta', 'orange']

# Group the data by charge state and iterate over each group
groups = df.groupby('q_ref')
count = 1.0
x = np.arange(1, 5)
q_ref_labels = [52., 54., 55., 59.]

for label, group in groups:
    # Extract the relevant data for the group
    elements = group['element_ref'].unique()
    calculated_values = group['sigmas_EL'].tolist()
    measured_values = group['sigma_measured_ref'].tolist()
    error_bars = [1.0e-19] * len(group)  # Assuming a constant error bar value

    # Determine the colors for the bars
    colors = [element_colors.get(element, custom_colors[i % len(custom_colors)]) for i, element in enumerate(elements)]

    # Create the bars for each element
    for i, element in enumerate(elements):
        shift_factor = 1 if label==54.0 else 0  # to adjust the bars 
        bar_position = count + (i - shift_factor ) * 0.2  # Adjust the position for multiple bars per label
        ax.bar(bar_position, calculated_values[i], width=0.2, color=colors[i], label=element)
        ax.errorbar(bar_position, measured_values[i], yerr=error_bars[i], fmt='s', color='black')

    # Store the color for each unique element
    for element, color in zip(elements, colors):
        element_colors[element] = color
    count += 1

# Set the x-axis labels as charge states
ax.set_xticks(x)
ax.set_xticklabels(q_ref_labels)

# Set the y-axis to a logarithmic scale
ax.set_yscale('log')

# Create a list of unique legend handles and labels
handles, labels = ax.get_legend_handles_labels()
unique_labels = list(set(labels))

# Create the legend with unique labels and handles
legend_elements = [plt.Line2D([0], [0], marker='s', color='black', linestyle='', label='Measured')]
legend_elements += [plt.Rectangle((0, 0), 1, 1, color=element_colors[label]) for label in unique_labels]
ax.legend(handles=legend_elements, labels=['Measured'] + unique_labels)

# Set the axis labels and title
ax.set_xlabel('Pb charge state')
ax.set_ylabel(r'$\sigma_{EL}$')
#ax.set_title('Comparison of Calculated and Measured Sigmas')

fig.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
fig.savefig('Output/Comparing_Pb_EL_cross_sections.png', dpi=250)