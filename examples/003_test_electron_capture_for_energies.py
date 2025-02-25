"""
Example case to calculate Electron Capture cross sections from the Schlachter formula
"""
from beam_gas_collisions import IonLifetimes
import matplotlib.pyplot as plt
import numpy as np

# Instantiate lifetime object
ion_beam = IonLifetimes(projectile='Pb54', machine='PS')
q_p = 54.
Z_p = 82.

# Initiate arrays
EC_energy_range = np.logspace(np.log10(1e2), np.log10(1e6), 50) # in keV/u 
sigmas_EL = np.zeros([2, len(EC_energy_range)])
sigmas_EC = np.zeros([2, len(EC_energy_range)])
sigmas_EC_reduced = np.zeros([2, len(EC_energy_range)])
E_tilde = np.zeros([2, len(EC_energy_range)])

Z_He = 2.0
Z_Ar = 18.0

# Iterate over the different energies
for i, e_kin_keV in enumerate(EC_energy_range):
    
    # Calculate EC cross section 
    sigmas_EC[0, i] = ion_beam.calculate_sigma_electron_capture(Z_He, q_p, e_kin_keV, SI_units=True)
    sigmas_EC[1, i] = ion_beam.calculate_sigma_electron_capture(Z_Ar, q_p, e_kin_keV, SI_units=True)
    
    # Calculate EL cross section
    sigmas_EL[0, i] = ion_beam.calculate_sigma_electron_loss(Z_He, Z_p, q_p, 1e-3*e_kin_keV, ion_beam.I_p, ion_beam.n_0, SI_units=True)
    sigmas_EL[1, i] = ion_beam.calculate_sigma_electron_loss(Z_Ar, Z_p, q_p, 1e-3*e_kin_keV, ion_beam.I_p, ion_beam.n_0, SI_units=True)
    
    # Calculate reduced cross sections
    sigmas_EC_reduced[0, i] = sigmas_EC[0, i] * 1/(q_p**0.5/(Z_He**1.8))
    sigmas_EC_reduced[1, i] = sigmas_EC[1, i] * 1/(q_p**0.5/(Z_Ar**1.8))
    
    # Calculate reduced energy
    E_tilde[0, i] =  (e_kin_keV)/(Z_He**(1.25)*q_p**0.7)
    E_tilde[1, i] =  (e_kin_keV)/(Z_Ar**(1.25)*q_p**0.7)
    
# Compute reduced cross sections and energy for experiments
e_kin_PS_inj_keV = 72.2e3
E_tilde_Pb54_on_He = e_kin_PS_inj_keV/(Z_He**1.25*q_p**0.7)
E_tilde_Pb54_on_Ar = e_kin_PS_inj_keV/(Z_Ar**1.25*q_p**0.7)
sigmas_EC_reduced_Pb54_on_He = ion_beam.calculate_sigma_electron_capture(Z_He, q_p, e_kin_PS_inj_keV, SI_units=True)/(q_p**0.5/(Z_He**1.8))
sigmas_EC_reduced_Pb54_on_Ar = ion_beam.calculate_sigma_electron_capture(Z_Ar, q_p, e_kin_PS_inj_keV, SI_units=True)/(q_p**0.5/(Z_Ar**1.8))

    
# Plot reduced cross sections over reduced energy
fig, ax = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)
ax.plot(E_tilde[0, :], sigmas_EC_reduced[0, :], color='r', label='He')
ax.plot(E_tilde[1, :], sigmas_EC_reduced[1, :], color='b', label='Ar')
ax.plot(E_tilde_Pb54_on_He, sigmas_EC_reduced_Pb54_on_He, ls='None', marker='o', ms=11, markerfacecolor='red', color='k', label='Pb54+ on He in PS')
ax.plot(E_tilde_Pb54_on_Ar, sigmas_EC_reduced_Pb54_on_Ar, ls='None', marker='o', ms=11, markerfacecolor='teal', color='k', label='Pb54+ on Ar in PS')
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_ylabel('Reduced cross section $\\tilde{\\sigma}$')
ax.set_xlabel('Reduced energy $\\tilde{E}$')
ax.legend()
ax.grid(alpha=0.55)

# Plot reduced cross section over reduced energy
fig2, ax2 = plt.subplots(1, 1, figsize=(8,6), constrained_layout=True)
ax2.plot(EC_energy_range*1e-3, sigmas_EC[0, :], color='r', label='EC: He')
ax2.plot(EC_energy_range*1e-3, sigmas_EC[1, :], color='b', label='EC: Ar')
ax2.plot(EC_energy_range*1e-3, sigmas_EL[0, :], color='r', ls='--', label='EL: He')
ax2.plot(EC_energy_range*1e-3, sigmas_EL[1, :], color='b', ls='--', label='EL: Ar')
ax2.axvline(x=72.2, label='PS injection energy for Pb$^{54+}$', lw=2.2, color='limegreen')
ax2.set_xscale('log')
ax2.set_yscale('log')
ax2.set_ylabel('Pb$^{54+}$ cross section $\\sigma$')
ax2.set_xlabel('Projectile energy E [MeV/u]')
ax2.legend()
ax2.grid(alpha=0.55)

plt.show()