#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import scipy.constants as constants
from pathlib import Path

data_folder = Path(__file__).resolve().parent.joinpath('../Data').absolute()

class BeamGasCollisions:
    """
    Class to calculate the electron loss and electron capture cross section of rest gas collisions from 
    semi-empirical Dubois, Shevelko and Schlachter formulae 
    """
    def __init__(self, p=None, molecular_fraction_array=None, projectile_data=None, provided_beta=True, T=298):
        """
        Parameters
        ----------
        beta : relativistic beta
        p : vacuum pressure in millibar 
        molecular_fraction_array: contains
            x_H2 : fraction of H2
            x_H2O : fraction of H2O
            x_CO : fraction of CO
            x_CH4 : fraction of CH4
            x_CO2 : fraction of CO2
            x_He : fraction of He
            x_O2 : fraction of O2
            x_Ar : fraction of Ar
        T : Temperature in kelvin. The default is 298.
        """
        
        # Fitting parameters obtained from the semi-empirical study by Weber (2016)
        self.par = [10.88, 0.95, 2.5, 1.1137, -0.1805, 2.64886, 1.35832, 0.80696, 1.00514, 6.13667]
        
        # Beam and accelerator setting
        self.K = constants.Boltzmann
        self.T = T # temperature in Kelvin 
        self.c_light = constants.c
        
        ### Set attributes if provided ###
        if p is not None:
            self.p = p*1e2 # convert mbar to Pascal  
        if molecular_fraction_array is not None:
            self.set_molecular_densities(molecular_fraction_array)
            self.molecular_densities_are_set = True
        else:
            self.molecular_densities_are_set = False
        if projectile_data is not None:
            self.set_projectile_data(projectile_data, provided_beta)
        else:
            self.exists_projetile_data = False
        self.lifetimes_are_calculated = False # state to indicate if lifetimes are calculated or not
    
    
    def set_projectile_data(self, projectile_data, provided_beta=True):
        """
        Sets the projectile data:
            Z_p : Z of projectile
            q : charge of projectile.
            e_kin : collision energy in MeV/u.
            I_p : first ionization potential of projectile in keV
            n_0: principal quantum number
            Can either provide: beta or m
                beta: relativistic beta 
                m: mass of atom in Dalton (atomic unit)
        """
        if provided_beta:
            self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, self.beta = projectile_data
            self.exists_beta = True
            self.exists_projetile_data = True
        else:
            self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, self.atomic_mass_in_u = projectile_data
            self.exists_projetile_data = True
            self.beta_from_mass()
            self.exists_beta = True
        
        
    def beta_from_mass(self):
        """
        Calculate relativistic beta factor from projectile data if only mass is given
        """
        if not self.exists_projetile_data:
            raise ValueError('Have to provide projectile data!') 
            
        # Calculate mass of ion in electron volt 
        self.mass_in_u_stripped = self.atomic_mass_in_u - (self.Z_p - self.q) * constants.physical_constants['electron mass in u'][0] 
        self.mass_in_eV =  self.mass_in_u_stripped * constants.physical_constants['atomic mass unit-electron volt relationship'][0]
        self.E_tot = self.mass_in_eV + 1e6*self.e_kin * self.mass_in_u_stripped# total kinetic energy in eV per particle at injection
        self.gamma = self.E_tot/self.mass_in_eV
        self.beta = self.beta_from_gamma(self.gamma)
        
    
    
    def beta_from_gamma(self, gamma):
        """
        Relativistic beta factor from gamma factor 
        """
        return np.sqrt(1 - 1/gamma**2)
        
    
    def set_molecular_densities(self, molecular_fraction_array, p=None):
        """
        Parameters
        ----------
        molecular_fraction_array: contains
            x_H2 : fraction of H2
            x_H2O : fraction of H2O
            x_CO : fraction of CO
            x_CH4 : fraction of CH4
            x_CO2 : fraction of CO2
            x_He : fraction of He
            x_O2 : fraction of O2
            x_Ar : fraction of Ar

        Raises
        ------
        ValueError
            If fraction of gases does not sum up to 1.0
        ValueError
            If pressure data is not provided when the 'p' attribute is not set
        
        Returns
        -------
        None.
        """
        # Check if 'p' attribute exists
        if p is None and not hasattr(self, 'p'):
            raise ValueError('Have to provide pressure data!')
        else:
            self.p = p*1e2 # convert mbar to Pascal  
        if sum(molecular_fraction_array) != 1.0:
            raise ValueError('Molecular fraction does not sum up!')
        if not np.all(molecular_fraction_array >= 0):
            raise ValueError('Cannot set negative molecular fraction')
            
        # Set the molecular fractions
        x_H2, x_H2O, x_CO, x_CH4, x_CO2, x_He, x_O2, x_Ar = molecular_fraction_array    
        
        # Calculate the molecular density for each gas 
        self.n_H2 = self.p * x_H2 / (self.K * self.T)
        self.n_H2O = self.p * x_H2O / (self.K * self.T)
        self.n_CO = self.p * x_CO / (self.K * self.T)
        self.n_CH4 = self.p * x_CH4 / (self.K * self.T)
        self.n_CO2 = self.p * x_CO2 / (self.K * self.T)
        self.n_He = self.p * x_He / (self.K * self.T)
        self.n_O2 = self.p * x_O2 / (self.K * self.T)
        self.n_Ar = self.p * x_Ar / (self.K * self.T)

        # Contain these into one array
        self.fractions = [self.n_H2,
                          self.n_H2O,
                          self.n_CO,
                          self.n_CH4,
                          self.n_CO2,
                          self.n_He,
                          self.n_O2,
                          self.n_Ar]


    def dubois(self, Z, Z_p, q, e_kin, I_p):
        """
        Calculate the electron loss cross section from DuBois et al. (2011)

        Parameters
        ----------
        Z : Z of target
        Z_p : Z of projectile
        q : charge of projectile.
        e_kin : collision energy in MeV/u.
        I_p : first ionization potential of projectile in keV

        Returns
        -------
        dubois : dubois contribution to cross section
        """
        par = self.par
        alpha = 7.2973525376e-3
        AU = 931.5016 # MeV
        m_e = 510.998 # keV
        #Ry = 0.0136 # Rydberg energy
        g = 1. + e_kin/AU # gamma factor
        #beta = np.sqrt(1. - 1./g**2) # beta factor
        #u = (beta/alpha)**2/(I_p/Ry)
        N_eff = min(10**(par[3]*np.log10(Z)+par[4]*np.log10(Z)**2), Z)
        F1 = min(abs(N_eff+par[0]*(Z-N_eff)*(g-1.)), Z)
        F2 = Z*(1.-np.sqrt(1.-F1/Z))
        F3 = -F2**par[1]/((np.sqrt(2.*e_kin/AU)+np.sqrt(2.*I_p/m_e))/alpha)
        dubois = F1 + (F2*np.exp(F3))**2
        return dubois
    
    
    def shevelko(self, Z_p, q, e_kin, I_p, n_0):
        """
        Calculate the electron loss cross section from Shevelko et al. (2011)

        Parameters
        ----------
        Z_p : Z of projectile
        q : charge of projectile.
        e_kin : collision energy in MeV/u.
        I_p : first ionization potential of projectile in keV
        n_0 : principle quantum number of outermost projectile electron 

        Returns
        -------
        shevelko : shevelko contribution to cross section
        """
        par = self.par
        alpha = 7.2973525376e-3
        AU = 931.5016 # MeV
        #m_e = 510.998 # keV
        Ry = 13.606 # Rydberg energy eV
        Ry = Ry/1e3 # convert to keV
        g = 1. + e_kin/AU # gamma factor
        beta = np.sqrt(1. - 1./g**2) # beta factor
        u = (beta/alpha)**2/(I_p/Ry)
        shevelko = par[5]*1e-16*u/(u**2.+par[6])*(Ry/I_p)**(par[7]+((q+par[2])/Z_p)*(1.-np.exp(-par[9]*u**2)))\
        *(1.+par[8]/n_0*np.log((u+1.)*g))
        return shevelko


    def calculate_sigma_electron_loss(self, Z, Z_p, q, e_kin, I_p, n_0, SI_units=True):
        """
        Calculates the electron loss cross section from the Shevelko-Dubois formulae
        - not meant for fully stripped ions, i.e. with electrons in the 1s shell
        - if q = Z, then sets EL cross section automatically to 0
         
        Parameters
        ----------
        Z : Z of target
        Z_p : Z of projectile
        q : charge of projectile
        e_kin : collision energy in MeV/u.
        I_p : first ionization potential of projectile in keV
        n_0 : principle quantum number of outermost projectile electron 

        Returns
        -------
        sigma_electron_loss : electron loss cross section in in m^2 (cm^2 if SI units is set to false)
        """
        if q == Z_p:
            sigma_electron_loss = 0.0
            #print("Fully stripped ion: setting sigma_EL to 0 for Z_p = {} and q = {}!".format(self.Z_p, self.q))
        else:
            sigma_electron_loss = self.dubois(Z, Z_p, q, e_kin, I_p)*self.shevelko(Z_p, q, e_kin, I_p, n_0)
        
        if SI_units:
            sigma_electron_loss *= 1e-4 
        return sigma_electron_loss
    
    
    def calculate_sigma_electron_capture(self, Z, q, e_kin_keV, SI_units=True):
        """
        Calculate electron capture cross section from Schlachter formula 
    
        Parameters
        ----------
        Z : Z of target
        q : charge of projectile
        e_kin_keV : collision energy in keV/u.
      
        Returns
        -------
        sigma_electron_capture electron capture cross section  in (cm^2 if SI units is set to false)
        """
        E_tilde = (e_kin_keV)/(Z**(1.25)*q**0.7)  
        sigma_tilde = 1.1e-8/E_tilde**4.8*(1 - np.exp(-0.037*E_tilde**2.2))*(1 - np.exp(-2.44*1e-5*E_tilde**2.6))
        sigma_electron_capture = sigma_tilde*q**0.5/(Z**1.8)
        if SI_units:
            sigma_electron_capture *= 1e-4 # convert to m^2
        return sigma_electron_capture
        
    
    def return_all_sigmas(self):
        """
        Return electron loss (EL) and electron capture (EC) cross sections after having calculated estimates lifetimes
        unit is in m^2 (SI units), for rest gases with non-zero contribution 
 
        Returns
        -------
        sigmas_EL 
        sigmas_EC
        """
        if not self.lifetimes_are_calculated:
            self.calculate_total_lifetime_full_gas()

        # All sigmas for electron loss and electron capture
        sigmas_EL = np.array([self.sigma_H2_EL, 
                              self.sigma_H2O_EL, 
                              self.sigma_CO_EL, 
                              self.sigma_CH4_EL, 
                              self.sigma_CO2_EL, 
                              self.sigma_He_EL, 
                              self.sigma_O2_EL,
                              self.sigma_Ar_EL])
        
        sigmas_EC = np.array([self.sigma_H2_EC, 
                              self.sigma_H2O_EC, 
                              self.sigma_CO_EC, 
                              self.sigma_CH4_EC, 
                              self.sigma_CO2_EC,
                              self.sigma_He_EC, 
                              self.sigma_O2_EC,
                              self.sigma_Ar_EC])

        # Create a list to store the non-zero fractions and corresponding cross sections
        non_zero_fractions = []
        non_zero_sigmas_EL = []
        non_zero_sigmas_EC = []

        # Iterate over the fractions and cross sections
        for fraction, sigma_EL, sigma_EC in zip(self.fractions, sigmas_EL, sigmas_EC):
            # Check if the fraction is non-zero
            if fraction != 0:
                # Append the non-zero fraction and corresponding cross sections to the respective lists
                non_zero_fractions.append(fraction)
                non_zero_sigmas_EL.append(sigma_EL)
                non_zero_sigmas_EC.append(sigma_EC)
        
        # Convert the non-zero cross sections to numpy arrays
        non_zero_sigmas_EL = np.array(non_zero_sigmas_EL)
        non_zero_sigmas_EC = np.array(non_zero_sigmas_EC)

        # Return the non-zero fractions and cross sections
        return non_zero_sigmas_EL, non_zero_sigmas_EC, non_zero_fractions
    
    
    def calculate_lifetime_on_single_gas(self, p, Z_t, projectile_data=None, atomicity=1):
        """
        Calculates beam lifetime from electron loss and electron capture from beam gas
        interactions with a single gas 
        
        Parameters
        ----------
        p : pressure in beam pipe in millibar 
        Z_t : Z of target
        projectile_data: optional, not needed if projectile data is already initiated 
        atomicity : number of atoms per molecule

        Returns
        -------
        Total lifetime tau in seconds
        """  
        # Find molecular density from pressure
        p_SI =  p*1e2 # convert mbar to Pascal (SI) 
        n = p_SI / (self.K * self.T)
        
        # Cross sections
        sigma_EL = atomicity * self.calculate_sigma_electron_loss(Z_t, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        e_kin_keV = self.e_kin * 1e3 # convert MeV to keV 
        sigma_EC = atomicity * self.calculate_sigma_electron_capture(Z_t, self.q, e_kin_keV, SI_units=True)
    
        # Lifetimes
        tau_EL = 1.0/(sigma_EL * n * self.beta * self.c_light)
        tau_EC = 1.0/(sigma_EC * n * self.beta * self.c_light)
        tau_tot_inv = 1/tau_EL + 1/tau_EC
        tau_tot = 1/tau_tot_inv 
        return tau_tot, sigma_EL, sigma_EC
    
    
    def calculate_total_lifetime_full_gas(self, projectile_data=None):
        """
        Calculates beam lifetime contributions for electron loss and electron capture from beam gas
        interactions with all compound gases
        
        Parameters
        ----------
        projectile_data: optional, not needed if projectile data is already initiated 
        
        Returns
        -------
        Total lifetime tau in seconds
        """
        if not self.molecular_densities_are_set:
            raise ValueError('Have to provide molecular composition data!') 
        if not self.exists_beta:
            raise ValueError('Beta has to be provided!') 
        if not self.exists_projetile_data:
            if projectile_data is not None:
                self.set_projectile_data(projectile_data)
            else:
                raise ValueError('Have to provide projectile data!') 
        
        # Atomic numbers of relevant gasess
        Z_H = 1.0
        Z_He = 2.0
        Z_C = 6.0
        Z_O = 8.0
        Z_Ar = 18.0
                
        # Calculate the electron loss (EL) cross sections from all atoms present in gas 
        sigma_H_EL = self.calculate_sigma_electron_loss(Z_H, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_He_EL = self.calculate_sigma_electron_loss(Z_He, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_C_EL = self.calculate_sigma_electron_loss(Z_C, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_O_EL = self.calculate_sigma_electron_loss(Z_O, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_Ar_EL = self.calculate_sigma_electron_loss(Z_Ar, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)

        # Estimate molecular EL cross section from additivity rule, multiplying with atomicity 
        # Eq (5) in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.67.022706
        self.sigma_H2_EL = 2.0 * sigma_H_EL
        self.sigma_He_EL = sigma_He_EL
        self.sigma_H2O_EL = 2.0 * sigma_H_EL + sigma_O_EL
        self.sigma_CO_EL = sigma_C_EL + sigma_O_EL
        self.sigma_CH4_EL = sigma_C_EL + 4 * sigma_H_EL
        self.sigma_CO2_EL = sigma_C_EL + 2 * sigma_O_EL
        self.sigma_O2_EL =  2 * sigma_O_EL
        self.sigma_Ar_EL = sigma_Ar_EL
        
        # Estimate the EL lifetime for collision with each gas type - if no gas or zero EL cross section, set lifetime to infinity
        tau_H2_EL = 1.0/(self.sigma_H2_EL *self.n_H2*self.beta*self.c_light) if self.n_H2 and self.sigma_H2_EL != 0.0 else np.inf
        tau_He_EL = 1.0/(self.sigma_He_EL *self.n_He*self.beta*self.c_light) if self.n_He and self.sigma_He_EL != 0.0 else np.inf 
        tau_H2O_EL = 1.0/(self.sigma_H2O_EL *self.n_H2O*self.beta*self.c_light) if self.n_H2O and self.sigma_H2O_EL != 0.0 else np.inf 
        tau_CO_EL = 1.0/(self.sigma_CO_EL *self.n_CO*self.beta*self.c_light) if self.n_CO and self.sigma_CO_EL != 0.0 else np.inf 
        tau_CH4_EL = 1.0/(self.sigma_CH4_EL *self.n_CH4*self.beta*self.c_light) if self.n_CH4 and self.sigma_CH4_EL != 0.0 else np.inf 
        tau_CO2_EL = 1.0/(self.sigma_CO2_EL *self.n_CO2*self.beta*self.c_light) if self.n_CO2 and self.sigma_CO2_EL != 0.0 else np.inf
        tau_O2_EL = 1.0/(self.sigma_O2_EL *self.n_O2*self.beta*self.c_light) if self.n_O2 and self.sigma_O2_EL != 0.0 else np.inf
        tau_Ar_EL = 1.0/(self.sigma_Ar_EL *self.n_Ar*self.beta*self.c_light) if self.n_Ar and self.sigma_Ar_EL != 0.0 else np.inf
        
        # Estimate the electron capture (EC) cross sections from all atoms present in gas 
        e_kin_keV = self.e_kin * 1e3 # convert MeV to keV 
        sigma_H_EC = self.calculate_sigma_electron_capture(Z_H, self.q, e_kin_keV, SI_units=True)
        sigma_He_EC = self.calculate_sigma_electron_capture(Z_He, self.q, e_kin_keV, SI_units=True)
        sigma_C_EC = self.calculate_sigma_electron_capture(Z_C, self.q, e_kin_keV, SI_units=True)
        sigma_O_EC = self.calculate_sigma_electron_capture(Z_O, self.q, e_kin_keV, SI_units=True)
        sigma_Ar_EC = self.calculate_sigma_electron_capture(Z_Ar, self.q, e_kin_keV, SI_units=True)

        # Also calculate the molecular EC cross section
        self.sigma_H2_EC = 2.0 * sigma_H_EC
        self.sigma_He_EC = sigma_He_EC
        self.sigma_H2O_EC = 2.0 * sigma_H_EC + sigma_O_EC
        self.sigma_CO_EC = sigma_C_EC + sigma_O_EC
        self.sigma_CH4_EC = sigma_C_EC + 4 * sigma_H_EC
        self.sigma_CO2_EC = sigma_C_EC + 2 * sigma_O_EC
        self.sigma_O2_EC =  2 * sigma_O_EC     
        self.sigma_Ar_EC = sigma_Ar_EC
        
        # Flag for calculated lifetimes
        self.lifetimes_are_calculated = True

        # Estimate the EL lifetime for collision with each gas type - if no gas, set lifetime to infinity
        tau_H2_EC = 1.0/(self.sigma_H2_EC *self.n_H2*self.beta*self.c_light) if self.n_H2 else np.inf
        tau_He_EC = 1.0/(self.sigma_He_EC *self.n_He*self.beta*self.c_light) if self.n_He else np.inf 
        tau_H2O_EC = 1.0/(self.sigma_H2O_EC *self.n_H2O*self.beta*self.c_light) if self.n_H2O else np.inf 
        tau_CO_EC = 1.0/(self.sigma_CO_EC *self.n_CO*self.beta*self.c_light) if self.n_CO else np.inf 
        tau_CH4_EC = 1.0/(self.sigma_CH4_EC *self.n_CH4*self.beta*self.c_light) if self.n_CH4 else np.inf 
        tau_CO2_EC = 1.0/(self.sigma_CO2_EC *self.n_CO2*self.beta*self.c_light) if self.n_CO2 else np.inf
        tau_O2_EC = 1.0/(self.sigma_O2_EC *self.n_O2*self.beta*self.c_light) if self.n_O2 else np.inf
        tau_Ar_EC = 1.0/(self.sigma_Ar_EC *self.n_Ar*self.beta*self.c_light) if self.n_Ar else np.inf 

        # Calculate the total cross section
        tau_tot_inv = 1/tau_H2_EL + 1/tau_H2_EC \
                    + 1/tau_He_EL + 1/tau_He_EC \
                    + 1/tau_H2O_EL + 1/tau_H2O_EC \
                    + 1/tau_CO_EL + 1/tau_CO_EC \
                    + 1/tau_CH4_EL + 1/tau_CH4_EC \
                    + 1/tau_CO2_EL + 1/tau_CO2_EC \
                    + 1/tau_O2_EL + 1/tau_O2_EC \
                    + 1/tau_Ar_EL + 1/tau_Ar_EC
        tau_tot = 1/tau_tot_inv
        
        return tau_tot 
    
    
    
class Data:
    """
    Contains all projectile, pressure and gas data
    """
    def __init__(self):
        self.gas_fractions = pd.read_csv('{}/Gas_fractions.csv'.format(data_folder), index_col=0)
        self.pressure_data = pd.read_csv('{}/Pressure_data.csv'.format(data_folder), index_col=0).T
        self.projectile_data = pd.read_csv('{}/Projectile_data.csv'.format(data_folder), index_col=0)