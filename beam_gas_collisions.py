#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import scipy.constants as constants

class beam_gas_collisions:
    """
    Class to calculate the electron loss and electron capture cross section of rest gas collisions from 
    semi-empirical Dubois, Shevelko and Schlachter formulae 
    """
    def __init__(self, p, molecular_fraction_array, projectile_data=None,T=298):
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
        T : Temperature in kelvin. The default is 298.
        """
        
        # Fitting parameters obtained from the semi-empirical study by Weber (2016)
        self.par = [10.88, 0.95, 2.5, 1.1137, -0.1805, 2.64886, 1.35832, 0.80696, 1.00514, 6.13667]
        
        # Beam and accelerator setting
        self.K = constants.Boltzmann
        self.T = T # temperature in Kelvin 
        self.c_light = constants.c
        self.p = p*1e2 # convert mbar to Pascal
        
        # Initiate data if available
        self.set_molecular_densities(molecular_fraction_array)
        if projectile_data is not None:
            self.set_projectile_data(projectile_data)
        else:
            self.exists_projetile_data = False
        self.lifetimes_are_calculated = False # state to indicate if lifetimes are calculated or not
    
    def set_projectile_data(self, projectile_data):
        """
        Sets the projectile data:
            Z_p : Z of projectile
            q : charge of projectile.
            e_kin : collision energy in MeV/u.
            I_p : first ionization potential of projectile in keV
            beta: relativistic beta 
        """
        self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, self.beta = projectile_data
        self.exists_projetile_data = True
        
        
    def set_molecular_densities(self, molecular_fraction_array):
        """
        Parameters
        ----------
        molecular_fraction_array: contains
        x_H2 : fraction of H2
        x_H2O : fraction of H2O
        x_CO : fraction of CO
        x_CH4 : fraction of CH4
        x_CO2 : fraction of CO2

        Raises
        ------
        ValueError
            If fraction of gases does not sum up to 1.0

        Returns
        -------
        None.
        """
        if sum(molecular_fraction_array) != 1.0:
            raise ValueError('Molecular fraction does not sum up!')
        if not np.all(molecular_fraction_array >= 0):
            raise ValueError('Cannot set negative molecular fraction')
        x_H2, x_H2O, x_CO, x_CH4, x_CO2 = molecular_fraction_array    
        
        # Calculate the molecular density for each gas 
        self.n_H2 = self.p * x_H2 / (self.K * self.T)
        self.n_H2O = self.p * x_H2O / (self.K * self.T)
        self.n_CO = self.p * x_CO / (self.K * self.T)
        self.n_CH4 = self.p * x_CH4 / (self.K * self.T)
        self.n_CO2 = self.p * x_CO2 / (self.K * self.T)


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
        e_kin_keV : collision energy in MeV/u.
      
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
        unit is in m^2 (SI units)
 
        Returns
        -------
        sigmas_EL 
        sigmas_EC
        """
        if not self.lifetimes_are_calculated:
            self.calculate_total_lifetime_full_gas()

        # Create vectors of sigmas for electron loss and electron capture
        sigmas_EL = np.array([self.sigma_H2_EL, self.sigma_H2O_EL, self.sigma_CO_EL, self.sigma_CH4_EL, self.sigma_CO2_EL])
        sigmas_EC = np.array([self.sigma_H2_EC, self.sigma_H2O_EC, self.sigma_CO_EC, self.sigma_CH4_EC, self.sigma_CO2_EC])
        return sigmas_EL, sigmas_EC
    
    
    def calculate_total_lifetime_full_gas(self, projectile_data=None):
        """
        Calculates beam lifetime contributions for electron loss and electron capture from beam gas
        interactions with all 
        
        Parameters
        ----------
        Z_p : Z of projectile
        q : charge of projectile
        e_kin : collision energy in MeV/u.
        I_p : first ionization potential of projectile in keV
        n_0 : principle quantum number of outermost projectile electron 
        
        Returns
        -------
        Total lifetime tau in seconds
        """
        if not self.exists_projetile_data:
            if projectile_data is not None:
                self.set_projectile_data(projectile_data)
            else:
                raise ValueError('Have to provide projectile data!')
        
        # Atomic numbers of relevant gasess
        Z_H = 1.0
        Z_O = 8.0
        Z_C = 6.0
                
        # Calculate the electron loss (EL) cross sections from all atoms present in gas 
        sigma_H_EL = self.calculate_sigma_electron_loss(Z_H, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_O_EL = self.calculate_sigma_electron_loss(Z_O, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
        sigma_C_EL = self.calculate_sigma_electron_loss(Z_C, self.Z_p, self.q, self.e_kin, self.I_p, self.n_0, SI_units=True)
                
        # Estimate molecular EL cross section from additivity rule
        # Eq (5) in https://journals.aps.org/pra/abstract/10.1103/PhysRevA.67.022706
        self.sigma_H2_EL = 2.0 * sigma_H_EL
        self.sigma_H2O_EL = 2.0 * sigma_H_EL + sigma_O_EL
        self.sigma_CO_EL = sigma_C_EL + sigma_O_EL
        self.sigma_CH4_EL = sigma_C_EL + 4 * sigma_H_EL
        self.sigma_CO2_EL = sigma_C_EL + 2 * sigma_O_EL 
        
        # Estimate the EL lifetime for collision with each gas type - if no gas, set lifetime to infinity
        tau_H2_EL = 1.0/(self.sigma_H2_EL *self.n_H2*self.beta*self.c_light) if self.n_H2 else np.inf 
        tau_H2O_EL = 1.0/(self.sigma_H2O_EL *self.n_H2O*self.beta*self.c_light) if self.n_H2O else np.inf 
        tau_CO_EL = 1.0/(self.sigma_CO_EL *self.n_CO*self.beta*self.c_light) if self.n_CO else np.inf 
        tau_CH4_EL = 1.0/(self.sigma_CH4_EL *self.n_CH4*self.beta*self.c_light) if self.n_CH4 else np.inf 
        tau_CO2_EL = 1.0/(self.sigma_CO2_EL *self.n_CO2*self.beta*self.c_light) if self.n_CO2 else np.inf 
        
        # Estimate the electron capture (EC) cross sections from all atoms present in gas 
        e_kin_keV = self.e_kin * 1e3 # convert MeV to keV 
        sigma_H_EC = self.calculate_sigma_electron_capture(Z_H, self.q, e_kin_keV, SI_units=True)
        sigma_O_EC = self.calculate_sigma_electron_capture(Z_O, self.q, e_kin_keV, SI_units=True)
        sigma_C_EC = self.calculate_sigma_electron_capture(Z_C, self.q, e_kin_keV, SI_units=True)

        # Also calculate the molecular EC cross section
        self.sigma_H2_EC = 2.0 * sigma_H_EC
        self.sigma_H2O_EC = 2.0 * sigma_H_EC + sigma_O_EC
        self.sigma_CO_EC = sigma_C_EC + sigma_O_EC
        self.sigma_CH4_EC = sigma_C_EC + 4 * sigma_H_EC
        self.sigma_CO2_EC = sigma_C_EC + 2 * sigma_O_EC        
        
        #
        self.lifetimes_are_calculated = True

        # Estimate the EL lifetime for collision with each gas type - if no gas, set lifetime to infinity
        tau_H2_EC = 1.0/(self.sigma_H2_EC *self.n_H2*self.beta*self.c_light) if self.n_H2 else np.inf 
        tau_H2O_EC = 1.0/(self.sigma_H2O_EC *self.n_H2O*self.beta*self.c_light) if self.n_H2O else np.inf 
        tau_CO_EC = 1.0/(self.sigma_CO_EC *self.n_CO*self.beta*self.c_light) if self.n_CO else np.inf 
        tau_CH4_EC = 1.0/(self.sigma_CH4_EC *self.n_CH4*self.beta*self.c_light) if self.n_CH4 else np.inf 
        tau_CO2_EC = 1.0/(self.sigma_CO2_EC *self.n_CO2*self.beta*self.c_light) if self.n_CO2 else np.inf 

        # Calculate the total cross section
        tau_tot_inv = 1/tau_H2_EL + 1/tau_H2_EC \
                    + 1/tau_H2O_EL + 1/tau_H2O_EC \
                    + 1/tau_CO_EL + 1/tau_CO_EC \
                    + 1/tau_CH4_EL + 1/tau_CH4_EC \
                    + 1/tau_CO2_EL + 1/tau_CO2_EC
        tau_tot = 1/tau_tot_inv
        
        return tau_tot 