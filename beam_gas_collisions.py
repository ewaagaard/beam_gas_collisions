#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np

class beam_gas_collisions:
    """
    Class to calculate the electron loss and electron capture cross section of rest gas collisions from 
    semi-empirical Dubois, Shevelko and Schlachter formulae 
    """
    def __init__(self):
        # Fitting parameters obtained from the semi-empirical study by Weber (2016)
        self.par = [10.88, 0.95, 2.5, 1.1137, -0.1805, 2.64886, 1.35832, 0.80696, 1.00514, 6.13667]
                 

    def DuBois(self, Z, Z_p, q, e_kin, I_p):
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
        DuBois : dubois contribution to cross section
        """
        par = self.par
        alpha = 7.2973525376e-3
        AU = 931.5016 # MeV
        m_e = 510.998 # keV
        Ry = 0.0136 # Rydberg energy
        g = 1. + e_kin/AU # gamma factor
        beta = np.sqrt(1. - 1./g**2) # beta factor
        u = (beta/alpha)**2/(I_p/Ry)
        N_eff = min(10**(par[3]*np.log10(Z)+par[4]*np.log10(Z)**2), Z)
        F1 = min(abs(N_eff+par[0]*(Z-N_eff)*(g-1.)), Z)
        F2 = Z*(1.-np.sqrt(1.-F1/Z))
        F3 = -F2**par[1]/((np.sqrt(2.*e_kin/AU)+np.sqrt(2.*I_p/m_e))/alpha)
        DuBois = F1 + (F2*np.exp(F3))**2
        return DuBois
    
    def Shevelko(self, Z_p, q, e_kin, I_p, n_0):
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
        Shevelko : shevelko contribution to cross section
        """
        par = self.par
        alpha = 7.2973525376e-3
        AU = 931.5016 # MeV
        m_e = 510.998 # keV
        Ry = 13.606 # eV
        Ry = Ry/1e3 # convert to keV
        g = 1. + e_kin/AU # gamma factor
        beta = np.sqrt(1. - 1./g**2) # beta factor
        u = (beta/alpha)**2/(I_p/Ry)
        Shevelko = par[5]*1e-16*u/(u**2.+par[6])*(Ry/I_p)**(par[7]+((q+par[2])/Z_p)*(1.-np.exp(-par[9]*u**2)))\
        *(1.+par[8]/n_0*np.log((u+1.)*g))
        return Shevelko


    def sigma_EL(self, Z, Z_p, q, e_kin, I_p, n_0, SI_units=True):
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
        sigma_EL : electron loss cross section in in m^2 (cm^2 if SI units is set to false)
        """
        sigma_EL = self.DuBois(Z, Z_p, q, e_kin, I_p)*self.Shevelko(Z_p, q, e_kin, I_p, n_0)
        if SI_units:
            sigma_EL *= 1e-4 
        return sigma_EL
    
    
    def sigma_EC(self, Z, q, e_kin_keV, SI_units=True):
        """
        Calculate electron capture cross section from Schlachter formula 
    
        Parameters
        ----------
        Z : Z of target
        q : charge of projectile
        e_kin_keV : collision energy in MeV/u.
      
        Returns
        -------
        sigma_EC: electron capture cross section  in (cm^2 if SI units is set to false)
        """
        E_tilde = (e_kin_keV)/(Z**(1.25)*q**0.7)  
        sigma_tilde = 1.1e-8/E_tilde**4.8*(1 - np.exp(-0.037*E_tilde**2.2))*(1 - np.exp(-2.44*1e-5*E_tilde**2.6))
        sigma_EC = sigma_tilde*q**0.5/(Z**1.8)
        if SI_units:
            sigma_EC *= 1e-4 # convert to m^2
        return sigma_EC
        