"""
Main container of Beam Gas Collisions class
"""
import numpy as np
import pandas as pd
import scipy.constants as constants
from pathlib import Path
import matplotlib.pyplot as plt

data_folder = Path(__file__).resolve().parent.joinpath('../data').absolute()

SMALL_SIZE = 16
MEDIUM_SIZE = 18
BIGGER_SIZE = 23
plt.rcParams["font.family"] = "serif"
plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)   # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)   # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)   # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


class DataObject:
    """
    Data container for projectile, pressure and gas data
    """
    def __init__(self, machine='PS'):
        """
        Loads and stores data for desired accelertor
        
        Parameters
        ----------
        machine : str
            Which CERN accelerator to load vacuum conditions from: 'LEIR', 'PS' or 'SPS'
        """
        if machine not in ['LEIR', 'PS', 'SPS']:
            raise ValueError("Machine has to be 'LEIR', 'PS', or 'SPS' !")
        
        # Load full data
        self.gas_fractions_all = pd.read_csv('{}/gas_fractions.csv'.format(data_folder), index_col=0)
        self.pressure_data = pd.read_csv('{}/pressure_data.csv'.format(data_folder), index_col=0).T
        self.projectile_data = pd.read_csv('{}/projectile_data.csv'.format(data_folder), index_col=0)
        
        # Store machine-specific pressure and gas data
        self.pressure_pascal = self.pressure_data[machine] * 1e2 # convert mbar to Pascal (SI unit)
        self.gas_frac = self.gas_fractions_all[machine]


class IonLifetimes:
    """
    Class to calculate the electron loss and electron capture cross section of rest gas collisions from 
    semi-empirical Dubois, Shevelko and Schlachter formulae 
    """
    def __init__(self, 
                 projectile='Pb54',
                 machine='PS',
                 T=298,
                 p=None,
                 molecular_fraction_array=None):
        """
        Parameters
        ----------
        projectile : str
            define ion and charge state (+). Available ions are 'He1', 'He2', 'O4', 'O8', 'Mg6', 'Mg7', 'Pb54'
        machine : str
            define CERN accelerator to load vacuum conditions from: 'LEIR', 'PS' or 'SPS'
        T : float
            temperature in kelvin. The default is 298.
        p : float
            pressure in mbar. Default is None, not needed if machine is specified
        molecular_fraction_array : np.ndarray
            fraction of molecular density in rest gas. Default is None, not needed if machine is specified
        """
        # Beam and accelerator setting
        self.K = constants.Boltzmann
        self.T = T # temperature in Kelvin 
        self.c_light = constants.c
        # Fitting parameters obtained from the semi-empirical study by Weber (2016)
        self.par = [10.88, 0.95, 2.5, 1.1137, -0.1805, 2.64886, 1.35832, 0.80696, 1.00514, 6.13667]
        self.set_projectile_and_machine(projectile=projectile, machine=machine, p=p, molecular_fraction_array=molecular_fraction_array)


    def set_projectile_and_machine(self, projectile='Pb54', machine='PS', p=None, molecular_fraction_array=None):
        """
        Update projectile data and parameters
        
        Parameters
        ----------
        projectile : str
            define ion and charge state (+). Available ions are 'He1', 'He2', 'O4', 'O8', 'Mg6', 'Mg7', 'Pb54'
        machine : str
            define CERN accelerator to load vacuum conditions from: 'LEIR', 'PS' or 'SPS'
        p : float
            pressure in mbar. Default is None, not needed if machine is specified
        molecular_fraction_array : np.ndarray
            fraction of molecular density in rest gas. Default is None, not needed if machine is specified
        """
        self.projectile = projectile

        if machine in ['LEIR', 'PS', 'SPS']:      
            # Load machine data: set pressure in Pascal and molecular density
            self.machine = machine
            data = DataObject(machine)
            self.p = data.pressure_pascal.values[0]
            self.set_molecular_densities(data.gas_frac)
            self.set_projectile_data(data)
            self.target_atoms = data.gas_frac.T.keys().values
            self.all_possible_projectiles = data.projectile_data.T.columns.values
        else:
            print("Machine has to be 'LEIR', 'PS', or 'SPS' for automatic data loading! Set pressures and molecular fractions manually!") 
            if (p is not None) & (molecular_fraction_array is not None):
                self.p = p
                self.set_molecular_densities(molecular_fraction_array)
            else:
                raise ValueError("If machine not 'LEIR', 'PS', or 'SPS', have to provide pressure!")      

        print('\nProjectile {} for machine {} at p = {} mbar\n'.format(self.projectile, self.machine, self.p / 1e2))
        
    
    def set_projectile_data(self, data):
        """
        Sets the projectile data:
            Z_p : atomic number Z of projectile
            A_p : mass number A of projectile
            q : charge of projectile.
            e_kin : collision energy in MeV/u.
            I_p : first ionization potential of projectile in keV
            n_0: principal quantum number
            Can either provide: beta or m
                beta: relativistic beta 
                m: mass of atom in Dalton (atomic unit)
                
        Parameters:
        -----------
        data : DataObject
        """
        projectile_data = np.array([data.projectile_data['Z_p'][self.projectile],
                                    data.projectile_data['A_p'][self.projectile],
                                    data.projectile_data['{}_q'.format(self.machine)][self.projectile],
                                    data.projectile_data['{}_Kinj'.format(self.machine)][self.projectile],
                                    data.projectile_data['I_p'][self.projectile], 
                                    data.projectile_data['n_0'][self.projectile], 
                                    data.projectile_data['{}_beta'.format(self.machine)][self.projectile]])
        
        self.Z_p, self.A_p, self.q, self.e_kin, self.I_p, self.n_0, self.beta = projectile_data
        self.gamma = data.projectile_data['{}_gamma'.format(self.machine)][self.projectile] # more exact
        print('\nProjectile initialized: {}\n{}\n'.format(self.projectile, data.projectile_data.loc[self.projectile]))


    def set_projectile_data_manually(self, projectile_data, beta_is_provided=False):
        """
        Sets the projectile data manually, and calculates relativistic beta

        Parameters:
        -----------
        projectile_data : np.ndarray
            Z_p : atomic number Z of projectile
            A_p : mass number A of projectile
            q : charge of projectile.
            e_kin : collision energy in MeV/u.
            I_p : first ionization potential of projectile in keV
            n_0: principal quantum number
            m: mass of atom in Dalton (atomic unit)
        beta_is_provided: bool
            whether relativistic beta is given, or only the atomic mass. Default is False.
        """
        if beta_is_provided:
            self.Z_p, self.A_p, self.q, self.e_kin, self.I_p, self.n_0, self.beta = projectile_data
        else:
            self.Z_p, self.A_p, self.q, self.e_kin, self.I_p, self.n_0, self.atomic_mass_in_u = projectile_data

            self.mass_in_u_stripped = self.atomic_mass_in_u - (self.Z_p - self.q) * constants.physical_constants['electron mass in u'][0] 
            self.mass_in_eV =  self.mass_in_u_stripped * constants.physical_constants['atomic mass unit-electron volt relationship'][0]
            self.E_tot = self.mass_in_eV + 1e6*self.e_kin * self.mass_in_u_stripped# total kinetic energy in eV per particle at injection
            self.gamma = self.E_tot/self.mass_in_eV
            self.beta = np.sqrt(1 - 1/self.gamma**2)


    def set_molecular_densities(self, molecular_fraction_array):
        """
        Specify fraction of molecular density in rest gas
        
        Parameters
        ----------
        molecular_fraction_array: np.ndarray
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
        """
        if sum(molecular_fraction_array) != 1.0:
            raise ValueError('Molecular fraction does not sum up!')
        if not np.all(molecular_fraction_array >= 0):
            raise ValueError('Cannot set negative molecular fraction')
            
        # Set the molecular fractions
        x_H2, x_H2O, x_CO, x_CH4, x_CO2, x_He, x_O2, x_Ar = molecular_fraction_array
        print('Molecular gas densities set to:\n{}'.format(molecular_fraction_array))
        
        # Calculate the molecular density for each gas 
        self.n_H2 = self.p * x_H2 / (self.K * self.T)
        self.n_H2O = self.p * x_H2O / (self.K * self.T)
        self.n_CO = self.p * x_CO / (self.K * self.T)
        self.n_CH4 = self.p * x_CH4 / (self.K * self.T)
        self.n_CO2 = self.p * x_CO2 / (self.K * self.T)
        self.n_He = self.p * x_He / (self.K * self.T)
        self.n_O2 = self.p * x_O2 / (self.K * self.T)
        self.n_Ar = self.p * x_Ar / (self.K * self.T)

        # Make array of target atom fractions and atomic numbers
        self.fractions = [self.n_H2,
                          self.n_H2O,
                          self.n_CO,
                          self.n_CH4,
                          self.n_CO2,
                          self.n_He,
                          self.n_O2,
                          self.n_Ar]
        # For molecular cross sections of atomic itneractions,
        # we approximate the target atomic number as the sum
        self.Z_t = [2., 10., 20., 16., 28., 2, 16., 18]


    def dubois(self, Z, Z_p, q, e_kin, I_p):
        """
        Calculate the electron loss cross section from DuBois et al. (2011)

        Parameters
        ----------
        Z : int
            Z of target
        Z_p : int
            Z of projectile
        q : int
            charge of projectile.
        e_kin : float
            collision energy in MeV/u.
        I_p : float
            first ionization potential of projectile in keV

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
        Z_p : int
            Z of projectile
        q : int
            charge of projectile.
        e_kin : float
            collision energy in MeV/u.
        I_p : float
            first ionization potential of projectile in keV
        n_0 : int
            principle quantum number of outermost projectile electron 

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
        
        
    def calculate_lifetime_on_single_gas(self, p, Z_t, atomicity=1):
        """
        Calculates beam lifetime from electron loss and electron capture from beam gas
        interactions with a single gas species.
        
        Parameters
        ----------
        p : float
            Pressure in beam pipe [mbar]
        Z_t : float
            Atomic number of target gas
        atomicity : int, optional
            Number of atoms per molecule (default=1)

        Returns
        -------
        tau_tot : float
            Total lifetime [s]
        sigma_EL : float
            Electron loss cross section [m^2]
        sigma_EC : float
            Electron capture cross section [m^2]
        """  
        # Find molecular density from pressure
        n = self._pressure_to_density(p)
        
        # Calculate cross sections
        sigma_EL = atomicity * self.calculate_sigma_electron_loss(Z_t, self.Z_p, self.q, 
                                                                self.e_kin, self.I_p, 
                                                                self.n_0, SI_units=True)
        sigma_EC = atomicity * self.calculate_sigma_electron_capture(Z_t, self.q, 
                                                                   self.e_kin * 1e3, 
                                                                   SI_units=True)
        
        # Calculate total lifetime
        tau_tot = self._calculate_lifetime_from_cross_sections(sigma_EL, sigma_EC, n)
        return tau_tot, sigma_EL, sigma_EC

    def _pressure_to_density(self, p):
        """Convert pressure in mbar to molecular density in m^-3"""
        p_SI = p * 1e2  # convert mbar to Pascal (SI)
        return p_SI / (self.K * self.T)

    def _calculate_lifetime_from_cross_sections(self, sigma_EL, sigma_EC, n):
        """Calculate total lifetime from cross sections and density"""
        tau_EL = 1.0/(sigma_EL * n * self.beta * self.c_light)
        tau_EC = 1.0/(sigma_EC * n * self.beta * self.c_light)
        return 1.0/(1/tau_EL + 1/tau_EC)
    
    
    def calculate_total_lifetime_full_gas(self):
        """
        Calculate total beam lifetime considering all gas species present in the 
        accelerator, including both electron loss and electron capture effects.
        
        The lifetime is calculated using:
            tau = 1 / (n * sigma * beta * c)
        where:
            - n is the molecular density
            - sigma is the total cross section (EC + EL)
            - beta is the relativistic beta factor
            - c is the speed of light
        
        Returns
        -------
        float
            Total beam lifetime in seconds
        
        Notes
        -----
        The calculation uses the gas fractions and pressure stored in the object,
        which must be set either through the constructor or manually before calling
        this method.
        
        Examples
        --------
        >>> beam = IonLifetimes(projectile='Pb54', machine='PS')
        >>> tau = beam.calculate_total_lifetime_full_gas()
        """
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
    
    
    def return_all_sigmas(self):
        """
        Return electron loss (EL) and electron capture (EC) cross sections after having calculated estimates lifetimes
        unit is in m^2 (SI units), for rest gases with non-zero contribution 
 
        Returns
        -------
        sigmas_EL : electron loss cross section
        sigmas_EC : electron capture (EC) cross section
        """
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
    

class BeamGasCollisions(IonLifetimes):
    """
    Class object to represent inelastic nuclear collisions and elastic Coulomb scattering,
    which are other processed possibly contributing to deteriorated beam lifetime. Inherits attributes
    and info from IonLifetimes class
    """
    
    def __init__(self, 
                 projectile='Pb54',
                 machine='PS',
                 T=298,
                 p=None,
                 molecular_fraction_array=None):
        
        super().__init__(projectile=projectile, machine=machine, T=T, 
                         p=p, molecular_fraction_array=molecular_fraction_array)  # instantiate Ion Lifetime class
        
        # Define required constants
        self.r0 = 1.35e-15 # from Westfall (1979)
        self.b0 = 0.83
        
        # Average SPS beta functions from Q26 lattice
        self.beta_avg = {'x': 57.32, 'y': 55.81}

        
    def compute_inelastic_nuclear_cross_sections(self, A_p : float, A_t : float):
        """
        Nuclear collision cross section from Bradt-Peters formula (1950)
        
        Parameters
        ----------
        A_p : float
            Projectile mass number
        A_t : float
            Target mass number
            
        Returns:
        -------
        sigma_n : float
            total inelastic nuclear collision cross section 
        """
        A_eff = 0.089 if A_t == 1.0 else A_t
        
        sigma_n = np.pi * self.r0**2 * (A_p**(1/3) + A_eff**(1/3) - self.b0)**2
        return sigma_n


    def get_lifetimes_from_nuclear_collisions(self):
        """
        Compute total beam lifetimes from inelastic nuclear collision cross sections with target nuclei

        Returns
        -------
        tau_tot : float
            total beam lifetime in seconds due to these processes
        N_dict : dict
            dictionary with sigma_N and gas fractions for non-zero values 
        """
        
        # Atomic numbers of relevant gasess
        A_H = 1.0
        A_He = 4.0
        A_C = 12.0
        A_O = 16.0
        A_Ar = 40.0
          
        # Sum nucleon of molecular targets for very rough approximation for Bradt-Peters formula, neglects molecular structure and
        # shadowing. Real cross section is most likely about a few percent higher for PS energy ranges (where they may be some nuclear 
        # resonances according to https://www.sciencedirect.com/science/article/pii/S0273117711007897)
        A_H2 = 2.0 * A_H
        A_H2O = 2.0 * A_H + A_O
        A_CO = A_C + A_O
        A_CH4 = A_C + 4.0 * A_H
        A_CO2 = A_C + 2.0 * A_O
        A_O2 = 2.0 * A_O

        # Compute cross section on targets
        # TO DO: already defined at start, instead make dictionary
        targets = ['H2', 'He', 'H20', 'H20', 'CO', 'CH4', 'CO2', 'O2', 'Ar']
        self.sigma_H2_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_H2)
        self.sigma_He_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_He)
        self.sigma_H2O_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_H2O)
        self.sigma_CO_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_CO)
        self.sigma_CH4_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_CH4)
        self.sigma_CO2_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_CO2)
        self.sigma_O2_N =  self.compute_inelastic_nuclear_cross_sections(self.A_p, A_O2)
        self.sigma_Ar_N = self.compute_inelastic_nuclear_cross_sections(self.A_p, A_Ar)
        
        # All sigmas for electron loss and electron capture
        sigmas_N = np.array([self.sigma_H2_N, self.sigma_H2O_N, self.sigma_CO_N, 
                             self.sigma_CH4_N, self.sigma_CO2_N, self.sigma_He_N, 
                             self.sigma_O2_N, self.sigma_Ar_N])
        
        # Create a list to store the non-zero fractions and corresponding cross sections
        non_zero_fractions = []
        non_zero_sigmas_N = []
        non_zero_target = []

        # Iterate over the fractions and cross sections
        for fraction, sigma_N, target in zip(self.fractions, sigmas_N, targets):
            # Check if the fraction is non-zero
            if fraction != 0:
                # Append the non-zero fraction and corresponding cross sections to the respective lists
                non_zero_fractions.append(fraction)
                non_zero_sigmas_N.append(sigma_N)
                non_zero_target.append(target)

        # Convert fractions to dictionary
        N_dict = {'Sigma_N': non_zero_sigmas_N, 'Targets': non_zero_target, 'Fraction': non_zero_fractions} 
    

        # Estimate the EL lifetime for collision with each gas type - if no gas or zero EL cross section, set lifetime to infinity
        tau_H2_N = 1.0/(self.sigma_H2_N *self.n_H2*self.beta*self.c_light) if self.n_H2 and self.sigma_H2_N != 0.0 else np.inf
        tau_He_N = 1.0/(self.sigma_He_N *self.n_He*self.beta*self.c_light) if self.n_He and self.sigma_He_N != 0.0 else np.inf 
        tau_H2O_N = 1.0/(self.sigma_H2O_N *self.n_H2O*self.beta*self.c_light) if self.n_H2O and self.sigma_H2O_N != 0.0 else np.inf 
        tau_CO_N = 1.0/(self.sigma_CO_N *self.n_CO*self.beta*self.c_light) if self.n_CO and self.sigma_CO_N != 0.0 else np.inf 
        tau_CH4_N = 1.0/(self.sigma_CH4_N *self.n_CH4*self.beta*self.c_light) if self.n_CH4 and self.sigma_CH4_N != 0.0 else np.inf 
        tau_CO2_N = 1.0/(self.sigma_CO2_N *self.n_CO2*self.beta*self.c_light) if self.n_CO2 and self.sigma_CO2_N != 0.0 else np.inf
        tau_O2_N = 1.0/(self.sigma_O2_N *self.n_O2*self.beta*self.c_light) if self.n_O2 and self.sigma_O2_N != 0.0 else np.inf
        tau_Ar_N = 1.0/(self.sigma_Ar_N *self.n_Ar*self.beta*self.c_light) if self.n_Ar and self.sigma_Ar_N != 0.0 else np.inf
        
        # Calculate the total cross section
        tau_tot_inv = 1/tau_H2_N + 1/tau_He_N + 1/tau_H2O_N + 1/tau_CO_N + 1/tau_CH4_N \
                    + 1/tau_CO2_N + 1/tau_O2_N + 1/tau_Ar_N 
        tau_tot = 1/tau_tot_inv
        
        sigmas_N_all = [self.sigma_H2_N, self.sigma_He_N, self.sigma_H2O_N, self.sigma_CO_N, self.sigma_CH4_N, self.sigma_CO2_N, self.sigma_O2_N, self.sigma_Ar_N]
        tau_N_all = [tau_H2_N, tau_He_N, tau_H2O_N, tau_CO_N, tau_CH4_N, tau_CO2_N, tau_O2_N, tau_Ar_N]
        print(f'\nCross sections with projectile {self.projectile}:')
        for i, sigma in enumerate(sigmas_N_all):
            if tau_N_all[i] != np.inf:
                print('{}: {:.4e} m^2, with tau = {:.4e}'.format(targets[i], sigma, tau_N_all[i]))
        print('--------------------------')
            
        return tau_tot, N_dict


    def calculate_elastic_emittance_growth_rate(self):
        """
        Calculates the normalized RMS emittance growth rate due to multiple elastic Coulomb scattering
        
        Parameters
        ----------

        Returns
        -------
        d_epsilon_dt : list
            normalized emittance growth rate in [m rad/s] in x and y for given projectile
        """
        if not hasattr(self, 'beta'):
            raise ValueError("Projectile beta must be set first!")
        
        
        # Classical proton radius
        r_p = constants.physical_constants['classical proton radius'][0]
                
        for plane in ['x', 'y']:
            
            # Get average beta function for specified plane
            beta_u = self.beta_avg[plane]
            i = 0
            for Z_t, n_t in zip(self.Z_t, self.fractions):
                if n_t > 0:
                    d_epsilon_dt = 2 * np.pi * self.gamma * beta_u * n_t * self.beta * constants.c * \
                                  (2 * self.Z_p * Z_t * r_p / (self.atomic_mass_in_u * self.beta**2 * self.gamma))**2 * \
                                  np.log(204 * Z_t**(-1/3))
                    print('de_{}/dt = {:.4e} for {} on {}'.format(plane, d_epsilon_dt, self.projectile, self.target_atoms[i]))
                else:
                    d_epsilon_dt = 0.0
                                  
                i += 1
        
        return d_epsilon_dt
    
    
    def calculate_full_elastic_emittance_growth(self):
        """
        Compute total emittance growth for all target gases in all machines, inversely adding the lifetimes

        Returns
        -------
        d_epsilon_dt: tuple
            total emittance growth rates due to elastic Coulomb scattering in X and Y
        """
        # Loop over all machines and projectiles
        for machine in ['LEIR', 'PS', 'SPS']:
            for projectile in self.all_possible_projectiles:
                self.set_projectile_and_machine(projectile=projectile, machine=machine)
                