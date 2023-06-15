#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dataframe generator for the different CERN accelerators 
- made from Reyes Fern
"""
import pandas as pd

########### GAS FRACTIONS ############
data_gf = [['H2',0.83,0.9,0.905],
           ['H2O',0.02,0.1,0.035],
           ['CO',0.04,0.0,0.025],
           ['CH4',0.05,0.0,0.025],
           ['CO2',0.06,0.0,0.01],
           ['He', 0., 0., 0.],
           ['O2', 0., 0., 0.,],
           ['Ar', 0., 0., 0.]]
gas_fractions = pd.DataFrame(data_gf,columns=['Gas','LEIR','PS','SPS'])
gas_fractions = gas_fractions.set_index('Gas')
gas_fractions.to_csv('Gas_fractions.csv')


########### PRESSURE DATA - in mbar ########### 
# For PS, According to Jose Antonio Ferreira Somoza (CERN), for PS 
# to get the correct pressure for hydrogen we need to multiply it with a factor 2.4.
data_pres = [['LEIR',1e-11],
             ['PS',2.4*5e-10],  # factor 2.4 
             ['SPS',1e-8]]
pressure = pd.DataFrame(data_pres, columns=['Machine','Pressure'])
pressure = pressure.set_index('Machine')
pressure.to_csv('Pressure_data.csv')

########### INJECTION ENERGY, BETA and PRINCIPAL QUANTUM NUMBERS ###########
# Data frame with kinetic energy (MeV), beta and charge per machine and projectile, ionization potential(keV), princip
# Ionization energy and principal quantum number can be found here: https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html
data_projectile =  [['He1', 4.2,67.02, 5722.74, 0.094612,0.360286,0.99,1, 1, 2, 2, 24.6, 1],
                    ['He2', 4.2,245.39,12279.29,0.094612,0.611310,0.99,2, 2, 2, 2, 24.6, 1],
                    ['O4',  4.2,67.08, 5723.50, 0.094657,0.360590,0.99,4, 4, 8, 8, 13.67,2],
                    ['O8',  4.2,245.58,12280.12,0.094657,0.611680,0.99,8, 8, 8, 8, 13.67,2],
                    ['Mg6', 4.2,67.09, 5723.75, 0.094672,0.360686,0.99,6, 6, 12,12,7.618,2],  # updated n to 2 from NIST
                    ['Mg7', 4.2,90.24, 6812.68, 0.094672,0.411251,0.99,7, 7, 12,12,7.618,2],  # updated n to 2 from NIST
                    ['Pb54',4.2,72.13, 5974.37, 0.094647,0.372499,0.99,54,54,82,82,5.414,3]]
df_projectile = pd.DataFrame(data_projectile,columns=['Projectile','LEIR_Kinj','PS_Kinj','SPS_Kinj','LEIR_beta','PS_beta','SPS_beta','LEIR_q','PS_q','SPS_q','Z','I_p','n_0'])
df_projectile = df_projectile.set_index('Projectile')
df_projectile.to_csv('Projectile_data.csv')