# Beam Gas Collisions

Ion beams in the CERN accelerator complex suffer from heavy losses due to various effects. These effects include charge-changing interactions when charge beam projectiles collide with residual gas molecules in the beam pipe, which cause ions to fall outside the accelerator acceptance. Neglecting other incoherent beam dynamics effects, the beam intensity $I$ can simplistically be modelled with

$$
I(t) = I(t_0)\times\exp\bigg(- \frac{t}{\tau}\bigg),  \qquad \textrm{where} \quad \tau = \frac{1}{\sigma\ n\ \beta \ c }
$$

where $\sigma$ is the charge-changing cross section, $\beta$ is the projectile relativistic beta factor, $c$ is the speed of light and $n$ is the molecular density in the beam pipe. The dominant contributions to the cross section $\sigma$ at LEIR, PS and SPS ion beam energies are 
- **electron capture (EC)** via pair production
- **electron loss (EL)** via electron and nucleus impact ionization
  
Electron capture can be modelled with the [Schlachter formula](https://link.aps.org/doi/10.1103/PhysRevA.27.3372). We estimate the electron loss cross section with a semi-empirical formula from [G. Weber](http://repository.gsi.de/record/201624) (p. 74) for total electron loss cross section of many-electron ions penetrating through matter, which combines studies from [Dubois](https://link.aps.org/doi/10.1103/PhysRevA.84.022702) and [Shevelko](https://www.sciencedirect.com/science/article/pii/S0168583X11003272).

![Rest_gas_collisions](https://github.com/ewaagaard/Beam-gas-collisions/assets/68541324/a83c3b9f-f020-4385-9003-c60dccd68c14)

The `beam_gas_collisions`package contains two classes:
- The `IonLifetimes()` class calculates the total cross sections from EC and EL interactions, given projectile properties, fractional molecular rest gas composition and pressure. The ion beam lifetimes $\tau$ can also be calculated, assuming that no other relevant charge-changing processes.
- The `DataObject()` class, which contains pressure, projectile and rest gas composition data for LEIR, PS and SPS. Specifying the projectile and machine `IonLifetimes(projectile='Pb54', machine='PS')` will automatically load the relevant data.

A full review (currently under construction) and preliminary results of charge-changing interactions in the CERN accelerator complex can be found on this [link](https://www.overleaf.com/read/pvkmfbzrfnxk).

## Installing the package

To directly start calculating beam lifetimes and cross sections, create an isolated virtual environment and perform a local `pip` install to use the beam_gas_collisions freely. Once having cloned the `beam_gas_collisions` repository, run:

```python
conda create --name test_venv python=3.11 numpy pandas scipy matplotlib
conda activate test_venv
python -m pip install -e beam_gas_collisions
```
Then the different scripts in the folder calculations can be executed.

## Getting started

Thanks to the `DataObject()` class, pressure and projectile data is loaded automatically for most relevant ions in LEIR, PS and SPS. For instance:
```python
from beam_gas_collisions import IonLifetimes

# Instantiate PS class object
PS = IonLifetimes(projectile='Pb54', machine='PS')
tau_Pb54_PS = PS.calculate_total_lifetime_full_gas()  # gives Pb54+ ion lifetime under nominal vacuum conditions
```
All EC and EL cross sections are implicitly calculated for non-zero fractions of rest gas inside the `calculate_total_lifetime_full_gas()` method, we can also calculate the cross sections explicitly:

### Calculating the cross sections

The respective EL and EC cross sections can be calculated from
```python
sigma_EL = PS.calculate_sigma_electron_loss(Z, Z_p, q, e_kin, I_p, n_0)
sigma_EC = PS.calculate_sigma_electron_capture(Z, q, e_kin_keV)
```
where the parameters are 
- `Z` is the $Z$ of the target atom
- `Z_p` is the Z of the projectile
- `q` is the charge of projectile
- `e_kin` is the collision energy in MeV/u
- `e_kin_keV` is the collision energy in keV/u
- `I_p` is the first ionization potential of projectile in keV
- `n_0` is principle quantum number of outermost projectile electron. `n_0` and `I_0` for each ion can be found on the [NIST database](https://physics.nist.gov/PhysRefData/ASD/ionEnergy.html).

### Manually calculating lifetime and cross sections for new ions

The class object can also be used for a hypothetical accelerator, as long as pressure, fractional molecular rest gas composition and projectile properties are provided. For instance, for U28+ in a new accelerator with know vacuum conditions, we calculate the ion lifetime:

```python
import numpy as np

# Rest gas composition of ['H2', 'H2O', 'CO', 'CH4', 'CO2', 'He', 'O2', 'Ar']
gas_fractions = np.array([0.758, 0.049, 0.026, 0.119, 0.002, 0.034, 0.004, 0.008]) # relative fraction
p = 1e-10 #mbar

# Instantiate class for ion lifetimes
ion_beam_U28 = IonLifetimes(machine=None, p=p, molecular_fraction_array=gas_fractions)

# Projectile_data for U28+
Z_p = 92.
q_p = 28.
Ip = 0.93
n0 = 5.0 # from physics.nist.gov
atomic_mass_U238_in_u = 238.050787 # from AME2016 atomic mass table 
projectile_data_U28 = np.array([Z_p, q_p, E_kin, Ip, n0, atomic_mass_U238_in_u])

# Calculate lifetimes and cross sections
ion_beam_U28.set_projectile_data_manually(projectile_data_U28)
tau_U28 = ion_beam_U28.calculate_total_lifetime_full_gas()
```
If the relativistic $\beta$ is known, simply replace it with the atomic mass in the projectile data and select `beta_is_provided = True`:

```python
projectile_data_U28 = np.array([Z_p, q_p, E_kin, Ip, n0, beta_U238])
ion_beam_U28.set_projectile_data_manually(projectile_data_U28, beta_is_provided = True)
```

### Calculating the ion beam lifetime interacting with a single gas

Suppose we are interested in calculating the ion beam lifetime on a single gas, e.g. Pb54+ on Ar (with `Z_t=18`) at `5e-10` mbar. We simply instantiate the `IonLifetimes()` for PS (or manually for any other accelerator), and calculate the lifetime on argon at the provided pressures. 
```python
PS = IonLifetimes(projectile='Pb54', machine='PS')
taus_Pb54_on_Ar = PS.calculate_lifetime_on_single_gas(p=5e-10, Z_t=18)
```

### Cross sections across the CERN accelerator complex 

In the module `examples/001_leir_ps_sps_lifetimes.py`, all the cross sections and total ion lifetimes are calculated for some considered future ions, as shown in these plots below. 

![Sigmas_on_H2_all_ions](https://github.com/ewaagaard/Beam-gas-collisions/assets/68541324/6bbd7e51-6faf-4d90-8ddc-1db29bf40a11)

![Rest_gas_lifetime_all_ions](https://github.com/ewaagaard/Beam-gas-collisions/assets/68541324/60f967a8-7bc4-4458-88f2-b27820b5976a)


