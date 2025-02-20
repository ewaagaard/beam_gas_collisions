Beam Gas Collisions Documentation
===============================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   usage
   api

Installation
-----------
.. code-block:: bash

   pip install beam_gas_collisions

Basic Usage
----------
.. code-block:: python

   from beam_gas_collisions import IonLifetimes
   
   # Calculate lifetime for Pb54+ in PS
   PS = IonLifetimes(projectile='Pb54', machine='PS')
   tau = PS.calculate_total_lifetime_full_gas() 