from setuptools import setup, find_packages

setup(
    name="beam_gas_collisions",
    packages=find_packages(),
    include_package_data=True,
    package_data={'beam_gas_collisions': ['data/*.csv']},
) 