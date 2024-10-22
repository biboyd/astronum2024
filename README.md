# Analysis for Astronum Proceedings 2024
Looking to extend some of the quick mixing analaysis inlcuded in the main paper.

* Add source term for Na23 from carbon burning
* Look at the impact of altering the convection size
* Make extendable to various configurations of initial model
* Add analysis for looking at other Urca pairs (e.g. A=21, A=25)

## Current working stuff
Rn we have one script and a couple of notebooks that were more exploratory and the scripts are now based off of.

### main script
So the current working script is `calc_optimal_ratio_v_Rconv.py`. This script takes in an init model and calcs:

* the quick mixing limit ratio for a range of Rconv values
* saves the array of ratios and the corresponding Rconv,Mconv values
* Makes a simple plot of ratios vs Rconv and Mconv.

This script assumes there is not ingestion of material from a growing convection zone (though this can be changed easily). 
Additionally, it assumes that ${}^{23}\mathrm{Na}$ is constanly being added by carbon burning at a rate of `7.17668016e+25 g/s`
this value is based on a large scale 3D simulation.

### jupyter notebooks
There are two notebooks, `estimated_a23_losses.ipynb` and `smarter_quick_mixing.ipynb`. The first notebook is looking at how different ratios of the Urca pair relates to the neutrinos losses one sees. And there is quite a bit of comparison to a 3D simulation model. In this notebook there is some attempts and finding the ratio in the limit of very quick mixing which would define a uniform composition. There is also some work trying to build off this to see how the ratio may change with convection zone size (though this is done kinda slopily).
At the end of the notebook there is some work to extend this to the A=21 Urca pair as well as the beginning of building a better calculation that accounts for the sources of A=23 via Carbon burning.

The Second notebook is work to create a more calculation of the quick mixing limit when we take into account the sources related to carbon burning or a growing convection zone. This results in a drastically different equilibrium ratio that can be close to an order of magnitude smaller in some cases. This uses a minimization algorithm to solve an equation analytically. The mathematical basis for this is laid out in the notebook but largely comes down to tracking the rate at which mass is changing for each urca nuclei, and finding an equilibrium Mass ratio such that the change in mass ratio is zero.

