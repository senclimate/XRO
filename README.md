[![](XRO_logo.png)](https://github.com/senclimate/XRO)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10681114.svg)](https://doi.org/10.5281/zenodo.10681114)
======

## Description

The **XRO** is an e**X**tended nonlinear **R**echarge **O**scillator model for El Niño-Southern Oscillation (ENSO) and other modes of variability in the global oceans. It builds on the legacies of the Hasselmann stochastic climate model capturing upper ocean memory in SST variability, and the recharge oscillator model for the oscillatory core dynamics of ENSO. It constitutes a parsimonious representation of the climate system in a reduced variable and parameter space that still captures the essential dynamics of interconnected global climate variability. 

For the detailed formulation of XRO model, please refer to our paper Zhao et al. (2024)[[1]](#1), currently under review in Nature.

This repository hosts the [python package](https://github.com/senclimate/XRO) for `XRO` model. We have designed `XRO` to be user-friendly, aiming to be a valuable tool not only for research but also for operational forecasting and as an educational resource in the classroom. We hope that XRO proves to be both a practical and accessible tool that enhances your research and teaching experiences. 

If you encounter problems in running `XRO` or have questions, please feel free to contact Sen Zhao (zhaos@hawaii.edu).

## XRO functionalities

[`XRO`](https://github.com/senclimate/XRO) model is implemented in `python` with dependencies on only [`numpy`](https://numpy.org/) and [`xarray`](https://docs.xarray.dev/en/stable/). Key functionalities include:

- `XRO.fit_matrix` is a precedure to train XRO parameters from the observational and climate model outputs
- `XRO.simulate` is a precedure to perform stochastic simulations with the trained parameters
- `XRO.reforecast`is a precedue to perform reforecasting or forecasting using the trained parameters and initial condictions

## Quick Start 

`XRO_Cookbook.ipynb` is a Jupyter Notebook showing how to use `XRO` and reproduce the analysis of Zhao et al. 2024[[1]](#1). To successfully run the example, these open-source python modules may be necessary: [`numpy`](https://numpy.org/), [`xarray`](https://docs.xarray.dev/en/stable/), [`climpred`](https://climpred.readthedocs.io/en/stable/), [`matplotlib`](https://matplotlib.org/), and [`datetime`](https://docs.python.org/3/library/datetime.html). 


## Acknowledgement

Kindly requested to cite the paper Zhao et al. (2024) [[1]](#1) and the code [[2]](#2) if use the XRO model in your published works.


## References
<a id="1">[1]</a> 
Zhao, S., Jin, F.-F., Stuecker, M.F., Thompson, P.R., Kug, J.-S., McPhaden, M.J., Cane, M.A., Wittenberg, A.T., Cai, W., (Under review). Explainable El Niño predictability from climate mode interactions. Nature.

<a id="2">[2]</a> 
Zhao, S. (2024). Extended nonlinear Recharge Oscillator (XRO) model for "Explainable El Niño predictability from climate mode interactions". Zenodo. https://doi.org/10.5281/zenodo.10681114
