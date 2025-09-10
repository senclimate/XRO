[![](XRO_logo.png)](https://github.com/senclimate/XRO)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10681114.svg)](https://doi.org/10.5281/zenodo.10681114)
======

## Description

The **XRO** is an e**X**tended nonlinear **R**echarge **O**scillator model for El Niño-Southern Oscillation (ENSO) and other modes of variability in the global oceans. It builds on the legacies of the Hasselmann stochastic climate model capturing upper ocean memory in SST variability, and the recharge oscillator model for the oscillatory core dynamics of ENSO. It constitutes a parsimonious representation of the climate system in a reduced variable and parameter space that still captures the essential dynamics of interconnected global climate variability. 

This is updated version of XRO, with the model equations can be written as:

$$
\frac{d}{dt} \begin{pmatrix} X_{\text{ENSO}} \\ X_M \end{pmatrix} = L \begin{pmatrix} X_{\text{ENSO}} \\ X_M \end{pmatrix} + \begin{pmatrix} N_{\text{ENSO}} \\ N_M \end{pmatrix} + \sigma_{\xi} G \xi, \quad (1)
$$


$$
\frac{d\xi}{dt} = -r_{\xi} \xi + w(t), \quad (2)
$$


where 


$$
X_{\text{ENSO}} = [T_{\text{ENSO}}, h ]
$$


and 


$$
X_M = [T_{\text{NPMM}}, T_{\text{SPMM}}, T_{\text{IOB}}, T_{\text{IOD}}, T_{\text{SIOD}}, T_{\text{TNA}}, T_{\text{ATL3}}, T_{\text{SASD}}]
$$


are state vectors of ENSO and other climate modes, respectively. This model allows for two-way interactions between ENSO and the other modes. $L$, $N$, $G(X)$ describe linear, nonlinear, multiplicative noise dynamics. For the detailed formulation of XRO model, please refer to our paper Zhao et al. (2024)[[1]](#1) in Nature ([v0.1](https://github.com/senclimate/XRO/tree/v0.1)). 

This repository hosts the [python package](https://github.com/senclimate/XRO) for `XRO` model. We have designed `XRO` to be user-friendly, aiming to be a valuable tool not only for research but also for operational forecasting and as an educational resource in the classroom. We hope that XRO proves to be both a practical and accessible tool that enhances your research and teaching experiences. 

If you encounter problems in running `XRO` or have questions, please feel free to contact Sen Zhao (zhaos@hawaii.edu).

---

## Installation
You can install XRO in two ways:

#### From PyPI

```pip install XRO```

#### From GitHub (latest development version)

```pip install git+https://github.com/senclimate/XRO.git```

## XRO functionalities

[`XRO`](https://github.com/senclimate/XRO) model is implemented in `python` with dependencies on only [`numpy`](https://numpy.org/) and [`xarray`](https://docs.xarray.dev/en/stable/). Key functionalities include:

- `XRO.fit_matrix` is a precedure to train nonlinear XRO parameters from the observational and climate model outputs
- `XRO.simulate` is a precedure to perform stochastic simulations with the trained parameters
- `XRO.reforecast`is a precedue to perform reforecasting or forecasting using the trained parameters and initial condictions


## Quick Start 

`XRO_Cookbook.ipynb` is a Jupyter Notebook showing how to use `XRO` and reproduce the analysis of Zhao et al. 2024[[1]](#1). To successfully run the example, these open-source python modules may be necessary: [`numpy`](https://numpy.org/), [`xarray`](https://docs.xarray.dev/en/stable/), [`climpred`](https://climpred.readthedocs.io/en/stable/), [`matplotlib`](https://matplotlib.org/), and [`datetime`](https://docs.python.org/3/library/datetime.html). 

---

## Applications

- The repository [**Recharge Oscillator (RO) Practical**](https://github.com/senclimate/RO_practical) for the [ENSO Winter School 2025](https://sites.google.com/hawaii.edu/enso-winter-school-2025/). The practical covers theoretical and computational aspects of the RO framework, its applications in ENSO simulations, and forecasting. When other climate modes are not considered, the `XRO` simplifies to the `RO`, making it well-suited for use in this practical context.

---
## Acknowledgement

Kindly requested to cite our paper Zhao et al. (2024) [[1]](#1) and code [[2]](#2) if use the XRO model in your published works.

## References
<a id="1">[1]</a> 
Zhao, S., Jin, F.-F., Stuecker, M.F., Thompson, P.R., Kug, J.-S., McPhaden, M.J., Cane, M.A., Wittenberg, A.T., Cai, W., (2024). Explainable El Niño predictability from climate mode interactions. Nature. https://doi.org/10.1038/s41586-024-07534-6 

<a id="2">[2]</a> 
Zhao, S. (2024). Extended nonlinear Recharge Oscillator (XRO) model for "Explainable El Niño predictability from climate mode interactions". Zenodo. https://doi.org/10.5281/zenodo.10681114
