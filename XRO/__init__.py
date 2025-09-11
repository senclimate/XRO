
# __init__.py for XRO

__version__ = "1.0.1"

from .core import XRO, gen_noise, gradient, variable_xarray_to_model, variable_model_to_xarray
from .visual import plot_above_below_shading, plot_fill_between, legend_combo
from .stats import *

