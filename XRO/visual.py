import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

###########################################

def time_axis(tim, offset=False, freq='MS'):
    """
    Convert xarray time coordinates into a continuous numeric time axis.

    Parameters
    ----------
    tim : xarray.DataArray or pandas.DatetimeIndex
        Time coordinate array with datetime values.
    offset : bool, optional, default=False
        If True, subtracts the first year from the time axis to start at 0.
    freq : {'MS', 'D'}, optional, default='MS'
        Frequency used for conversion:
        - 'MS': Monthly start, fractional year based on month.
        - 'D': Daily, fractional year based on day of year.

    Returns
    -------
    xarray.DataArray
        Numeric time axis in fractional years.
    """
    if freq=='MS':
        xtime = tim.dt.year + (tim.dt.month-0.5)/12.
        if offset:
            xtime = xtime - tim.dt.year[0]
    elif freq=='D':
        xtime = tim.dt.year + (tim.dt.dayofyear-0.5)/365
        if offset:
            xtime = xtime - tim.dt.year[0]
    return xtime


def plot_above_below_shading(x_ds, above=0.5, below=-0.5, c='black',
                             above_c='#E5301D', below_c='#301D80', alpha_fill=0.8,
                             xtime=None, ax=None,
                             **kwargs):
    """
    Plot a time series with shaded regions for values above and below thresholds.

    Parameters
    ----------
    x_ds : xarray.DataArray or array-like
        Time series data to plot.
    above : float, optional, default=0.5
        Threshold above which values are shaded with `above_c`.
    below : float, optional, default=-0.5
        Threshold below which values are shaded with `below_c`.
    c : str, optional, default='black'
        Color of the line plot.
    above_c : str, optional, default='#E5301D'
        Fill color for values above the threshold.
    below_c : str, optional, default='#301D80'
        Fill color for values below the threshold.
    alpha_fill : float, optional, default=0.8
        Transparency of shaded regions.
    xtime : array-like, optional
        Pre-computed time axis. If None, will be derived from `x_ds.time`.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    **kwargs
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    if xtime is None:
        x_axis = time_axis(x_ds.time)
    else:
        x_axis = xtime

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    ax.plot(x_axis, x_ds, c=c, **kwargs)
    ax.fill_between(x_axis, x_ds, above, where=x_ds>=above, interpolate=True, color=above_c, alpha=alpha_fill)
    ax.fill_between(x_axis, x_ds, below, where=x_ds<=below, interpolate=True, color=below_c, alpha=alpha_fill)
    return ax


def plot_fill_between(x_ds, dim='member', c='orangered', alpha=0.2, xtime=None, ax=None, option=None, **kwargs):
    """
    Plot ensemble mean with shaded uncertainty (standard deviation or quantiles).

    Parameters
    ----------
    x_ds : xarray.DataArray
        Ensemble time series data.
    dim : str, optional, default='member'
        Dimension over which to compute ensemble statistics.
    c : str, optional, default='orangered'
        Color of the mean line and shaded region.
    alpha : float, optional, default=0.2
        Transparency of shaded region.
    xtime : array-like, optional
        Pre-computed time axis. If None, will be derived from `x_ds.time`.
    ax : matplotlib.axes.Axes, optional
        Axis to plot on. If None, a new figure and axis are created.
    option : float, optional
        If None, uncertainty is ±1 standard deviation. Otherwise, interpreted as
        a quantile (e.g., 0.1 for 10–90% interval).
    **kwargs
        Additional keyword arguments passed to `ax.plot`.

    Returns
    -------
    matplotlib.axes.Axes
        Axis containing the plot.
    """
    if xtime is None:
        x_axis = time_axis(x_ds.time)
    else:
        x_axis = xtime

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(8, 4))

    if option is None:
        x_lo = x_ds.mean(dim) - x_ds.std(dim)
        x_up = x_ds.mean(dim) + x_ds.std(dim)
    else:
        x_lo = x_ds.quantile(option, dim=dim)
        x_up = x_ds.quantile(1-option, dim=dim)
        
    ax.plot(x_axis, x_ds.mean(dim), c=c, **kwargs)
    ax.fill_between(x_axis, x_lo, x_up, fc=c, alpha=alpha)
    return ax


def _unique_ordered(elements):
    """
    Return unique elements in the order of first appearance.

    Parameters
    ----------
    elements : list
        Input list of elements.

    Returns
    -------
    list
        Unique elements preserving original order.
    """
    seen = set()
    result = []
    for element in elements:
        if element not in seen:
            seen.add(element)
            result.append(element)
    return result


def legend_combo(ax, reverse=False, **kwargs):
    """
    Combine duplicate legend labels into grouped entries.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axis containing plotted elements with labels.
    reverse : bool, optional, default=False
        If True, reverse the legend order.
    **kwargs
        Additional keyword arguments passed to `ax.legend`.

    Returns
    -------
    None
        Modifies the legend of the given axis in place.
    """
    handler, labeler = ax.get_legend_handles_labels()
    hd = []
    labli = _unique_ordered(labeler)
    for lab in labli:
        comb = [h for h,l in zip(handler,labeler) if l == lab]
        hd.append(tuple(comb))
    if reverse:
        ax.legend(hd[::-1], labli[::-1], **kwargs)
    else:
        ax.legend(hd, labli, **kwargs)
