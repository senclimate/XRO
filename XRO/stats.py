import numpy as np
import xarray as xr


######################################################################
def compute_skew(da, dim):
    """Compute skewness along the specified dimension."""
    mean_da = da.mean(dim=dim)
    std_da = da.std(dim=dim)
    skewness = ((da - mean_da) / std_da) ** 3
    return skewness.mean(dim=dim)

@xr.register_dataarray_accessor("skew")
@xr.register_dataset_accessor("skew")
class SkewAccessor:
    '''
        calculate the skewness using xarray 
    '''
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, dim='time'):
        """
        Compute the skewness along the given dimension.
        """
        if isinstance(self._obj, xr.DataArray):
            return compute_skew(self._obj, dim)

        elif isinstance(self._obj, xr.Dataset):
            skews = {}
            for name, da in self._obj.data_vars.items():
                skews[name] = compute_skew(da, dim)
            return xr.Dataset(skews)

class GroupBySkewAccessor:
    def __init__(self, groupby_object):
        self._groupby = groupby_object

    def __call__(self, dim='time'):
        if isinstance(self._groupby._obj, (xr.Dataset, xr.DataArray)):
            return self._groupby.map(lambda x: compute_skew(x, dim=dim))

        raise TypeError("The groupby object must be a DatasetGroupBy or DataArrayGroupBy")

xr.core.groupby.DataArrayGroupBy.skew = property(lambda self: GroupBySkewAccessor(self))
xr.core.groupby.DatasetGroupBy.skew = property(lambda self: GroupBySkewAccessor(self))

######################################################################

def compute_kurtosis(da, dim):
    """Compute kurtosis along the specified dimension."""
    mean_da = da.mean(dim=dim)
    std_da = da.std(dim=dim)
    kurt = ((da - mean_da) / std_da) ** 4
    return kurt.mean(dim=dim) - 3  # Excess kurtosis
    
@xr.register_dataarray_accessor("kurt")
@xr.register_dataset_accessor("kurt")
class KurtAccessor:
    """
    Calculate the kurtosis using xarray.
    """
    def __init__(self, xarray_obj):
        self._obj = xarray_obj

    def __call__(self, dim='time'):
        """
        Compute kurtosis along the given dimension.
        """
        if isinstance(self._obj, xr.DataArray):
            return compute_kurtosis(self._obj, dim)

        elif isinstance(self._obj, xr.Dataset):
            kurts = {}
            for name, da in self._obj.data_vars.items():
                kurts[name] = compute_kurtosis(da, dim)
            return xr.Dataset(kurts)

class GroupByKurtAccessor:
    def __init__(self, groupby_object):
        self._groupby = groupby_object

    def __call__(self, dim='time'):
        if isinstance(self._groupby._obj, (xr.DataArray, xr.Dataset)):
            return self._groupby.map(lambda x: compute_kurtosis(x, dim=dim))
        raise TypeError("The groupby object must be a DatasetGroupBy or DataArrayGroupBy")

xr.core.groupby.DataArrayGroupBy.kurt = property(lambda self: GroupByKurtAccessor(self))
xr.core.groupby.DatasetGroupBy.kurt = property(lambda self: GroupByKurtAccessor(self))

######################################################################

def _xcorrnan(x, y, demean=True, maxlags=12):
    '''
    %find the cross correlation with missing values
    %formula is c(k) = 1/(N-k) * sum((X(t)-X')(Y(t+k)-Y'))/(std(X) * std(Y)) for k = 0,1,......,N-1
    %           c(k) = 1/(N-k) * sum((Y(t)-Y')(X(t+k)-X'))/(std(X) * std(Y)) for k =-1,-2......,-(N-1)
    % Input parameters:
    %        X, Y : two vectors among which the cross correlation is to be computed (type: real)
    %       maxlags (optional): maximum time lag it will look for. default value is minimum length among X and Y.
    % Output parameters:
    %        c: cross correlation score
    %        lags: corresponding lags of c values
    %        cross_cov: cross covariance score
    credit from https://github.com/kleinberg-lab/FLK-NN/blob/master/xcorr_w_miss.m
    '''
    Nx = len(x)
    if Nx != len(y):
        raise ValueError('x and y must be equal length')

    if demean:
        x = np.array(x - np.nanmean(x))
        y = np.array(y - np.nanmean(y))

    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    x_std = np.nanstd(x)
    y_std = np.nanstd(y)

    lags = np.arange(-maxlags, maxlags + 1, dtype=np.int32)
    c = np.full_like(lags, np.nan, dtype=float)
    cross_cov = np.full_like(lags, np.nan, dtype=float)

    for k in lags:
        if k > 0:
            tempCrossCov = np.nanmean(y[:-k] * x[k:])
        elif k == 0:
            tempCrossCov = np.nanmean(x * y)
        else:
            tempCrossCov = np.nanmean(x[:k] * y[-k:])
        cross_cov[k+maxlags] = tempCrossCov
        c[k+maxlags] = tempCrossCov/(x_std*y_std)
    xr_c = xr.DataArray(c, coords=[lags], dims=["lag"])
    return xr_c


def xcorr(xr_var1, xr_var2, maxlags=12, units='month', dim='time'):
    '''
    xarray lead-lag correlations
    '''
    lags = np.arange(-maxlags, maxlags + 1, dtype=np.int32)
    lags_out = xr.DataArray(lags, coords={'lag': lags}, dims=['lag'], name=['lag'],
                            attrs={'long_name': 'lag', 'units': units, '_FillValue': -32767.})  # 'axis': 'T',

    xr_corr = xr.apply_ufunc(_xcorrnan, xr_var1, xr_var2,
                             input_core_dims=[[dim], [dim]],
                             output_core_dims=[['lag']],
                             kwargs={'maxlags': maxlags},
                             vectorize=True)
    xr_corr = xr_corr.assign_coords({"lag": lags_out})
    xr_corr.encoding['_FillValue'] = 1.e+20
    xr_corr.encoding['dtype'] = 'float32'
    return xr_corr
    

######################################################################

def pmtm(ds_x, dim='time', dt=1/12., nw=4, cl=0.95, nfft=None, scale_by_freq=True, lag1_r=None,
         return_dataset=False):
    """
    Thomson’s multitaper power spectral density (PSD) estimate, 
    PSD significance level estimate, Confidence Intervals

    PSD estimate modified from Peter Huybers's matlab code, pmtmPH.m
    PSD significance level estimate modified from NCAR/NCL specx_ci 

    Parameters
    ----------
    x : numpy array
        Time series to analyze
    dt : float
        Sampling interval, default is 1.
    nw : float
        The time bandwidth product, available 
    cl : float
        Confidence interval to calculate and display
    nfft: int
        Default 
    lag1_r: float
        significance test for AR(1) lag1_r value
    return_dataset: True or False
        return xarray dataset

    Returns
    -------
    x_P : xarray DataArray
        PSD estimate via the multi-taper method.
    x_P_sig : xarray DataArray
        PSD significance level estimate.
    x_ci : xarray DataArray
        Confidence intervals.
    """
    P_ds, s_ds, Psig_ds, ci_ds = xr.apply_ufunc(_pmtm, ds_x, input_core_dims=[[dim]],
                             output_core_dims=[['freq'], ['freq'], ['freq'], ['freq', 'bound', ]],
                             kwargs={'dt': dt, 'nw': nw, 'cl': cl,
                                     'nfft': nfft, 'scale_by_freq': scale_by_freq, 'lag1_r': lag1_r},
                             vectorize=True)
    if len(s_ds.dims)==1:
        P_ds = P_ds.assign_coords({'freq': s_ds})
        Psig_ds = Psig_ds.assign_coords({'freq': s_ds})
        ci_ds = ci_ds.assign_coords({'freq': s_ds})
    else:
        mem_dim = list(s_ds.dims)[0]
        P_ds = P_ds.assign_coords({'freq': s_ds.isel({mem_dim: 0}).values})
        Psig_ds = Psig_ds.assign_coords({'freq': s_ds.isel({mem_dim: 0}).values})
        ci_ds = ci_ds.assign_coords({'freq': s_ds.isel({mem_dim: 0}).values})

    if return_dataset:
        out_ds = xr.Dataset({'P': P_ds, 'Psig': Psig_ds, 'Pci': ci_ds})
        out_ds.attrs['dt'] = dt
        out_ds.attrs['nw'] = nw
        out_ds.attrs['cl'] = cl
        out_ds.attrs['scale_by_freq'] = scale_by_freq
        return out_ds
    else:
        return P_ds, Psig_ds, ci_ds



def _pmtm(x, dt=1, nw=4, cl=0.95, nfft=None, scale_by_freq=True):
    '''
        numpy of pmtm
    '''

    from scipy.signal import windows
    from scipy.stats import chi2

    if nfft is None:
        nfft = np.shape(x)[0]

    nx = np.shape(x)[0]
    k = min(np.round(2.*nw), nx)
    k = int(max(k-1, 1))
    s = np.arange(0, 1/dt, 1/(nfft*dt))

    # Compute the discrete prolate spheroidal sequences
    [E, V] = windows.dpss(nx, nw, k, return_ratios=True)
    E = E.T

    # Compute the windowed DFTs.
    Pk = np.abs(np.fft.fft(E*x[:, np.newaxis], nfft, axis=0))**2

    # Iteration to determine adaptive weights
    if k > 1:
        sig2 = np.dot(x[np.newaxis, :], x[:, np.newaxis])[0][0]/nx # power
        # initial spectrum estimate
        P = ((Pk[:, 0] + Pk[:, 1])/2)[:, np.newaxis]
        Ptemp = np.zeros((nfft, 1))
        P1 = np.zeros((nfft, 1))
        tol = .0005*sig2/nfft
        a = sig2*(1-V)

        while (np.sum(np.abs(P - P1)/nfft) > tol):
            b = np.repeat(P, k, axis=-1)/(P*V[np.newaxis, :] + np.ones((nfft, 1))*a[np.newaxis, :])
            wk = (b**2) * (np.ones((nfft, 1))*V[np.newaxis, :])
            P1 = (np.sum(wk*Pk, axis=-1)/np.sum(wk, axis=-1))[:, np.newaxis]

            Ptemp = np.empty_like(P1)
            Ptemp[:] = P1
            P1 = np.empty_like(P)
            P1[:] = P
            P = np.empty_like(Ptemp)
            P[:] = Ptemp

        # Determine equivalent degrees of freedom, see Percival and Walden 1993.
        v = ((2*np.sum((b**2)*(np.ones((nfft, 1))*V[np.newaxis, :]), axis=-1)**2) /
             np.sum((b**4)*(np.ones((nfft, 1))*V[np.newaxis, :]**2), axis=-1))

    else:
        P = np.empty_like(Pk)
        P[:] = Pk
        v = 2*np.ones((nfft, 1))

    select = (np.arange(0, (nfft + 1)/2.)).astype('int')
    P = P[select].flatten()
    s = s[select].flatten()
    v = v[select].flatten()

    # Whether the resulting density values should be scaled by the scaling frequency, which gives density in units of 1/day, 1/year, or 1/Hz.
    # This allows for integration over the returned frequency values
    if scale_by_freq:
        P *= dt

    # Chi-squared 95% confidence interval
    # approximation from Chamber's et al 1983; see Percival and Walden p.256, 1993
    ci = np.empty((np.shape(v)[0], 2))
    ci[:, 0] = 1./(1-2/(9*v) - 1.96*np.sqrt(2/(9*v)))**3
    ci[:, 1] = 1./(1-2/(9*v) + 1.96*np.sqrt(2/(9*v)))**3

    # red noise significance levels
    # PSD significance level estimate modified from NCAR/NCL specx_ci 
    alpha = np.corrcoef(x[:-1], x[1:])[0, 1]
    dof = v
    P_red = (1 - alpha**2) / (1 + alpha**2 - 2 * alpha * np.cos(2 * np.pi * s * dt))
    # rescale
    scale = np.sum(P)/np.sum(P_red)
    quantile = chi2.ppf(cl, dof)
    P_sig = P_red * scale * quantile / dof

    x_s = xr.DataArray(s, dims=['freq'], coords={'freq': s})
    x_P = xr.DataArray(P, dims=['freq'], coords={'freq': s})
    x_P_sig = xr.DataArray(P_sig, dims=['freq'], coords={'freq': s})
    x_ci = xr.DataArray(ci, dims=['freq', 'bound'], coords={'freq': s, 'bound': [0, 1]})
    return x_P, x_s, x_P_sig, x_ci

