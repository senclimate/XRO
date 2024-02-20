############################################################################
#
#  XRO: A nonlinear e*X*tended *R*echarge *O*scillator model 
#
############################################################################

import numpy as np
import xarray as xr

class XRO(object):
    """
        class XRO object
    """
    #---------------------------------
    def __init__(self, ncycle=12, ac_order=2, is_forward=True, taus=None, maxlags=2):
        """
        initialize the XRO
            ncycle  : cycles per year, ncycle=12 is monthly data, ncycle=52 is weekly data, ncycle=365 is daily data
            ac_order: degree of accountted seasonal cycle, 0: annual mean, 1: annual cycle, 2: semi-annual cycle
            is_forward: use forward differencing when calculating the gradient, otherwise using center differencing
            taus    : [1, 1, 1] time step for lag corvariance for each ac_order
            noise fitting parameters:
            maxlags : maxlags is the number lags to fit the noise memory by treating the residual as red noise
        """
        if taus is None:
            taus = np.zeros(
                ac_order+1, dtype=int) if is_forward else np.ones(ac_order+1, dtype=int)

        if len(taus) <= ac_order:
            raise ValueError(
                "Length of 'taus' must be larger than 'ac_order'!")

        self.ncycle = ncycle
        self.omega = 2 * np.pi    # annual freqeucny (year^-1)
        self.ac_order = ac_order
        self.is_forward = is_forward
        self.taus = taus
        self.maxlags = maxlags

    #---------------------------------
    def _preprocess(self, X, Y, time=None, is_remove_nan=True):
        '''
            preprocess the X, Y
            remove
        '''
        rank_x, ntim_x = X.shape
        rank_y, ntim_y = Y.shape

        # assure that x and y has same time axis
        if ntim_x != ntim_y:
            raise ValueError(
                f'Input X {ntim_x} and Y {ntim_y} must have the same time dimension!')
        else:
            ntim = ntim_x

        # time axis value
        if time is None:
            tim = np.arange(1./self.ncycle/2., ntim/self.ncycle,
                            step=1./self.ncycle, dtype=float)
        else:
            tim = time

        # cycle ids (1...ncycle)
        nyear = int(np.ceil(len(tim)/self.ncycle))
        cycle_idx = xr.DataArray(np.array(list(range(1, self.ncycle+1))*nyear)[0:len(tim)],
                                 dims=['time'], coords={'time': tim})

        # remove nan in X Y's time domain, and select the tim and cycle_idx consistently
        if is_remove_nan:
            try:
                ind = np.isfinite(X[1, :] + Y[1, :])
            except:
                ind = np.isfinite(X[0, :] + Y[0, :])

            X = X[:, ind]
            Y = Y[:, ind]
            cycle_idx = cycle_idx.loc[ind]
            tim = tim[ind]
            ntim = X.shape[1]

        # prepare the time arrays with same shape of y and x
        t2d_y = np.zeros(shape=Y.shape)
        for ir in range(rank_y):
            t2d_y[ir, :] = tim
        t2d_x = np.zeros(shape=X.shape)
        for ir in range(rank_x):
            t2d_x[ir, :] = tim

        # time step: delta time [make sure no nan in the first two samples]
        delta_tim = tim[1] - tim[0]

        # cycle DataArray
        # this can dealing with the time series that started from any calendar month
        _cyls = np.arange(tim[0], tim[0]+1, step=1./self.ncycle, dtype=float)
        cycle = xr.DataArray(_cyls, dims={'cycle': _cyls}, coords={
                             'cycle': _cyls}, name='cycle')
        cycle.attrs['long_name'] = 'cycle'

        # _cyls_shift since forward differencing has deltaT/2 shift
        if self.is_forward:
            cycle_shift = _cyls - delta_tim*0.5
        else:
            cycle_shift = _cyls

        # rank of ac_order + 1
        _ac_rank = np.arange(0, self.ac_order+1, step=1, dtype=np.int32)
        ac_rank = xr.DataArray(_ac_rank, dims={'ac_rank': _ac_rank}, coords={
                               'ac_rank': _ac_rank}, name='ac_rank')
        ac_rank.attrs['long_name'] = 'ac_rank'
        ac_rank.encoding['dtype'] = 'int32'

        # cos()
        _cossin = np.arange(0, 2 * self.ac_order + 1, step=1, dtype=np.int32)
        cossin = xr.DataArray(_cossin, dims={'cossin': _cossin}, coords={
                              'cossin': _cossin}, name='cossin')
        cossin.attrs['long_name'] = 'coeff cos(nωt): 0-ac_order; coeff for sin(nωt): ac_order+1 ~ 2*ac_order + 1'
        cossin.encoding['dtype'] = 'int32'

        return X, Y, tim, ntim, cycle_idx, t2d_y, t2d_x, delta_tim, cycle, cycle_shift, ac_rank, cossin

#---------------------------------
    def __compute__(self, X, Y, time=None, is_remove_nan=True):
        """
        Core XRO fitting precedure

        Compute the operators L from Y = L * X + ξ

        Considering the following approximation for the linear and periodic operator:

        L=L_0+L_1^c  cos⁡〖(ωt)〗+L_2^c  cos⁡〖(2ωt)〗+L_3^c  cos⁡〖(3ωt)〗
          +L_1^s  sin⁡〖(ωt)〗+L_2^s  sin⁡〖(2ωt)〗+L_3^s  sin⁡〖(3ωt)〗

        G_n^c=cos⁡(nωt)Y(t)*X^T (t-d)
        G_n^s=sin⁡(nωt)Y(t)*X^T (t-d)
        C_n^c=cos⁡(nωt)X(t)*X^T (t-d)
        C_n^s=sin⁡(nωt)X(t)*X^T (t-d)

        To solve the matrix:
        G_n^c=∑_(m=0)^3〖L_m^c*((C_(n+m)^c+C_(n-m)^c)/2) 〗+∑_(m=1)^3〖L_m^s*((C_(n+m)^s-C_(n-m)^s)/2) 〗, n=0,1,2,3
        G_n^s=∑_(m=0)^3〖L_m^c*((C_(n+m)^s+C_(n-m)^s)/2) 〗+∑_(m=1)^3〖L_m^s*((C_(n-m)^c-C_(n+m)^c)/2) 〗, n=1,2,3
        L=G*C^(-1)

        This precedure is equivalent to linear regression but considering the time lag [taus] in covariance.
        See the fitting considering annual cycle in Supporting Information of Zhao et al. (2019)
        See details of a specific case (Y=dX/dt) in Appendix of Chen and Jin (2021)

        Zhao, S., Jin, F.-F., & Stuecker, M. F. (2019). Improved Predictability of the Indian Ocean Dipole Using Seasonally
        Modulated ENSO Forcing Forecasts. Geophysical Research Letters, 46(16), 9980–9990. https://doi.org/10.1029/2019GL084196

        Chen, H.-C., & Jin, F.-F. (2021). Simulations of ENSO Phase-Locking in CMIP5 and CMIP6. Journal of Climate, 
        34(12), 5135–5149. https://doi.org/10.1175/JCLI-D-20-0874.1

        INPUTs: 
            Y [rank_y, ntim]: Y samples 
            X [rank_x, ntim]: X samples
            time [ntim]: time indices, float, must be continuious but can has nan in X and Y 

        Returns: xarray.Dataset
            Lcoef  [rank_y, rank_x, cossin]   : L opeator coefficents 
            Lcomp  [rank_y, rank_x, ncycle, ac_order+1] : L opeator components (annual mean, annual cycle, semi-annual, ...)
            Lac    [rank_y, rank_x, ncycle]             : L opeator (total sum)
            Y [rank_y, ntim*]                      : Y for non-nan values
            X [rank_x, ntim*]                      : X for non-nan values
            Y_fit [rank_y, ntim*]                  : L * X fit for non-nan values
            corr:  [rank_y]  : correlation of fit performance
            xi_stdac [rank_y,ncycle] : standard deviation of fit residual as a function of cycle
            xi_std   [rank_y,ncycle] : RMSE/ standard deviation of fit residual 
            xi_a1:  [rank_y] : fit residual lag=1 auto-correlation
            xi_lambda: [rank_y] : fit residual decorrelation rate
            Y_stdac: [rank_y,ncycle] : standard deviation of Y as a function of cycle
            Yfit_stdac: [rank_y,ncycle] : standard deviation of fit as a function of cycle
        """
        # STEP 1: preprocess the data
        X, Y, tim, ntim, cycle_idx, t2d_y, t2d_x, delta_tim, cycle, cycle_shift, ac_rank, cossin = self._preprocess(
            X, Y, time=time, is_remove_nan=is_remove_nan)
        rank_x, _ = X.shape
        rank_y, _ = Y.shape

        # STEP 2: caclualte the operators
        # create matrix A * X = b
        ncol = self.ac_order*2+1
        G = np.full((rank_y, rank_x*ncol), np.nan)
        C = np.full((rank_x*ncol, rank_x*ncol), np.nan)

        for n in range(ncol):
            if (n <= self.ac_order):
                d = self.taus[n]
                G[:, rank_x*n:rank_x*n +
                    rank_x] = _Gn_cos(Y, X, t2d_y, n, self.omega, d)
                for m in range(ncol):
                    if (m <= self.ac_order):
                        C[rank_x*m:rank_x*m+rank_x, rank_x*n:rank_x*n+rank_x] = np.add(_Cn_cos(
                            X, t2d_x, n+m, self.omega, d), _Cn_cos(X, t2d_x, n-m, self.omega, d)) / 2
                    else:
                        mm = m - self.ac_order
                        C[rank_x*m:rank_x*m+rank_x, rank_x*n:rank_x*n+rank_x] = np.add(_Cn_sin(
                            X, t2d_x, n+mm, self.omega, d), -_Cn_sin(X, t2d_x, n-mm, self.omega, d)) / 2
            else:
                nn = n-self.ac_order
                d = self.taus[nn]
                G[:, rank_x*n:rank_x*n +
                    rank_x] = _Gn_sin(Y, X, t2d_y, nn, self.omega, d)
                for m in range(ncol):
                    if (m <= self.ac_order):
                        C[rank_x*m:rank_x*m+rank_x, rank_x*n:rank_x*n+rank_x] = np.add(_Cn_sin(
                            X, t2d_x, nn+m, self.omega, d), _Cn_sin(X, t2d_x, nn-m, self.omega, d)) / 2
                    else:
                        mm = m - self.ac_order
                        C[rank_x*m:rank_x*m+rank_x, rank_x*n:rank_x*n+rank_x] = np.add(-_Cn_cos(
                            X, t2d_x, nn+mm, self.omega, d), _Cn_cos(X, t2d_x, nn-mm, self.omega, d)) / 2

        L = _solve_L_with_zero(G, C)
        Lcoef = np.reshape(L, (rank_y, rank_x, ncol), 'F')

        # Y fit
        Y_fit = np.zeros(shape=Y.shape)
        for m in range(ncol):
            if (m <= self.ac_order):
                Y_fit = Y_fit + \
                    np.dot(Lcoef[:, :, m], X) * np.cos(m * self.omega * tim)
            else:
                mm = m - self.ac_order
                Y_fit = Y_fit + \
                    np.dot(Lcoef[:, :, m], X) * np.sin(mm * self.omega * tim)

        # from Lcoef to get L_ac
        # L opeator comp: annual cycle, semi-annual cycle, and so on.
        L_ac = np.zeros((rank_y, rank_x, self.ncycle))
        L_comp = np.zeros((rank_y, rank_x, self.ncycle, self.ac_order+1))
        for i in range(rank_y):
            for j in range(rank_x):
                L_comp[i, j, :, 0] = Lcoef[i, j, 0] * \
                    np.cos(0 * self.omega * cycle_shift)
                for k in range(1, self.ac_order+1):
                    L_comp[i, j, :, k] = Lcoef[i, j, k] * np.cos(
                        k * self.omega * cycle_shift) + Lcoef[i, j, k + self.ac_order] * np.sin(k * self.omega * cycle_shift)

                for m in range(ncol):
                    if (m <= self.ac_order):
                        L_ac[i, j, :] = L_ac[i, j, :] + Lcoef[i, j, m] * \
                            np.cos(m * self.omega * cycle_shift)
                    else:
                        mm = m - self.ac_order
                        L_ac[i, j, :] = L_ac[i, j, :] + Lcoef[i, j, m] * \
                            np.sin(mm * self.omega * cycle_shift)

        # output the operators as xarray Dataset
        axis_rankx = np.arange(1, rank_x + 1, dtype=np.int32)
        axis_ranky = np.arange(1, rank_y + 1, dtype=np.int32)
        xr_Lac = xr.DataArray(L_ac, dims={'ranky': axis_ranky, 'rankx': axis_rankx, 'cycle': cycle},
                              coords={'ranky': axis_ranky, 'rankx': axis_rankx, 'cycle': cycle})
        xr_Lcomp = xr.DataArray(L_comp, dims={'ranky': axis_ranky, 'rankx': axis_rankx, 'cycle': cycle, 'ac_rank': ac_rank},
                                coords={'ranky': axis_ranky, 'rankx': axis_rankx, 'cycle': cycle, 'ac_rank': ac_rank})
        xr_Lcoef = xr.DataArray(Lcoef, dims={'ranky': axis_ranky, 'rankx': axis_rankx, 'cossin': cossin},
                                coords={'ranky': axis_ranky, 'rankx': axis_rankx, 'cossin': cossin})
        xr_Yraw = xr.DataArray(Y, dims={'ranky': axis_ranky, 'time': tim}, coords={
                               'ranky': axis_ranky, 'time': tim})
        xr_Yfit = xr.DataArray(Y_fit, dims={'ranky': axis_ranky, 'time': tim}, coords={
                               'ranky': axis_ranky, 'time': tim})
        xr_X = xr.DataArray(X, dims={'rankx': axis_rankx, 'time': tim}, coords={
                                'rankx': axis_rankx, 'time': tim})

        # Step 3: fit the noise
        # Y fit performance, correlation and RMSE
        xr_corr = xr.corr(xr_Yraw, xr_Yfit, dim='time')

        # residual terms
        xr_residual = xr_Yraw - xr_Yfit
        xr_residual['cycle_idx'] = cycle_idx
        xr_Yraw['cycle_idx'] = cycle_idx
        xr_Yfit['cycle_idx'] = cycle_idx

        # noise memory (red noise lag1 correlation)
        xi_a1 = _calc_a1(xr_residual, maxlags=self.maxlags)
        xi_lambda = - np.log(xi_a1) / delta_tim

        # noise memory (red noise standard devition)
        xr_stdac = xr_residual.groupby('cycle_idx').std('time')
        xr_stdac = xr_stdac.rename(
            {'cycle_idx': 'cycle'}).assign_coords({'cycle': cycle})
        xi_std = xr_residual.std(dim='time').expand_dims(
            dim={"cycle": cycle}).transpose(..., 'cycle')

        y_stdac = xr_Yraw.groupby('cycle_idx').std('time').rename({'cycle_idx': 'cycle'}).assign_coords({'cycle': cycle})
        yfit_stdac = xr_Yfit.groupby('cycle_idx').std('time').rename({'cycle_idx': 'cycle'}).assign_coords({'cycle': cycle})

        ds_out = xr.Dataset({'Lac': xr_Lac,
                             'Lcomp': xr_Lcomp,
                             'Lcoef': xr_Lcoef,
                             'X': xr_X,
                             'Y': xr_Yraw.drop('cycle_idx'),
                             'Yfit': xr_Yfit.drop('cycle_idx'),
                             'corr': xr_corr,
                             'xi_std': xi_std,
                             'xi_stdac': xr_stdac,
                             'xi_a1': xi_a1,
                             'xi_lambda': xi_lambda,
                             'Y_stdac': y_stdac,
                             'Yfit_stdac': yfit_stdac,
                             })
        return ds_out

    #---------------------------------
    def get_norm_fit(self, fit_ds):
        """
            Returns normalized Lac and noises
        """
        rank_y, rank_x = len(fit_ds.ranky), len(fit_ds.rankx)
        stddev_X_np = np.std(fit_ds['X'].to_numpy(), axis=1)
        normLac = fit_ds['Lac'].copy(deep=True)
        normXi_std = fit_ds['xi_std'].copy(deep=True)
        normXi_stdac = fit_ds['xi_stdac'].copy(deep=True)

        normY_stdac = fit_ds['Y_stdac'].copy(deep=True)
        normYfit_stdac = fit_ds['Yfit_stdac'].copy(deep=True)
        
        for j in range(rank_y):
            for i in range(rank_x):
                normLac.loc[dict(ranky=j+1, rankx=i+1)] = fit_ds['Lac'].loc[dict(ranky=j+1, rankx=i+1)] * stddev_X_np[i]/stddev_X_np[j]
                normXi_std.loc[dict(ranky=j+1)] = fit_ds['xi_std'].loc[dict(ranky=j+1)] / stddev_X_np[j]
                normXi_stdac.loc[dict(ranky=j+1)] = fit_ds['xi_stdac'].loc[dict(ranky=j+1)] / stddev_X_np[j]
                normY_stdac.loc[dict(ranky=j+1)] = fit_ds['Y_stdac'].loc[dict(ranky=j+1)] / stddev_X_np[j]
                normYfit_stdac.loc[dict(ranky=j+1)] = fit_ds['Yfit_stdac'].loc[dict(ranky=j+1)] / stddev_X_np[j]
            
        norm_fit = xr.Dataset({'normLac': normLac,
                              'normxi_std': normXi_std,
                              'normxi_stdac': normXi_stdac,
                              'normY_stdac': normY_stdac,
                              'normYfit_stdac': normYfit_stdac,
                              })
        return norm_fit

    #---------------------------------
    def _get_var_names(self, X, var_names=None):
        X_np = _convert_to_numpy(X)
        n_var = X_np.shape[0]
        axis_rank = np.arange(1, n_var + 1, dtype=np.int32)
        if var_names:
            if len(var_names) != n_var:
                raise ValueError(f'Input var_names must have the same dimension with Input X!')
            # add prefix "X" to numeric var_names and convert to string
            var_names = ['X' + str(name) if isinstance(name, (int, float)) else str(name) for name in var_names]
            ds_names = xr.DataArray(var_names, dims=['ranky'], coords={'ranky': axis_rank})
        elif isinstance(X, xr.Dataset):
            ds_names = xr.DataArray(list(X.data_vars), dims=['ranky'], coords={'ranky': axis_rank})
        else:
            ds_names = xr.DataArray([f"X{i}" for i in axis_rank], dims=['ranky'], coords={'ranky': axis_rank})
        return xr.Dataset({'var_names': ds_names})
        
    #---------------------------------
    def fit(self, X, Y, time=None, is_remove_nan=True):
        """
            Y = L * X + ξ
        """
        Y_np = _convert_to_numpy(Y)
        X_np = _convert_to_numpy(X)
        fit_out = self.__compute__(X_np, Y_np, time=time, is_remove_nan=is_remove_nan)
        if len(fit_out.ranky) == len(fit_out.rankx) and len(fit_out.ranky) > 1:
            xr_norm = self.get_norm_fit(fit_out)
            fit_out = xr.merge([fit_out, xr_BWJ, xr_norm])
        return fit_out
        
    #---------------------------------
    def fit_matrix(self, X, var_names=None, maskb=None, n_th=1, time=None):
        """
        This module is fitting nonlinear model with T**2, T*H nonlinearity

        dX/dt = L * X + maskb * Ib * X^2  + X * X(1) * maskth
            X: state vector          [nrank, ntime]
            L: linear operator       [nrank, nrank, cycle]
            Ib : diagonal quadratic coefficient    [nrank, cycle]
            maskb: diagonal quadratic mask,        [nrank]
            ξ: red noise 

        make sure that first two elements in X are : ENSO T & H

        Returns:
            linear ouputs like fit
            Nonlinear fit parameters NLb for X**2
            Nonlinear fit parameter NLth X[1]*X[2] in the first equation
        """
        NL_return_vars = ['Lac', 'Lcoef', 'Lcomp']
        xr_var_names = self._get_var_names(X, var_names=var_names)
        full_vars = xr_var_names['var_names'].values

        XN = _convert_to_numpy(X)
        YN = gradient(XN, axis=1, is_forward=self.is_forward, ncycle=self.ncycle)
        XN2 = XN**2 #[rank_x, ntime]
        XTH = XN[0, :]*XN[1, :] #[ntime]

        n_var = XN.shape[0]

        ## check maskb 
        if maskb is None:
            maskb_vars = full_vars
        else:
            maskb_vars = maskb
        mask_b = get_mask_array(full_vars=full_vars, mask_vars=maskb_vars)

        loop_index = np.arange(0, n_var, step=1, dtype=np.int32)
        for i in loop_index:
            if i<=n_th-1:
                if mask_b[i] == 1:
                    var_XN = np.concatenate([XN, np.stack([XN2[i, :]], axis=0), np.stack([XTH], axis=0) ], axis=0)
                else:
                    var_XN = np.concatenate([XN, np.stack([XTH], axis=0) ], axis=0)
            else:
                if mask_b[i] == 1:
                    var_XN = np.concatenate([XN, np.stack([XN2[i, :]], axis=0) ], axis=0)
                else:
                    var_XN = XN

            var_YN  = np.stack([YN[i, :]],axis=0)
            var_res = self.fit(var_XN, var_YN, time=time)
            var_resL = var_res.sel(rankx=slice(1, n_var)).assign_coords(ranky=[i+1])

            if i<=n_th-1:
                if mask_b[i] == 1:
                    var_resNLb = var_res.sel(rankx=n_var+1).assign_coords(ranky=[i+1])[NL_return_vars].drop('rankx')
                    var_resTH = var_res.sel(rankx=n_var+2).assign_coords(ranky=[i+1])[NL_return_vars].drop('rankx')
                else:
                    var_resNLb = var_resL[NL_return_vars].sel(rankx=1).assign_coords(ranky=[i+1]).drop('rankx')*0.
                    var_resTH = var_res.sel(rankx=n_var+1).assign_coords(ranky=[i+1])[NL_return_vars].drop('rankx')
            else:
                if mask_b[i] == 1:
                    var_resNLb = var_res.sel(rankx=n_var+1).assign_coords(ranky=[i+1])[NL_return_vars].drop('rankx')
                else:
                    var_resNLb = var_resL[NL_return_vars].sel(rankx=1).assign_coords(ranky=[i+1]).drop('rankx')*0.

            if i==0:
                res_L = var_resL
                res_NLb = var_resNLb
                if i<=n_th-1:
                    res_NLth = var_resTH
                else:
                    res_NLth = xr.zeros_like(res_NLb)
            else:
                res_L = xr.concat([res_L, var_resL], dim='ranky')
                res_NLb = xr.concat([res_NLb, var_resNLb], dim='ranky')
                if i<=n_th-1:
                    res_NLth = xr.concat([res_NLth, var_resTH], dim='ranky')

        res_NLth = res_NLth.rename({'ranky': 'ranky_ro'})

        #
        fit_X = res_L['X'].sel(ranky=1).drop('ranky')
        res_L['X'] = fit_X

        # get eigs and norm operators
        xr_norm = self.get_norm_fit(res_L)

        # rename operators NLb and NLc
        for var in res_NLb.data_vars:
            res_NLb = res_NLb.rename({var: 'NLb_' + var})

        for var in res_NLth.data_vars:
            res_NLth = res_NLth.rename({var: 'NLth_' + var})

        fit_out = xr.merge([res_L, xr_norm, res_NLb, res_NLth, xr_var_names])
        return fit_out

    @staticmethod
    def _retrieve_fit_parameters(fit_ds, is_xi_stdac=False):
        '''
            get fit parameters from the fit_ds for the intergration
            required: linear
                Lac
                xi_stdac, xi_a1
            optional: nonlinear
                NLb_Lac
                NLth_Lac
        '''
        Lac_da = fit_ds['Lac']
        Noise_ds = fit_ds[['xi_stdac', 'xi_std', 'xi_a1']]
        NLb_da = fit_ds.get('NLb_Lac', xr.zeros_like(Lac_da.sel(rankx=1).drop('rankx')))
        NLth_da = fit_ds.get('NLth_Lac', xr.zeros_like(Lac_da.sel(rankx=1, ranky=slice(1, 1)).drop({'rankx'}).rename({'ranky': 'ranky_ro'})))

        Lac = Lac_da.values
        a1 = Noise_ds['xi_a1'].values
        
        if is_xi_stdac:
            stddev = Noise_ds['xi_stdac'].values
        else:
            stddev = Noise_ds['xi_std'].values
        NLb = NLb_da.values

        # Handle NLth
        NLth_T = np.zeros_like(NLb)
        NLth_H = np.zeros_like(NLb)
        if len(NLth_da.ranky_ro) > 1:
            NLth_T[0, :] = NLth_da.sel(ranky_ro=1).values
            NLth_H[1, :] = NLth_da.sel(ranky_ro=2).values
        else:
            NLth_T[0, :] = NLth_da.values
        return Lac, a1, stddev, NLb, NLth_T, NLth_H


    @staticmethod
    def _integration_core(X, m_Lac, b, th_T, th_H, noise, dt, nstep):
        tmp_X = np.zeros_like(X)  # Initialize with zeros
        for k in range(nstep):
            nl_terms = b[:, None] * X**2  + th_T[:, None] * X * X[1, :] + th_H[:, None] * X * X[0, :]
            dX = (np.dot(m_Lac, X) + nl_terms + noise) * dt
            X += dX
            tmp_X += X / nstep
        return X, tmp_X

    #---------------------------------
    def simulate(self, fit_ds, X0_ds, nyear=10, nstep=10, 
                 ncopy=1, seed=None, noise_type='red',
                 time=None, is_xi_stdac=False):
        '''

        Integration of XRO model include linear and nonlinear

        '''
        var_names = list(X0_ds.data_vars)
        X0 = X0_ds.to_array().to_numpy()

        ncycle = len(fit_ds['Lac'].cycle)
        rank_y = len(fit_ds['Lac'].ranky)
        dt = 1.0 / ncycle / nstep
        n_time = ncycle * nyear

        Lac, a1, stddev, NLb, NLth_T, NLth_H = self._retrieve_fit_parameters(fit_ds, is_xi_stdac)

        # use discrete noise equation
        noise_ds = gen_noise(stddev=stddev, nyear=nyear, ncopy=ncopy, seed=seed, noise_type=noise_type, a1=a1)

        # Reshape X0 to consider ensemble members
        X = np.tile(X0, (ncopy, 1)).T
        YY = np.zeros((rank_y, n_time, ncopy), dtype=float)

        for i in range(nyear):
            for j in range(ncycle):
                m_Lac = Lac[:, :, j]
                b, th_T, th_H = NLb[:, j], NLth_T[:, j], NLth_H[:, j]
                noise = noise_ds[:, i * ncycle + j, :].values
                X, tmp_X = self._integration_core(X, m_Lac, b, th_T, th_H, noise, dt, nstep)
                YY[:, i * ncycle + j, :] = tmp_X
        coords = {'ranky': fit_ds.ranky, 'time': time or noise_ds.time, 'member': noise_ds.member}
        YY_ds = xr.DataArray(YY, dims=['ranky', 'time', 'member'], coords=coords)
        fcst_ds = variable_model_to_xarray(YY_ds, var_names)
        return fcst_ds

    #---------------------------------
    def _integration_forecast(self, fit_ds, X0, t0_cycle, n_month=12, nstep=10,
                         ncopy=1, seed=None, noise_type='red', is_xi_stdac=False):
        '''

        Integration of XRO model include linear and nonlinear

        '''
        ncycle = len(fit_ds.cycle)
        rank_y = len(fit_ds.ranky)
        dt = 1.0 / ncycle / nstep
        n_time = int(ncycle*n_month/12)
        nyear = int(np.ceil(n_month/12))+1
        members = np.arange(0, ncopy, step=1).astype(np.int32)

        Lac, a1, stddev, NLb, NLth_T, NLth_H = self._retrieve_fit_parameters(fit_ds, is_xi_stdac)
        if noise_type == 'zero':
            noise_ds = np.zeros(shape=(rank_y, n_time+1, ncopy), dtype=float)
        else:
            noise_full = gen_noise(stddev=stddev, nyear=nyear, ncopy=ncopy, seed=seed, noise_type=noise_type, a1=a1)
            noise_ds = noise_full.values[:, t0_cycle:n_time+t0_cycle+1, :]   #[ranky, time, member]

        tim = np.arange(0, n_month/12.+0.5/ncycle, step=1./ncycle, dtype=float)
        # Reshape X0 to consider ensemble members
        X = np.tile(X0[:, np.newaxis], (1, ncopy))
        YY = np.zeros((rank_y, n_time + 1, ncopy), dtype=float)
        YY[:, 0, :] = X
        for i in range(n_time):
            j = np.mod(i+t0_cycle, ncycle)
            m_Lac = Lac[:, :, j]
            b, th_T, th_H = NLb[:, j], NLth_T[:, j], NLth_H[:, j]
            noise = noise_ds[:, i, :]
            tmp_X = np.zeros_like(X)  # Initialize with zeros
            X, tmp_X = self._integration_core(X, m_Lac, b, th_T, th_H, noise, dt, nstep)
            YY[:, i+1, :] = tmp_X
        coords = {'ranky': fit_ds.ranky, 'lead': tim, 'member': members}
        YY_ds = xr.DataArray(YY, dims=['ranky', 'lead', 'member'], coords=coords)
        return YY_ds

    #---------------------------------
    def reforecast(self, fit_ds, init_ds, n_month=12, nstep=10,
                         ncopy=1, seed=None, noise_type='red', is_xi_stdac=False):
        '''
            reforecast using XRO model
        '''
        var_names = list(init_ds.data_vars)
        X = variable_xarray_to_model(init_ds, ncycle=len(fit_ds.cycle))

        for i, init_t in enumerate(range(len(X.time))):
            t0_cycle = np.mod(init_t, len(fit_ds.cycle))
            X0 = X.isel(time=init_t).values
            t0 = X.time[init_t]
            tmp_fcst = self._integration_forecast(fit_ds, X0, t0_cycle, n_month=n_month, nstep=nstep, 
                                                  ncopy=ncopy, seed=seed, noise_type=noise_type, is_xi_stdac=is_xi_stdac)
            tmp_fcst = tmp_fcst.assign_coords({'time': t0})
            if i == 0:
                out_fcst = tmp_fcst
            else:
                out_fcst = xr.concat([out_fcst, tmp_fcst], dim='time')

        lead_axis = np.arange(0, len(out_fcst.lead), step=1, dtype=np.int32)
        xr_lead = xr.DataArray(lead_axis, dims={'lead': lead_axis}, coords={'lead': lead_axis}, attrs={'units': 'months', 'long_name': 'Lead'})
        out_fcst['lead'] = xr_lead

        fcst_ds = variable_model_to_xarray(out_fcst, var_names)
        fcst_ds['time'] = init_ds.time
        fcst_ds = fcst_ds.rename({'time': 'init'})
        fcst_ds['init'].encoding['_FillValue'] = None

        if ncopy == 1:
            fcst_ds = fcst_ds.squeeze('member', drop=True)
        return fcst_ds



############################################################################
# base functions
def _Gn_cos(Y, X, t2d, n, omega, d=1):
    '''
        cos(n * omega * t) Y(t) * XT(t-d)
    '''
    if d >= 0:
        return np.dot(np.cos(n * omega * t2d[:, d:-1]) * Y[:, d:-1], X[:, 0:(-d-1)].T)
    else:
        return np.nan


def _Gn_sin(Y, X, t2d, n, omega, d=1):
    '''
        sin(n * omega * t) Y(t) * XT(t-d)
    '''
    if d >= 0:
        return np.dot(np.sin(n * omega * t2d[:, d:-1]) * Y[:, d:-1], X[:, 0:(-d-1)].T)
    else:
        return np.nan


def _Cn_cos(X, t2d, n, omega, d=1):
    '''
        cos(n * omega * t)X(t) * XT(t-d)
    '''
    if d >= 0:
        return np.dot(np.cos(n * omega * t2d[:, d:-1]) * X[:, d:-1], X[:, 0:-d-1].T)
    else:
        return np.nan


def _Cn_sin(X, t2d, n, omega, d=1):
    '''
        sin(n * omega * t)X(t) * XT(t-d)
    '''
    if d >= 0:
        return np.dot(np.sin(n * omega * t2d[:, d:-1]) * X[:, d:-1], X[:, 0:-d-1].T)
    else:
        return np.nan


def _solve_L_with_zero(G, C):
    '''
    matrices G and C with zero columns and rows
    '''
    # print(G.shape, C.shape)

    # Identify the rows and columns that are entirely zero in both G and C
    zero_rows_G = np.all(G == 0, axis=1)
    zero_cols_G = np.all(G == 0, axis=0)
    zero_rows_C = np.all(C == 0, axis=1)
    zero_cols_C = np.all(C == 0, axis=0)

    # Get the indices of non-zero rows and non-zero columns
    nonzero_rows = np.where(~zero_rows_G)[0]
    nonzero_cols = np.where(~zero_cols_C)[0]

    # Create submatrices of G and C without zero rows and columns
    G_nonzero = G[~zero_rows_G][:, ~zero_cols_G]
    C_nonzero = C[~zero_rows_C][:, ~zero_cols_C]

    # Initialize the full-sized L matrix with zeros or NaN
    L = np.zeros_like(G)
    # Calculate L using the non-zero submatrices
    try:
        L_nonzero = np.dot(G_nonzero, np.linalg.inv(C_nonzero))
    except np.linalg.inv.LinAlgError:
        # Handle singular matrix or non-invertible case here
        L_nonzero = np.full((len(nonzero_rows), len(nonzero_cols)), np.nan)

    # Fill the corresponding positions in L with values from L_nonzero
    L[nonzero_rows.reshape(-1, 1), nonzero_cols] = L_nonzero
    return L


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
        x = np.array( x - np.nanmean(x) )
        y = np.array( y - np.nanmean(y) )
    
    if maxlags is None:
        maxlags = Nx - 1

    if maxlags >= Nx or maxlags < 1:
        raise ValueError('maglags must be None or strictly '
                         'positive < %d' % Nx)

    x_std  = np.nanstd(x)
    y_std  = np.nanstd(y)

    lags = np.arange(-maxlags, maxlags + 1, dtype=np.int32)
    c    = np.full_like(lags, np.nan, dtype=float)
    cross_cov = np.full_like(lags, np.nan, dtype=float)
    
    for k in lags:
        if k>0:
            tempCrossCov = np.nanmean(  y[:-k] * x[k:]  )
        elif k==0:
            tempCrossCov = np.nanmean(  x * y )
        else:
            tempCrossCov = np.nanmean( x[:k] * y[-k:] )
        cross_cov[ k+maxlags ] = tempCrossCov
        c[k+maxlags] = tempCrossCov/(x_std*y_std)
    xr_c = xr.DataArray(c, coords=[lags], dims = ["lag"])
    return xr_c


def xcorr(xr_var1, xr_var2, maxlags=12, units='month', dim='time'):
    '''
    xarray lead-lag correlations
    '''
    lags     = np.arange(-maxlags, maxlags + 1, dtype=np.int32)
    lags_out = xr.DataArray( lags, coords = {'lag': lags}, dims = ['lag'], name = ['lag'], 
                attrs = {'long_name':'lag', 'units': units, '_FillValue':-32767.}) #  'axis': 'T',

    xr_corr = xr.apply_ufunc(_xcorrnan, xr_var1, xr_var2,
                             input_core_dims=[[dim], [dim]],
                             output_core_dims=[['lag']],
                             kwargs={'maxlags': maxlags},
                             vectorize=True)
    xr_corr = xr_corr.assign_coords({"lag": lags_out})
    xr_corr.encoding['_FillValue'] = 1.e+20
    xr_corr.encoding['dtype']      = 'float32'
    return xr_corr

def _calc_a1(x, maxlags=6):
    '''
        calc the lag-1 correlation using multiple lags which is much accurate than using only lag=1

        red noise sequence x_j from a white noise w_j
            x_1   = w_1
            x_j+1 = r * x_j + sqrt(1-r*r) w_j+1, j>=1

        Using properties of normal distributions, it is easily shown that x_j+1 is n(0, 1)(Gaussian) 
        and that the lag-1 correlation coefficient of x_j+1 and x_j is r. It is also easy to show 
        by induction that 
        the correlation coefficient of x_j+p and x_j for for p>1, r^p = exp(-p log r) = exp(-R*p*δt),
        where R=-log(r)/δt is the the decorrelation rate.
        The autocovariance sequence of red noise thus decays exponentially with lag.
        The lag at which the autocorrelation drops to 1/e is τ = R^(−1).

        See Lecture 11: White and red noise by Christopher S. Bretherton 
        https://atmos.washington.edu/~breth/classes/AM582/lect/lect8-notes.pdf
    '''
    acc = xcorr(x, x, maxlags=maxlags).sel(lag=slice(1, maxlags))
    a1 = np.power(acc, 1/acc.lag)
    return a1.mean('lag')


def _convert_to_numpy(data):
    """
    Convert the given data (either xr.Dataset, xr.DataArray, or numpy array) to numpy format.
    """
    # Convert xr.Dataset to a numpy array.
    if isinstance(data, xr.Dataset):
        return data.to_array().to_numpy()

    # Convert xr.DataArray to a numpy array. If it's 1-dimensional, reshape it.
    elif isinstance(data, xr.DataArray):
        data_np = data.to_numpy()
        return data_np[np.newaxis, :] if len(data.shape) == 1 else data_np

    # If data is already a numpy array and it's 1-dimensional, reshape it.
    else:
        return data[np.newaxis, :] if len(data.shape) == 1 else data


def gradient(arr, axis=-1, is_forward=True, ncycle=12):
    """
    Compute the difference of an array.

    Parameters:
    - arr: numpy array
    - is_forward is True forward or elsewise center 
    - axis: int, the axis along which differences are computed.

    Returns:
    - diff_arr: numpy array of differences.
    """
    # Initialize a placeholder for differences with the same shape as arr
    diff_arr = np.empty_like(arr)

    if is_forward:
        # All points except the last: forward difference
        slice_obj = [slice(None)] * arr.ndim
        slice_obj[axis] = slice(0, -1)
        diff_arr[tuple(slice_obj)] = arr.take(indices=range(
            1, arr.shape[axis]), axis=axis) - arr.take(indices=range(0, arr.shape[axis]-1), axis=axis)

        # Last point: backward difference
        slice_obj[axis] = -1
        diff_arr[tuple(slice_obj)] = arr.take(
            indices=-1, axis=axis) - arr.take(indices=-2, axis=axis)

    else:
        # Interior points: central differences
        slice_obj = [slice(None)] * arr.ndim
        slice_obj[axis] = slice(1, -1)
        diff_arr[tuple(slice_obj)] = (arr.take(indices=range(2, arr.shape[axis]),
                                               axis=axis) - arr.take(indices=range(0, arr.shape[axis]-2), axis=axis)) / 2.0

        # First point: forward difference
        slice_obj[axis] = 0
        diff_arr[tuple(slice_obj)] = arr.take(
            indices=1, axis=axis) - arr.take(indices=0, axis=axis)

        # Last point: backward difference
        slice_obj[axis] = -1
        diff_arr[tuple(slice_obj)] = arr.take(
            indices=-1, axis=axis) - arr.take(indices=-2, axis=axis)

    # compute the dX/dt and convert units of year^-1
    return diff_arr * ncycle


def get_mask_array(full_vars, mask_vars):
    """
        example:
        get_mask_array(full_vars=['X1', 'X2', 'X3'], mask_vars=['X1'])
        # returns [1, 0, 0]
        get_mask_array(full_vars=['X1', 'X2', 'X3'], mask_vars=['X1', 'X3'])
        # returns [1, 0, 1]
    """
    return np.array([1 if var in mask_vars else 0 for var in full_vars]).astype(int)


def gen_noise(stddev, nyear=50, ncopy=1, init=None, seed=None, noise_type='white', a1=None):
    '''
        generate noise with amplitude of seasonal standard deviation (seastd)

        INPUT: 
               stddev is seasonal standard deviation [:, ncycle]
               noise_type can be 'white' or 'red'. If 'red', a1 must be provided.

        OUTPUT:
              xr.DataArray  noise [:, nyear*ncycle, ncopy] 
    '''
    if seed is not None:
        np.random.seed(seed)

    ranky, ncycle = stddev.shape
    tim = np.arange(1./ncycle/2., nyear, step=1./ncycle, dtype=float)
    axis_ranky = np.array(np.arange(1, ranky+1, step=1, dtype=np.int32))

    if init is not None:
        red = init[:, np.newaxis]
    else:
        red = stddev[:, 0, np.newaxis] * np.random.normal(size=(ranky, ncopy))

    noise_ts = np.full(shape=(ranky, nyear*ncycle, ncopy), fill_value=np.nan, dtype=float)

    if noise_type == 'red':
        if a1 is None:
            raise ValueError("For red noise, a1 must be provided.")
        amp = np.sqrt(1-a1*a1)[:, np.newaxis]
        for iy in range(nyear):
            for ic in range(ncycle):
                red = a1[:, np.newaxis] * red + amp * np.random.normal(size=(ranky, ncopy))
                noise_ts[:, iy*ncycle+ic, :] = stddev[:, ic, np.newaxis] * red
    elif noise_type == 'white':
        for iy in range(nyear):
            for ic in range(ncycle):
                noise_ts[:, iy*ncycle+ic, :] = stddev[:, ic, np.newaxis] * np.random.normal(size=(ranky, ncopy))
    else:
        raise ValueError("Invalid noise_type. Must be 'white' or 'red'.")

    if ncycle==12:
        time = xr.cftime_range('0001-01', periods=nyear*ncycle, freq='MS')
    else:
        time = tim
    members = np.arange(0, ncopy, step=1).astype(np.int32)
    noise = xr.DataArray(noise_ts, dims=['ranky', 'time', 'member'], coords={'ranky': axis_ranky, 'time': time, 'member': members})
    return noise


def variable_xarray_to_model(xr_ds, ncycle=12):
    """
        xr.Dataset to the model X with [nrank, ntime]
    """
    X = xr_ds.to_array().to_numpy()
    rank_y = X.shape[0]
    ntim = X.shape[1]
    axis_ranky = np.arange(1, rank_y+1, step=1, dtype=np.int32)
    tim = np.arange(1./ncycle/2., ntim/ncycle,
                    step=1./ncycle, dtype=np.float32)
    return xr.DataArray(X, dims={'ranky': axis_ranky, 'time': tim}, coords={'ranky': axis_ranky, 'time': tim})


def variable_model_to_xarray(model_X, var_names):
    """
        model variable to individual components
    """
    for k, var in enumerate(var_names):
        tmp_var = model_X.sel(ranky=k+1).drop('ranky')
        if k == 0:
            model_ds = xr.Dataset({var: tmp_var})
        else:
            model_ds[var] = tmp_var
    return model_ds

class _AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__