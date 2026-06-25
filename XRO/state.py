
import os
import numpy as np
import xarray as xr


def calc_XRO_indices(sst_a: xr.DataArray, h_a: xr.DataArray) -> xr.Dataset:
    """
    Compute XRO-related SST indices and WWV from SST and heat content fields.

    Parameters
    ----------
    sst_a : xr.DataArray
        sea surface temperature (time, lat, lon).
    h_a : xr.DataArray
        heat content / ssh / thermocline proxy (time, lat, lon)

    Returns
    -------
    xr.Dataset
        Dataset containing SST indices + WWV.
    """

    # -----------------------------------------------------
    # Define SST regions
    # -----------------------------------------------------
    sst_regions = {
        'Nino34': {'latS': -5, 'latN': 5, 'lonW': 190, 'lonE': 240},
        'NPMM': {'latS': 10, 'latN': 25, 'lonW': 200, 'lonE': 240},
        'SPMM': {'latS': -25, 'latN': -15, 'lonW': 250, 'lonE': 270},
        'IOB': {'latS': -20, 'latN': 20, 'lonW': 40, 'lonE': 100},
        'IODe': {'latS': -10, 'latN': 0, 'lonW': 90, 'lonE': 110},
        'IODw': {'latS': -10, 'latN': 10, 'lonW': 50, 'lonE': 70},
        'SIODe': {'latS': -30, 'latN': -5, 'lonW': 90, 'lonE': 120},
        'SIODw': {'latS': -25, 'latN': -10, 'lonW': 65, 'lonE': 85},
        'SASD_SW': {'latS': -45, 'latN': -35, 'lonW': 300, 'lonE': 360},
        'SASD_NE': {'latS': -30, 'latN': -20, 'lonW': 320, 'lonE': 380},
        'ATL3': {'latS': -3, 'latN': 3, 'lonW': 340, 'lonE': 360},
        'TNA': {'latS': 5, 'latN': 25, 'lonW': 305, 'lonE': 345},
    }

    # -----------------------------------------------------
    # Area-mean SST indices
    # -----------------------------------------------------
    ssti_a = area_average_regionDicts(sst_a, sst_regions)

    # Derived indices
    ssti_a["IOD"] = ssti_a["IODw"] - ssti_a["IODe"]
    ssti_a["SIOD"] = ssti_a["SIODe"] - ssti_a["SIODw"]
    ssti_a["SASD"] = ssti_a["SASD_SW"] - ssti_a["SASD_NE"]
    ssti_a["IOD"].attrs["long_name"] = "Indian Ocean Dipole"
    ssti_a["SIOD"].attrs["long_name"] = "Southern Indian Ocean Dipole"
    ssti_a["SASD"].attrs["long_name"] = "South Atlantic Subtropical Dipole"

    # -----------------------------------------------------
    # WWV (heat content in equatorial Pacific)
    # -----------------------------------------------------
    eqPac = {'latS': -5, 'latN': 5, 'lonW': 120, 'lonE': 280}
    wwvi_a = area_average(h_a, region=eqPac).to_dataset(name="WWV")

    # -----------------------------------------------------
    # Align time
    # -----------------------------------------------------
    ssti_a, wwvi_a = xr.align(ssti_a, wwvi_a, join="inner")

    XRO_vars = ["Nino34", "WWV", "NPMM", "SPMM", "IOB", "IOD", "SIOD", "TNA", "ATL3", "SASD"]
    return xr.merge([ssti_a, wwvi_a])[XRO_vars]


##################################################################################


def area_average(x, region=None):
    '''
        cos-weighted area averaged fields
    '''
    if region is None:
        x_subset = x
    else:
        x_subset = _select_region(x, region)

    w = np.cos(np.deg2rad(x_subset.lat))
    w = w.broadcast_like(x_subset)
    aave = (x_subset * w).mean(dim=('lat', 'lon'))/w.mean(dim=('lat', 'lon'))

    if isinstance(region, dict):
        region_name = region.get('name') or region.get('long_name')
        if region_name:
            aave.attrs['long_name'] = region_name
    return aave


def area_average_regionDicts(x, Rdicts):
    '''
        cos-weigthed area averaged fields using dicts
    '''
    tmp_arrs = []
    for skey, region in Rdicts.items():
        tmp_ds = area_average(x, region=region)
        tmp_ds.name = skey
        tmp_arrs.append( tmp_ds.to_dataset() )
    return xr.merge(tmp_arrs)


def _select_region(x, region):
    '''
        select region boxes
    '''
    if isinstance(region, str):
        try:
            Rbox   = _box_region_array(region)
            R_latS = Rbox['latS']
            R_latN = Rbox['latN']
            R_lonW = Rbox['lonW']
            R_lonE = Rbox['lonE']
            x_sel = x.sel(lat=slice(R_latS, R_latN), lon=slice(R_lonW, R_lonE))

        except ValueError:
            print("box_region_array error: undefined region string!")

    elif isinstance(region, dict) and len(region)>=4:
        R_latS = region['latS']
        R_latN = region['latN']
        R_lonW = region['lonW']
        R_lonE = region['lonE']
        x_sel = x.sel(lat=slice(R_latS, R_latN), lon=slice(R_lonW, R_lonE))

    elif isinstance(region, (list, tuple))  and len(region)>=4:
        R_lonW = region[0]
        R_lonE = region[1]
        R_latS = region[2]
        R_latN = region[3]
        x_sel = x.sel(lat=slice(R_latS, R_latN), lon=slice(R_lonW, R_lonE))
    else:

        raise ValueError('error in select_region: unsupported region type!')

    return x_sel


def _box_region_array(Rstr = 'Nino34'):
    '''
        return box region array of latS, latN, lonW, lonE
    '''
    region = {}

    ## ENSO SST indices
    region['Nino34']  = {'latS': -5, 'latN': 5, 'lonW': 190, 'lonE': 240, 'name': 'Niño3.4 (5°S-5°N, 170°W-120°W)'}
    region['Nino3']   = {'latS': -5, 'latN': 5, 'lonW': 210, 'lonE': 270, 'name': 'Niño3 (5°S-5°N, 150°W-90°W)'}
    region['Nino4']   = {'latS': -5, 'latN': 5, 'lonW': 160, 'lonE': 210, 'name': 'Niño4 (5°S-5°N, 160°E-150°W)'}
    region['Nino2']   = {'latS': -5, 'latN': 0, 'lonW': 270, 'lonE': 280, 'name': 'Niño1 (5°S-0°, 90°W-80°W)'}
    region['Nino1']   = {'latS': -10, 'latN': -5, 'lonW': 270, 'lonE': 280, 'name': 'Niño2 (10°S-5°S, 90°W-80°W)'}
    region['Nino12']  = {'latS': -10, 'latN': 0, 'lonW': 270, 'lonE': 280, 'name': 'Niño1+2 (10°S-0°, 90°W-80°W)'}
    region['Nino4W']  = {'latS': -5, 'latN': 5, 'lonW': 160, 'lonE': 185, 'name': 'Niño4W (5°S-5°N, 160°E-175°W)'}
    region['Nino4E']  = {'latS': -5, 'latN': 5, 'lonW': 185, 'lonE': 210, 'name': 'Niño4E (5°S-5°N, 175°W-150°W)'}
    region['ColdTongue'] = {'latS': -6, 'latN': 6, 'lonW': 180, 'lonE': 270, 'name': 'ColdTongue (6°S-6°N, 180°-90°W)'}

    ## EMI 
    ## EMI SST indice
    # ;https://www.jamstec.go.jp/frcgc/research/d1/iod/publications/modoki-ashok.pdf
    # ;Ashok et al (2007) paper “El Nino Modoki and its Possible Teleconnection.”
    # ; regions A (165E-140W, 10S-10N), B (110W-70W, 15S-5N), and C (125E-145E, 10S-20N), 
    region['EMI_C']  = {'latS': -10, 'latN': 10, 'lonW': 165, 'lonE': 220, 'name': 'EMI_C', 'name': 'EMI_C (10°S-10°N, 165°E-140°W)'}
    region['EMI_W']  = {'latS': -10, 'latN': 20, 'lonW': 125, 'lonE': 145, 'name': 'EMI_W', 'name': 'EMI_W (10°S-20°N, 125°E-145°E)'}
    region['EMI_E']  = {'latS': -15, 'latN': 5, 'lonW': 250, 'lonE': 290, 'name': 'EMI_E', 'name': 'EMI_E (15°S-5°N, 110°W-70°W)'}

    ## ENSO presursors 
    ## NPMM and SPMM indices
    region['NPMM']      = {'latS': 10, 'latN': 25, 'lonW': 200, 'lonE': 240, 'name': 'NPMM (10°N-25°N, 160°W-120°W)'}            # modfied by myself looking at MCA_sst best correlation region box
    region['NPMM_box0'] = {'latS': 15, 'latN': 20, 'lonW': 200, 'lonE': 240, 'name': 'NPMM_box0 (15°N-20°N, 160°W-120°W)'}       # modfied by myself looking at MCA_sst best correlation region box
    region['NPMM_box1'] = {'latS': 15, 'latN': 25, 'lonW': 210, 'lonE': 240, 'name': 'NPMM_Amaya2019 (15°N-25°N, 150°W-120°W)'}  # Amaya 2019
    # Amaya, D. J. (2019). The Pacific Meridional Mode and ENSO: a Review. Current Climate Change Reports, 5(4), 296–307.https://doi.org/10.1007/s40641-019-00142-x
    region['NPMM_box2'] = {'latS': 15, 'latN': 25, 'lonW': 220, 'lonE': 240, 'name': 'NPMM_Zhang2014 (15°N-25°N, 140°W-120°W)'}  # Zhang et al. 2014
    # Zhang, H., Deser, C., Clement, A., & Tomas, R. (2014). Equatorial signatures of the Pacific Meridional Modes: Dependence on mean climate state. Geophysical Research Letters, 41(2), 568–574. https://doi.org/10.1002/2013GL058842
    region['NPMM_box3'] = {'latS': 8, 'latN': 25, 'lonW': 200, 'lonE': 240, 'name': 'NPMM_Richter2022 (8°N-25°N, 160°W-120°W)'}  #  Richter et al. 2022
    ##

    region['WNP']   = {'latS': 18, 'latN': 28, 'lonW': 120, 'lonE': 132, 'name': 'WNP (18°N-28°N, 120°E-132°E)'} 
    # Wang, S.-Y., L’Heureux, M., & Chia, H.-H. (2012). ENSO prediction one year in advance using western North Pacific sea surface temperatures. 
    # Geophysical Research Letters, 39(5). https://doi.org/10.1029/2012GL050909

    region['SPMM']      = {'latS': -25, 'latN': -15, 'lonW': 250, 'lonE': 270, 'name': 'SPMM (25°S-15°S, 110°W-90°W)'}      # Zhang et al. (2014) SPMM

    ##----------------------
    # Indian Ocean
    region['IOB'] = {'latS': -20, 'latN': 20, 'lonW': 40, 'lonE': 100, 'name': 'IOB (20°S-20°N, 40°E–100°E)'}

    ## Saji
    region['IODe'] = {'latS': -10, 'latN': 0, 'lonW': 90, 'lonE': 110, 'name': 'IODe (10°S-0°, 90°E–110°E)'}
    region['IODw'] = {'latS': -10, 'latN': 10, 'lonW': 50, 'lonE': 70, 'name': 'IODw (10°S-10°N, 50°E–70°E)'}

    ## Southern Indian Ocean Dipole (SIOD) 
    ## Jo et al. 2022 Southern Indian Ocean Dipole as a trigger for Central Pacific El Niño since the 2000s
    region['SIODe'] = {'latS': -30, 'latN': -5, 'lonW': 90, 'lonE': 120, 'name': 'southeastern IO (30°S-5°S, 90°E–120°E)'}
    region['SIODw'] = {'latS': -25, 'latN': -10, 'lonW': 65, 'lonE': 85, 'name': 'southcentral IO (10°S-10°N, 50°E–70°E)'}

    region['Ningaloo'] = {'latS': -28, 'latN': -22, 'lonW': 108, 'lonE': 116, 'name': 'Ningaloo (28°S-22°S, 108°E–116°E)'} ## Kataoka et al. 2014

    ##----------------------
    ## Atlantic Ocean
    region['ATL3'] = {'latS': -3, 'latN': 3, 'lonW': 340, 'lonE': 360, 'name': 'ATL3 (3°S-3°N, 20°W–0°)'}     # Zebiak 1993
    region['ATLW'] = {'latS': -3, 'latN': 3, 'lonW': 315, 'lonE': 335, 'name': 'ATLW (3°S-3°N, 45°W–25°W)'}   # Tokinaga and Shang-Ping Xie 2011

    ## Enfield et al. (JGR, 1999)
    region['TNA'] = {'latS': 5, 'latN': 25, 'lonW': 305, 'lonE': 345, 'name': 'TNA (5°N-25°N, 55°W–15°W)'}        #  55°W - 15°W, 5°N - 25°N
    region['TSA'] = {'latS': -20, 'latN': 0, 'lonW': 330, 'lonE': 370, 'name': 'TSA (20°S-0°, 30°W–10°E)'}        #  30°W - 10°E, 20°S - 0°.

    ##  Chang, Ji, and Li (Nature, 1997),
    region['TNA1'] = {'latS': 5, 'latN': 20, 'lonW': 320, 'lonE': 340, 'name': 'TNA1 (5°N-20°N, 40°W–20°W)'}    #  40°W - 20°W, 5°N - 20°N.
    region['TSA1'] = {'latS': -20, 'latN': -5, 'lonW': 345, 'lonE': 365, 'name': 'TSA1 (20°S-5°S, 15°W–5°E)'}  # 15°W - 5°E, 20°S - 5°S.

    region['NTA_Zhang'] = {'latS': 0, 'latN': 15, 'lonW': 270, 'lonE': 360, 'name': 'NTA_Zhang (0°-15°N, 90°W–0°W)'}    #  0°–15°N, 90°–0°W) 
    region['NTA_JZ22'] = {'latS': 0, 'latN': 15, 'lonW': 285, 'lonE': 360, 'name': 'NTA_JZ22 (0°-25°N, 75°W–0°W)'}    #  (0°–25°N, 0°–75°W)
    ## see Fig. 1 in Zhang, W., Jiang, F., Stuecker, M. F., Jin, F.-F., & Timmermann, A. (2021). Spurious North Tropical Atlantic precursors to El Niño. Nature Communications, 12(1), 3096. https://doi.org/10.1038/s41467-021-23411-6

    ## MORIOKA et al. 2011 On the Growth and Decay of the Subtropical Dipole Mode in the South Atlantic
    region['SASD1_SW'] = {'latS': -40, 'latN': -30, 'lonW': 330, 'lonE': 350, 'name': 'SASD1_SW (40°S-30°S, 30°W–10°W)'}
    region['SASD1_NE'] = {'latS': -25, 'latN': -15, 'lonW': 340, 'lonE': 360, 'name': 'SASD1_NE (25°S-15°S, 20°W–0°E)'}

    ## Ham, Y.-G., H.-J. Lee, H.-S. Jo, S.-G. Lee, W. Cai, and R. R. Rodrigues, 2021: Inter-Basin Interaction Between Variability in the South Atlantic Ocean 
    ## and the El Niño/Southern Oscillation. Geophysical Research Letters, 48, e2021GL093338, https://doi.org/10.1029/2021GL093338.
    region['SASD2_SW'] = {'latS': -45, 'latN': -35, 'lonW': 300, 'lonE': 360, 'name': 'SASD2_SW (45°S-35°S, 60°W–0°W)'}
    region['SASD2_NE'] = {'latS': -30, 'latN': -20, 'lonW': 320, 'lonE': 380, 'name': 'SASD2_NE (30°S-20°S, 40°W–20°E)'}

    ## Nnamchi, H. C., J. Li, and R. N. C. Anyadike, 2011: Does a dipole mode really exist in the South Atlantic Ocean? Journal of Geophysical Research: Atmospheres, 116, https://doi.org/10.1029/2010JD015579.
    region['SAOD_SWP'] = {'latS': -40, 'latN': -25, 'lonW': 320, 'lonE': 350, 'name': 'SAOD_SWP (45°S-35°S, 40°W–10°W)'}
    region['SAOD_NEP'] = {'latS': -15, 'latN': 0, 'lonW': 340, 'lonE': 370, 'name': 'SAOD_NEP (15°S-0°S, 20°W–10°E)'}

    region['eqPac'] = {'latS': -5, 'latN': 5, 'lonW': 120, 'lonE': 280, 'name': 'eqPac (5°S-5°N, 120°E-80°W)'}  #
    
    ## origioanl defintion: https://www.pmel.noaa.gov/elnino/upper-ocean-heat-content-and-enso
    region['eqPacW0'] = {'latS': -5, 'latN': 5, 'lonW': 120, 'lonE': 205, 'name': 'eqPacE0 (5°S-5°N, 120°E-155°W)'}  #
    region['eqPacE0'] = {'latS': -5, 'latN': 5, 'lonW': 205, 'lonE': 280, 'name': 'eqPacW0 (5°S-5°N, 155°W-80°W)'}   #

    ## defintion in Zhao et al. (2021), west and east are the same size http://onlinelibrary.wiley.com/doi/abs/10.1029/2021GL094366
    region['eqPacW'] = {'latS': -5, 'latN': 5, 'lonW': 120, 'lonE': 200, 'name': 'eqPacE (5°S-5°N, 120°E-160°W)'}  #
    region['eqPacE'] = {'latS': -5, 'latN': 5, 'lonW': 200, 'lonE': 280, 'name': 'eqPacW (5°S-5°N, 160°W-80°W)'}   #

    ## this defintion may works for SSH data
    region['eqPacW1'] = {'latS': -5, 'latN': 5, 'lonW': 120, 'lonE': 180, 'name': 'eqPacW1 (5°S-5°N, 120°E-180°)'}  #
    region['eqPacE1'] = {'latS': -5, 'latN': 5, 'lonW': 180, 'lonE': 280, 'name': 'eqPacE1 (5°S-5°N, 180°-80°W)'}   #

    # WP: 110°E - 180, 5°S-5°N; 
    # EP: 180-80°W, 5°S - 5°N
    region['EP'] = {'latS': -5, 'latN': 5, 'lonW': 180, 'lonE': 280, 'name': 'EP (5°S-5°N, 180°-280°)'}
    region['WP'] = {'latS': -5, 'latN': 5, 'lonW': 110, 'lonE': 180, 'name': 'WP (5°S-5°N, 110°-180°)'}
    region['IWP'] = {'latS': -5, 'latN': 5, 'lonW': 80, 'lonE': 150, 'name': 'IWP (5°S-5°N, 80°-150°)'}
    region['warmpool']  = {'latS': -30, 'latN': 30, 'lonW': 50, 'lonE': 200, 'name': 'warmpool (30°S-30°N, 50°-200°)'}
    region['WP3_Seager'] = {'latS': -3, 'latN': 3, 'lonW': 140, 'lonE': 170, 'name': 'WP3 (3°S-3°N, 140°-170°)'}
    region['EP3_Seager'] = {'latS': -3, 'latN': 3, 'lonW': 170, 'lonE': 270, 'name': 'EP3 (3°S-3°N, 170°-270°)'}
    region['offeqPac_N'] = {'latS': 3, 'latN': 9, 'lonW': 170, 'lonE': 240, 'name': 'offeqP_N (3°N-9°N, 170°-270°)'}
    region['offeqPac_S'] = {'latS': -9, 'latN': -3, 'lonW': 170, 'lonE': 240, 'name': 'offeqP_S (9°S-3°S, 170°-270°)'}
    region['eqPac3'] = {'latS': -3, 'latN': 3, 'lonW': 170, 'lonE': 240, 'name': 'offeqP_S (9°S-3°S, 170°-270°)'}

    # Southern Ocean: 45°S - 75°S
    region['SouthernOcean']   = {'latS': -60, 'latN': -45, 'lonW': 0, 'lonE': 359.5, 'name': 'SouthernOcean (75°S-45°S, 0°-360°)'}
    # SEP: 140°W-70°W, 65°S-45°S
    # SWP near SPCZ: 40°–30°S, 190°–210°E
    region['SO_SEP'] = {'latS': -65, 'latN': -45, 'lonW': 220, 'lonE': 290, 'name': 'SEP (65°S-45°S, 220°-290°)'}
    region['SO_SWP'] = {'latS': -40, 'latN': -30, 'lonW': 190, 'lonE': 210, 'name': 'SWP (40°S-30°S, 190°-210°)'}

    return region.get(Rstr, "nothing")


##################################################################################


def lonFlip(ds, deg=0):
    """
    Flips the longitude coordinates of a dataset around the specified degree
    and sorts by longitude. Copies the attributes of the dataset and coordinates.

    Parameters:
    ds (xarray.Dataset or xarray.DataArray): The input dataset or data array.
    deg (int): The degree around which to flip longitudes. Default is 0.

    Returns:
    xarray.Dataset or xarray.DataArray: The modified dataset with flipped and sorted longitude.
    """
    # Copy the original attributes of the longitude coordinates
    lon_attrs = ds['lon'].attrs.copy()

    # Create the modified longitude coordinates
    new_lon = (((ds.lon + deg) % 360) - deg)

    # Assign new coordinates with attributes copied & Sort by longitude
    ds = ds.assign_coords(lon=new_lon).sortby('lon')

    # Restore the original attributes to the new longitude coordinates
    ds['lon'].attrs = lon_attrs
    return ds


##################################################################################

# https://cds.climate.copernicus.eu/datasets/reanalysis-oras5

class CDS_ORAS5_Realtime:
    """ORAS5 realtime fields and XRO indices."""
    
    URLS = {
        "consolidated": "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/reanalysis_oras5/consolidated/geoChunked.zarr",
        "operational": "https://arco.datastores.ecmwf.int/cadl-arco-geo-001/arco/reanalysis_oras5/operational/geoChunked.zarr",
    }

    VAR_MAP = {
        "sosstsst": "sst",
        "so20chgt": "d20",
    }

    def __init__(self, token: str | None = None):
        """
        Parameters
        ----------
        token : str, optional
            ECMWF CDS/API token. If None, will be read from ~/.cdsapirc.
        """

        self.token = token or self._read_cds_token()

        if not self.token:
            raise ValueError(
                "❌ No CDS/API token found. "
                "Please provide token explicitly or set it in ~/.cdsapirc"
            )

        self.storage_options = {
            "headers": {"Authorization": f"Bearer {self.token}"}
        }

        self._ds = None

    # -----------------------------------------------------
    # Token reader
    # -----------------------------------------------------
    def _read_cds_token(self) -> str | None:
        """
        Read token from ~/.cdsapirc.

        Expected format:
            url: https://cds.climate.copernicus.eu/api
            key: <uid>:<api-key>
        """
        path = os.path.expanduser("~/.cdsapirc")

        if not os.path.exists(path):
            return None

        try:
            with open(path, "r") as f:
                for line in f:
                    if line.strip().startswith("key:"):
                        return line.split("key:")[1].strip()
        except Exception:
            return None

        return None

    # -----------------------------------------------------
    # Data loader: We only need SST and D20 in XRO states
    # -----------------------------------------------------
    def load_variables(self, time_slice=None):
        """Load SST and D20 fields."""

        if self._ds is None:
            ds = xr.concat(
                [
                    xr.open_zarr(
                        self.URLS["consolidated"],
                        consolidated=True,
                        storage_options=self.storage_options,
                    ),
                    xr.open_zarr(
                        self.URLS["operational"],
                        consolidated=True,
                        storage_options=self.storage_options,
                    ),
                ],
                dim="time",
            )

            ds = ds.rename({"latitude": "lat", "longitude": "lon"})[self.VAR_MAP.keys()].rename(self.VAR_MAP)

            # convert lon from -180-180 to 0-360
            ds = lonFlip(ds, deg=0)

            if time_slice is None:
                self._ds = ds
            else:
                self._ds = ds.sel(time=time_slice)

        return self._ds
