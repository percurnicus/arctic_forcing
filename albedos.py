import numpy as np
import pandas as pd


class Albedo:

    def __init__(self, zeniths, albedos):
        zeniths = np.array(zeniths)
        albedos = np.array(albedos)
        self._zeniths = zeniths[~np.isnan(zeniths)].copy()
        self._albedos = albedos[~np.isnan(albedos)].copy()
        sort_inds = np.argsort(self._zeniths)
        self._zeniths = self._zeniths[sort_inds]
        self._albedos = self._albedos[sort_inds]

    @property
    def zeniths(self):
        return self._zeniths.copy()

    @property
    def albedos(self):
        return self._albedos.copy()

    def get_albedo(self, zeniths):
        mask = zeniths <= 90
        albedo = np.zeros(zeniths.shape)
        albedo[mask] = np.interp(
            x=zeniths[mask],
            xp=self._zeniths,
            fp=self._albedos,
        )
        return albedo


class Albedos:

    def __init__(self, filepath=None):
        if filepath is None:
            filepath = 'Albedos.csv'
        self._filepath = filepath
        self._albedos = pd.read_csv(filepath)
        self._clear_ocean = Albedo(
            zeniths=self._albedos['Clear Sky Over Ocean X'].values,
            albedos=self._albedos['Clear Sky Over Ocean Y'].values
        )
        self._cloud_bright_ice = Albedo(
            zeniths=self._albedos['Cloud Over Bright Sea Ice X'].values,
            albedos=self._albedos['Cloud Over Bright Sea Ice Y'].values
        )
        self._cloud_ocean = Albedo(
            zeniths=self._albedos['Cloud Over Ocean X'].values,
            albedos=self._albedos['Cloud Over Ocean Y'].values
        )
        self._cloud_dark_ice = Albedo(
            zeniths=self._albedos['Cloud Over Dark Sea Ice X'].values,
            albedos=self._albedos['Cloud Over Dark Sea Ice Y'].values)
        self._clear_bright_ice = Albedo(
            zeniths=self._albedos['Clear Sky Over Bright Ice X'].values,
            albedos=self._albedos['Clear Sky Over Bright Ice Y'].values
        )
        self._clear_dark_ice = Albedo(
            zeniths=self._albedos['Clear Sky Over Dark Ice X'].values,
            albedos=self._albedos['Clear Sky Over Dark Ice Y'].values,
        )

    def get_ice_albedo(self, zeniths, ice_thickness, temperature,
                       clear_sky=True, sea_albedo=None):
        if clear_sky:
            bright_ice = self._clear_bright_ice
            dark_ice = self._clear_dark_ice
        else:
            bright_ice = self._cloud_bright_ice
            dark_ice = self._cloud_dark_ice

        if sea_albedo is None:
            sea_albedo = self.get_sea_albedo(zeniths, clear_sky)

        bright_albedo = bright_ice.get_albedo(zeniths)
        dark_albedo = dark_ice.get_albedo(zeniths)

        albedo = np.zeros(zeniths.shape)

        has_melt = temperature >= (-1 + 273.15)
        is_thin_ice = ice_thickness < 0.5

        thin_dry = ~has_melt & is_thin_ice
        thin_melt = has_melt & is_thin_ice
        thick_dry = ~has_melt & ~is_thin_ice
        thick_melt = has_melt & ~is_thin_ice

        fh = np.arctan(4 * ice_thickness) / np.arctan(4 * .5)
        fh[fh > 1] = 1

        albedo[thick_melt] = dark_albedo[thick_melt]
        albedo[thick_dry] = bright_albedo[thick_dry]

        albedo[thin_dry] = (
            sea_albedo[thin_dry] * (1 - fh[thin_dry]) +
            bright_albedo[thin_dry] * fh[thin_dry]
        )
        albedo[thin_melt] = (
            sea_albedo[thin_melt] * (1 - fh[thin_melt]) +
            bright_albedo[thin_melt] * fh[thin_melt]
        )

        return albedo

    def get_sea_albedo(self, zeniths, clear_sky=True):
        if clear_sky:
            ocean = self._clear_ocean
        else:
            ocean = self._cloud_ocean
        albedo = ocean.get_albedo(zeniths)
        return albedo
