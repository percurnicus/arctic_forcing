from datetime import timedelta

import pysolar
import numpy as np

from cmip5 import CMIP5, CltCMIP5


class DataSet:

    def __init__(self, sic_path, sit_path, tas_path, clt_path, sic_scale,
                 clt_scale):
        print('loading data')
        self.sic = CMIP5(sic_path, sic_scale)
        self.sit = CMIP5(sit_path)
        self.tas = CMIP5(tas_path)
        self.clt = CltCMIP5(clt_path, clt_scale)

        model_cmips = (self.sic, self.sit, self.tas)

        self.start_date = max([cmip.start_date for cmip in model_cmips])
        self.start_date_np = np.datetime64(
            self.start_date.replace(tzinfo=None)
        )
        for cmip in model_cmips:
            cmip.set_delta(self.start_date)

        self.clt.set_delta(self.start_date)

        model_cmips = (self.sic, self.sit, self.tas, self.clt)

        lowest_res = min(model_cmips, key=lambda cmip: cmip.lats.size)
        self.lats, self.lons = lowest_res.lats.copy(), lowest_res.lons.copy()
        print('Setting Uniform Grid')
        for cmip in model_cmips:
            cmip.set_grid_data(self.lats, self.lons)

        for cmip in model_cmips:
            cmip.set_interpolation()

        self.areas = self._get_areas()

    def get_zeniths(self, times):
        times = times.astype('timedelta64[s]')
        dates = self.start_date_np + times
        dates = dates.reshape((dates.size, 1, 1))
        altitude = pysolar.solar.get_altitude_fast(self.lats, self.lons, dates)
        zenith = 90 - altitude
        return zenith

    def get_zeniths_scalar(self, time):
        date = self.start_date + timedelta(seconds=time)
        altitude = pysolar.solar.get_altitude_fast(self.lats, self.lons, date)
        zenith = 90 - altitude
        return zenith

    @staticmethod
    def lat_lon_area(lat1, lat2, lon1, lon2):
        R = 6.3781e6
        rlat1 = np.deg2rad(lat1)
        rlat2 = np.deg2rad(lat2)
        circ_area = (np.pi / 180) * R**2
        lon_diff = np.abs(lon1 - lon2)
        between_lats = np.abs(np.sin(rlat1) - np.sin(rlat2))
        return circ_area * between_lats * lon_diff

    def _get_areas(self):
        lons, lats = self.lons, self.lats
        lons = lons[0]
        lats = lats[:, 0]
        lon_step = (lons[1] - lons[0]) * .5
        lat_step = (lats[1] - lats[0]) * .5
        lon = 90
        lon1 = lon - lon_step
        lon2 = lon + lon_step
        areas = self.lat_lon_area(
            lat1=lats - lat_step,
            lat2=lats + lat_step,
            lon1=lon1,
            lon2=lon2,
        )
        areas[0] = self.lat_lon_area(
            lat1=lats[0],
            lat2=(lats + lat_step)[0],
            lon1=lon1,
            lon2=lon2,
        )
        areas[-1] = self.lat_lon_area(
            lat1=(lats + lat_step)[-2],
            lat2=lats[-1],
            lon1=lon1,
            lon2=lon2,
        )
        areas = np.tile(areas.reshape((len(areas), 1)), len(lons))
        return areas
