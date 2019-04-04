import os
from contextlib import closing
from datetime import datetime

import netCDF4
import numpy as np
from scipy.interpolate import griddata, interp1d


class CMIP5:

    def __init__(self, filep, scale=1):
        self._filep = filep
        if isinstance(filep, list):
            name = filep[0]
        else:
            name = filep
        key = os.path.basename(name).split('_')[0]
        if key.endswith('nc'):
            key = os.path.basename(name).split('.')[0]
        with closing(netCDF4.MFDataset(filep)) as ds:
            lats = ds.variables['lat'][:]
            lons = ds.variables['lon'][:]
            lons[lons < 0] = lons[lons < 0] + 360
            arctic_mask = np.array(lats >= 65)
            if lats.ndim == 1:
                lats = np.array(lats[arctic_mask])
                self.lons, self.lats = np.meshgrid(lons, lats)
                self.data = ds.variables[key][:][:, arctic_mask, ...]
            else:
                arctic_mask = np.any(arctic_mask, axis=1)
                self.lats = np.array(lats[arctic_mask, ...])
                self.lons = np.array(lons[arctic_mask, ...])
                self.data = ds.variables[key][:][:, arctic_mask, ...]
            self.data = self.data * scale
            self.data = np.ma.array(
                self.data,
                mask=np.ma.getmaskarray(self.data)
            )
            ds_time = ds.variables['time']
            self.dates = netCDF4.num2date(ds_time[:], ds_time.units)
            self.start_date = self.dates[0]
            self.end_date = self.dates[-1]
            self.times = np.array(
                [
                    (date - self.start_date).total_seconds()
                    for date in self.dates
                ]
            )
            self.units = ds.variables[key].units
            self.long_name = ds.variables[key].long_name
        self.key = key
        self._interp_seconds = None
        self._interp_mask = None
        self._delta = None

    def __repr__(self):
        return f'{self.__class__.__name__}({self._filep})'

    @property
    def mask(self):
        mask = np.ma.getmaskarray(self.data[0])
        for data in self.data[1:]:
            mask = mask & np.ma.getmaskarray(data)
        return mask

    def get_data(self, time):
        time = time + self._delta
        data = self._interp_seconds(time)
        mask = self._interp_mask(time)
        return np.ma.array(data, mask=mask)

    def get_date(self, time):
        delta = np.timedelta64(int(self._delta), 's')
        time = time.astype('timedelta64[s]')
        date = (np.datetime64(self.start_date, 's') + delta + time)
        return date

    def set_interpolation(self):
        self._interp_seconds = interp1d(
            self.times,
            self.data,
            axis=0,
            fill_value='extrapolate',
        )
        self._interp_mask = interp1d(
            self.times,
            np.ma.getmaskarray(self.data),
            axis=0,
            fill_value='extrapolate',
        )

    def set_delta(self, ref_date):
        self._delta = int((ref_date - self.start_date).total_seconds())

    def set_grid_data(self, lats, lons):
        same_lats = np.array_equal(lats, self.lats)
        same_lons = np.array_equal(lons, self.lons)
        if same_lats and same_lons:
            return
        lats = lats.copy()
        lons = lons.copy()

        datas = []
        points = self.lats.copy().flatten(), self.lons.copy().flatten()
        for datain in self.data:
            data = griddata(
                points=points,
                values=datain.flatten(),
                xi=(lats, lons),
                method='nearest'
            )
            datas.append(data)
        self.data = np.ma.vstack(
            [np.ma.expand_dims(d, axis=0) for d in datas]
        )

        self.lons = lons
        self.lats = lats


class CltCMIP5(CMIP5):

    leap_years = np.arange(1972, 3000, 4).astype('str').astype('datetime64[Y]')

    def __init__(self, filep, scale=1):
        super().__init__(filep, scale)
        dates = []
        groups = []
        assert self.dates.astype('datetime64[Y]')[0] in self.leap_years
        for i in range(4):
            group_years = self.leap_years + i
            group_mask = np.isin(
                self.dates.astype('datetime64[Y]'),
                group_years
            )
            group_dates = self.dates[group_mask]
            num_years = np.unique(group_dates.astype('datetime64[Y]')).size
            days_in_year = 365 if i > 0 else 366
            data_per_day = group_dates.size // (days_in_year * num_years)
            group_shape = (num_years, days_in_year, data_per_day)
            group_data = self.data[group_mask, ...]
            data_shape = tuple(list(group_shape) + list(group_data.shape)[1:])
            year_data = group_data.reshape(data_shape)
            avg_data = np.mean(year_data, axis=0)
            final_shape = tuple(
                [data_per_day * days_in_year] + list(avg_data.shape)[2:]
            )
            data = avg_data.reshape(final_shape)
            groups.append(data)
            dates.append(group_dates[:data_per_day * days_in_year])
        self.dates = np.concatenate(dates)
        self.data = np.concatenate(groups)
        self.start_date = self.dates[0]
        self.times = np.array(
            [
                (date - self.start_date).total_seconds()
                for date in self.dates
            ]
        )

    def set_delta(self, ref_date):
        self._delta = np.datetime64(ref_date)
        diff = (self.leap_years - np.datetime64(ref_date, 'Y')).astype(int)
        ref_leap_year = ref_date.year - int(np.abs(diff[diff < 0]).min())
        ref_leap_year = np.datetime64(str(ref_leap_year), 's')
        self._ref_leap_year = ref_leap_year

    def _fix_future_time(self, time):
        if np.all(time <= self.times[-1]):
            return time
        four_years = np.timedelta64(365 * 3 + 366, 'D')
        mask = time > self.times[-1]
        time_to_change = time[mask]
        dates = self._ref_leap_year + time_to_change.astype('timedelta64[s]')
        new_dates = dates - four_years.astype('timedelta64[s]')
        new_time = (new_dates - self._ref_leap_year).astype('timedelta64[s]')
        new_time = new_time.astype(int)
        time[mask] = self._fix_future_time(new_time)
        return time

    def _fix_past_time(self, time):
        if np.all(time >= self.times[0]):
            return time
        four_years = np.timedelta64(365 * 3 + 366, 'D')
        mask = time < self.times[0]
        time_to_change = time[mask]
        dates = self._ref_leap_year + time_to_change.astype('timedelta64[s]')
        new_dates = dates + four_years.astype('timedelta64[s]')
        new_time = (new_dates - self._ref_leap_year).astype('timedelta64[s]')
        new_time = new_time.astype(int)
        time[mask] = self._fix_past_time(new_time)
        return time

    def get_data(self, time):
        time = time.astype('timedelta64[s]')
        dates = self._delta + time
        time = (dates - self._ref_leap_year).astype('timedelta64[s]')
        time = time.astype(int)
        time = self._fix_future_time(time)
        time = self._fix_past_time(time)
        data = self._interp_seconds(time)
        mask = self._interp_mask(time)
        return np.ma.array(data, mask=mask)

    def get_date(self, time):
        time = time.astype('timedelta64[s]')
        dates = self._delta + time
        time = (dates - self._ref_leap_year).astype('timedelta64[s]')
        time = time.astype(int)
        time = self._fix_future_time(time)
        time = self._fix_past_time(time)
        dates = np.datetime64(self.start_date) + time.astype('timedelta64[s]')
        dates = dates.astype(datetime)
        return dates
