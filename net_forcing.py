import dateutil
from datetime import timedelta

import numpy as np
import pandas as pd
from tqdm import tqdm


def _get_E(times, delta_t, data_set, albedos):
    zeniths = data_set.get_zeniths(times)
    cos_zeniths = np.cos(np.deg2rad(zeniths))
    cos_zeniths[cos_zeniths < 0] = 0

    ice_data = data_set.sic.get_data(times)
    cloud_data = data_set.clt.get_data(times)
    thickness = data_set.sit.get_data(times)
    temperature = data_set.tas.get_data(times)

    cloud_data[cloud_data.mask] = 0

    sic_dates = pd.to_datetime(data_set.sic.get_date(times))
    for cmip in (data_set.sit, data_set.tas, data_set.clt):
        assert np.array_equal(
            sic_dates,
            pd.to_datetime(cmip.get_date(times)),
        )
    clt_dates = pd.to_datetime(data_set.clt.get_date(times))
    assert np.array_equal(clt_dates.day, sic_dates.day)
    assert np.array_equal(clt_dates.month, sic_dates.month)
    assert np.array_equal(clt_dates.hour, sic_dates.hour)
    assert np.array_equal(clt_dates.minute, sic_dates.minute)
    assert np.array_equal(clt_dates.second, sic_dates.second)

    a_Ocld = albedos.get_sea_albedo(zeniths, clear_sky=False)
    a_Oclr = albedos.get_sea_albedo(zeniths, clear_sky=True)
    a_Icld = albedos.get_ice_albedo(
        zeniths=zeniths,
        ice_thickness=thickness,
        temperature=temperature,
        clear_sky=False,
        sea_albedo=a_Ocld,
    )
    a_Iclr = albedos.get_ice_albedo(
        zeniths=zeniths,
        ice_thickness=thickness,
        temperature=temperature,
        clear_sky=True,
        sea_albedo=a_Oclr,
    )

    phi = (1 - a_Icld) * (ice_data * cloud_data)
    beta = (1 - a_Iclr) * (ice_data * (1 - cloud_data))
    omega = (1 - a_Ocld) * ((1 - ice_data) * cloud_data)
    xi = (1 - a_Oclr) * ((1 - ice_data) * (1 - cloud_data))

    E = cos_zeniths * (phi + beta + omega + xi)
    assert (~E.mask).any()
    E.mask = thickness.mask

    return E


def _get_E_tot(start_date, delta_t, data_set, albedos):
    end_date = start_date + dateutil.relativedelta.relativedelta(years=1)
    # stop_date = end_date - timedelta(seconds=delta_t)

    date = start_date

    time = (start_date - data_set.start_date).total_seconds()

    E_integral = np.ma.array(
        np.zeros(data_set.sic.mask.shape),
        mask=data_set.sic.mask,
    )
    default_chunk_size = 1 * 24 * 60 * 60  # 1 day at a time
    # Trapezoidal integration: dx / 2 * (f(x_{i-1}) + f(x_i))
    with tqdm(total=(end_date - start_date).total_seconds()) as pbar:
        while date < end_date:
            seconds_remaining = (end_date - date).total_seconds()
            if seconds_remaining < default_chunk_size:
                chunk_size = seconds_remaining
            else:
                chunk_size = default_chunk_size
            times = np.arange(time, time + chunk_size + delta_t, delta_t)
            E = _get_E(
                times=times,
                delta_t=delta_t,
                data_set=data_set,
                albedos=albedos,
            )
            # E_integral = E_integral + np.sum(E * delta_t, axis=0)
            chunk_integral = np.trapz(E, dx=delta_t, axis=0)
            E_integral = E_integral + chunk_integral
            # Increase by the chunk size so the last date is repeated
            date += timedelta(seconds=chunk_size)
            time += chunk_size

            pbar.update(chunk_size)

    S = 1365
    E_integral = E_integral * S * data_set.areas

    E_tot = np.sum(E_integral.data[~E_integral.mask])
    return E_tot


def get_radiative_forcing(start_date, delta_t, data_set, albedos):
    E_tot = _get_E_tot(
        start_date=start_date,
        delta_t=delta_t,
        data_set=data_set,
        albedos=albedos,
    )

    end_date = start_date + dateutil.relativedelta.relativedelta(years=1)
    year_secs = (end_date - start_date).total_seconds()
    earth_surface_area = data_set.lat_lon_area(-90, 90, 0, 360)

    forcing = E_tot / (earth_surface_area * year_secs)

    return forcing
