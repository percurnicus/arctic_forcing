import json
from datetime import datetime

import dateutil
import numpy as np

from albedos import Albedos
from data_set import DataSet
from net_forcing import get_radiative_forcing

SIC_PATH = 'sic_day_GFDL-CM3_rcp45_r1i1p1_20[56]*'
SIT_PATH = 'sit_day_GFDL-CM3_rcp45_r1i1p1*.nc'
TAS_PATH = 'tas_3hr_GFDL-CM3_rcp45_r1i1p1_*.nc'
# CLT_PATH = 'clt_3hr_GFDL-CM3_rcp45_r1i1p1_*.nc'
CLT_PATH = 'tcdc.eatm.gauss.19[89]*.nc'

BEGIN_DATE = datetime(2056, 1, 1, 0)
NUM_YEARS = 10

DELTA_T = 150


if __name__ == '__main__':
    print('Creating DataSet')
    data_set = DataSet(
        sic_path=SIC_PATH,
        sit_path=SIT_PATH,
        tas_path=TAS_PATH,
        clt_path=CLT_PATH,
        sic_scale=.01,
        clt_scale=.01
    )
    print('Getting Albedos')
    albedos = Albedos()
    year = dateutil.relativedelta.relativedelta(years=1)
    rad_start_dates = [BEGIN_DATE + year * n for n in range(NUM_YEARS)]
    forcings = []
    for rad_start_date in rad_start_dates:
        forcing = get_radiative_forcing(
            start_date=rad_start_date,
            delta_t=DELTA_T,
            data_set=data_set,
            albedos=albedos,
        )
        print(rad_start_date, forcing)
        forcings.append(forcing)

    out = {
        date.isoformat(): forcing for date, forcing
        in zip(rad_start_dates, forcings)
    }
    out['mean'] = np.mean(forcings)
    out['std'] = np.std(forcings)
    with open(f'ice_free_delta_t_{DELTA_T}.json', 'w') as stream:
        json.dump(out, stream, indent=4)
