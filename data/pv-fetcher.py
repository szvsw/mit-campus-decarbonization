import os

import numpy as np
import pandas as pd
import plotly.express as px
import requests
from dotenv import load_dotenv

load_dotenv()

if __name__ == "__main__":
    # cambridge MA lat lon
    lat = 42.3736
    lon = -71.1071

    response = requests.get(
        "https://developer.nrel.gov/api/pvwatts/v8.json",
        params={
            "api_key": os.getenv("PVWATTS_API_KEY"),
            "module_type": 0,
            "losses": 0,
            "system_capacity": 0.19,  # 150W panel is roughly a m^2
            "dc_ac_ratio": 1,
            "array_type": 1,
            "tilt": 0,
            "azimuth": 180,
            "lat": lat,
            "lon": lon,
            "inv_eff": 99,
            "timeframe": "hourly",
            # TODO: Do we need to change gcr?
        },
    )

    res = response.json()
    pv_sqm_kW = np.array(res["outputs"]["ac"]) / 1000

    total_roof_area_ft2 = 2.5e6  # https://www.arrowstreet.com/portfolio/sustainable-roof-study/#:~:text=The%20campus%20is%20169%2Dacres,percent%20of%20MIT's%20land%20area.
    total_roof_area_m2 = total_roof_area_ft2 * 0.092903

    pv_100_kW = pv_sqm_kW * total_roof_area_m2
    pv_50_kW = pv_100_kW * 0.50
    pv_75_kW = pv_100_kW * 0.75

    timestamp = pd.date_range(start="2024-01-01", periods=8760, freq="h")
    print(len(timestamp), len(pv_sqm_kW), len(pv_100_kW), len(pv_50_kW), len(pv_75_kW))
    df = pd.DataFrame(
        {
            "kW per sqm": pv_sqm_kW,
            "kW 100 coverage": pv_100_kW,
            "kW 50 coverage": pv_50_kW,
            "kw 75 coverage": pv_75_kW,
        },
        index=timestamp,
    )
    df.to_hdf("data/pv.hdf", key="pv")
