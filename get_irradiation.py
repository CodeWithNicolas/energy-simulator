import requests
import numpy as np


def get_hourly_irradiation(lat, lon, year=None):
    """Fetch hourly global irradiation (GHI) from Open-Meteo ERA5 API."""
    url = "https://archive-api.open-meteo.com/v1/era5"
    if year is None:
        from datetime import datetime

        year = datetime.utcnow().year - 1
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "shortwave_radiation",
        "start_date": f"{year}-01-01",
        "end_date": f"{year}-12-31",
        "timezone": "UTC",
    }
    r = requests.get(url, params=params)
    if not r.ok:
        raise RuntimeError(
            f"Open-Meteo ERA5 API request failed: {r.status_code} {r.text}"
        )
    data = r.json()
    try:
        ghi = np.array(data["hourly"]["shortwave_radiation"], dtype=float) / 1000.0
        if len(ghi) == 8784:
            # Remove the 24 hours of Feb 29th (hours 1416 to 1439)
            ghi = np.delete(ghi, np.s_[1416:1440])
        return ghi
    except Exception as e:
        print(f"Available keys in Open-Meteo response: {list(data.keys())}")
        raise RuntimeError(f"Could not parse Open-Meteo response: {e}")
