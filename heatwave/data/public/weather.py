"""Weather helpers: locations, variables and wet-bulb temperature.

Wet bulb: Stull (2011), "Wet-Bulb Temperature from Relative Humidity and Air Temperature",
J. Appl. Meteor. Climatol. 50, 2267-2269, doi:10.1175/JAMC-D-11-0143.1 (valid for RH 5-99 %, T -20..50 degC).
"""

from __future__ import annotations

import numpy as np

# name -> (latitude, longitude, description). Points chosen near the grid regions of the other series.
LOCATIONS = {
    "bayarea": (37.35, -121.96, "Santa Clara, California (CAISO NP15 region, Bay Area data-centre cluster)"),
    "london": (51.51, -0.13, "London, United Kingdom (GB grid)"),
    "frankfurt": (50.11, 8.68, "Frankfurt am Main, Germany (DE-LU bidding zone, large data-centre cluster)"),
}


def wet_bulb_stull(t_c, rh_pct):
    """Wet-bulb temperature in degC from air temperature (degC) and relative humidity (%), Stull (2011)."""
    t = np.asarray(t_c, dtype="float64")
    rh = np.clip(np.asarray(rh_pct, dtype="float64"), 0.0, 100.0)
    return (
        t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
        + np.arctan(t + rh)
        - np.arctan(rh - 1.676331)
        + 0.00391838 * rh**1.5 * np.arctan(0.023101 * rh)
        - 4.686035
    )
