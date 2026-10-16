import numpy as np, pandas as pd

TYPES = ['A1','A2','A3','B1','B2','B3','B4','C1','C2','C3']
VAL = {'A': 20, 'B': 5, 'C': 1}


def add_step_feats(d):
    """d: rows of one or many series sorted by step; needs sun/moon columns."""
    d = d.sort_values(['series_id', 'step_id']).copy()
    g = d.groupby('series_id')
    d['sun_prev'] = g.sun_altitude.shift(1)
    d['sun_next'] = g.sun_altitude.shift(-1)
    d['moon_prev'] = g.moon_altitude.shift(1)
    d['moon_next'] = g.moon_altitude.shift(-1)
    # new night: sun alt jumps from near-horizon morning to evening
    d['dsun_next'] = d.sun_next - d.sun_altitude
    d['dmoon_next'] = d.moon_next - d.moon_altitude
    d['dillum_next'] = g.moon_illumination.shift(-1) - d.moon_illumination
    return d


def obj_feats(d):
    d = d.copy()
    a = d['action'].fillna('REST')
    isobj = ~a.str.startswith('REST')
    d['cls'] = np.where(isobj, a.str[0], 'R')
    d['cls_i'] = d.cls.map({'A': 0, 'B': 1, 'C': 2, 'R': 3})
    d['type_i'] = a.map({t: i for i, t in enumerate(TYPES)}).fillna(-1)
    d['moon_term'] = d.moon_illumination * np.clip(d.moon_altitude, 0, None) / np.maximum(d.moon_separation, 1)
    d['airmass'] = 1 / np.sin(np.radians(np.clip(d.object_altitude, 5, 90)))
    d['dT_amb'] = d.temperature - d.ambient_temperature
    return d
