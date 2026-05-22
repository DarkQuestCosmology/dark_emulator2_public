import numpy as np
import scipy
import frozendict
import functools

import re
from packaging import version

cache_size = 64

scipy_ver = re.sub("[^0-9^.]", "", scipy.__version__)
if version.parse(scipy_ver) >= version.parse('1.10.0'):
    from scipy.interpolate import RegularGridInterpolator as rgi

    def create_interp2d(x, y, xydata, method="cubic"):
        # method: ["linear", "cubic"]
        return rgi((y, x), xydata, method=method, bounds_error=False, fill_value=None)

    def call_interp2d(x, y, xyinterp): return xyinterp((y, x))
else:
    from scipy.interpolate import interp2d
    def create_interp2d(x, y, xydata, method="cubic"): return interp2d(x, y, xydata, kind=method)

    def call_interp2d(x, y, xyinterp):
        if y.ndim > 1: y = np.squeeze(y)
        return xyinterp(x, y)


def split_spacing(s):
    # Blanks enclosed in () are not split.
    return re.split(r'\s+(?![^\(]*\))', s)


def read_header_label(filename):
    # read first line only
    with open(filename) as f:
        head = f.readline().rstrip()
        head = head.replace("#", "")
        head = head.lstrip()
    head = split_spacing(head)
    return head


def dict_to_sarray(d, dtype=np.float64):
    if isinstance(d, list):
        l_length = len(d)
        d_length = len(d[0])
        param_dtype = np.dtype(
            {'names': tuple(d[0].keys()), 'formats': (dtype,) * d_length})
        sarray = np.array([tuple(d[ix].values())
                          for ix in range(l_length)], dtype=param_dtype)
    else:
        param_dtype = np.dtype(
            {'names': tuple(d.keys()), 'formats': (dtype,) * len(d)})
        sarray = np.array(tuple(d.values()), param_dtype)
    return sarray


def sarray_to_dict(sa):
    sa = np.asarray(sa)
    scalar_input = False
    if sa.ndim == 0:
        sa = sa[np.newaxis]  # Makes x 1D
        scalar_input = True

    sa_range = range(len(sa))
    names = sa.dtype.names
    d = [{ix: sa[isa][ix] for ix in names} for isa in sa_range]

    if scalar_input:
        return np.squeeze(d)
    return d


def is_num(s):
    try:
        float(s)
    except ValueError:
        return False
    else:
        return True


# <https://stackoverflow.com/questions/6358481/using-functools-lru-cache-with-dictionary-arguments>
def freezeargs(func):
    """Transform mutable dictionnary
    Into immutable
    Useful to be compatible with cache
    """
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        args = tuple([frozendict.frozendict(arg) if isinstance(arg, dict) else arg for arg in args])
        kwargs = {k: frozendict.frozendict(v) if isinstance(v, dict) else v for k, v in kwargs.items()}
        return func(*args, **kwargs)
    return wrapped
