# Based on gptorch code by Steven Atkinson (steven@atkinson.mn)
# ------------------------------------------------
# Pablo Moreno-Munoz (pmoreno@tsc.uc3m.es)
# Universidad Carlos III de Madrid
# January 2020
# ------------------------------------------------

import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

_lim_val = np.finfo(np.float64).max
_lim_val_exp = np.log(_lim_val)
_lim_val_square = np.sqrt(_lim_val)
#_lim_val_cube = cbrt(_lim_val)
_lim_val_cube = np.nextafter(_lim_val**(1/3.0), -np.inf)
_lim_val_quad = np.nextafter(_lim_val**(1/4.0), -np.inf)
_lim_val_three_times = np.nextafter(_lim_val/3.0, -np.inf)

def squared_distance(x1, x2=None):
    """
    Given points x1 [n1 x d1] and x2 [n2 x d2], return a [n1 x n2] matrix with
    the pairwise squared distances between the points.
    Entry (i, j) is sum_{j=1}^d (x_1[i, j] - x_2[i, j]) ^ 2
    """
    if x2 is None:
        return squared_distance(x1, x1)

    x1s = x1.pow(2).sum(1, keepdim=True)
    x2s = x2.pow(2).sum(1, keepdim=True)

    r2 = x1s + x2s.t() -2.0 * x1 @ x2.t()

    # Prevent negative squared distances using torch.clamp
    # NOTE: Clamping is for numerics.
    # This use of .detach() is to avoid breaking the gradient flow.
    return r2 - (torch.clamp(r2, max=0.0)).detach()

