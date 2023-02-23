import os
import torch
import numpy as np

_lim_val = np.finfo(np.float64).max
_lim_val_exp = np.log(_lim_val)
_lim_val_square = np.sqrt(_lim_val)
#_lim_val_cube = cbrt(_lim_val)
_lim_val_cube = np.nextafter(_lim_val**(1/3.0), -np.inf)
_lim_val_quad = np.nextafter(_lim_val**(1/4.0), -np.inf)
_lim_val_three_times = np.nextafter(_lim_val/3.0, -np.inf)

def safe_exp(f):
    if torch.is_tensor(f):
        clamp_f = torch.clamp(f, min=-np.inf, max=_lim_val_exp)
        output = torch.exp(clamp_f)
    else:
        clamp_f = np.clip(f, -np.inf, _lim_val_exp)
        output = np.exp(clamp_f)
    return output

def safe_square(f):
    if torch.is_tensor(f):
        clamp_f = torch.clamp(f, min=-np.inf, max=_lim_val_square)
        output = clamp_f**2
    else:
        clamp_f = np.clip(f, -np.inf, _lim_val_square)
        output = clamp_f**2
    return output

def safe_cube(f):
    if torch.is_tensor(f):
        clamp_f = torch.clamp(f, min=-np.inf, max=_lim_val_cube)
        output = clamp_f**3
    else:
        clamp_f = np.clip(f, -np.inf, _lim_val_cube)
        output = clamp_f**3
    return output

def safe_quad(f):
    if torch.is_tensor(f):
        clamp_f = torch.clamp(f, min=-np.inf, max=_lim_val_quad)
        output = clamp_f**4
    else:
        clamp_f = np.clip(f, -np.inf, _lim_val_quad)
        output = clamp_f**4
    return output

def safe_sqrt(f):
    if torch.is_tensor(f):
        clamp_f = torch.clamp(f, min=1e-3, max=_lim_val_quad)
        output = torch.sqrt(clamp_f)
    else:
        clamp_f = np.clip(f, -np.inf, _lim_val_quad)
        output = np.sqrt(clamp_f)
    return output
