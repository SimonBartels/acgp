import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader

def true_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        5*torch.cos(7*np.pi*x + 2.4*np.pi)
    return y

def smooth_function(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi)
    return y

def smooth_function_bias(x):
    y = 4.5*torch.cos(2*np.pi*x + 1.5*np.pi) - \
        3*torch.sin(4.3*np.pi*x + 0.3*np.pi) + \
        3.0*x - 7.5
    return y
