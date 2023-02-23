import numpy as np
import matplotlib.pyplot as plt


def acgp_color(steps=5):
    return plt.cm.Reds(np.linspace(0.4, 0.8, steps))


def cglb_color(steps=3):
    return plt.cm.Blues(np.linspace(0.4, 0.8, steps))


def exact_color(steps=3):
    return plt.cm.Greys(np.linspace(0.4, 0.8, steps))


def svgp_color(steps=3):
    return plt.cm.Greens(np.linspace(0.4, 0.8, steps))
