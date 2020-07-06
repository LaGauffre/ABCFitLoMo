# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 17:16:18 2020

@author: Pat and Pierre-O
"""

# This tells Python to just use one thread for internal
# math operations. When it tries to be smart and do multithreading
# it usually just stuffs everything up and slows us down.
import os
for env in ["MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "OMP_NUM_THREADS"]:
    os.environ[env] = "1"

# Reminder that we need a relatively new version of numpy to make
# use of the latest pseudorandom number generation algorithms.
import numpy
if int(numpy.__version__.split('.')[1]) < 17:
    raise RuntimeError("Need Numpy version >= 1.17")

# Similarly, we need a new version of scipy.stats to make
# use of the latest gaussian_kde (where we can set the seed for resample).
import scipy
if int(scipy.__version__.split('.')[1]) < 4:
    raise RuntimeError("Need Scipy version >= 1.4.0")

import matplotlib.pyplot as plt # For plotting
import numpy as np # For fast math

# New way to get pseudorandom numbers
from numpy.random import SeedSequence, default_rng, Generator, PCG64

import dill

# The main part of our code
import abcre

import pandas as pd
import seaborn as sns
import pymc3 as pm

import subprocess
import shutil

def crop_fig(filename):
    cmd = f"pdf-crop-margins {filename} -o cropped.pdf"
    proc = subprocess.Popen(cmd.split())
    proc.wait()
    shutil.move("cropped.pdf", filename)
    
def save_cropped(filename):
    plt.savefig(filename.replace("pdf", "pgf"), pad_inches=0)
    plt.savefig(filename, pad_inches=0)
    crop_fig(filename)
    plt.show()
    
plt.rcParams['figure.figsize'] = (5.0, 2.0)
plt.rcParams['figure.dpi'] = 350
plt.rcParams['savefig.bbox'] = "tight"
plt.rcParams['font.family'] = "serif"

# This is the default color scheme
colors = [
    "tab:blue",
#     "tab:orange", # but this yellow/orange color is hideous.
    "tab:green",
    "tab:red",
    "tab:purple",
    "tab:brown",
    "tab:pink",
    "tab:gray",
    "tab:olive",
    "tab:cyan",
]

import cycler
plt.rcParams["axes.prop_cycle"] = cycler.cycler(color=colors)

priorColor = colors[3]

trueStyle = {"color": "black", "linestyle": "--"}
mleStyle = {"color": "black", "linestyle": ":"}

from ttictoc import tic, toc

def draw_prior(prior, axs):
    lines = []
    for i, priorI in enumerate((prior.marginals)):
        priorL = priorI.isf(1)
        priorR = priorI.isf(0)
        
        if priorL > 0:
            xlimL = priorL * 0.9
        elif priorL == 0:
            if priorR == 1 or priorR == 1.0:
                xlimL = -0.1
            else:
                xlimL = -1
        else:
            xlimL = priorL * 1.1

        xlimR = priorR * 1.1

        xs = np.linspace(xlimL, xlimR, 100)
        xs = np.sort(
            np.concatenate(
                (
                    xs,
                    [
                        priorL,
                        priorL - 1e-8,
                        priorL + 1e-8,
                        priorR,
                        priorR - 1e-8,
                        priorR + 1e-8,
                    ],
                )
            )
        )

        (priorLine,) = axs[i].plot(xs, priorI.pdf(xs), label="Prior", color=priorColor, alpha=0.75, zorder = 0)
        lines.append(priorLine)

    return(lines)
