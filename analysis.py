#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 18/12/2023
"""

import numpy as np 
import jax
import matplotlib.pyplot as plt
from jax import vmap, jit
from nn import batched_predict_force
from featurize import featurize
from tqdm import tqdm
from glob import glob


def make_test_preds(params, testloader):
    ## Predict forces for test set based on coordinates
    f_pred_all = []
    f_label_all = []
    label_feat = []
    calc_feat = []

    ## Create jitted version of predict function
    # partial_pred = jax.tree_util.Partial(batched_predict_force, params=params)
    jitted_pred = jit(batched_predict_force)

    ## Loop over all batches of the testloader
    for i, (x, feat, _, f_proj, div, f) in tqdm(enumerate(testloader), total=len(testloader)):
        ## Predict force for entire batch
        f_pred_projected = jitted_pred(params, x, f_proj, div)
        f_pred_all.append(f_pred_projected)
        f_label_all.append(f)
        label_feat.append(feat)
        ## Calculate features for entire batch
        calc_feat.append(vmap(featurize)(x))

    return np.concatenate(f_pred_all), \
            np.concatenate(f_label_all), \
            np.concatenate(label_feat), \
            np.concatenate(calc_feat)


def make_label_pred_plot(ax, feat, labels, preds, plot_stride=1):
    ax.plot(
        feat[::plot_stride], labels[::plot_stride], 
        'o', markersize=1, 
        label="Labels"
        )
    
    ax.plot(
        feat[::plot_stride], preds[::plot_stride],
        'o', markersize=2, 
        label="Predicted"
        )
    
    return ax


def load_sim_coords(
        sim_folder: str, 
        sim_file_name: str,
        numfiles: int = None,
        ) -> np.ndarray:
    """Function to load simulation coordinates

    Parameters
    ----------
    sim_folder
        Base folder for simulation output
    sim_file_name
        Template file name for simulation output

    Returns
    -------
    coords
        Numpy array of stacked coordinates
    """
    ## Define number of files if not given already
    if numfiles is None:
        numfiles = len(sorted(glob(f'{sim_folder}/*')))

    coords = []
    ## Compile all coordinates
    for i in range(numfiles):
        coords.append(np.load(f'{sim_folder}/{sim_file_name.format(i)}', allow_pickle=True))

    return np.concatenate(coords)

def make_fe(phi, psi, bins=100):
    """Function to make free energy from 2 arrays

    Parameters
    ----------
    phi
        Array of phi values
    psi
        Array of psi values
    bins, optional
        Number of bins or list of bins, by default 100

    Returns
    -------
        free energy, raw histogram, bins in x, bins in y
    """
    hist, bins_1, bins_2 = np.histogram2d(
        phi, psi, 
        bins=bins
        )
    
    fe = -np.log(hist/hist.sum())

    return fe, hist, bins_1, bins_2

def make_fe_plot(
        bins_1, 
        bins_2, 
        fe,
        levels=10,
        cmap='rainbow'
        ):
    """Function to make FE contour plot, given bins and FE vals

    Parameters
    ----------
    bins_1
        Bins along 1st dimension
    bins_2
        Bins along 2nd dimension
    fe
        Free energy corresponding to bins
    levels, optional
        Number of levels in contour plot, by default 10

    Returns
    -------
    fig, ax
        Matplotlib figure and axes 
    """

    fig, ax = plt.subplots()

    ## Make contour plot
    ax.contourf(
        bins_1[:-1], bins_2[:-1],
        fe.T - fe.T.min(),
        levels=levels,
        cmap=cmap,
    )

    return fig, ax