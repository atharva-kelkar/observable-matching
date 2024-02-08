#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import numpy as np

def gaussian_densities(x, centers, kernel_size, periodic=False):
    if np.ndim(x) == 1 and np.ndim(centers) == 1:
        x = np.expand_dims(x, 1)
        centers = np.expand_dims(centers, 1)
    square_dists = np.sum((np.expand_dims(x, 2) - np.expand_dims(centers.T, 0))**2, axis=1)
    if periodic:
        I = square_dists > np.pi**2
        square_dists[I] = (2*np.pi - np.sqrt(square_dists[I]))**2
    state_features = (1.0/kernel_size) * np.exp(-square_dists / (2*kernel_size**2))
    
    return state_features


def gaussian_memberships(x, centers, kernel_size, periodic=False):
    state_features = gaussian_densities(x, centers, kernel_size, periodic=periodic)
    # normalize
    state_features /= state_features.sum(axis=1, keepdims=True)
    
    return state_features


def histogram_density(x, nbins=100):
    """ Density estimate using a histogram """
    xgrid = np.linspace(x.min(), x.max(), 100)
    hist, bin_edges = np.histogram(x, xgrid)
    bin_positions = 0.5*(bin_edges[:-1]+bin_edges[1:])
    x_density = hist / hist.sum()
    
    return bin_positions, x_density


def gaussian_kde(x, n=100, kernel_size_multiplier=2.0, periodic=False):
    """ Kernel density estimate in a Gaussian basis
    
    Parameters
    ----------
    x : samples
    n : number of points to evaluate the density at
    kernel_size_multiplier : width of Gaussians in terms of multiple of the distance between subsequent grid points
    
    Returns
    -------
    centers : positions of Gaussian means
    kernel_size : standard deviation of Gaussian
    kde : density in the Gaussian basis.
    
    """
    
    centers = np.linspace(np.min(x), np.max(x), n)
    kernel_size = kernel_size_multiplier*(centers[1]-centers[0])
    kde = gaussian_memberships(x, centers, kernel_size, periodic=periodic).sum(axis=0)
    kde /= kde.sum()    
    
    return centers, kernel_size, kde

def gaussian_kde_adaptive(x, n_adaptive_centers, n=50, periodic=False):
    """ Kernel density estimate in a Gaussian basis
    
    Parameters
    ----------
    x : samples
    n : number of points to evaluate the density at
    
    Returns
    -------
    centers : positions of Gaussian means
    kernel_size : standard deviation of Gaussian
    kde : density in the Gaussian basis.
    
    """
    n_data_per_center = int(x.size / n)
    adaptive_centers = []
    adaptive_widths = []
    x_sorted = np.sort(x)
    print(x_sorted)
    
    for i in range(n_adaptive_centers):
        batch = x_sorted[i*n_data_per_center:(i+1)*n_data_per_center]
        adaptive_centers.append(np.mean(batch))
        adaptive_widths.append(np.std(batch))
    adaptive_centers = np.array(adaptive_centers)
    adaptive_widths = np.array(adaptive_widths)

    kde = gaussian_densities(x, adaptive_centers, adaptive_widths, periodic=periodic).sum(axis=0)
    kde /= kde.sum()    
    
    return adaptive_centers, adaptive_widths, kde


def gaussian_kde_adaptive2(x, ncenter_max=50, ndata_min=100, periodic=False):
    """ Kernel density estimate in a Gaussian basis
    
    Parameters
    ----------
    x : samples
    ncenter_max : the maximum number of centers to evaluate the density at
    ndata_min : the minimum number of datapoints for each center - centers will be grown until this number is reached
    
    Returns
    -------
    centers : positions of Gaussian means
    kernel_size : standard deviation of Gaussian
    kde : density in the Gaussian basis.
    
    """
    n_data_per_center = int(x.size / ncenter_max)
    adaptive_centers = []
    adaptive_widths = []
    x_sorted = np.sort(x)
    x_range = x_sorted[-1] - x_sorted[0]
    x_binsize = x_range / ncenter_max
    last_cut_x = x_sorted[0]
    last_cut_idx = 0
    
    while(last_cut_idx < x.size):
        # find next separator
        #print('cut', last_cut_x + x_binsize)
        new_cut_idx = np.searchsorted(x_sorted, last_cut_x + x_binsize)
        #print(new_cut_idx)
        # if there's not enough data, move separator to achieve ndata_min.
        new_cut_idx = max(new_cut_idx, last_cut_idx + ndata_min)
        # if we are at the end, just merge into one bin
        if new_cut_idx > x.size - ndata_min + 1:
            new_cut_idx = x.size
        #print(new_cut_idx)
        # cut data and define center and width
        batch = x_sorted[last_cut_idx:new_cut_idx]
        #print(batch.size)
        adaptive_centers.append(np.mean(batch))
        adaptive_widths.append(np.std(batch))
        # update counters
        last_cut_idx = new_cut_idx
        last_cut_x = batch[-1]
        
    adaptive_centers = np.array(adaptive_centers)
    adaptive_widths = np.array(adaptive_widths)

    kde = gaussian_densities(x, adaptive_centers, adaptive_widths, periodic=periodic).sum(axis=0)
    kde /= kde.sum()    
    
    return adaptive_centers, adaptive_widths, kde    

def density2force(x, density, beta=1, periodic=False):
    # fe = - beta^-1 * log(density)
    energy = - beta**(-1) * np.log(density)
    I = np.isfinite(density)
    d_energy = energy[I][1:] - energy[I][:-1]
    d_x = x[I][1:] - x[I][:-1]
    forces = -d_energy / d_x
    if periodic:
        force_start = (energy[I][0]-energy[I][-1]) / (2*np.pi - (x[I][-1]-x[I][0]))
        forces = np.concatenate([[force_start], forces, [force_start]])
    else:
        forces = np.concatenate([[2*forces[0]-forces[1]], forces, [2*forces[-1]-forces[-2]]])
    # average neighboring forces
    forces = 0.5 * (forces[1:] + forces[:-1])
    
    return forces

def compute_forces(x, grid, grid_forces):
    """ Interpolate forces for values x if we have discretized forces on a grid """
    binsize = grid[1]-grid[0]
    xbin = (x-grid[0]) / binsize
    i_lower = (np.floor(xbin)).astype(int)
    i_lower = np.maximum(i_lower, 0)
    i_upper = (np.ceil(xbin)).astype(int)
    i_upper = np.minimum(i_upper, grid.size-1)
    forces = grid_forces[i_lower] + ((x - grid[i_lower])/binsize) * (grid_forces[i_upper] - grid_forces[i_lower])
    return forces

def average_to_bins(x, y, xgrid):
    """ Averages the signal y to a regular 1d-grid """
    binsize = xgrid[1]-xgrid[0]
    xbin = (x-xgrid[0]) / binsize
    i_lower = (np.floor(xbin)).astype(int)
    i_lower = np.maximum(i_lower, 0)
    i_upper = (np.ceil(xbin)).astype(int)
    i_upper = np.minimum(i_upper, xgrid.size-1)
    grid_y = np.zeros(xgrid.size)
    for i in range(xgrid.size):
        grid_y[i] = np.mean(np.concatenate([y[i_lower==i], y[i_upper==i]]))
    return grid_y
