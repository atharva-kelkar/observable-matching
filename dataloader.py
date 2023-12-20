#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import torch
import numpy as np


def custom_collate(batch):
    transposed_data = list(zip(*batch))
    
    coords = np.array(transposed_data[0])
    ics = np.array(transposed_data[1])
    f_cg = np.array(transposed_data[2])
    f_proj = np.array(transposed_data[3])
    div_term = np.array(transposed_data[4])
    kde_forces = np.array(transposed_data[5])
    
    return coords, ics, f_cg, f_proj, div_term, kde_forces

def create_test_train_datasets(
        aladi_crd,
        all_x,
        aladi_cg_force,
        cg_atoms,
        force_proj_arr,
        div_arr,
        interpolated_forces,
        training_test_ratio = 5,
        batch_size = 64,
        test_batch_size = 1024 * 10,
        ):
    
    # train/test indices
    I_test = np.arange(0, all_x.shape[0], training_test_ratio)
    I_train = np.array(list(set(list(np.arange(0, all_x.shape[0]))) - set(list(I_test))))

    ## Make training and test datasets
    trainset = torch.utils.data.TensorDataset(torch.tensor(aladi_crd[I_train][:, cg_atoms]),
                                          torch.tensor(all_x[I_train]), # all_x is array of all internal coordinates
                                          torch.tensor(aladi_cg_force[I_train]),
                                          torch.tensor(force_proj_arr[I_train]),
                                          torch.tensor(div_arr[I_train]),
                                          torch.tensor(interpolated_forces[I_train].astype(np.float32)))

    testset = torch.utils.data.TensorDataset(torch.tensor(aladi_crd[I_test][:, cg_atoms]),
                                            torch.tensor(all_x[I_test]), # all_x is array of all internal coordinates
                                            torch.tensor(aladi_cg_force[I_test]),
                                            torch.tensor(force_proj_arr[I_test]),
                                            torch.tensor(div_arr[I_test]),
                                            torch.tensor(interpolated_forces[I_test].astype(np.float32)),
                                            )

    ## Make torch DataLoaders
    trainloader = torch.utils.data.DataLoader(trainset,
                                            batch_size=batch_size, 
                                            shuffle=True, 
                                            collate_fn=custom_collate,
                                            drop_last=True,
                                            )

    testloader = torch.utils.data.DataLoader(testset, 
                                            batch_size=test_batch_size, 
                                            shuffle=True, 
                                            collate_fn=custom_collate,
                                            drop_last=True,
                                            )
    
    return trainloader, testloader


def select_sample_phipsi(phi_traj, psi_traj, phi_value, psi_value, tolerance):
    idx = np.where(np.logical_and(np.logical_and(phi_traj > phi_value-tolerance, phi_traj < phi_value+tolerance),
                                  np.logical_and(psi_traj > psi_value-tolerance, psi_traj < psi_value+tolerance)))[0]
    if len(idx) == 0:
        return None
    else:
        return np.random.choice(idx, size=1)[0]
    

def select_start_configs(
        ngrid, 
        phi_traj, 
        psi_traj,
        aladi_crd,
        cg_atoms
        ):
    # Make grid of equally spaced points from -pi to +pi
    agrid = np.linspace(-np.pi, np.pi, int(ngrid+1))[:-1] + (np.pi/ngrid)
    # Define tolerance based on grid size
    tolerance = (np.pi/(2*ngrid))
    # Blank lists to append points to
    phipsi = []
    sample = []
    # Loop over all phi points
    for phi in agrid:
        # Loop over all psi grid points
        for psi in agrid:
            # Select samples of phi and psi from the grid
            idx = select_sample_phipsi(phi_traj, psi_traj, phi, psi, tolerance)
            # If something is sampled, attach to lists
            if idx is not None:
                phipsi.append([phi, psi])
                sample.append(aladi_crd[idx][cg_atoms])
    # Stack list together
    x0_ala2_phipsicover = np.stack(sample)
    phipsi = np.array(phipsi)
    
    return x0_ala2_phipsicover, phipsi