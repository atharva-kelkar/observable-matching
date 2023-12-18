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