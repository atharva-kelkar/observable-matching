#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 18/12/2023
"""

import numpy as np 
import matplotlib.pyplot as plt
from jax import vmap, jit
from dataloader import create_test_train_datasets
from analysis import make_test_preds, make_label_pred_plot
from plotting_functions import load_default_params, format_plot

if __name__ == "__main__":

    cg_atoms = np.array([5,7,9,11,15,17]) - 1 # [4, 6, 8, 10, 14, 16]

    ## File labels
    preloaded_data_out_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/precomputed_dataset.npz'
    model_name = 'mode=cv+cg_cgcvrat=0.10:0.90_bs=256_n_layers=4_width=256_startLR=0.001_endLR=0.0001_epochs=50'
    test_model_file = f'/home/mi/atkelkar/python_codes/projected_force_matching/models/{model_name}.pkl'
    to_save_plot = True

    ## Load parameters of saved model
    params = np.load(test_model_file, allow_pickle=True)

    print('Loading pre-computed dataset...')
    preloaded_data = np.load(preloaded_data_out_file)
    aladi_crd = preloaded_data['coords']
    aladi_cg_force = preloaded_data['cg_force']
    force_proj_arr = preloaded_data['f_proj_arr']
    div_arr = preloaded_data['div_arr']
    interpolated_forces = preloaded_data['kde_forces']
    all_feats = preloaded_data['all_feats']

    print('Creating train and test datasets...')
    trainloader, testloader = create_test_train_datasets(
        aladi_crd,
        all_feats,
        aladi_cg_force,
        cg_atoms,
        force_proj_arr,
        div_arr,
        interpolated_forces,
        test_batch_size = 1024 * 10
    )

    print('Calculating labels for all features...')
    f_pred, f_labels, feat_label, feat_calc = make_test_preds(params, testloader)
    
    print('Making plots...')
    fig, axes = plt.subplots(ncols=2, figsize=(10,5))

    ## Plot torsions
    for i, ax in enumerate(axes):
        ax = make_label_pred_plot(
            ax, 
            feat_label[:, i],
            f_labels[:, i],
            f_pred[:, i],
            plot_stride=10
            )
        kwargs = load_default_params(f'Torsion {i}', 'Projected force')
        kwargs['want_legend'] = 1
        ax = format_plot(ax, **kwargs)  

    if to_save_plot:
        fig.savefig(f'plots/torsions_{model_name}.png',
                    bbox_inches='tight',
                    dpi=200
                    )
        
    fig, axes = plt.subplots(ncols=5, nrows=2, figsize=(15, 6))
    ## Plot distances
    for i, ax in enumerate(axes.flatten()):
        ax = make_label_pred_plot(
            ax, 
            feat_label[:, i+2],
            f_labels[:, i+2],
            f_pred[:, i+2],
            plot_stride=10
            )
        kwargs = load_default_params(f'Dist {i}', 'Projected force')
        kwargs['want_legend'] = 1
        ax = format_plot(ax, **kwargs)

    if to_save_plot:
        fig.tight_layout()
        fig.savefig(f'plots/dists_{model_name}.png',
                    bbox_inches='tight',
                    dpi=200
                    )
