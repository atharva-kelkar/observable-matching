#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 13/12/2023
"""

import numpy as np
import mdtraj as md
from density import gaussian_kde, gaussian_kde_adaptive2, density2force, compute_forces
from projection import calc_force_proj_operator, calc_div_operator
from dataloader import create_test_train_datasets
from nn import init_MLP, weighted_update, weighted_update_with_cg
import jax
from jax import jit, vmap
import jax.numpy as jnp
from tqdm import tqdm
import pickle
import sys
import os
from functools import partial
from featurize import ala2_featurize

def load_ala2_data(aladi_file, top_file, forcemap_file, stride=1):
    aladi_raw = np.load(aladi_file)

    aladi_crd = aladi_raw['coordinates']
    aladi_forces = aladi_raw['forces']
    aladi_pdb = aladi_raw['pdb']

    aladi_crd = aladi_crd[::stride]

    top = md.load(top_file).topology

    # load mdtraj Trajectory, converting coordinates in nanometers
    traj = md.Trajectory(0.1*aladi_crd, top)

    # load basic force map
    forcemap = np.load(forcemap_file)

    return aladi_crd, traj, aladi_forces, forcemap


def calc_density_and_force_from_trajectory(
        feat_traj, 
        kde_func, 
        periodic,
        beta,
        **kwargs,
        ):
    # KDE for torsions:
    kde_centers = []
    kde_kernelsizes = []
    kde_densities = []
    kde_forces = []
    for t in range(feat_traj.shape[1]):
        ## Choose only the torsion you want to work with
        curr_feat = feat_traj[:, t]
        ## Estimate distribution with KDE estimate
        centers, kernel_size, kde = kde_func(curr_feat, periodic=periodic, **kwargs)
        ## Append centers to list
        kde_centers.append(centers)
        ## Append kernel size to list
        kde_kernelsizes.append(kernel_size)
        ## Append kernel density to list
        kde_densities.append(kde)
        ## Append force calculated from density to list
        kde_forces.append(density2force(centers, kde, periodic=periodic, beta=beta))
        #acenters, akernel_size, akde = gaussian_kde_adaptive(torsions, n=50, periodic=True)

    return kde_centers, kde_kernelsizes, kde_densities, kde_forces


def interpolate_forces(feats, kde_centers, kde_forces, list_to_append):
    for i in range(feats.shape[1]):
        list_to_append.append(compute_forces(feats[:, i], kde_centers[i], kde_forces[i]))
    

def force_projection_calculator(
        aladi_crd, 
        cg_atoms, 
        torsion_cg_idx, 
        dist_cg_idx,
        fproj_out_file='',
        det_G_out_file='',
        to_save_output=True
        ):
    ## Jit and vmap force projection operator
    force_calc = jax.tree_util.Partial(calc_force_proj_operator, featurize=ala2_featurize)
    jit_force_calc = jit(vmap(force_calc, in_axes=(0)))
    
    ## Calculate output used jitted function
    _, sum_G_inv_dot_q, det_G = jit_force_calc(
        aladi_crd[:, cg_atoms], 
        )
    
    ## Convert JITted output to correct shape
    jit_out_np = np.array(sum_G_inv_dot_q)
    det_G_arr = np.array(det_G)

    ## Convert NumPy array into proper shape
    n_feat, n, n_beads, n_dim = jit_out_np.shape
    jit_out = np.zeros((n, n_feat, n_beads, n_dim))
    for i in tqdm(range(n)):
        jit_out[i] = jit_out_np[:, i]
        
    force_proj_arr = jit_out.astype(np.float32)
    
    ## Save output array
    if to_save_output:
        print(f'***** OUT FILE IS {fproj_out_file} *****')
        np.save(
            fproj_out_file,
            force_proj_arr
        )

        print(f'***** OUT FILE IS {det_G_out_file} *****')
        np.save(
            det_G_out_file,
            det_G_arr
        )

    return force_proj_arr, det_G_arr

def divergence_calculator(
        aladi_crd, 
        cg_atoms,
        torsion_cg_idx,
        dist_cg_idx, 
        out_dir, 
        n_splits=100
        ):

    ## Make directory if needed
    os.makedirs(out_dir, exist_ok=True)
    ## Jit divergence operator
    div_operator = jax.tree_util.Partial(calc_div_operator, featurize=ala2_featurize, nfeatures=12)
    jit_div_operator = jit(vmap(div_operator, in_axes=(0)))
    ## Calculate divergence for 1/100th of the dataset and save as individual npy files
    for n_split in tqdm(range(n_splits)):
        a = jit_div_operator(
                aladi_crd[int((n_split) * len(aladi_crd) // n_splits) : int((n_split + 1) * len(aladi_crd) // n_splits), cg_atoms],
            )
        np.save(
            f'{out_dir}/jacobian_{n_split}.npy',
            np.array(a)
        )

def load_divergence(out_dir, aladi_crd, n_splits=100):
    div_arr = []
    for n_split in tqdm(range(n_splits)):
        ## Load current jacobian
        curr_jac = np.load(f'{out_dir}/jacobian_{n_split}.npy')
        div_arr.append(
            np.trace(
                curr_jac.reshape(
                    curr_jac.shape[0], curr_jac.shape[1], 
                    np.prod(curr_jac.shape[2:4]), 
                    np.prod(curr_jac.shape[4:6])
                ),
                axis1=2, axis2=3
            )
        )
        
    div_arr = np.concatenate(div_arr, axis=1).T
    assert len(div_arr) == len(aladi_crd)
    return div_arr


def trainer(
        trainloader,
        params,
        loss_wts,
        lrs, 
        n_epochs, 
        to_save_model,
        out_model_name,
        model_state,
        train_mode='cv',
        cg_cv_wts=jnp.array([0.5, 0.5]),
        ):
    if train_mode == 'cv':
        ## Fix weights as a static argument
        wtd_update = jax.tree_util.Partial(weighted_update, wts=loss_wts)
    elif train_mode == 'cv+cg':
        wtd_update = jax.tree_util.Partial(weighted_update_with_cg, wts=loss_wts, cg_cv_wts=cg_cv_wts)
    ## JIT function
    jit_wtd_update = jit(wtd_update)

    ## Training loop
    for lr in lrs:
        for epoch in tqdm(range(n_epochs)):
            for i, (x, _, f_cg, f_proj, det_G_weight, div, f_cv) in tqdm(enumerate(trainloader), total=len(trainloader)):
                if i % 250 == 1:
                    losses_cv.append(loss_cv)
                    grad_losses_cv.append(grad_loss_cv)
                    if train_mode == 'cv+cg':
                        losses_cg.append(loss_cg)
                        grad_losses_cg.append(grad_loss_cg)
                    # print(f'Running for i={i}, loss value of {losses[-1]}')
                if train_mode == 'cv':
                    loss, params, grad_loss = jit_wtd_update(params, x, f_cv, f_proj, det_G_weight, div, lr)
                elif train_mode == 'cv+cg':
                    # print(f'shape of x from training loop is: {x.shape}')
                    loss_cg, loss_cv, params, grad_loss_cg, grad_loss_cv = jit_wtd_update(params, x, f_cg, f_cv, f_proj, div, lr)
            print(loss_cg, loss_cv)
    # def weighted_update_with_cg                                                        (param, x, f_cg, y, f_proj, det_G_weight, div, lr, wts, cg_cv_wts):

    if to_save_model:
        ## Save model parameters
        pickle.dump(params, open(f'models/{out_model_name}.pkl', 'wb'))
        pickle.dump(model_state, open(f'models/state_{out_model_name}.pkl', 'wb'))
        ## Save losses
        # np.save(f'models/losses_{out_model_name}.npy', losses)
        pickle.dump(
            {
                'loss_cv' : losses_cv,
                'loss_cg' : losses_cg,
                'grad_loss_cg' : grad_losses_cg,
                'grad_losses_cv' : grad_losses_cv,
            },
            open(f'models/losses_{out_model_name}.pkl', 'wb') 
        )

def make_model_name(train_mode, cg_cv_ratio, batch_size, model_layers, lrs, n_epochs, stride, train_seed):
    n_layers = len(model_layers) - 2
    lr_start = lrs[0]
    lr_end = lrs[-1]
    width = model_layers[1]
    cg, cv = cg_cv_ratio
    model_name = f'mode={train_mode}_cgcvrat={cg:.3f}:{cv:.3f}_bs={batch_size}_n_layers={n_layers}_width={width}_startLR={lr_start}_endLR={lr_end}_epochs={n_epochs}_stride={stride}'

    state = {'n_layers': n_layers,
             'model_layers': model_layers,
             'cg_cv_ratio': cg_cv_ratio,
             'lrs': lrs,
             'n_epochs': n_epochs,
             'train_mode': train_mode,
             'model_name': model_name,
             'dataset_stride': stride,
             'train_seed': train_seed,
             }
    
    return state, model_name

def load_model_state(state_model_file):
    state = np.load(state_model_file, allow_pickle=True)
    return state['train_mode'], state['cg_cv_ratio'], state['model_layers'], state['lrs'], state['n_epochs']


if __name__ == "__main__":

    ## Debug section
    aladi_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine-dipeptide-1Mx1ps-with-force.npz'
    top_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine_1mn.pdb'
    forcemap_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/force_maps/basic_force_map.npy'
    state_model_file = 'models/state_mode=cv+cg_cgcvrat=0.10:0.90_bs=64_n_layers=4_width=256_startLR=0.001_endLR=0.0001_epochs=50.pkl'
    
    stride = 1
    ## Flag variables
    calculate_fproj_arr, calculate_div_arr = True, True
    save_fproj_arr, save_div_arr = True, True
    to_save_int_output = True
    to_load_precomputed_dataset = False
    to_load_model_state = True
    to_train_model = True
    to_restart_training = False
    to_save_model = True
    ## Model training variables
    if to_load_model_state:
        train_mode, cg_cv_ratio, model_layers, lrs, n_epochs = load_model_state(state_model_file)
        ## Manual override to save different model
        n_epochs = 49
    else:
        train_mode = 'cv+cg' # 'cv' or 'cv+cg'
        cg_cv_ratio = jnp.array([0.1, 0.9])
        model_layers = [12, 256, 256, 256, 256, 1]
        lrs = [0.001, 0.0001]
        n_epochs = 50
    
    batch_size = 64
    train_seed = 0
    restart_model_name = 'mode=cv+cg_cgcvrat=0.10:0.90_bs=64_n_layers=4_width=256_startLR=0.001_endLR=0.001_epochs=10' # 'simple_jax_model_4_hidden_128_noLastBias_scale_0.1_LR0.001'
    model_state, out_model_name = make_model_name(train_mode, cg_cv_ratio, batch_size, model_layers, lrs, n_epochs, stride, train_seed)
    # out_model_name = f'{out_model_name}_RestartTrain'
    print(f'Model name is {out_model_name}')

    ## Index arrays    
    cg_atoms = np.array([5,7,9,11,15,17])-1
    bond_cg_idx = [[0,1], [1,2], [2,3], [2,4], [4,5]]
    angle_cg_idx = [[0,1,2], [1,2,3], [1,2,4], [3,2,4], [2,4,5]]
    torsion_cg_idx = [[0,1,2,4], [1,2,4,5]]
    ## List pairs of atoms in all bonds and angles
    pairs = [list(cg_atoms[b]) for b in bond_cg_idx] + [[cg_atoms[aci[0]],cg_atoms[aci[2]]] for aci in angle_cg_idx]
    ## Make NumPy arrays of indices
    torsion_cg_idx = np.array(torsion_cg_idx)
    dist_cg_idx = np.array(bond_cg_idx + [[a[0], a[2]] for a in angle_cg_idx])


    ## Output files
    fproj_out_file = f'/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/all_force_proj_operators_stride{stride}.npy'
    det_G_out_file = f'/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/all_det_G_stride{stride}.npy'
    div_out_dir = f'/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence_stride{stride}'
    preloaded_data_out_file = f'/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/precomputed_dataset_stride{stride}.npz'

    if to_load_precomputed_dataset is False:
        print('Loading trajectory...')
        ## Load coordinates, trajectory, forces, forcemap
        aladi_crd, traj, aladi_forces, forcemap = load_ala2_data(aladi_file, top_file, forcemap_file, stride=stride)

        ## Compute dihedrals
        print('Computing dihedrals...')
        torsion_traj = md.compute_dihedrals(traj, [cg_atoms[t] for t in torsion_cg_idx])
        phi_traj = torsion_traj[:, 0]
        psi_traj = torsion_traj[:, 1]

        ## Calculate KDE and AKDE densities, forces, etc.
        print('Computing forces along torsions...')
        torsion_kde_centers, torsion_kde_kernelsizes, torsion_kde_densities, torsion_kde_forces = calc_density_and_force_from_trajectory(
            torsion_traj, 
            gaussian_kde,
            periodic=True,
        )

        torsion_akde_centers, torsion_akde_kernelsizes, torsion_akde_densities, torsion_akde_forces = calc_density_and_force_from_trajectory(
            torsion_traj, 
            gaussian_kde_adaptive2,
            periodic=True,
            **{'ncenter_max':100, # Default for AKDE
                'ndata_min':100, # Default for AKDE
            }
        )

        dists = md.compute_distances(traj, pairs)

        # change units to Angstrom:
        dists *= 10.0

        ## Calculate KDE and AKDE densities, forces, etc.
        print('Computing forces along pairwise distances...')
        dist_kde_centers, dist_kde_kernelsizes, dist_kde_densities, dist_kde_forces = calc_density_and_force_from_trajectory(
            dists, 
            gaussian_kde,
            periodic=False,
        )

        dist_akde_centers, dist_akde_kernelsizes, dist_akde_densities, dist_akde_forces = calc_density_and_force_from_trajectory(
            dists, 
            gaussian_kde_adaptive2,
            periodic=False,
            **{'ncenter_max':100, # Default for AKDE
                'ndata_min':100, # Default for AKDE
            }
        )

        ## Interpolate all forces
        print('Interpolating forces...')
        interpolated_forces = []

        interpolate_forces(
            torsion_traj, 
            torsion_kde_centers, 
            torsion_kde_forces,
            interpolated_forces
            )
        
        interpolate_forces(
            dists, 
            dist_kde_centers, 
            dist_kde_forces,
            interpolated_forces
            )
        
        ## Stack features together
        all_feats = np.hstack([torsion_traj, dists])

        ## Stack interpolated forces
        interpolated_forces = np.vstack(interpolated_forces).T
            
        ## Calculate force projection operator
        if calculate_fproj_arr:
            print('Calculating force projection operator...')
            force_proj_arr, det_G_arr = force_projection_calculator(
                aladi_crd=aladi_crd,
                cg_atoms=cg_atoms,
                torsion_cg_idx=torsion_cg_idx,
                dist_cg_idx=dist_cg_idx,
                fproj_out_file=fproj_out_file,
                det_G_out_file=det_G_out_file,
                to_save_output=save_fproj_arr
                )
        else:
            force_proj_arr = np.load(fproj_out_file)
            det_G_arr = np.load(det_G_out_file)
        
        ## Calculate divergence operator
        if calculate_div_arr:
            print('Calculating *divergence* of force projection operator...')
            
            divergence_calculator(
                aladi_crd,
                cg_atoms,
                torsion_cg_idx,
                dist_cg_idx,
                div_out_dir,
                n_splits=100
            )
        ## Load divergence operator from saved files in any case
        print('Loading divergence operator...')
        div_arr = load_divergence(
            div_out_dir, 
            aladi_crd, 
            n_splits=100
        )

        ## Calculate aggregated CG force
        aladi_cg_force = forcemap @ aladi_forces

        ## Make reweighting term for each force output
        # TODO Leaving this as ones right now because subspace clustering would be super-sparse
        # Check again later
        det_G_weight_arr = np.ones((len(aladi_crd)))

        if to_save_int_output:
            print('Saving precomputed output data')
            preloaded_data = np.savez(
                preloaded_data_out_file,
                coords=aladi_crd,
                cg_force=aladi_cg_force,
                f_proj_arr=force_proj_arr,
                det_G_weight=det_G_weight_arr,
                div_arr=div_arr,
                kde_forces=interpolated_forces,
                all_feats=all_feats,

            )
    elif to_load_precomputed_dataset is True:
        print('Loading pre-computed dataset...')
        preloaded_data = np.load(preloaded_data_out_file)
        aladi_crd = preloaded_data['coords']
        aladi_cg_force = preloaded_data['cg_force']
        force_proj_arr = preloaded_data['f_proj_arr']
        det_G_weight_arr = preloaded_data['det_G_weight'][:, None]
        div_arr = preloaded_data['div_arr']
        interpolated_forces = preloaded_data['kde_forces']
        all_feats = preloaded_data['all_feats']

    ## Normalize interpolated forces by factor of 20 -- why?
    # use this factor to normalize loss
    force_norms = jnp.sqrt(np.mean((interpolated_forces)**2, axis=0))
    loss_wts = 1 / force_norms

    ## Make dataloaders
    print('Creating training and test datasets...')
    trainloader, testloader = create_test_train_datasets(
        aladi_crd,
        all_feats,
        aladi_cg_force,
        cg_atoms,
        force_proj_arr,
        det_G_weight_arr,
        div_arr,
        interpolated_forces,
        batch_size=batch_size
    )

    ## Initialize neural network parameters
    key = jax.random.PRNGKey(train_seed)
    params = init_MLP(model_layers, key)

    ## Train neural network
    losses_cg = []
    losses_cv = []
    grad_losses_cg = []
    grad_losses_cv = []
    loss_norm = []

    ## Load weights if training is to be restarted
    if to_restart_training:
        params = pickle.load(open(f'models/{restart_model_name}.pkl', 'rb'))
        # losses = np.load(f'models/losses_{restart_model_name}.npy')
        # losses = list(losses)

    if to_train_model:
        print('Training model...')
        trainer(
            loss_wts=loss_wts,
            trainloader=trainloader,
            params=params,
            lrs=lrs,
            n_epochs=n_epochs,
            to_save_model=to_save_model,
            out_model_name=out_model_name,
            train_mode=train_mode,
            cg_cv_wts=cg_cv_ratio,
            model_state=model_state
        )










