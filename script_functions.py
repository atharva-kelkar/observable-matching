import numpy as np
import mdtraj as md
from density import gaussian_kde, gaussian_kde_adaptive2, density2force, compute_forces
from projection import calc_force_proj_operator, calc_div_operator
from dataloader import create_test_train_datasets
from nn import init_MLP, weighted_update
import jax
from jax import jit, vmap
import jax.numpy as jnp
from tqdm import tqdm
import pickle
import sys
from functools import partial

def load_ala2_data(aladi_file, top_file, forcemap_file):
    aladi_raw = np.load(aladi_file)

    stride = 1
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
        **kwargs
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
        kde_forces.append(density2force(centers, kde, periodic=periodic))
        #acenters, akernel_size, akde = gaussian_kde_adaptive(torsions, n=50, periodic=True)

    return kde_centers, kde_kernelsizes, kde_densities, kde_forces

# def ala2_main(
#         aladi_file: str,
#         top_file: str,
#         forcemap_file: str,
#         cg_atoms: np.ndarray = np.array([5,7,9,11,15,17])-1,
#         torsion_cg_idx: list = [[0,1,2,4], [1,2,4,5]],

#     ):

def interpolate_forces(feats, kde_centers, kde_forces, list_to_append):
    for i in range(feats.shape[1]):
        list_to_append.append(compute_forces(feats[:, i], kde_centers[i], kde_forces[i]))
    

def force_projection_calculator(
        aladi_crd, 
        cg_atoms, 
        torsion_cg_idx, 
        dist_cg_idx,
        out_file
        ):
    ## Jit and vmap force projection operator
    jit_force_calc = jit(vmap(calc_force_proj_operator, in_axes=(0, None, None)))
    
    ## Calculate output used jitted function
    output = jit_force_calc(
        aladi_crd[:, cg_atoms], 
        torsion_cg_idx,
        dist_cg_idx
        )
    
    ## Convert JITted output to correct shape
    jit_out_np = np.array(output[1])

    ## Convert NumPy array into proper shape
    n = len(output[1][0])
    jit_out = np.zeros((len(output[1][0]), 12, 6, 3))
    for i in tqdm(range(n)):
        jit_out[i] = jit_out_np[:, i]
        
    force_proj_arr = jit_out.astype(np.float32)
    ## Save output array
    print(f'***** OUT FILE IS {out_file} *****')
    np.save(
        out_file,
        force_proj_arr
    )

    return force_proj_arr

def divergence_calculator(
        aladi_crd, 
        cg_atoms,
        torsion_cg_idx,
        dist_cg_idx, 
        out_dir, 
        n_splits=100
        ):

    jit_div_operator = jit(vmap(calc_div_operator, in_axes=(0, None, None)))

    ## Calculate divergence for 1/100th of the dataset and save as individual npy files
    for n_split in tqdm(range(n_splits)):
        a = jit_div_operator(
                aladi_crd[int((n_split) * 0.01*10**6) : int((n_split + 1) * 0.01*10**6), cg_atoms],
                torsion_cg_idx,
                dist_cg_idx
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
        ):
    ## Fix weights as a static argument
    wtd_update = jax.tree_util.Partial(weighted_update, wts=loss_wts)
    ## JIT function
    jit_wtd_update = jit(wtd_update)
    ## Training loop
    for lr in lrs:
        for epoch in tqdm(range(n_epochs)):
            for i, (x, _, _, f_proj, div, f) in tqdm(enumerate(trainloader), total=len(trainloader)):
                if i % 250 == 1:
                    losses.append(loss)
                    # print(f'Running for i={i}, loss value of {losses[-1]}')
                
                loss, params, grad_loss = jit_wtd_update(params, x, f, f_proj, div, lr)
            print(loss)
    
    if to_save_model:
        ## Save model parameters
        pickle.dump(params, open(f'models/{out_model_name}.pkl', 'wb'))
        ## Save losses
        np.save(f'models/losses_{out_model_name}.npy', losses)

def make_model_name(model_layers, lrs, n_epochs):
    n_layers = len(model_layers) - 2
    lr_start = lrs[0]
    lr_end = lrs[-1]
    width = model_layers[1]
    return f'n_layers={n_layers}_width={width}_startLR={lr_start}_endLR={lr_end}_epochs={n_epochs}'

if __name__ == "__main__":

    ## Debug section
    aladi_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine-dipeptide-1Mx1ps-with-force.npz'
    top_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine_1mn.pdb'
    forcemap_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/force_maps/basic_force_map.npy'
    calculate_fproj_arr, calculate_div_arr = False, False
    to_save_int_output = False
    to_load_precomputed_dataset = True
    model_layers = [12, 128, 128, 128, 128, 1]
    to_train_model = True
    to_restart_training = True
    restart_model_name = 'n_layers=4_width=128_startLR=0.001_endLR=0.001' # 'simple_jax_model_4_hidden_128_noLastBias_scale_0.1_LR0.001'
    lrs = [0.0005]
    n_epochs = 30
    to_save_model = True
    out_model_name = make_model_name(model_layers, lrs, n_epochs)
    

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
    fproj_out_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/all_force_proj_operators.npy'
    div_out_dir = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence'
    preloaded_data_out_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/precomputed_dataset.npz'

    if to_load_precomputed_dataset is False:
        print('Loading trajectory...')
        ## Load coordinates, trajectory, forces, forcemap
        aladi_crd, traj, aladi_forces, forcemap = load_ala2_data(aladi_file, top_file, forcemap_file)

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
            force_proj_arr = force_projection_calculator(
                aladi_crd,
                cg_atoms,
                torsion_cg_idx,
                dist_cg_idx,
                fproj_out_file
                )
        else:
            force_proj_arr = np.load(fproj_out_file)
        
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

        if to_save_int_output:
            print('Saving precomputed output data')
            preloaded_data = np.savez(
                preloaded_data_out_file,
                coords=aladi_crd,
                cg_force=aladi_cg_force,
                f_proj_arr=force_proj_arr,
                div_arr=div_arr,
                kde_forces=interpolated_forces,
                all_feats=all_feats
            )
    elif to_load_precomputed_dataset is True:
        print('Loading pre-computed dataset...')
        preloaded_data = np.load(preloaded_data_out_file)
        aladi_crd = preloaded_data['coords']
        aladi_cg_force = preloaded_data['cg_force']
        force_proj_arr = preloaded_data['f_proj_arr']
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
        div_arr,
        interpolated_forces,
    )

    ## Initialize neural network parameters
    seed = 0
    key = jax.random.PRNGKey(seed)
    params = init_MLP(model_layers, key)

    ## Train neural network
    losses = []
    loss_norm = []

    ## Load weights if training is to be restarted
    if to_restart_training:
        params = pickle.load(open(f'models/{restart_model_name}.pkl', 'rb'))
        losses = np.load(f'models/losses_{restart_model_name}.npy')
        losses = list(losses)

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
        )










