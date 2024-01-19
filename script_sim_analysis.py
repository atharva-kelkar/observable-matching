#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 21/12/2023
"""

import numpy as np
from analysis import load_sim_coords, make_fe, make_fe_plot, make_dist_hist_plots
import mdtraj as md


if __name__ == "__main__":
    import_home = '/import/a12/users/atkelkar/data/AlanineDipeptide/simulation_output/projected_force_simulations/'
    stride = 2
    ref_dataset_file = f'/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/feature_divergence/precomputed_dataset_stride{stride}.npz'
    model_name = 'mode=cv+cg_cgcvrat=0.10:0.90_bs=64_n_layers=4_width=256_startLR=0.001_endLR=0.0001_epochs=50'
    run_index = 7
    sim_folder = f'{import_home}/{model_name}_run{run_index}'
    cg_atoms = np.array([5,7,9,11,15,17])-1
    torsion_cg_idx = [[0,1,2,4], [1,2,4,5]]
    bond_cg_idx = [[0,1], [1,2], [2,3], [2,4], [4,5]]
    angle_cg_idx = [[0,1,2], [1,2,3], [1,2,4], [3,2,4], [2,4,5]]
    ## List pairs of atoms in all bonds and angles
    pairs = [list(b) for b in bond_cg_idx] + [[aci[0], aci[2]] for aci in angle_cg_idx]
    
    to_save_rplot, to_save_traj = True, True

    ## Load reference data
    preloaded_data = np.load(ref_dataset_file)
    ref_coords = preloaded_data['coords'][:, cg_atoms]

    ## Load simulation coordinates
    coords = load_sim_coords(
        sim_folder=sim_folder,
        sim_file_name='sim_pos_{}.pkl',
        numfiles=64
    )

    ## Load topology
    top_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine_1mn.pdb'
    top = md.load(top_file)

    ## Filter only CG atoms
    top = top.atom_slice(cg_atoms).topology
    
    ## Make trajectory
    ref_traj = md.Trajectory(ref_coords * 0.1, top)
    traj = md.Trajectory(coords * 0.1, top)

    ## Calculate torsions
    torsion_traj = md.compute_dihedrals(traj, torsion_cg_idx)
    phi_traj = torsion_traj[:, 0]
    psi_traj = torsion_traj[:, 1]

    ## Plot free energy
    fe, hist, bins_phi, bins_psi = make_fe( phi_traj, psi_traj )

    ## Make Ramachandran plot
    fig, ax = make_fe_plot(bins_phi, bins_psi, fe)
    
    ## Save Ramachandran plot
    if to_save_rplot:
        fig.savefig(f'plots/sim_rplot_{model_name}_run{run_index}.png',
                    bbox_inches='tight',
                    dpi=200
                    )
        
    if to_save_traj:
        traj.save_xtc(f'{sim_folder}/all_sims.xtc')
        traj[0].save_pdb(f'{sim_folder}/sim.pdb')

    ## Calculate pairwise distances
    ref_pairwise_distances = md.compute_distances(ref_traj, pairs)
    pairwise_distances = md.compute_distances(traj, pairs)
    
    ## Make histogram plots
    print(ref_pairwise_distances.shape, pairwise_distances.shape)
    fig, ax = make_dist_hist_plots(ref_pairwise_distances, pairwise_distances)

    ## Save histogram plot
    if to_save_rplot:
        fig.savefig(f'plots/sim_distHists_{model_name}_run{run_index}.png',
                    bbox_inches='tight',
                    dpi=200
                    )
    