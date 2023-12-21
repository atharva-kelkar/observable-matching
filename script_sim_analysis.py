#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 21/12/2023
"""

import numpy as np
from analysis import load_sim_coords, make_fe, make_fe_plot
import mdtraj as md


if __name__ == "__main__":
    import_home = '/import/a12/users/atkelkar/data/AlanineDipeptide/simulation_output/projected_force_simulations/'
    model_name = 'mode=cv+cg_cgcvrat=0.10:0.90_bs=64_n_layers=4_width=256_startLR=0.001_endLR=0.0001_epochs=50'
    sim_folder = f'{import_home}/{model_name}'
    cg_atoms = np.array([5,7,9,11,15,17])-1
    torsion_cg_idx = [[0,1,2,4], [1,2,4,5]]
    to_save_rplot, to_save_traj = True, True

    ## Load simulation coordinates
    coords = load_sim_coords(
        sim_folder=sim_folder,
        sim_file_name='sim_{}.pkl',
        numfiles=64
    )

    ## Load topology
    top_file = '/import/a12/users/atkelkar/data/AlanineDipeptide/all_atom_data/raw_trajectory/alanine_1mn.pdb'
    top = md.load(top_file)

    ## Filter only CG atoms
    top = top.atom_slice(cg_atoms).topology
    
    ## Make trajectory
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
        fig.savefig(f'plots/sim_rplot_{model_name}.png',
                    bbox_inches='tight',
                    dpi=200
                    )
        
    if to_save_traj:
        traj.save_xtc(f'{sim_folder}/all_sims.xtc')
        traj[0].save_pdb(f'{sim_folder}/sim.pdb')