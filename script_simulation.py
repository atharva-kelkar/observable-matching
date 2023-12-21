#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 20/12/2023
"""

import numpy as np
from jax import lax, jit
import jax.numpy as jnp
import jax
from simulate import forcefield, step_fn, init_state, timestep
import time
import pickle
from script_training import load_ala2_data
import mdtraj as md
from dataloader import select_start_configs
import os

def simulator(
        x0,
        params,
        nstep_ala2,
        write_every,
        sim_out_file,
        dt,
        timestep
        ):
    
    ## Freeze partial arguments
    partial_step_fn = jax.tree_util.Partial(step_fn, dt=dt, write_every=write_every)

    ## Loop over all starting configurations
    for i in range(x0.shape[0]):
        state = init_state(
            jnp.array(x0[i]), 
            jnp.array(forcefield(params, jnp.array(x0[i])))
        )

        log = {
            'time': np.zeros((nstep_ala2,)),
            'H': np.zeros((nstep_ala2,)),
            'position': np.zeros((nstep_ala2 // write_every,) + state.position.shape),
            'force': np.zeros((nstep_ala2 // write_every,) + state.position.shape),
            'noise': np.zeros((nstep_ala2 // write_every,) + state.position.shape),
            'velocities': np.zeros((nstep_ala2 // write_every,) + state.position.shape)
        }

        tic = time.time()
        ## Run simulation
        state, log, _ = lax.fori_loop(0, nstep_ala2, partial_step_fn, (state, log, timestep))
        ## Record time needed
        print(f'Time required is {time.time() - tic:.2f} seconds')

        ## Save output as pickle
        pickle.dump(log['position'], open(sim_out_file.format(i), 'wb')) # open(f'{import_home}/sim_1mn_frames_dt_1em5_massRescale_{i}.pkl', 'wb'))


if __name__ == "__main__":

    import_home = '/import/a12/users/atkelkar/data/AlanineDipeptide'
    aladi_file = f'{import_home}/all_atom_data/raw_trajectory/alanine-dipeptide-1Mx1ps-with-force.npz'
    top_file = f'{import_home}/all_atom_data/raw_trajectory/alanine_1mn.pdb'
    forcemap_file = f'{import_home}/force_maps/basic_force_map.npy'
    model_name = 'mode=cv+cg_cgcvrat=0.10:0.90_bs=64_n_layers=4_width=256_startLR=0.001_endLR=0.0001_epochs=50'
    sim_dir = f'{import_home}/simulation_output/projected_force_simulations/{model_name}'
    ## Make storage directory
    os.mkdir(sim_dir)
    sim_out_file = f'{sim_dir}/sim_{{}}.pkl'
    
    ## Set simulation parameters
    ngrid = 10
    nstep_ala2 = 2 * 10**5
    dt = 1e-5
    D = 1
    write_every = 10
    beta = 1.6776
    friction = 1
    vscale = np.exp(- dt * friction)
    noisescale = np.sqrt(1 - vscale * vscale)

    ## Define CG atoms
    cg_atoms = np.array([5,7,9,11,15,17])-1

    ## Load coordinates and topology
    aladi_crd, traj, aladi_forces, forcemap = load_ala2_data(aladi_file, top_file, forcemap_file)

    ## Define CG topology
    cg_topo = traj.atom_slice(cg_atoms).topology

    ## Compute dihedrals
    torsion_cg_idx = [[0,1,2,4], [1,2,4,5]] #, [1,2,3,4]]  # phi, psi, out-of-plane improper around Ca
    torsion_traj = md.compute_dihedrals(traj, [cg_atoms[t] for t in torsion_cg_idx])
    phi_traj = torsion_traj[:, 0]
    psi_traj = torsion_traj[:, 1]

    ## Load model
    params = pickle.load(open(f'models/{model_name}.pkl', 'rb'))

    ## Get starting points
    x0_ala2_phipsicover, phipsi = select_start_configs(
        ngrid, 
        phi_traj, 
        psi_traj,
        aladi_crd,
        cg_atoms
    )

    # ## Set initial state
    # x = x0_ala2_phipsicover[0]
    # force = forcefield(params, x)
    # state = init_state(jnp.array(x), jnp.array(force))
    ## Define timestep function for simulation
    timestep = jax.tree_util.Partial(timestep,
        params=params,
        dt=dt,
        beta=beta,
        vscale=vscale,
        noisescale=noisescale,
    )

    ## Run simulation
    print(f'Running {len(x0_ala2_phipsicover)} simulations...')
    simulator(
        x0_ala2_phipsicover,
        params=params,
        nstep_ala2=nstep_ala2,
        write_every=write_every,
        sim_out_file=sim_out_file,
        dt=dt,
        timestep=timestep
        )