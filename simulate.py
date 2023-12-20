#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Atharva Kelkar
@date: 20/12/2023
"""

import dataclasses
import numpy as np
import jax
from jax import grad, jit, lax
import jax.numpy as jnp
from nn import predict_energy

"""Functions adapted from https://github.com/jax-md/jax-md"""

def dataclass(clz):
    """Create a class which can be passed to functional transformations.

    Jax transformations such as `jax.jit` and `jax.grad` require objects that are
    immutable and can be mapped over using the `jax.tree_util` methods.

    The `dataclass` decorator makes it easy to define custom classes that can be
    passed safely to Jax.

    Args:
    clz: the class that will be transformed by the decorator.
    Returns:
    The new class.
    """
    clz.set = lambda self, **kwargs: dataclasses.replace(self, **kwargs)
    data_clz = dataclasses.dataclass(frozen=True)(clz)
    meta_fields = []
    data_fields = []
    for name, field_info in data_clz.__dataclass_fields__.items():
        is_static = field_info.metadata.get('static', False)
        if is_static:
            meta_fields.append(name)
        else:
            data_fields.append(name)

    def iterate_clz(x):
        meta = tuple(getattr(x, name) for name in meta_fields)
        data = tuple(getattr(x, name) for name in data_fields)
        return data, meta

    def clz_from_iterable(meta, data):
        meta_args = tuple(zip(meta_fields, meta))
        data_args = tuple(zip(data_fields, data))
        kwargs = dict(meta_args + data_args)
        return data_clz(**kwargs)

    jax.tree_util.register_pytree_node(data_clz,
                                     iterate_clz,
                                     clz_from_iterable)

    return data_clz


def static_field():
    return dataclasses.field(metadata={'static': True})

replace = dataclasses.replace
asdict = dataclasses.asdict
astuple = dataclasses.astuple
is_dataclass = dataclasses.is_dataclass
fields = dataclasses.fields
field = dataclasses.field

def unpack(dc) -> tuple:
    return tuple(getattr(dc, field.name) for field in dataclasses.fields(dc))

Array = jnp.ndarray

@dataclass
class State:
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    chain: The variables describing the Nose-Hoover chain.
    """
    position: Array
    force: Array
    noise: Array


@dataclass
class VeloState:
    """State information for an NVT system with a Nose-Hoover chain thermostat.

    Attributes:
    position: The current position of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    momentum: The momentum of particles. An ndarray of floats
      with shape `[n, spatial_dimension]`.
    force: The current force on the particles. An ndarray of floats with shape
      `[n, spatial_dimension]`.
    mass: The mass of the particles. Can either be a float or an ndarray
      of floats with shape `[n]`.
    chain: The variables describing the Nose-Hoover chain.
    """
    position: Array
    force: Array
    noise: Array
    velocities: Array
    masses: Array

def forcefield(params, x):
    """Function that returns force on each position given NN params"""
    return - grad( predict_energy, argnums=1 )( params, x )

@jit
def init_state(x, f):
    """Function to initialize state of simulation

    Parameters
    ----------
    x : jnp.array
        Initial position
    f : jnp.array
        Initial velocity

    Returns
    -------
    state
        State with initialized position and velocity
    """
    masses = jnp.array([12, 14, 12, 12, 12, 14]) / 418.4
    state = VeloState(x, f, jnp.zeros_like(x), jnp.zeros_like(x), masses)
    return state


def timestep(
        state,
        params,
        dt,
        beta,
        vscale,
        noisescale,

        ):
    """Timestep method for Langevin dynamics
    
    Parameters
    ----------
    state:
        State at t

    Returns
    -------
    state:
        State at t+1
    """
    v_old = state.velocities
    masses = state.masses
    x_old = state.position

    # Calc forces
    forces = forcefield(params, x_old)

    # B
    v_new = v_old + 0.5 * dt * forces / masses[:, jnp.newaxis]

    # A (position update)
    x_new = x_old + v_new * dt * 0.5

    # O (noise)
    noise = jnp.sqrt(1.0 / beta / masses[:, jnp.newaxis])
    noise = noise * jnp.array(
        np.random.normal(
            size=x_new.shape
        )
    )

    v_new = v_new * vscale + noisescale * noise

    # A
    x_new = x_new + v_new * dt * 0.5

    state = state.set(position=x_new)
    forces = forcefield(params, state.position)

    # B
    v_new = v_new + 0.5 * dt * forces / masses[:, jnp.newaxis]
    state = state.set(velocities=v_new)

    return state

def step_fn(i, state_and_log, dt, write_every):
    """Function to take 1 step in time

    Parameters
    ----------
    i : int
        Time index
    state_and_log : tuple(State, dict)
        Tuple containing State object and log dictionary
    dt : float
        Time step
    write_every : int
        Number of steps to write log
    """
    state, log, timestep = state_and_log
#     state, force = state
    
    t = i * dt
    ## Register time
    log['time'].at[i].set(t)
    
    ## Record position if record time
    log['position'] = lax.cond(i % write_every == 0,
                               lambda p: \
                               p.at[i // write_every].set(state.position),
                               lambda p: p,
                               log['position']
                              )
    
    log['force'] = lax.cond(i % write_every == 0,
                               lambda p: \
                               p.at[i // write_every].set(state.force),
                               lambda p: p,
                               log['force']
                              )
    
    log['noise'] = lax.cond(i % write_every == 0,
                               lambda p: \
                               p.at[i // write_every].set(state.noise),
                               lambda p: p,
                               log['noise']
                              )
    
    log['velocities'] = lax.cond(i % write_every == 0,
                               lambda p: \
                               p.at[i // write_every].set(state.velocities),
                               lambda p: p,
                               log['velocities']
                              )
    
#     state = apply(state)
    state = timestep(state)
    
    return state, log, timestep