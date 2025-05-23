import numpy as np
import jax
from jax import lax, random, jit, vmap
import jax.numpy as jnp
import matplotlib.pyplot as plt

import jax.scipy.stats as jstats
from jax.scipy.special import logsumexp, expit as sigmoid
from typing import List, Tuple, Dict, Any
from functools import partial

# helper function, not used for now
def univariate_normal_sample(key, mean, scale, shape):
    snormal_variates = random.normal(key, shape)
    return mean + scale * snormal_variates

@jit
def filter_step(
    key: jax.Array,
    obs: jax.Array,                             # y
    particles: jax.Array, 
    input_weights: jax.Array,                   # w
    logit_inputs: jax.Array,                    # x_t
    dynamics_matrix: jax.Array,                 # F
    dynamics_var: jax.Array,                    # Q
    dynamics_mean: jax.Array | None = None,     # mu
):

    n_particles, dim_z = particles.shape
    key, subkey_dynamics, subkey_resample = random.split(key, 3)

    if dynamics_mean is None:
        dynamics_mean = jnp.zeros(dim_z)

    dynamics_noise = random.multivariate_normal(
        subkey_dynamics, dynamics_mean, dynamics_var, (n_particles, )
    )

    # to make the dimensions work for einsum
    if logit_inputs.ndim == 1:
        _logit_inputs = logit_inputs[jnp.newaxis, :]
    else:
        _logit_inputs = logit_inputs

    predicted_particles = jnp.einsum('kj,nj->nk', dynamics_matrix, particles, optimize="optimal") + dynamics_noise
    weights_z = input_weights + predicted_particles
    logits = jnp.einsum('kj,nj->nk', _logit_inputs, weights_z, optimize="optimal").squeeze() # x_t * (w + z_t)
    probs = sigmoid(logits)

    y_t = obs.astype(jnp.int32)
    log_weights = jstats.bernoulli.logpmf(y_t, p=probs)
    inds = random.categorical(subkey_resample, log_weights, shape=(n_particles,))
    new_particles = predicted_particles[inds]

    return key, new_particles, log_weights, inds

def run_particle_filter(
    obs: jax.Array,
    n_particles: int,
    input_weights: jax.Array,
    logit_input_sequence: jax.Array,
    initial_var: jax.Array,
    initial_mean: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_var: jax.Array,
    dynamics_mean: jax.Array,
    seed:int
):
    key = random.PRNGKey(seed)

    def scan_step(carry, current_vars):
        key, particles_prev = carry
        current_obs, current_input = current_vars 

        new_key, resampled_particles, log_weights, inds = filter_step(
            key             = key,
            obs             = current_obs,
            particles       = particles_prev,
            input_weights   = input_weights,
            logit_inputs    = current_input,
            dynamics_matrix = dynamics_matrix,
            dynamics_var    = dynamics_var,
            dynamics_mean   = dynamics_mean
        )

        new_carry = (new_key, resampled_particles)
        scan_collection = (resampled_particles, log_weights, inds)
        return new_carry, scan_collection

    key, key_scan_init, key_particle_init = random.split(key, 3)

    initial_particles = random.multivariate_normal(
        key_particle_init, initial_mean, initial_var, (n_particles, )
    )
    
    initial_carry = (key_scan_init, initial_particles)
    scan_inputs = (obs, logit_input_sequence)
    _, scan_outputs = lax.scan(scan_step, initial_carry, scan_inputs)

    resampled_particles, log_weights, inds = scan_outputs

    filter_results = {
        "z0"    : initial_particles,
        "z1toT" : resampled_particles,
        "logw"  : log_weights, 
        "inds"  : inds
    }

    return filter_results
    
@jit
def backward_simulation_step(
    carry: jax.Array,
    scan_outputs: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_var: jax.Array
):

    key, smoothed_particles_next = carry
    key, subkey = random.split(key)

    resampled_particles, log_forward_weights = scan_outputs
    
    means_next_given_t = jnp.einsum('kj,nj->nk', dynamics_matrix, resampled_particles, optimize="optimal")
    
    log_trans_probs = jstats.multivariate_normal.logpdf(
        smoothed_particles_next, means_next_given_t, dynamics_var
    )
    
    log_backward_weights = log_forward_weights + log_trans_probs
    ancestor_ind = random.categorical(subkey, log_backward_weights)
    smoothed_particle = resampled_particles[ancestor_ind]

    new_carry = (key, smoothed_particle)
    per_step_output = smoothed_particle 
    
    return new_carry, per_step_output


def run_backward_simulation(
    key: jax.Array,
    initial_particles: jax.Array,
    resampled_particles_sequence: jax.Array,
    log_weights_sequence: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_var: jax.Array
) -> jax.Array:

    T_plus_1, _, _ = resampled_particles_sequence.shape
    T = T_plus_1 - 1

    key, subkey_init, subkey_scan = random.split(key, 3)

    final_filter_log_weights = log_weights_sequence[T-1]
    j_T = random.categorical(subkey_init, final_filter_log_weights)
    z_tilde_T = resampled_particles_sequence[T, j_T, :]

    particles_to_scan = jnp.concatenate([initial_particles[jnp.newaxis, :], resampled_particles_sequence[:-1]])
    weights_to_scan = log_weights_sequence             
    scan_inputs = (particles_to_scan, weights_to_scan)

    scan_fn = partial(backward_simulation_step, dynamics_matrix=dynamics_matrix, dynamics_var=dynamics_var)
    initial_carry = (subkey_scan, z_tilde_T)

    _, backward_scan_outputs = lax.scan(
        scan_fn, initial_carry, scan_inputs, reverse=True
    )

    smoothed_trajectory = jnp.concatenate(
        [backward_scan_outputs,
         z_tilde_T[jnp.newaxis, :]],         
        axis=0
    )

    return smoothed_trajectory

def run_particle_smoother_multiple(
    n_samples: int, 
    key: jax.Array,
    initial_particles: jax.Array,
    resampled_particles_sequence: jax.Array,
    log_weights_sequence: jax.Array,
    dynamics_matrix: jax.Array,
    dynamics_var: jax.Array
    ) -> jax.Array:
    keys = random.split(key, n_samples)

    vectorized_backward_sim = vmap(
        run_backward_simulation,
        in_axes=(0, None, None, None, None, None) 
    )

    all_smoothed_trajectories = vectorized_backward_sim(
        keys, initial_particles, resampled_particles_sequence, log_weights_sequence, dynamics_matrix, dynamics_var
    )

    return all_smoothed_trajectories 
