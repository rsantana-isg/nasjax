import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import equinox as eqx
import scipy.io

def sample_predictive_posterior_AllenCahn(N_SAMPLES, model, key_sample, data_root):

    Exact_2D, X, T, Nt, Nx = analytical_data_AllenCahn(data_root)

    keys = jax.random.split(key_sample,N_SAMPLES)

    X_flat = X.ravel()
    T_flat = T.ravel()
    U_samples_array = []

    for k in keys:
        u = model.out_u_only(X_flat, T_flat, k)[0]
        U_samples_array.append(u)

    U_samples_array = jnp.stack(U_samples_array)

    U_mean_flat = jnp.mean(U_samples_array,axis=0)
    U_mean_2D = U_mean_flat.reshape(Nt,Nx)

    U_std_flat = jnp.std(U_samples_array, axis=0)
    U_std_2D = U_std_flat.reshape(Nt, Nx)
    
    Exact_flat = Exact_2D.reshape(-1,1)


    return U_samples_array, Exact_2D, Exact_flat, U_mean_flat, U_mean_2D, U_std_flat, U_std_2D, X, T, Nx, Nt


def sample_predictive_posterior_Burgers(N_SAMPLES, model, key_sample, data_root):

    Exact_2D, X, T, Nt, Nx = analytical_data_Burgers(data_root)

    keys = jax.random.split(key_sample,N_SAMPLES)

    X_flat = X.ravel()
    T_flat = T.ravel()
    U_samples_array = []

    for k in keys:
        u = model.out_u_only(X_flat, T_flat, k)[0]
        U_samples_array.append(u)

    U_samples_array = jnp.stack(U_samples_array)

    U_mean_flat = jnp.mean(U_samples_array,axis=0)
    U_mean_2D = U_mean_flat.reshape(Nt,Nx)

    U_std_flat = jnp.std(U_samples_array, axis=0)
    U_std_2D = U_std_flat.reshape(Nt, Nx)
    
    Exact_flat = Exact_2D.ravel()


    return U_samples_array, Exact_2D, Exact_flat, U_mean_flat, U_mean_2D, U_std_flat, U_std_2D, X, T, Nx, Nt




def analytical_data_AllenCahn(data_root):

    mat_data = scipy.io.loadmat(data_root)

    t_coords = mat_data['tt'].flatten()[:,None]
    x_coords = mat_data['x'].flatten()[:,None]
    Exact_sol_2D = np.real(mat_data['uu']).T

    X, T = np.meshgrid(x_coords, t_coords)

    Nt = len(t_coords)
    Nx = len(x_coords)

    return Exact_sol_2D, X, T, Nt, Nx


def analytical_data_Burgers (data_root):

    mat_data = scipy.io.loadmat(data_root) 

    t_coords = mat_data['t'].flatten()
    x_coords = mat_data['x'].flatten()
    Exact_sol_2D = np.real(mat_data['usol']).T

    X, T  = np.meshgrid(x_coords, t_coords)

    Nt = len(t_coords)
    Nx = len(x_coords)


    return Exact_sol_2D, X, T, Nt, Nx
