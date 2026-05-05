import jax
import jax.numpy as jnp
import equinox as eqx
import matplotlib.pyplot as plt


z_jic = jnp.array([1.96, 1.64, 1.0, 0.67]) #Just in case; #%95, %90, %68, %50 PICP



def compute_deterministic_metrics (U_samples_array, U_exact_flat):

    U_mean = jnp.mean(U_samples_array, axis=0).squeeze()
    U_std  = jnp.std (U_samples_array, axis=0).squeeze()

    MAE = jnp.mean(jnp.abs(U_mean-U_exact_flat))

    MSE  = jnp.mean((U_mean-U_exact_flat)**2)
    RMSE = jnp.sqrt(MSE)

    RELATIVE_ERROR_MAP  = jnp.abs(U_mean - U_exact_flat) / (jnp.abs(U_exact_flat)+ 1E-12)
    RELATIVE_ERROR_MEAN = jnp.mean(RELATIVE_ERROR_MAP)

    RELATIVE_L2 = jnp.linalg.norm(U_mean - U_exact_flat) / jnp.linalg.norm(U_exact_flat)

    MAX_ERROR = jnp.max(jnp.abs(U_mean - U_exact_flat)) 

    R2_SCORE = (1 - 
                (jnp.sum((U_exact_flat-U_mean)**2))/
                (jnp.sum((U_exact_flat-jnp.mean(U_exact_flat))**2))
                )
    return MAE, MSE, RMSE, RELATIVE_ERROR_MAP, RELATIVE_ERROR_MEAN, RELATIVE_L2, MAX_ERROR, R2_SCORE, U_mean, U_std


### NEGATIVE LGO LIKELIHOOD

def compute_nll_single_point (y_true_iteration, mu_iteration, sigma_iteration):
    negative_log_likelihood_iteration = (0.5 * jnp.log(2 * jnp.pi * sigma_iteration**2 ) +
    0.5 * ((y_true_iteration-mu_iteration)**2/(sigma_iteration**2)))
    return negative_log_likelihood_iteration

@jax.jit
def compute_nll (y_true, mu, sigma):
    negative_log_likelihood = jax.vmap(compute_nll_single_point, in_axes = (0,0,0))(y_true,mu,sigma)
    return jnp.mean(negative_log_likelihood), negative_log_likelihood

### CRPS: CONTINOUS RANKED PROBAILITY SCORE


def compute_CRPS_ordered(sample_iteration, y_true):
    S = sample_iteration.shape[0]
    term1 = jnp.mean(jnp.abs(sample_iteration - y_true))
    x_sorted = jnp.sort(sample_iteration)
    i = jnp.arange(1, S+1)
    coeff = (2*i - S - 1)
    term2 = jnp.sum(coeff * x_sorted) / (S**2)

    return term1 - term2

def compute_CRPS_single_point (sample_iteration,y_true):
    S = sample_iteration.shape[0]

    term1 = jnp.mean (jnp.abs(sample_iteration-y_true))

    diff = jnp.abs( sample_iteration[:,None] - sample_iteration[None,:])
    term2 = jnp.mean(diff)/2

    return term1 - term2

@jax.jit
def compute_CRPS (samples, y_true):
    CRPS_per_point = jax.vmap(compute_CRPS_ordered, in_axes = (1,0))(samples,y_true)
    return jnp.mean(CRPS_per_point), CRPS_per_point

#### PICP: PREDICTION INTERVAL COVERAGE PROBABILITY
### MPIW: MEAN PREDICTION INTERVAL WIDTH

def compute_PICP_and_MPIW_single_point (y_true_iteration, mu_iteration,sigma_iteration,z_iteration, ALPHA_Winkler):

        lower = mu_iteration - z_iteration * sigma_iteration
        upper = mu_iteration + z_iteration * sigma_iteration

        width = upper - lower

        inside_iteration = (y_true_iteration >= lower) & (y_true_iteration <= upper)

        penalty_lower = (2 / ALPHA_Winkler) * (lower - y_true_iteration)
        penalty_upper = (2 / ALPHA_Winkler) * (y_true_iteration - upper)

        winkler = jnp.where(
            inside_iteration,
            width,
            jnp.where(
                y_true_iteration < lower,
                width + penalty_lower,
                width + penalty_upper
            )
        )

        return inside_iteration, width, winkler


def compute_PICP_and_MPIW_per_z (y_true, mu, sigma, z_value, ALPHA_Winkler):

    inside, width, winkler_score = jax.vmap(compute_PICP_and_MPIW_single_point,in_axes=(0,0,0,None,None))(y_true,mu,sigma,z_value,ALPHA_Winkler)

    return jnp.mean(inside), inside, jnp.mean(width), width, jnp.mean(winkler_score)

@jax.jit
def compute_PICP_and_MPIW (y_true, mu, sigma, z, ALPHA_Winkler):

    inside_mean, inside, width_mean, width, Winkler_Score = jax.vmap(compute_PICP_and_MPIW_per_z,in_axes=(None,None,None,0,0))(y_true,mu,sigma,z, ALPHA_Winkler)

    return inside_mean, inside, width_mean, width, Winkler_Score



def compute_probabilistic_metrics (U_samples_array, U_exact_flat, Nt, Nx, z = z_jic, ALPHA_Winkler = 0.05):

    U_mean = jnp.mean(U_samples_array, axis=0).squeeze()
    U_std  = jnp.std (U_samples_array, axis=0).squeeze()
    U_samples = U_samples_array.squeeze()

    NLL_mean, NLL_map = compute_nll(U_exact_flat, U_mean, U_std)
    NLL_2D = NLL_map.reshape(Nt,Nx)

    CRPS_mean, CRPS_map = compute_CRPS(U_samples, U_exact_flat)
    #CRPS_2D = CRPS_map.reshape(Nt,Nx)

    PICP_mean, PICP_map, MPIW_mean, MPIW_map, Winkler_Score = compute_PICP_and_MPIW(U_exact_flat, U_mean, U_std, z, ALPHA_Winkler)
    #PICP_2D = PICP_map.reshape(len(z),Nt,Nx)
    #MPIW_2D = MPIW_map.reshape(len(z),Nt,Nx)


    return (NLL_mean, NLL_2D, 
            CRPS_mean, CRPS_map, 
            PICP_mean, PICP_map, MPIW_mean, MPIW_map, Winkler_Score)



