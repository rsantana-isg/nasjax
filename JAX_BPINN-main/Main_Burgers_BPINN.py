import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import scipy.io
import time
import matplotlib.pyplot as plt
#from Main_BPINN_Functions.Aux_functions_Main_AC_BPINN import get_data, network_selection, define_sigma_likelihood, initialize_local_lambdas, initialize_global_lambdas, make_step_None, logs_init, update_logs
from Main_BPINN_Functions.PINN_AC_BPINN import PINN_Bayesian_AC
from Main_BPINN_Functions.Prior_MLP_functions import (sns_pi, sns_sigma_spike, sns_sigma_slab, mean_Laplace, beta_Laplace, mean_Gaussian, sigma_Gaussian)
import wandb
from Main_BPINN_Functions.Sample_Results import sample_predictive_posterior_Burgers
from Main_BPINN_Functions.StatisticMetrics import compute_deterministic_metrics, compute_probabilistic_metrics


### Burgers equation, Take it into account.
from Main_BPINN_Functions.Aux_functions_Main_Burgers_BPINN import get_data, network_selection, define_sigma_likelihood, initialize_local_lambdas, initialize_global_lambdas, make_step_None, logs_init, update_logs, validation_step

project_name = "Burgers_Hidden_Layers_BPINN"
file_name = 'Burgers_BPINN_'

initial_seed_JAX = 1234
main_key = jax.random.PRNGKey(initial_seed_JAX)

# prior_type='SnS'
# prior_type='Laplace'
prior_type='Gaussian'

local_lambda_tech = 'N' # Possible options : 'N'one,'Likelihood_RBA'/'Residual_RBA', ###'SA' No SA implemented yet.
global_lambda_tech = 'N' # Possible options : 'N'one,'LRA', 'NTK'

periodic_BC_hard_constrainst = 'N' # Possible options : 'Y'es,'N'o
enhanced_MLP = 'Y' # Possible options : 'Y'es,'N'o

main_key, init_key = jax.random.split(main_key)
activation_function = 'tanh' # Possible options : 'tanh', 'relu', 'selu'
# layers_dims = [2, 128, 128, 128, 128, 128, 1]
# layers_dims = [2, 20, 20, 1]

data_root = './Main_BPINN_Functions/Data/burgers_shock.mat'

IC_VALUE = 100

GAMMA_RBA = 0.999
LR_RBA = 0.001

HIDDEN_WIDTH = 50
N_HIDDEN_LAYERS = 2

# layers_dims = [2, 128, 128, 128, 128, 128, 1]
layers_dims = [2] + [HIDDEN_WIDTH] * N_HIDDEN_LAYERS + [1]
LEARNING_RATE = 0.001
N_EPOCHS = 500
EPS = 0.0001
ALPHA = 0.9
GAMMA = 0.9
LEARNING_RATE_SA = 0.001
GLOBAL_LAMBDA_FREQ = 100
K_SAMPLES_PER_EPOCH = 1
Inference_N_SAMPLES = 500
VAL_FREQ = 25
N_VAL_SAMPLES = 50
N_VAL_POINTS = 50
ALPHA_INTERVAL = 0.05

save_filename = (
    f"{file_name}"
    f"RBA_{local_lambda_tech}"
    f"_Prior_{prior_type}"
    f"_hBC_{periodic_BC_hard_constrainst}"
    f"_mMLP_{enhanced_MLP}"
    f"_Hidden_{N_HIDDEN_LAYERS}"
    f"_Width_{HIDDEN_WIDTH}"
)

sigma_likelihood = define_sigma_likelihood()

batch_data , main_key = get_data(main_key)

network_type , layer_dims, L_Fourier, M_Fourier = network_selection(periodic_BC_hard_constrainst,enhanced_MLP,layers_dims)

local_lambdas = initialize_local_lambdas(batch_data)

global_lambdas = initialize_global_lambdas(IC_VALUE)


main_key, init_key = jax.random.split(main_key)

model = PINN_Bayesian_AC(layers_dims,activation_function, prior_type, network_type, L_Fourier, M_Fourier, init_key)
optimizer = optax.adam(learning_rate = LEARNING_RATE)
opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

# wandb.init(
#      project = project_name,
#      name = save_filename,
#      config = {
#         "prior_type": prior_type,
#         "activation_function": activation_function,
#         "layers_dims": layers_dims,
#         "learning_rate": LEARNING_RATE,
#         "N_EPOCHS": N_EPOCHS,
#         "GAMMA_RBA": GAMMA_RBA,
#         "LR_RBA": LR_RBA,
#         "GLOBAL_LAMBDA_FREQ": GLOBAL_LAMBDA_FREQ,
#         "K_SAMPLES_PER_EPOCH":K_SAMPLES_PER_EPOCH,
#         "Inference_N_SAMPLES":Inference_N_SAMPLES,
#         "IC_VALUE":IC_VALUE,
#         "local_lambda_tech": local_lambda_tech,
#         "global_lambda_tech": global_lambda_tech,
#         "network type": network_type,
#         "sigma_likelihood": sigma_likelihood["SIGMA_pde"],
#         "sigma_ic": sigma_likelihood["SIGMA_ic"],
#         "sigma_bc_l": sigma_likelihood["SIGMA_bc_l"],
#         "sigma_bc_r": sigma_likelihood["SIGMA_bc_r"],
#         "prior_SnS_pi": sns_pi,
#         "prior_SnS_spike_sigma": sns_sigma_spike,
#         "prior_SnS_slab_sigma": sns_sigma_slab,
#         "prior_Laplace_mean": mean_Laplace,
#         "prior_Laplace_beta": beta_Laplace,
#         "prior_Gaussian_mean": mean_Gaussian,
#         "prior_Gaussian_sigma": sigma_Gaussian,
#         "initial_seed_JAX":initial_seed_JAX,
#         "filename":file_name,
#         'VAL_FREQ' : VAL_FREQ, 
#         'N_VAL_SAMPLES': N_VAL_SAMPLES,
#         'N_VAL_POINTS': N_VAL_POINTS,
#         'ALPHA_INTERVAL': ALPHA_INTERVAL,
#         'HIDDEN_WIDTH': HIDDEN_WIDTH,
#         'N_HIDDEN_LAYERS': N_HIDDEN_LAYERS
#         }

# )

loss_logs = []

epoch_times = []
start_training_time = time.time()

logs_history = logs_init()

for epoch in range (N_EPOCHS):

        iter_start = time.time()

        update_lambdas = (epoch % GLOBAL_LAMBDA_FREQ == 0 )

        main_key, fwd_key = jax.random.split(main_key)

        model, opt_state, loss_logs, local_lambdas, global_lambdas = make_step_None(
            model, optimizer, opt_state, batch_data, local_lambdas, global_lambdas, local_lambda_tech, sigma_likelihood, fwd_key,
            K_SAMPLES_PER_EPOCH,GAMMA_RBA,LR_RBA)
        
        



        logs_history = update_logs(loss_logs, logs_history, global_lambdas, local_lambdas)

        if epoch % 25 == 0:
            print(f"Epoch: {epoch}, loss: {loss_logs['negative_ELBO']}")

        # wandb.log({
        #       "epoch":epoch,
        #       "negative_ELBO":loss_logs["negative_ELBO"],
        #       "log_lik_pde":loss_logs["log_lik_pde"],
        #       "log_lik_ic":loss_logs["log_lik_ic"],
        #       "log_lik_bc_l":loss_logs["log_lik_bc_l"],

        #       "log_lik_bc_r":loss_logs["log_lik_bc_r"],
        #       "unaltered_log_lik_pde": loss_logs["unaltered_log_lik_pde"],
        #       "unaltered_log_lik_ic": loss_logs["unaltered_log_lik_ic"],
        #       "unaltered_log_lik_bc_l": loss_logs["unaltered_log_lik_bc_l"],
        #       "unaltered_log_lik_bc_r": loss_logs["unaltered_log_lik_bc_r"],            

        #       "total_log_prior":loss_logs["total_log_prior"],
        #       "total_log_var_posterior":loss_logs["total_log_var_posterior"],
        #       "total_KL": loss_logs["total_log_var_posterior"] - loss_logs["total_log_prior"],

        #       "lambda_g_pde":global_lambdas["lambda_lik_pde"],
        #       "lambda_g_ic":global_lambdas["lambda_lik_ic"],
        #       "lambda_g_bc_l":global_lambdas["lambda_lik_bc_l"],
        #       "lambda_g_bc_r":global_lambdas["lambda_lik_bc_r"],
        #       "lambda_g_prior":global_lambdas["lambda_prior"],
        #       "lambda_g_var_posterior":global_lambdas["lambda_var_posterior"],

        #       "lambda_l_pde":jnp.mean(local_lambdas["lambda_pde"]),
        #       "lambda_l_ic":jnp.mean(local_lambdas["lambda_ic"]),
        #       "lambda_l_bc_l":jnp.mean(local_lambdas["lambda_bc_l"]),
        #       "lambda_l_bc_r":jnp.mean(local_lambdas["lambda_bc_r"]),
        # })

        iter_end = time.time()
        
        # if epoch % VAL_FREQ == 0:
        #     main_key, sample_key = jax.random.split(main_key)
        #     REL_L2_val, MPIW_val, PICP_val, var_mean_val = validation_step (model, batch_data, sample_key, N_VAL_SAMPLES, N_VAL_POINTS, ALPHA_INTERVAL)
        #     wandb.log({
        #          "val_REL_L2":REL_L2_val,
        #          "val_MPIW":MPIW_val,
        #          "val_PICP":PICP_val,
        #          "val_var_mean":var_mean_val
        #     })

        #     eqx.tree_serialise_leaves(f"{save_filename}_epoch_{epoch}.eqx", model)
            
        #     logs_history_val = {name: np.array(values, dtype=np.float32) 
        #                for name, values in logs_history.items()}
            
        #     np.savez(
        #         f"{save_filename}_epoch_{epoch}.npz",
        #         **logs_history_val,
        #         N_EPOCHS=np.array(N_EPOCHS),
        #         PRIOR_TYPE = prior_type,
        #         ACTIVATION_FUNC = activation_function,
        #         LEARNING_RATE=np.array(LEARNING_RATE),
        #         # LEARNING_RATE_SA=np.array(LEARNING_RATE_SA),
        #         GLOBAL_LAMBDA_FREQ=np.array(GLOBAL_LAMBDA_FREQ),
        #         layer_dims = np.array(layer_dims),
        #         activation_function = activation_function
        #         # BC_loss_weight=np.array(BC_loss_weight),
        #         # IC_loss_weight=np.array(IC_loss_weight),
        #     )
            
            

        if epoch > 0 :
            epoch_times.append(iter_end-iter_start)

end_training_time = time.time()
total_training_time = end_training_time - start_training_time

mean_iter_time = sum(epoch_times) / len(epoch_times)
std_iter_time = (sum((t - mean_iter_time)**2 for t in epoch_times) / len(epoch_times))**0.5

# wandb.log({
#       "total_training_time":total_training_time,
#       "mean_iter_time":mean_iter_time,
#       "std_iter_time":std_iter_time,
# })

eqx.tree_serialise_leaves(save_filename + ".eqx", model)

# model_artifact = wandb.Artifact("trained_model",type="model")
# model_artifact.add_file(save_filename + ".eqx")
# wandb.log_artifact(model_artifact)

logs_history_np = {name: np.array(values, dtype=np.float32) 
           for name, values in logs_history.items()}
        
np.savez(
    save_filename + ".npz",
    **logs_history_np,
    epoch_times=np.array(epoch_times, dtype=np.float64),
    total_training_time=np.array(total_training_time, dtype=np.float64),
    mean_iter_time=np.array(mean_iter_time, dtype=np.float64),
    std_iter_time=np.array(std_iter_time, dtype=np.float64),
    N_EPOCHS=np.array(N_EPOCHS),
    PRIOR_TYPE = prior_type,
    ACTIVATION_FUNC = activation_function,
    LEARNING_RATE=np.array(LEARNING_RATE),
    # LEARNING_RATE_SA=np.array(LEARNING_RATE_SA),
    GLOBAL_LAMBDA_FREQ=np.array(GLOBAL_LAMBDA_FREQ),
    layer_dims = np.array(layer_dims),
    activation_function = activation_function
    # BC_loss_weight=np.array(BC_loss_weight),
    # IC_loss_weight=np.array(IC_loss_weight),
)

# logs_artifact = wandb.Artifact(name="training_logs",type="dataset")
# logs_artifact.add_file(save_filename + ".npz")
# wandb.log_artifact(logs_artifact)

# main_key, pred_posterior_key = jax.random.split(main_key)

# U_sampled_2D, U_Exact_2D, U_Exact_flat, U_mean_flat, U_mean_2D, U_std_flat, U_std_2D, X, T, Nx, Nt = sample_predictive_posterior_Burgers(Inference_N_SAMPLES,model, pred_posterior_key, data_root)

# MAE, MSE, RMSE, RELATIVE_ERROR_MAP, RELATIVE_ERROR_MEAN, RELATIVE_L2, MAX_ERROR, R2_SCORE, U_mean, U_std = compute_deterministic_metrics(U_sampled_2D.squeeze(),U_Exact_flat.squeeze())

# Z_VALUES = jnp.array([1.96, 1.64, 1.0, 0.67])
# CONF_Levels = jnp.array([0.95, 0.90, 0.68, 0.50])
# ALPHA_WINKLER = 1 - CONF_Levels

# (NLL_mean, NLL_2D,
#     CRPS_mean, CRPS_map,
#     PICP_mean, PICP_map, MPIW_mean, MPIW_map, Winkler_Score) = compute_probabilistic_metrics(U_sampled_2D.squeeze(),U_Exact_flat.squeeze(), Nt, Nx, Z_VALUES, ALPHA_WINKLER)


# wandb.log({
#       "MAE": float(MAE),
#       "MSE": float(MSE),
#       "RMSE": float(RMSE),
#       "RELATIVE_ERROR_MEAN": float(RELATIVE_ERROR_MEAN),
#       "RELATIVE_L2": float(RELATIVE_L2),
#       "MAX_ERROR": float(MAX_ERROR),
#       "R2_SCORE": float(R2_SCORE),
#       "NLL_mean": float(NLL_mean),  
#       "CRPS_mean": float(CRPS_mean),
# })

# for i, z in enumerate(Z_VALUES):
#     wandb.log({
#         f"PICP_{z:.2f}": float(PICP_mean[i]),
#         f"MPIW_{z:.2f}": float(MPIW_mean[i]),
#     })

# def to_np (x):
#     return np.array(jax.device_get(x))

# results_dict = {

#     # --- Deterministic metrics ---
#     "MAE": float(MAE),
#     "MSE": float(MSE),
#     "RMSE": float(RMSE),
#     "RELATIVE_ERROR_MEAN": float(RELATIVE_ERROR_MEAN),
#     "RELATIVE_L2": float(RELATIVE_L2),
#     "MAX_ERROR": float(MAX_ERROR),
#     "R2_SCORE": float(R2_SCORE),

#     # --- Probabilistic metrics ---
#     "NLL_mean": float(NLL_mean),
#     "CRPS_mean": float(CRPS_mean),
#     "PICP_mean": to_np(PICP_mean),
#     "MPIW_mean": to_np(MPIW_mean),

#     # --- 2D fields ---
#     "U_mean_2D": to_np(U_mean_2D),
#     "U_std_2D": to_np(U_std_2D),
#     "NLL_2D": to_np(NLL_2D),

#     # --- Coordinates ---
#     "X": to_np(X),
#     "T": to_np(T),

#     # --- Experiment metadata ---
#     "Nt": Nt,
#     "Nx": Nx,
#     "N_SAMPLES": Inference_N_SAMPLES,
#     "prior_type": prior_type,
#     "activation_function": activation_function,
#     "layers_dims": np.array(layers_dims),
#     "learning_rate": LEARNING_RATE,
#     "N_EPOCHS": N_EPOCHS,
# }


# np.savez_compressed(
#     save_filename + "_Result_Analysis.npz",
#     **results_dict
# )

# wandb.finish()
