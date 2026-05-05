import jax
import jax.numpy as jnp
import equinox as eqx
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

def initial_analytic_condition(x):
    return x**2 * jnp.cos(jnp.pi*x)

def get_data (main_key):
    
    

    mat_data = scipy.io.loadmat('./Main_BPINN_Functions/Data/burgers_shock.mat')

    t_mat = mat_data['t'].flatten()[:,None]
    x_mat = mat_data['x'].flatten()[:,None]
    Exact_mat = np.real(mat_data['usol']).T
    X_mat, T_mat = np.meshgrid(x_mat,t_mat)
    X_star = np.hstack((X_mat.flatten()[:,None], T_mat.flatten()[:,None]))    #1. column: x values. 2. column: t values
    u_star = Exact_mat.flatten()[:,None]

    x_data = jnp.array(X_star[:,0:1])
    t_data = jnp.array(X_star[:,1:2])
    u_data = jnp.array(u_star)

    # We will define the collocation points for the neural network

    N_r   = 10000 # Number of residual collocation points
    N_ic  = 400 # Number of initial condition space points
    N_bc_l= 400 # Number of left boundary condtions time point
    N_bc_r= 400 # Number of right boundary condtions time point

    main_key, k1, k2, k3, k4, k5 = jax.random.split(main_key, 6)

    #  Define the spatial and temporal domain for the Burgers equation
    # x_min, x_max = -1.0, 1.0
    # t_min, t_max = 0.0, 1.0

    x_min, x_max = x_mat.min(), x_mat.max()
    t_min, t_max = t_mat.min(), t_mat.max()

    # PDE residual points
    x_pde=jax.random.uniform(key=k1,shape=(N_r,1),minval=x_min+0.001,maxval=x_max-0.001)
    t_pde=jax.random.uniform(key=k2,shape=(N_r,1),minval=t_min+0.001,maxval=t_max-0.001)
    u_pde=jnp.zeros_like(x_pde)


    #Creation of Initial Conditions collocation points
    x_ic=jax.random.uniform(key=k3,shape=(N_ic,1),minval=x_min,maxval=x_max)
    t_ic=jnp.zeros_like(x_ic)
    u_ic=initial_analytic_condition(x_ic)

    #Creation of the left boundary collocation points
    t_bc_l=jax.random.uniform(key=k4,shape=(N_bc_l,1),minval=t_min,maxval=t_max)
    x_bc_l=-jnp.ones_like(t_bc_l)
    u_bc_l=jnp.zeros_like(t_bc_l)


    #Creation of right boundary collocation points
    t_bc_r=jax.random.uniform(key=k5,shape=(N_bc_r,1),minval=t_min,maxval=t_max)
    x_bc_r=jnp.ones_like(t_bc_r)
    u_bc_r=jnp.zeros_like(t_bc_r)

    batch = {
        "x_ic": x_ic, "t_ic": t_ic,
        "x_bc_l": x_bc_l, "t_bc_l": t_bc_l,
        "x_bc_r": x_bc_r, "t_bc_r": t_bc_r,
        "x_pde": x_pde, "t_pde": t_pde,
        "x_data": x_data, "t_data": t_data,

        "u_ic": u_ic, "u_bc_l": u_bc_l, "u_bc_r": u_bc_r, "u_pder": u_pde, "u_data": u_data,
        }

    return batch , main_key


def network_selection (periodic_BC, enhanced_MLP, layers_dims, L = 2.0 , M = 10):
    if periodic_BC == 'Y' and enhanced_MLP == 'Y':
        network_type = 'Fourier_Emb_Enhanced_MLP'
        layers_dims[0] = layers_dims[0] * M + 2
    elif periodic_BC == 'Y' and enhanced_MLP == 'N':
        network_type = 'Fourier_Emb_MLP'
        layers_dims[0] = layers_dims[0] * M + 2
    elif periodic_BC == 'N' and enhanced_MLP == 'Y':
        network_type = 'Enhanced_MLP'
    elif periodic_BC == 'N' and enhanced_MLP == 'N':
        network_type = 'Vanilla'
    return network_type, layers_dims, L, M

def define_sigma_likelihood ():
    sigma_likelihood = {
        # 'SIGMA_ic' : jnp.array(1.0/(2.0*jnp.pi*1000.0),dtype=jnp.float32), 
        # 'SIGMA_pde' : jnp.array(1.0/(2.0*jnp.pi*1000.0),dtype=jnp.float32),  
        # 'SIGMA_bc_l' : jnp.array(1.0/(2.0*jnp.pi*1000.0),dtype=jnp.float32), 
        # 'SIGMA_bc_r' : jnp.array(1.0/(2.0*jnp.pi*1000.0),dtype=jnp.float32)

        'SIGMA_ic'       : jnp.array(0.005,dtype=jnp.float32), 
        'SIGMA_pde'      : jnp.array(0.001,dtype=jnp.float32),  
        'SIGMA_bc_l'     : jnp.array(0.005,dtype=jnp.float32), 
        'SIGMA_bc_r' : jnp.array(0.005,dtype=jnp.float32)  
    } 
    return sigma_likelihood

def initialize_local_lambdas(batch_data):
    local_lambdas ={
        "lambda_pde"    : jnp.ones_like(batch_data["x_pde"].shape[0], dtype=jnp.float32),
        "lambda_bc_l"   : jnp.ones_like(batch_data["x_bc_l"].shape[0], dtype=jnp.float32),
        "lambda_bc_r"   : jnp.ones_like(batch_data["x_bc_l"].shape[0], dtype=jnp.float32),
        "lambda_ic"     : jnp.ones_like(batch_data["x_ic"].shape[0], dtype=jnp.float32),
    }
    return local_lambdas

def initialize_global_lambdas (IC_VALUE):
    global_lambdas = {
        "lambda_lik_pde"      : jnp.array(1.0, dtype=jnp.float32),
        "lambda_lik_bc_r"     : jnp.array(1.0, dtype=jnp.float32),
        "lambda_lik_bc_l"     : jnp.array(1.0, dtype=jnp.float32),
        "lambda_lik_ic"       : jnp.array(IC_VALUE, dtype=jnp.float32),
        'lambda_prior'        : jnp.array(1.0, dtype=jnp.float32),
        'lambda_var_posterior': jnp.array(1.0, dtype=jnp.float32)
    }
    return global_lambdas


def RBA_apply (residuum, old_lambda, GAMMA = 0.999, L_R = 0.001, EPS = 1E-12):
    
    new_lambda = GAMMA * old_lambda + L_R * jnp.abs( residuum / (jnp.max(jnp.abs(residuum)) + EPS) )

    new_lambda = jax.lax.stop_gradient(new_lambda)
    
    new_residuum = new_lambda * residuum
        
    return new_residuum, new_lambda

def res_pde_Burgers (model, x, t, sampled_params):
    u,du_dx,du_dt,du_dxx = model.out_u_pde(x,t,sampled_params)
    residuum = du_dt + u * du_dx -(0.01/jnp.pi)*du_dxx - 0
    return residuum

def res_bc_Burgers (model, x, t, sampled_params):
    u, _, _ , _ = model.out_u_only(x,t,sampled_params)
    residuum  = u - 0
    return residuum

def res_ic_Burgers (model, x, t, sampled_params):
    u, _, _ , _ = model.out_u_only(x,t,sampled_params)
    residuum = u - -jnp.sin(jnp.pi*x)
    return residuum

def gaussian_log_likelihood (residuum, sigma):
        gauss_likelihood = jnp.sum(-0.5*jnp.log(2.0*jnp.pi*jnp.square(sigma))-(jnp.square(residuum)/(2.0*jnp.square(sigma))))
        return gauss_likelihood

def gaussian_log_likelihood_RBA (residuum, sigma, lambdas):
        gauss_likelihood_RBA = -0.5*jnp.log(2.0*jnp.pi*jnp.square(sigma/lambdas))-(jnp.square(residuum)/(2.0*jnp.square(sigma)))
        return jnp.sum(gauss_likelihood_RBA)

def likelihood_RBA (residuum, sigma, old_lambdas, GAMMA = 0.999, L_R = 0.001, EPS = 1E-12):
       
    likelihood_residual = (1/(2*sigma**2)) * (residuum)**2

    new_lambdas = GAMMA * old_lambdas + L_R * jnp.abs( likelihood_residual / (jnp.max(jnp.abs(likelihood_residual)) + EPS) )

    new_lambdas = jax.lax.stop_gradient(new_lambdas)

    likelihood = (-0.5*jnp.log(2.0*jnp.pi*jnp.square(sigma))-(jnp.square(residuum)/(2.0*jnp.square(sigma))))

    total_RBA_likelihood = jnp.sum(new_lambdas * likelihood) 

    return total_RBA_likelihood, new_lambdas

def single_sample_loss (model, batch_data, local_lambdas, global_lambdas, local_lambda_tech, sigma_likelihood, fwd_key, GAMMA_RBA, LR_RBA):
    EPS = 1E-12

    x_ic   = batch_data["x_ic"]
    t_ic   = batch_data["t_ic"]
    x_bc_l = batch_data["x_bc_l"]
    t_bc_l = batch_data["t_bc_l"]
    x_bc_r = batch_data["x_bc_r"]
    t_bc_r = batch_data["t_bc_r"]
    x_pde  = batch_data["x_pde"]
    t_pde  = batch_data["t_pde"]
    x_data = batch_data["x_data"]
    t_data = batch_data["t_data"]

    sampled_params, sampled_sum_log_prior, sampled_sum_log_var_posterior = model.u_model.sample_prior_var_posterior(fwd_key, sample = True)

    res_pde_pre = res_pde_Burgers(model, x_pde, t_pde, sampled_params)

    res_bc_l_pre = res_bc_Burgers (model, x_bc_l,t_bc_l,sampled_params)

    res_bc_r_pre = res_bc_Burgers (model, x_bc_r, t_bc_r,sampled_params)

    res_ic_pre = res_ic_Burgers(model, x_ic, t_ic, sampled_params)


    if local_lambda_tech == 'Residual_RBA':

        new_res_pde, new_local_lambda_pde = RBA_apply(res_pde_pre,local_lambdas['lambda_pde'],GAMMA_RBA,LR_RBA,EPS)
        new_res_bc_l, new_local_lambda_bc_l = RBA_apply(res_bc_l_pre,local_lambdas['lambda_bc_l'],GAMMA_RBA,LR_RBA,EPS)
        new_res_bc_r, new_local_lambda_bc_r = RBA_apply(res_bc_r_pre,local_lambdas['lambda_bc_r'],GAMMA_RBA,LR_RBA,EPS)
        new_res_ic, new_local_lambda_ic = RBA_apply(res_ic_pre,local_lambdas['lambda_ic'],GAMMA_RBA,LR_RBA,EPS)

        sampled_log_lik_pde      = gaussian_log_likelihood_RBA(new_res_pde, sigma_likelihood['SIGMA_pde'],new_local_lambda_pde)
        sampled_log_lik_bc_l     = gaussian_log_likelihood_RBA(new_res_bc_l, sigma_likelihood['SIGMA_bc_l'],new_local_lambda_bc_l)
        sampled_log_lik_bc_r = gaussian_log_likelihood_RBA(new_res_bc_r, sigma_likelihood['SIGMA_bc_r'],new_local_lambda_bc_r)
        sampled_log_lik_ic       = gaussian_log_likelihood_RBA(new_res_ic, sigma_likelihood['SIGMA_ic'],new_local_lambda_ic)

        unaltered_log_lik_pde      = gaussian_log_likelihood(res_pde_pre, sigma_likelihood['SIGMA_pde'])
        unaltered_log_lik_bc_l     = gaussian_log_likelihood(res_bc_l_pre, sigma_likelihood['SIGMA_bc_l'])
        unaltered_log_lik_bc_r = gaussian_log_likelihood(res_bc_r_pre, sigma_likelihood['SIGMA_bc_r'])
        unaltered_log_lik_ic       = gaussian_log_likelihood(res_ic_pre, sigma_likelihood['SIGMA_ic'])

    # elif local_lambda_tech == 'SA':
    #     new_res_pde, new_local_lambda_pde = SA_apply(res_pde_pre,local_lambdas_SA.lambda_pde)
    #     new_res_bc_u, new_local_lambda_bc_u = SA_apply(res_bc_u_pre,local_lambdas_SA.lambda_bc_u)
    #     new_res_bc_du_dx, new_local_lambda_bc_du_dx = SA_apply(res_bc_du_dx_pre,local_lambdas_SA.lambda_bc_du_dx)
    #     new_res_ic, new_local_lambda_ic = SA_apply(res_ic_pre,local_lambdas_SA.lambda_ic)

    elif local_lambda_tech == 'Likelihood_RBA':

        sampled_log_lik_pde,  new_local_lambda_pde          = likelihood_RBA(res_pde_pre, sigma_likelihood['SIGMA_pde'], local_lambdas['lambda_pde'], GAMMA_RBA, LR_RBA, EPS)
        sampled_log_lik_bc_l, new_local_lambda_bc_l         = likelihood_RBA(res_bc_l_pre, sigma_likelihood['SIGMA_bc_l'], local_lambdas['lambda_bc_l'], GAMMA_RBA, LR_RBA, EPS)
        sampled_log_lik_bc_r, new_local_lambda_bc_r = likelihood_RBA(res_bc_r_pre, sigma_likelihood['SIGMA_bc_r'], local_lambdas['lambda_bc_r'], GAMMA_RBA, LR_RBA, EPS)
        sampled_log_lik_ic, new_local_lambda_ic             = likelihood_RBA(res_ic_pre, sigma_likelihood['SIGMA_ic'], local_lambdas['lambda_ic'], GAMMA_RBA, LR_RBA, EPS)

        unaltered_log_lik_pde      = gaussian_log_likelihood(res_pde_pre, sigma_likelihood['SIGMA_pde'])
        unaltered_log_lik_bc_l     = gaussian_log_likelihood(res_bc_l_pre, sigma_likelihood['SIGMA_bc_l'])
        unaltered_log_lik_bc_r = gaussian_log_likelihood(res_bc_r_pre, sigma_likelihood['SIGMA_bc_r'])
        unaltered_log_lik_ic       = gaussian_log_likelihood(res_ic_pre, sigma_likelihood['SIGMA_ic'])

    else:
        new_res_pde, new_local_lambda_pde = res_pde_pre, local_lambdas['lambda_pde']
        new_res_bc_l, new_local_lambda_bc_l = res_bc_l_pre, local_lambdas['lambda_bc_l']
        new_res_bc_r, new_local_lambda_bc_r = res_bc_r_pre, local_lambdas['lambda_bc_r']
        new_res_ic, new_local_lambda_ic = res_ic_pre, local_lambdas['lambda_ic']

        sampled_log_lik_pde      = gaussian_log_likelihood(new_res_pde, sigma_likelihood['SIGMA_pde'])
        sampled_log_lik_bc_l     = gaussian_log_likelihood(new_res_bc_l, sigma_likelihood['SIGMA_bc_l'])
        sampled_log_lik_bc_r = gaussian_log_likelihood(new_res_bc_r, sigma_likelihood['SIGMA_bc_r'])
        sampled_log_lik_ic       = gaussian_log_likelihood(new_res_ic, sigma_likelihood['SIGMA_ic'])

        unaltered_log_lik_pde      = sampled_log_lik_pde
        unaltered_log_lik_bc_l     = sampled_log_lik_bc_l
        unaltered_log_lik_bc_r = sampled_log_lik_bc_r
        unaltered_log_lik_ic       = sampled_log_lik_ic



    return ({'log_lik_pde':sampled_log_lik_pde,
            'log_lik_ic':sampled_log_lik_ic,
            'log_lik_bc_l':sampled_log_lik_bc_l,
            'log_lik_bc_r':sampled_log_lik_bc_r,
            'log_prior':sampled_sum_log_prior,
            'log_var_posterior':sampled_sum_log_var_posterior,
            'unaltered_log_lik_pde':unaltered_log_lik_pde,
            'unaltered_log_lik_ic':unaltered_log_lik_ic,
            'unaltered_log_lik_bc_l':unaltered_log_lik_bc_l,
            'unaltered_log_lik_bc_r':unaltered_log_lik_bc_r},
            {'lambda_pde':new_local_lambda_pde,
            'lambda_bc_l':new_local_lambda_bc_l,
            'lambda_bc_r':new_local_lambda_bc_r,
            'lambda_ic':new_local_lambda_ic
            })

@eqx.filter_jit
def loss_terms (model, batch_data, local_lambdas, global_lambdas, local_lambda_tech, sigma_likelihood, fwd_key, K_SAMPLES, GAMMA_RBA, LR_RBA):

    keys = jax.random.split(fwd_key, K_SAMPLES)

    vmapped_fn = jax.vmap(lambda k: single_sample_loss(model, batch_data, local_lambdas, global_lambdas, local_lambda_tech, 
                                                       sigma_likelihood, k, GAMMA_RBA, LR_RBA))
    
    sampled_results, sampled_local_lambdas = vmapped_fn(keys)

    total_log_lik_pde = jnp.mean(sampled_results['log_lik_pde'])
    total_log_lik_ic = jnp.mean(sampled_results['log_lik_ic'])
    total_log_lik_bc_l = jnp.mean(sampled_results['log_lik_bc_l'])
    total_log_lik_bc_r = jnp.mean(sampled_results['log_lik_bc_r'])
    
    unaltered_total_log_lik_pde = jnp.mean(sampled_results['unaltered_log_lik_pde'])
    unaltered_total_log_lik_ic = jnp.mean(sampled_results['unaltered_log_lik_ic'])
    unaltered_total_log_lik_bc_l = jnp.mean(sampled_results['unaltered_log_lik_bc_l'])
    unaltered_total_log_lik_bc_r = jnp.mean(sampled_results['unaltered_log_lik_bc_r'])



    total_log_prior = jnp.mean(sampled_results['log_prior'])
    total_log_var_posterior = jnp.mean(sampled_results['log_var_posterior'])

    total_log_likelihood = (
        global_lambdas['lambda_lik_pde'] * total_log_lik_pde +
        global_lambdas['lambda_lik_bc_l'] * total_log_lik_bc_l +
        global_lambdas['lambda_lik_bc_r'] * total_log_lik_bc_r +
        global_lambdas['lambda_lik_ic'] * total_log_lik_ic
    )

    kl_divergence = (
        global_lambdas['lambda_var_posterior'] * total_log_var_posterior -
        global_lambdas['lambda_prior'] * total_log_prior
    )

    neg_ELBO = kl_divergence - total_log_likelihood

    log_loss = {
        'log_lik_pde': total_log_lik_pde,
        'log_lik_ic': total_log_lik_ic,
        'log_lik_bc_l':total_log_lik_bc_l,
        'log_lik_bc_r':total_log_lik_bc_r,
        'unaltered_log_lik_pde': unaltered_total_log_lik_pde,
        'unaltered_log_lik_bc_l':unaltered_total_log_lik_bc_l,
        'unaltered_log_lik_bc_r':unaltered_total_log_lik_bc_r,
        'unaltered_log_lik_ic':unaltered_total_log_lik_ic,
        'total_log_prior': total_log_prior,
        'total_log_var_posterior' : total_log_var_posterior,
        'negative_ELBO': neg_ELBO
    }

    new_local_lambdas = {
        'lambda_pde':jnp.mean(sampled_local_lambdas['lambda_pde'],axis=0),
        'lambda_bc_l':jnp.mean(sampled_local_lambdas['lambda_bc_l'],axis=0),
        'lambda_bc_r':jnp.mean(sampled_local_lambdas['lambda_bc_r'],axis=0),
        'lambda_ic':jnp.mean(sampled_local_lambdas['lambda_ic'],axis=0)
    }

    return neg_ELBO, (log_loss, new_local_lambdas)

@eqx.filter_jit
def make_step_None (model, optimizer1, state1, batch_data, local_lambdas, global_lambdas, local_lambda_tech, sigma_likelihood, fwd_key, K_SAMPLES, GAMMA_RBA, LR_RBA):

    new_global_lambdas = global_lambdas

    (loss, (loss_log, new_local_lambdas)), grads = eqx.filter_value_and_grad(loss_terms, has_aux = True)(model, batch_data, local_lambdas, global_lambdas, local_lambda_tech, sigma_likelihood, fwd_key,
                                                                                                         K_SAMPLES,GAMMA_RBA,LR_RBA)

    updates, new_opt1_state = optimizer1.update(grads,state1,model)
    new_model = eqx.apply_updates (model, updates)


    return new_model, new_opt1_state, loss_log, new_local_lambdas, new_global_lambdas

def logs_init ():
    log = {
    "negELBO_history"            : [],
    "log_lik_bc_l_history"            : [],
    "log_lik_bc_r_history"        : [],
    "log_lik_ic_history"              : [],
    "unaltered_log_lik_pde": [],
    "unaltered_log_lik_ic": [],
    "unaltered_log_lik_bc_l": [],
    "unaltered_log_lik_bc_r": [], 
    "log_lik_pde_history"             : [],
    "log_prior_history"             : [],
    "log_var_posterior_history"             : [],
    "lambda_g_pde_history"        : [],
    "lambda_g_ic_history"       : [],
    "lambda_g_bc_l_history"     : [],
    "lambda_g_bc_r_history" : [],
    "lambda_g_prior_history" : [],
    "lambda_g_var_posterior_history" : [],
    "lambda_l_pde_history"        : [],
    "lambda_l_ic_history"       : [],
    "lambda_l_bc_l_history"     : [],
    "lambda_l_bc_r_history" : [],
    #"lambda_l_pde_SA_history"        : [],
    #"lambda_l_ic_SA_history"       : [],
    #"lambda_l_bc_u_SA_history"     : [],
    #"lambda_l_bc_du_dx_SA_history" : [],
    }
    return log

def update_logs (logs, logs_history, global_lambdas, local_lambdas):
    logs_history["negELBO_history"].append(float(logs["negative_ELBO"]))
    logs_history["log_lik_bc_l_history"].append(float(logs["log_lik_bc_l"]))
    logs_history["log_lik_bc_r_history"].append(float(logs["log_lik_bc_r"]))
    logs_history["log_lik_ic_history"].append(float(logs["log_lik_ic"]))
    logs_history["log_lik_pde_history"].append(float(logs["log_lik_pde"]))
    logs_history["unaltered_log_lik_pde"].append(float(logs["unaltered_log_lik_pde"]))
    logs_history["unaltered_log_lik_ic"].append(float(logs["unaltered_log_lik_ic"]))
    logs_history["unaltered_log_lik_bc_l"].append(float(logs["unaltered_log_lik_bc_l"]))
    logs_history["unaltered_log_lik_bc_r"].append(float(logs["unaltered_log_lik_bc_r"]))
    logs_history["log_prior_history"].append(float(logs["total_log_prior"]))
    logs_history["log_var_posterior_history"].append(float(logs["total_log_var_posterior"]))
    logs_history["lambda_g_pde_history"].append(float(global_lambdas["lambda_lik_pde"]))
    logs_history["lambda_g_ic_history"].append(float(global_lambdas["lambda_lik_ic"]))
    logs_history["lambda_g_bc_l_history"].append(float(global_lambdas["lambda_lik_bc_l"]))
    logs_history["lambda_g_bc_r_history"].append(float(global_lambdas["lambda_lik_bc_r"]))
    logs_history["lambda_g_prior_history"].append(float(global_lambdas["lambda_prior"]))
    logs_history["lambda_g_var_posterior_history"].append(float(global_lambdas["lambda_var_posterior"]))
    logs_history["lambda_l_pde_history"].append(float(jnp.mean(local_lambdas["lambda_pde"])))
    logs_history["lambda_l_ic_history"].append(float(jnp.mean(local_lambdas["lambda_ic"])))
    logs_history["lambda_l_bc_l_history"].append(float(jnp.mean(local_lambdas["lambda_bc_l"])))
    logs_history["lambda_l_bc_r_history"].append(float(jnp.mean(local_lambdas["lambda_bc_r"])))
    #logs_history["lambda_l_pde_SA_history"].append(float(jnp.mean(local_lambdas_SA.lambda_pde)))
    #logs_history["lambda_l_ic_SA_history"].append(float(jnp.mean(local_lambdas_SA.lambda_ic)))
    #logs_history["lambda_l_bc_u_SA_history"].append(float(jnp.mean(local_lambdas_SA.lambda_bc_u)))
    #logs_history["lambda_l_bc_du_dx_SA_history"].append(float(jnp.mean(local_lambdas_SA.lambda_bc_du_dx)))


    return logs_history


def predictive_samples (model, x, t, sample_key, n_samples = 25):
    keys = jax.random.split(sample_key, n_samples)
    
    def single_sample (key):
        sampled_params, _, _ = model.u_model.sample_prior_var_posterior(key,sample=True)
        u, _, _, _ , _, _ = model.out_u_only (x,t,sampled_params)
        return u.squeeze()
    
    preds = jax.vmap(single_sample)(keys)
    return preds

def get_validation_data (batch_data, key, n_val):

    x_data = batch_data['x_data']
    t_data = batch_data['t_data']
    u_data = batch_data['u_data']

    N = x_data.shape[0]

    idx = jax.random.choice(key, N, shape = (n_val, ), replace = False)

    X_val = x_data[idx]
    T_val = t_data[idx]
    U_true_val = u_data[idx]
    return X_val.squeeze(), T_val.squeeze(), U_true_val.squeeze()

def compute_interval_metrics (pred_samples, U_true_val, ALPHA_Interval = 0.05):
    lower = jnp.percentile(pred_samples, 100*(ALPHA_Interval/2), axis=0)
    upper = jnp.percentile(pred_samples, 100*(1-ALPHA_Interval/2), axis=0)

    lower=lower.squeeze()
    upper=upper.squeeze()
    
    MPIW_val = jnp.mean(upper-lower)
    PICP_val = jnp.mean((U_true_val >= lower) & (U_true_val <= upper))

    var_mean_val = jnp.mean(jnp.var(pred_samples,axis=0))
    return lower, upper, MPIW_val, PICP_val, var_mean_val

def validation_step (model, batch_data, sample_key, N_VAL_SAMPLES, N_VAL_POINTS, ALPHA_INTERVAL):

    sample_key, data_key = jax.random.split(sample_key)

    X_val, T_val, U_true_val = get_validation_data(batch_data, data_key, N_VAL_POINTS)

    pred_samples = predictive_samples(model, X_val, T_val, sample_key, N_VAL_SAMPLES)

    mean_val = jnp.mean(pred_samples,axis=0)

    REL_L2_val = jnp.linalg.norm(mean_val-U_true_val)/ jnp.linalg.norm(U_true_val)

    lower_val, upper_val, MPIW_val, PICP_val, var_mean_val = compute_interval_metrics(pred_samples, U_true_val, ALPHA_INTERVAL)

    return REL_L2_val, MPIW_val, PICP_val, var_mean_val