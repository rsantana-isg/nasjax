import jax 
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np
import matplotlib.pyplot as plt
import time
from Main_BPINN_Functions.Prior_MLP_functions import network_weight_bias_init, define_activation_function, define_prior_type, sample_step, gaussian_log_prob, Fourier_feature_embedding_1D


class MLP_Bayesian_layer (eqx.Module):

    inputs : int = eqx.field(static=True)
    outputs : int = eqx.field(static=True)
    current_activation : object = eqx.field(static=True)
    weight_prior : object = eqx.field(static=True)
    bias_prior : object = eqx.field(static=True)
    weight_mu : jnp.ndarray
    weight_rho : jnp.ndarray
    bias_mu : jnp.ndarray
    bias_rho : jnp.ndarray

    def __init__ (self, input_dim, output_dim, activation_function, prior_type, key_init):

        self.inputs = input_dim
        self.outputs = output_dim

        self.current_activation = define_activation_function(activation_function)

        self.weight_mu, self.weight_rho, self.bias_mu, self.bias_rho = network_weight_bias_init(self.inputs, self.outputs, key_init)

        self.weight_prior, self.bias_prior = define_prior_type(prior_type)

    def log_var_posterior (self, params):
        
        weight = params['weight']
        weight_log_var_posterior = gaussian_log_prob(weight, self.weight_mu, self.weight_rho)

        bias = params['bias']
        bias_log_var_posterior = gaussian_log_prob(bias, self.bias_mu, self.bias_rho)

        layer_total_log_var_posterior = weight_log_var_posterior + bias_log_var_posterior

        return layer_total_log_var_posterior
    
    def log_prior (self, params):

        weight = params['weight']
        weight_log_prior = self.weight_prior.log_prob(weight)

        bias = params['bias']
        bias_log_prior = self.bias_prior.log_prob(bias)

        layer_log_prior = weight_log_prior + bias_log_prior

        return layer_log_prior
    
    def sample_params(self, step_key, sample=True):

        if sample:

            key_weight, key_bias = jax.random.split(step_key)

            weight = sample_step(self.weight_mu, self.weight_rho, key_weight) 
            bias   = sample_step(self.bias_mu, self.bias_rho, key_bias)

        else:
            weight = self.weight_mu
            bias = self.bias_mu

        params = ({
            'weight' : weight,
            'bias'   : bias
        })

        return params

    
    @eqx.filter_jit
    def __call__ (self, inputs, params):

        weight = params['weight']
        bias = params['bias']
        
        output_pre_act = inputs @ weight + bias
        output = self.current_activation(output_pre_act)
        return output
    
def MLP_creation_by_layer (n_layers, layer_dims_list, activation_function, prior_type, layers, keys):
    for i in range(1, n_layers + 1):
        input_dim = layer_dims_list[i-1]
        output_dim = layer_dims_list[i]

        current_activation = activation_function if i < n_layers else None

        MLP_layer = MLP_Bayesian_layer(
            input_dim,
            output_dim,
            current_activation,
            prior_type,
            keys[i-1]
        )

        layers.append(MLP_layer)
    return layers


class MLP_Bayesian_Vanilla (eqx.Module):

    layer_dims_list : tuple = eqx.field(static=True)
    n_layers : int = eqx.field(static=True)
    activation_function : str = eqx.field(static=True)
    prior_type : str = eqx.field(static=True)
    layers_list : tuple [MLP_Bayesian_layer, ...] = eqx.field()

    def __init__ (self, layer_dims_list, activation_function, prior_type, init_key):

        self.layer_dims_list = tuple(layer_dims_list)
        self.n_layers = len(layer_dims_list) - 1
        self.activation_function = activation_function
        self.prior_type = prior_type

        keys = jax.random.split(init_key,self.n_layers)
        layers = []
        layers = MLP_creation_by_layer (self.n_layers, self.layer_dims_list, self.activation_function, self.prior_type, layers, keys)

        self.layers_list = tuple (layers)

    def sample_prior_var_posterior (self, step_key, sample=True):

        step_keys = jax.random.split(step_key, len(self.layers_list))

        net_sampled_params = []
        total_log_prior = jnp.array(0.0)
        total_log_var_posterior = jnp.array(0.0)


        for layer, k in zip(self.layers_list,step_keys):

            sampled_params = layer.sample_params(k, sample)
            net_sampled_params.append(sampled_params)
            layer_log_prior         = layer.log_prior(sampled_params)
            layer_log_var_posterior = layer.log_var_posterior(sampled_params)

            total_log_prior = total_log_prior + layer_log_prior
            total_log_var_posterior = total_log_var_posterior + layer_log_var_posterior

        return net_sampled_params, total_log_prior, total_log_var_posterior

    @eqx.filter_jit
    def __call__ (self, x, t, sampled_params):

        output = jnp.hstack([x,t])
        

        for layer,params in zip(self.layers_list,sampled_params):
            output = layer(output,params)

        return output

class MLP_Bayesian_Enhanced (eqx.Module):

    layer_dims_list : tuple = eqx.field(static=True)
    n_layers : int = eqx.field(static=True)
    activation_function : str = eqx.field(static=True)
    prior_type : str = eqx.field(static=True)
    layers_list : tuple [MLP_Bayesian_layer, ...] = eqx.field()

    w_1_mu : jnp.ndarray
    w_1_rho : jnp.ndarray
    b_1_mu : jnp.ndarray
    b_1_rho : jnp.ndarray
    w_2_mu : jnp.ndarray
    w_2_rho : jnp.ndarray
    b_2_mu : jnp.ndarray
    b_2_rho : jnp.ndarray
    U_weight_prior : object = eqx.field(static=True)
    U_bias_prior : object = eqx.field(static=True)
    V_weight_prior : object = eqx.field(static=True)
    V_bias_prior : object = eqx.field(static=True)

    def __init__ (self, layer_dims_list, activation_function, prior_type, init_key):

        self.layer_dims_list = tuple(layer_dims_list)
        self.n_layers = len(layer_dims_list) - 1
        self.activation_function = activation_function
        self.prior_type = prior_type

        keys = jax.random.split(init_key,self.n_layers)
        layers = []
        layers = MLP_creation_by_layer (self.n_layers, self.layer_dims_list, self.activation_function, self.prior_type, layers, keys)

        self.layers_list = tuple (layers)

        #### Initialization of U and V, Enhanced MLP

        init_key, Enh_key = jax.random.split(init_key,2)
        Enhanced_MLP_key_1, Enhanced_MLP_key_2 = jax.random.split(Enh_key)

        input_layer_dim = layer_dims_list[0]
        layer_1_dim     = layer_dims_list[1]

        self.w_1_mu, self.w_1_rho, self.b_1_mu, self.b_1_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_1)
        self.w_2_mu, self.w_2_rho, self.b_2_mu, self.b_2_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_2)

        self.U_weight_prior, self.U_bias_prior = define_prior_type(prior_type)
        self.V_weight_prior, self.V_bias_prior = define_prior_type(prior_type)
    


    def sample_prior_var_posterior (self, step_key, sample=True):

        step_key, Enh_key = jax.random.split(step_key, 2)

        step_keys = jax.random.split(step_key, len(self.layers_list))

        net_sampled_params = []
        total_log_prior = jnp.array(0.0)
        total_log_var_posterior = jnp.array(0.0)
       

        Enh_k1, Enh_k2, Enh_k3, Enh_k4 = jax.random.split(Enh_key, 4)

        w_1 = sample_step(self.w_1_mu, self.w_1_rho, Enh_k1)
        b_1 = sample_step(self.b_1_mu, self.b_1_rho, Enh_k2)

        w_2 = sample_step(self.w_2_mu, self.w_2_rho, Enh_k3)
        b_2 = sample_step(self.b_2_mu, self.b_2_rho, Enh_k4)

    
        for layer, k in zip(self.layers_list,step_keys):

            sampled_params = layer.sample_params(k, sample)
            net_sampled_params.append(sampled_params)
            layer_log_prior         = layer.log_prior(sampled_params)
            layer_log_var_posterior = layer.log_var_posterior(sampled_params)

            total_log_prior = total_log_prior + layer_log_prior
            total_log_var_posterior = total_log_var_posterior + layer_log_var_posterior

        U_log_prior = self.U_weight_prior.log_prob(w_1) + self.U_bias_prior.log_prob(b_1)
        V_log_prior = self.V_weight_prior.log_prob(w_2) + self.V_bias_prior.log_prob(b_2)

        U_log_var_posterior = gaussian_log_prob(w_1, self.w_1_mu, self.w_1_rho) + gaussian_log_prob(b_1, self.b_1_mu, self.b_1_rho)
        V_log_var_posterior = gaussian_log_prob(w_2, self.w_2_mu, self.w_2_rho) + gaussian_log_prob(b_2, self.b_2_mu, self.b_2_rho)

        net_sampled_params[0]['w_1'] = w_1
        net_sampled_params[0]['b_1'] = b_1
        net_sampled_params[0]['w_2'] = w_2
        net_sampled_params[0]['b_2'] = b_2

        total_log_prior = total_log_prior + U_log_prior + V_log_prior
        total_log_var_posterior = total_log_var_posterior + U_log_var_posterior + V_log_var_posterior

        return net_sampled_params, total_log_prior, total_log_var_posterior

    @eqx.filter_jit
    def __call__ (self, x, t, sampled_params):

        output = jnp.hstack([x,t])

        w_1 = sampled_params[0]['w_1']
        b_1 = sampled_params[0]['b_1']

        w_2 = sampled_params[0]['w_2']
        b_2 = sampled_params[0]['b_2']

        U = self.layers_list[0].current_activation(jnp.dot(output, w_1) + b_1)
        V = self.layers_list[0].current_activation(jnp.dot(output, w_2) + b_2)
        

        for layer,params in zip(self.layers_list[:-1],sampled_params[:-1]):
            output_pre_enh = layer(output,params)
            alpha_1_enh = 1 - output_pre_enh
            alpha_2_enh = output_pre_enh
            output = jnp.multiply(alpha_1_enh, U) + jnp.multiply(alpha_2_enh, V)

        output = self.layers_list[-1](output,sampled_params[-1])

        return output
    
# class MLP_Bayesian_Enhanced (eqx.Module):

#     layer_dims_list : tuple = eqx.field(static=True)
#     n_layers : int = eqx.field(static=True)
#     activation_function : str = eqx.field(static=True)
#     prior_type : str = eqx.field(static=True)
#     layers_list : tuple [MLP_Bayesian_layer, ...] = eqx.field()
#     w_1_mu : jnp.ndarray
#     w_1_rho : jnp.ndarray
#     b_1_mu : jnp.ndarray
#     b_1_rho : jnp.ndarray
#     w_2_mu : jnp.ndarray
#     w_2_rho : jnp.ndarray
#     b_2_mu : jnp.ndarray
#     b_2_rho : jnp.ndarray
#     U_weight_prior : object = eqx.field(static=True)
#     U_bias_prior : object = eqx.field(static=True)
#     V_weight_prior : object = eqx.field(static=True)
#     V_bias_prior : object = eqx.field(static=True)

#     def __init__ (self, layer_dims_list, activation_function, prior_type, init_key):

#         self.layer_dims_list = tuple(layer_dims_list)
#         self.n_layers = len(layer_dims_list) - 1
#         self.activation_function = activation_function
#         self.prior_type = prior_type

#         init_key, Enhanced_MLP_key_1, Enhanced_MLP_key_2 = jax.random.split(init_key, 3)
#         input_layer_dim = layer_dims_list[0]
#         layer_1_dim = layer_dims_list[1]

#         ### Initialization of the trainable parameters of the probabilistic distributions of U and V. Same initialization as the rest of the trainable parameters of the networks. This could (or should) be changed in the future.

#         self.w_1_mu, self.w_1_rho, self.b_1_mu, self.b_1_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_1)
#         self.w_2_mu, self.w_2_rho, self.b_2_mu, self.b_2_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_2)

#         keys = jax.random.split(init_key,self.n_layers)
#         layers = []
#         layers = MLP_creation_by_layer (self.n_layers, self.layer_dims_list, self.activation_function, self.prior_type, layers, keys)

#         self.layers_list = tuple (layers)

#         self.U_weight_prior, self.U_bias_prior = define_prior_type(self.prior_type)
#         self.V_weight_prior, self.V_bias_prior = define_prior_type(self.prior_type)

#     @eqx.filter_jit
#     def __call__ (self, x, t, step_key, sample = True ):

#         output = jnp.hstack([x,t])
#         total_log_prior = jnp.array(0.0)
#         total_log_var_posterior = jnp.array(0.0)

#         step_key, Enh_k1, Enh_k2, Enh_k3, Enh_k4 = jax.random.split(step_key, 5)

#         w_1 = sample_step(self.w_1_mu, self.w_1_rho, Enh_k1)
#         b_1 = sample_step(self.b_1_mu, self.b_1_rho, Enh_k2)

#         w_2 = sample_step(self.w_2_mu, self.w_2_rho, Enh_k3)
#         b_2 = sample_step(self.b_2_mu, self.b_2_rho, Enh_k4)

#         U = self.layers_list[0].current_activation(jnp.dot(output, w_1) + b_1)
#         V = self.layers_list[0].current_activation(jnp.dot(output, w_2) + b_2)

#         step_keys = jax.random.split(step_key, len(self.layers_list))

#         for layer, k in zip(self.layers_list[:-1],step_keys):
#             output_pre_transformer, layer_log_prior, layer_log_var_posterior = layer(output,sample, k)

#             alpha_1 = 1 - output_pre_transformer

#             alpha_2 = output_pre_transformer

#             output = jnp.multiply(alpha_1, U) + jnp.multiply(alpha_2, V)

#             total_log_prior += layer_log_prior
#             total_log_var_posterior += layer_log_var_posterior

#         output_pre_last_layer = output
#         last_layer_weight_key, last_layer_bias_key = jax.random.split(step_keys[-1], 2)
#         last_layer_weight = sample_step(self.layers_list[-1].weight_mu, self.layers_list[-1].weight_rho, last_layer_weight_key)
#         last_layer_bias = sample_step(self.layers_list[-1].bias_mu, self.layers_list[-1].bias_rho, last_layer_bias_key)

#         output = output_pre_last_layer @ last_layer_weight + last_layer_bias

#         last_layer_log_prior = (self.layers_list[-1].weight_prior.log_prob(last_layer_weight) +
#                                 self.layers_list[-1].bias_prior.log_prob(last_layer_bias))
        
#         last_layer_log_var_posterior = (gaussian_log_prob(last_layer_weight, self.layers_list[-1].weight_mu, self.layers_list[-1].weight_rho) +
#                                         gaussian_log_prob(last_layer_bias, self.layers_list[-1].bias_mu, self.layers_list[-1].bias_rho))            
        
#         U_log_prior = self.U_weight_prior.log_prob(w_1) +self.U_bias_prior.log_prob(b_1)
#         U_log_var_posterior = (gaussian_log_prob(w_1,self.w_1_mu,self.w_1_rho) +
#                                gaussian_log_prob(b_1,self.b_1_mu,self.b_1_rho))

#         V_log_prior = self.V_weight_prior.log_prob(w_2) +self.V_bias_prior.log_prob(b_2)
#         V_log_var_posterior = (gaussian_log_prob(w_2,self.w_2_mu,self.w_2_rho) +
#                                gaussian_log_prob(b_2,self.b_2_mu,self.b_2_rho))

#         total_log_prior = total_log_prior + last_layer_log_prior + U_log_prior + V_log_prior
#         total_log_var_posterior = total_log_var_posterior + last_layer_log_var_posterior + U_log_var_posterior + V_log_var_posterior

#         return output, total_log_prior, total_log_var_posterior
    
# class MLP_Bayesian_FourierEmbedding (eqx.Module):

#     layer_dims_list : tuple = eqx.field(static=True)
#     n_layers : int = eqx.field(static=True)
#     activation_function : str = eqx.field(static=True)
#     prior_type : str = eqx.field(static=True)
#     layers_list : tuple [MLP_Bayesian_layer, ...] = eqx.field()
#     L_FourierEmb : float = eqx.field(static=True)
#     M_FourierEmb : int = eqx.field(static=True)

#     def __init__ (self, layer_dims_list, activation_function, prior_type, L_Fourier, M_Fourier, init_key):

#         self.layer_dims_list = tuple(layer_dims_list)
#         self.n_layers = len(layer_dims_list) - 1
#         self.activation_function = activation_function
#         self.prior_type = prior_type

#         self.L_FourierEmb = L_Fourier
#         self.M_FourierEmb = M_Fourier

#         keys = jax.random.split(init_key,self.n_layers)
#         layers = []
#         layers = MLP_creation_by_layer (self.n_layers, self.layer_dims_list, self.activation_function, self.prior_type, layers, keys)

#         self.layers_list = tuple (layers)

#     @eqx.filter_jit
#     def __call__ (self, x, t, step_key, sample = True ):

#         H = Fourier_feature_embedding_1D(x, t, self.L_FourierEmb, self.M_FourierEmb)

#         output = H
#         total_log_prior = jnp.array(0.0)
#         total_log_var_posterior = jnp.array(0.0)

#         step_keys = jax.random.split(step_key, len(self.layers_list))

#         for layer, k in zip(self.layers_list,step_keys):
#             output, layer_log_prior, layer_log_var_posterior = layer(output,sample, k)

#             total_log_prior += layer_log_prior
#             total_log_var_posterior += layer_log_var_posterior

#         return output, total_log_prior, total_log_var_posterior
    
    
# class MLP_Bayesian_Enhanced_Fourier_Embedding (eqx.Module):

#     layer_dims_list : tuple = eqx.field(static=True)
#     n_layers : int = eqx.field(static=True)
#     activation_function : str = eqx.field(static=True)
#     prior_type : str = eqx.field(static=True)
#     layers_list : tuple [MLP_Bayesian_layer, ...] = eqx.field()
#     L_FourierEmb : float = eqx.field(static=True)
#     M_FourierEmb : int = eqx.field(static=True)
#     w_1_mu : jnp.ndarray
#     w_1_rho : jnp.ndarray
#     b_1_mu : jnp.ndarray
#     b_1_rho : jnp.ndarray
#     w_2_mu : jnp.ndarray
#     w_2_rho : jnp.ndarray
#     b_2_mu : jnp.ndarray
#     b_2_rho : jnp.ndarray
#     U_weight_prior : object = eqx.field(static=True)
#     U_bias_prior : object = eqx.field(static=True)
#     V_weight_prior : object = eqx.field(static=True)
#     V_bias_prior : object = eqx.field(static=True)

#     def __init__ (self, layer_dims_list, activation_function, prior_type, L_Fourier, M_Fourier, init_key):

#         self.layer_dims_list = tuple(layer_dims_list)
#         self.n_layers = len(layer_dims_list) - 1
#         self.activation_function = activation_function
#         self.prior_type = prior_type

#         self.L_FourierEmb = L_Fourier
#         self.M_FourierEmb = M_Fourier

#         init_key, Enhanced_MLP_key_1, Enhanced_MLP_key_2 = jax.random.split(init_key, 3)
#         input_layer_dim = layer_dims_list[0]
#         layer_1_dim = layer_dims_list[1]

#         ### Initialization of the trainable parameters of the probabilistic distributions of U and V. Same initialization as the rest of the trainable parameters of the networks. This could (or should) be changed in the future.

#         self.w_1_mu, self.w_1_rho, self.b_1_mu, self.b_1_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_1)
#         self.w_2_mu, self.w_2_rho, self.b_2_mu, self.b_2_rho = network_weight_bias_init(input_layer_dim, layer_1_dim, Enhanced_MLP_key_2)

#         keys = jax.random.split(init_key,self.n_layers)
#         layers = []
#         layers = MLP_creation_by_layer (self.n_layers, self.layer_dims_list, self.activation_function, self.prior_type, layers, keys)

#         self.layers_list = tuple (layers)

#         self.U_weight_prior, self.U_bias_prior = define_prior_type(self.prior_type)
#         self.V_weight_prior, self.V_bias_prior = define_prior_type(self.prior_type)

#     @eqx.filter_jit
#     def __call__ (self, x, t, step_key, sample = True ):

#         H = Fourier_feature_embedding_1D(x, t, self.L_FourierEmb, self.M_FourierEmb)

#         output = H
#         total_log_prior = jnp.array(0.0)
#         total_log_var_posterior = jnp.array(0.0)

#         step_key, Enh_k1, Enh_k2, Enh_k3, Enh_k4 = jax.random.split(step_key, 5)

#         w_1 = sample_step(self.w_1_mu, self.w_1_rho, Enh_k1)
#         b_1 = sample_step(self.b_1_mu, self.b_1_rho, Enh_k2)

#         w_2 = sample_step(self.w_2_mu, self.w_2_rho, Enh_k3)
#         b_2 = sample_step(self.b_2_mu, self.b_2_rho, Enh_k4)

#         U = self.layers_list[0].current_activation(jnp.dot(output, w_1) + b_1)
#         V = self.layers_list[0].current_activation(jnp.dot(output, w_2) + b_2)

#         step_keys = jax.random.split(step_key, len(self.layers_list))

#         for layer, k in zip(self.layers_list[:-1],step_keys):
#             output_pre_transformer, layer_log_prior, layer_log_var_posterior = layer(output,sample, k)

#             alpha_1 = 1 - output_pre_transformer

#             alpha_2 = output_pre_transformer

#             output = jnp.multiply(alpha_1, U) + jnp.multiply(alpha_2, V)

#             total_log_prior += layer_log_prior
#             total_log_var_posterior += layer_log_var_posterior

#         output_pre_last_layer = output
#         last_layer_weight_key, last_layer_bias_key = jax.random.split(step_keys[-1], 2)
#         last_layer_weight = sample_step(self.layers_list[-1].weight_mu, self.layers_list[-1].weight_rho, last_layer_weight_key)
#         last_layer_bias = sample_step(self.layers_list[-1].bias_mu, self.layers_list[-1].bias_rho, last_layer_bias_key)

#         output = output_pre_last_layer @ last_layer_weight + last_layer_bias

#         last_layer_log_prior = (self.layers_list[-1].weight_prior.log_prob(last_layer_weight) +
#                                 self.layers_list[-1].bias_prior.log_prob(last_layer_bias))
        
#         last_layer_log_var_posterior = (gaussian_log_prob(last_layer_weight, self.layers_list[-1].weight_mu, self.layers_list[-1].weight_rho) +
#                                         gaussian_log_prob(last_layer_bias, self.layers_list[-1].bias_mu, self.layers_list[-1].bias_rho))            

#         U_log_prior = self.U_weight_prior.log_prob(w_1) +self.U_bias_prior.log_prob(b_1)
#         U_log_var_posterior = (gaussian_log_prob(w_1,self.w_1_mu,self.w_1_rho) +
#                                gaussian_log_prob(b_1,self.b_1_mu,self.b_1_rho))

#         V_log_prior = self.V_weight_prior.log_prob(w_2) +self.V_bias_prior.log_prob(b_2)
#         V_log_var_posterior = (gaussian_log_prob(w_2,self.w_2_mu,self.w_2_rho) +
#                                gaussian_log_prob(b_2,self.b_2_mu,self.b_2_rho))

#         total_log_prior = total_log_prior + last_layer_log_prior + U_log_prior + V_log_prior
#         total_log_var_posterior = total_log_var_posterior + last_layer_log_var_posterior + U_log_var_posterior + V_log_var_posterior


#         return output, total_log_prior, total_log_var_posterior