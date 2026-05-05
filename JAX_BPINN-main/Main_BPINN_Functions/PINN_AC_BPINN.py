import jax
import jax.numpy as jnp
import equinox as eqx
from Main_BPINN_Functions.MLP_AC_BPINN import MLP_Bayesian_Vanilla, MLP_Bayesian_Enhanced # MLP_Bayesian_FourierEmbedding, MLP_Bayesian_Enhanced_Fourier_Embedding

def reshape_to_residual (inputs):
    return inputs.reshape(-1,1)

class PINN_Bayesian_AC (eqx.Module):

    layers_dims_list : tuple
    u_model : eqx.Module = eqx.field()

    def __init__ (self, layers_dims, activation_function, prior_type, network_type, L_Fourier, M_Fourier, init_key):

        self.layers_dims_list = tuple (layers_dims)
        if network_type == 'Vanilla':
            self.u_model = MLP_Bayesian_Vanilla (layers_dims, activation_function, prior_type, init_key)
        elif network_type == 'Enhanced_MLP':
            self.u_model = MLP_Bayesian_Enhanced(layers_dims, activation_function, prior_type, init_key)
        # elif network_type == 'Fourier_Emb_MLP':
        #     self.u_model = MLP_Bayesian_FourierEmbedding(layers_dims, activation_function, prior_type, L_Fourier, M_Fourier, init_key)
        # elif network_type == 'Fourier_Emb_Enhanced_MLP':
        #     self.u_model = MLP_Bayesian_Enhanced_Fourier_Embedding(layers_dims, activation_function, prior_type, L_Fourier, M_Fourier, init_key)
  
    
    def u_only (self, x, t, sampled_params):
        u = self.u_model(x, t, sampled_params)
        u = jnp.squeeze(u)
        return u
    
    @eqx.filter_jit
    def u_pde_terms (self, x, t, sampled_params):
        
        def u_scalar(xx,tt, sampled_params): 
            u = self.u_only(xx,tt, sampled_params)
            return u
        
        u = u_scalar (x,t,sampled_params)
        du_dx = jax.grad(lambda xx: u_scalar(xx,t, sampled_params))(x)
        du_dt = jax.grad(lambda tt: u_scalar(x,tt, sampled_params))(t)
        du_dxx = jax.grad(jax.grad(lambda xx: u_scalar(xx,t, sampled_params)))(x)
                          
        return u, du_dx, du_dt, du_dxx
    
    @eqx.filter_jit
    def out_u_only (self, x, t, sampled_params):
        u = jax.vmap(self.u_only, in_axes = (0,0,None))(x.ravel(),t.ravel(), sampled_params)
        du_dx = jnp.zeros_like(u)
        du_dt = jnp.zeros_like(u)
        du_dxx = jnp.zeros_like(u)

        u = reshape_to_residual(u)
        du_dx = reshape_to_residual(du_dx)
        du_dt = reshape_to_residual(du_dt)
        du_dxx = reshape_to_residual(du_dxx)

        return u, du_dx, du_dt, du_dxx
    
    @eqx.filter_jit
    def out_u_pde (self, x, t, sampled_params):
        u, du_dx, du_dt, du_dxx = jax.vmap(self.u_pde_terms, in_axes = (0,0,None))(x.ravel(),t.ravel(), sampled_params)

        u = reshape_to_residual(u)
        du_dx = reshape_to_residual(du_dx)
        du_dt = reshape_to_residual(du_dt)
        du_dxx = reshape_to_residual(du_dxx)

        return u, du_dx, du_dt, du_dxx