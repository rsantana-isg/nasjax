import jax
import jax.numpy as jnp
import equinox as eqx

### Functions used to define the prior type

mean_Laplace = 0.0
beta_Laplace = 1.0

mean_Gaussian = 0.0
sigma_Gaussian = 1.0

sns_pi = 0.5
sns_sigma_spike = jnp.exp(-6.0)
sns_sigma_slab = jnp.exp(0.0)


### Second approach to have more flexible priors.

# mean_Gaussian = 0.0
# sigma_Gaussian = 3.0

# sns_pi = 0.2
# sns_sigma_spike = jnp.exp(-3.0)
# sns_sigma_slab = jnp.exp(1.0)



class Laplace_Prior:
    def __init__(self, mean_L = mean_Laplace, beta_L= beta_Laplace):
        self.mu = mean_L
        self.beta = beta_L
    
    def log_prob (self, inputs):
        log_prob_tensor = -jnp.log(2.0 * self.beta) - (jnp.abs(inputs-self.mu) / self.beta)
        return jnp.sum(log_prob_tensor)

class Gaussian_Prior:
    def __init__(self, mean_G = mean_Gaussian, sigma_G = sigma_Gaussian):
        self.mean = mean_G
        self.sigma = sigma_G
        
    def log_prob (self, inputs):
        log_prob_tensor = -jnp.log(jnp.sqrt(2.0*jnp.pi))-jnp.log(self.sigma)-(jnp.square(inputs-self.mean)/(2.0*jnp.square(self.sigma)))
        return jnp.sum(log_prob_tensor)
    
class ScaleMixtureGaussian:
    def __init__(self,pi = sns_pi, sigma1 = sns_sigma_spike, sigma2 = sns_sigma_slab):
        self.pi = pi
        self.sigma_1 = sigma1
        self.sigma_2 = sigma2 
    
    def log_prob_normal (self,inputs,mean,sigma):
        log_prob_tensor = -0.5*jnp.log(2.0*jnp.pi*jnp.square(sigma))-(jnp.square(inputs-mean)/(2.0*jnp.square(sigma)))
        return log_prob_tensor
    
    def log_prob (self, inputs):
        prob1 = jnp.exp(self.log_prob_normal(inputs, mean=0.,sigma=self.sigma_1))
        prob2 = jnp.exp(self.log_prob_normal(inputs, mean=0.,sigma=self.sigma_2))
        log_prob_tensor = jnp.log(self.pi*prob1 + (1-self.pi)*prob2)
        return jnp.sum(log_prob_tensor)
    
def define_prior_type(prior_type):
    if prior_type == 'SnS':
        weight_prior = ScaleMixtureGaussian()
        bias_prior = ScaleMixtureGaussian()
    elif prior_type == 'Laplace':
        weight_prior = Laplace_Prior()
        bias_prior = Laplace_Prior()
    else:
        weight_prior = Gaussian_Prior()
        bias_prior = Gaussian_Prior()
    return weight_prior, bias_prior

### Function used to initialize the network trainable parameters
    
def network_weight_bias_init (input_dim, output_dim, key_init):

    INIT_mu_min_val = -0.2
    INIT_mu_max_val = 0.2

    INIT_rho_min_val = - 5
    INIT_rho_max_val = - 4

    k1, k2, k3, k4 = jax.random.split(key_init, 4)

    weight_shape = (input_dim,output_dim)
    weight_mu = jax.random.uniform(k1, weight_shape, minval = INIT_mu_min_val, maxval = INIT_mu_max_val )
    weight_rho = jax.random.uniform(k2, weight_shape, minval = INIT_rho_min_val, maxval = INIT_rho_max_val )

    bias_shape = (output_dim,)
    bias_mu = jax.random.uniform(k3, bias_shape, minval = INIT_mu_min_val, maxval = INIT_mu_max_val )
    bias_rho = jax.random.uniform(k4, bias_shape, minval = INIT_rho_min_val, maxval = INIT_rho_max_val )

    return weight_mu, weight_rho, bias_mu, bias_rho


### Function to define the activation function used in the network

def define_activation_function (activation_function):
    if activation_function is not None:
        current_activation = getattr (jax.nn, activation_function, lambda x:x)
    else:
        current_activation = lambda x: x
    return current_activation

### Functions used for sampling when doing a forward pass

def sigma_sample (rho):
    return jnp.log1p(jnp.exp(rho))

def epsilon_sample (key, inputs):
    return jax.random.normal(key, shape = inputs.shape, dtype = inputs.dtype)

def sample_step (mu, rho, key):
    return mu + sigma_sample(rho) * epsilon_sample(key, mu)

def gaussian_log_prob (inputs, mu, rho):
    sigma = sigma_sample(rho)
    log_prob_tensor = -0.5*jnp.log(2.0*jnp.pi*jnp.square(sigma))-(jnp.square(inputs-mu)/(2.0*jnp.square(sigma)))
    return jnp.sum(log_prob_tensor)

### Functions used to Hard-Constrain Periodic Boundary Conditions

def Fourier_feature_embedding_1D (x, t, L, M):
        
        # x = x [...,None]
        # t = t [...,None]
        
        w = 2.0 * jnp.pi / L
        k = jnp.arange(1, M+1)
        feature_embedding = jnp.hstack([
            1,
            jnp.cos(k*w*x),
            jnp.sin(k*w*x),
            t
        ])  
        return feature_embedding

def prior_information (prior_type):
    prior = {
        ['mean_Laplace']:[],
        ['beta_Laplace']:[],
        ['mean_Gaussian'] : [],
        ['sigma_Gaussian'] : [],
        ['sns_pi'] : [],
        ['sns_sigma_spike'] :[],
        ['sns_sigma_slab'] :[]
        }
    
    if prior_type == 'SnS':
        prior['sns_pi'] = sns_pi
        prior['sns_sigma_spike'] = sns_sigma_spike
        prior['sns_sigma_slab'] = sns_sigma_slab
        prior['mean_Laplace'] = ''
        prior['beta_Laplace'] = ''
        prior['mean_Gaussian'] = ''
        prior['sigma_Gaussian'] = ''
    elif prior_type == 'Laplace':
        prior['sns_pi'] = ''
        prior['sns_sigma_spike'] = ''
        prior['sns_sigma_slab'] = ''
        prior['mean_Laplace'] = mean_Laplace
        prior['beta_Laplace'] = beta_Laplace
        prior['mean_Gaussian'] = ''
        prior['sigma_Gaussian'] = ''
    else:
        prior['sns_pi'] = ''
        prior['sns_sigma_spike'] = ''
        prior['sns_sigma_slab'] = ''
        prior['mean_Laplace'] = ''
        prior['beta_Laplace'] = ''
        prior['mean_Gaussian'] = mean_Laplace
        prior['sigma_Gaussian'] = sigma_Gaussian

    return sns_pi, 


