# sampler.py
from pyro.infer import MCMC, NUTS
from model import phylogenetic_model
import config

def run_inference(quartets):
    nuts_kernel = NUTS(phylogenetic_model)
    mcmc = MCMC(
        nuts_kernel, 
        num_samples=config.SAMPLES, 
        warmup_steps=config.WARMUP
    )
    
    mcmc.run(
        N=config.N, 
        K=config.K, 
        quartets=quartets, 
        lmbda=config.LMBDA, 
        sigma_u=config.SIGMA_U, 
        tau=config.TAU
    )
    return mcmc