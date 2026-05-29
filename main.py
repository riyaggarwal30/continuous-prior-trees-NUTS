# main.py
import config
from quartet_penalty import get_fixed_quartets
from sampler import run_inference
import arviz as az
import torch

def main():
    print(f"Initializing experiment with N={config.N}, B={config.B}...")
    
    # 1. Freeze quartets
    quartets = get_fixed_quartets(config.N, config.B, config.SEED)
    
    # 2. Run Sampler
    mcmc = run_inference(quartets)
    
    # 3. Basic Diagnostics
    data = az.from_pyro(mcmc)
    summary = az.summary(data, var_names=["u"])
    print("\nInference Summary (first 5 params):")
    print(summary.head())
    
    # 4. Save results
    samples = mcmc.get_samples()
    torch.save(samples["u"], "outputs/u_samples.pt")
    print("\nSamples saved to outputs/u_samples.pt")

if __name__ == "__main__":
    main()