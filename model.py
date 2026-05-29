# model.py
import pyro
import pyro.distributions as dist
from hyperbolic import poincare_map, compute_distance_matrix
from quartet_penalty import soft_four_point_penalty

def phylogenetic_model(N, K, quartets, lmbda=1.0, sigma_u=1.0, tau=0.1):
    # 1. Base Prior
    u = pyro.sample("u", 
                    dist.Normal(0, sigma_u).expand([N, K]).to_event(1))
    
    # 2. Hyperbolic Map
    x = poincare_map(u)
    
    # 3. Induced Distances
    D = compute_distance_matrix(x)
    
    # 4. Tree Penalty
    avg_penalty = soft_four_point_penalty(D, quartets, tau)
    
    # 5. Potential Energy adjustment
    pyro.factor("tree_force", -lmbda * avg_penalty)
    
    return D