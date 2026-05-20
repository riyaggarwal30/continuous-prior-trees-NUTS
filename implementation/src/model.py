# src/model.py
import torch
import pyro
import pyro.distributions as dist
from src.geometry import compute_distance_matrix
from src.penalty import get_fixed_quartets, soft_four_point_penalty

class PhylogeneticPrior:
    def __init__(self, N, K, B, seed=42):
        self.N = N
        self.K = K
        self.B = B
        #quartet subsets need to remain fixed across NUTS chains
        #this is because NUTS assumes deterministic potential energy,
        # if quartets are resampled at every leapfrog step, potential energy will change stochastically
        self.fixed_indices = get_fixed_quartets(N, B, seed)

    def initialize(self, lmbda=1.0, sigma_u=1.0, tau=0.1):
        """
        This function generates the penalized prior landscape for NUTS to explore.

        Args:
        self: Instance containing N, B, and K values, plus fixed_indices.
        lmbda: Strength of penalty measure
        sigma_u: variance of unconstrained base prior
        tau: scalar value controlling temperature/ smoothness of relaxation (larger values make gradient smoother for NUTS)

        Returns:
        D : torch.Tensor - induced pairwise distance matrix 

        """
        # 1. Coordinate Base Prior
        u = pyro.sample("u", dist.Normal(0, sigma_u).expand([self.N, self.K]).to_event(1))

        # 2. Hyperbolic Radial Projection
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
        u_norm_stable = torch.clamp(u_norm, min=1e-7)
   
        x_direction = u / u_norm_stable
        x_magnitude = torch.tanh(u_norm)
        x = x_direction * x_magnitude

        # 3. Pairwise Metric Construction
        D = pyro.deterministic("D", compute_distance_matrix(x))

        # 4. Global Structural Penalization Force
        # Only compute and apply the penalty if lambda is active
        if lmbda != 0.0:
            individual_q_penalties = soft_four_point_penalty(D, self.fixed_indices, tau)
            avg_penalty = individual_q_penalties.mean()
            pyro.factor("tree_force", -lmbda * avg_penalty)
            
        return D
    
        # 4. Global Structural Penalization Force
        individual_q_penalties = soft_four_point_penalty(D, self.fixed_indices, tau)
        avg_penalty = individual_q_penalties.mean()

        pyro.factor("tree_force", -lmbda * avg_penalty)
        return D