# hyperbolic.py
import torch

def poincare_map(u, eps=1e-7):
    """Maps unconstrained R^K vectors to the Poincaré Ball."""
    u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
    u_norm_stable = torch.clamp(u_norm, min=eps)
    
    x_direction = u / u_norm_stable
    x_magnitude = torch.tanh(u_norm)
    return x_direction * x_magnitude

def compute_distance_matrix(x, eps=1e-7):
    """Computes pairwise hyperbolic distances in the Poincaré Ball."""
    x_sq_norm = torch.sum(x**2, dim=-1)
    dist_sq = torch.cdist(x, x, p=2)**2
    
    denom = (1 - x_sq_norm.unsqueeze(1)) * (1 - x_sq_norm.unsqueeze(0))
    arg = 1 + 2 * dist_sq / torch.clamp(denom, min=eps)
    
    return torch.acosh(torch.clamp(arg, min=1.0 + eps))