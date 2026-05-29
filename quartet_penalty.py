# quartet_penalty.py
import torch

def get_fixed_quartets(N, B, seed=42):
    """Samples B random quartets once and freezes them."""
    torch.manual_seed(seed)
    quartets = []
    for _ in range(B):
        q = torch.randperm(N)[:4]
        quartets.append(q)
    return torch.stack(quartets)

def soft_four_point_penalty(D, quartets, tau=0.1):
    """Computes the differentiable tree-likeness penalty."""
    a, b, c, d = quartets[:, 0], quartets[:, 1], quartets[:, 2], quartets[:, 3]
    
    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]
    
    sums = torch.stack([s1, s2, s3], dim=-1) # [B, 3]
    
    penalty = 0
    for k in range(3):
        s_curr = sums[:, k]
        other_idx = [j for j in range(3) if j != k]
        
        # zeta_tau(s_k - s_j)
        z1 = tau * torch.log(1 + torch.exp((s_curr - sums[:, other_idx[0]]) / tau))
        z2 = tau * torch.log(1 + torch.exp((s_curr - sums[:, other_idx[1]]) / tau))
        penalty += (z1 * z2)
        
    return penalty.mean() # Returns expected violation over the batch