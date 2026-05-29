import torch
import numpy as np



def get_fixed_quartets(N: int, B: int, seed=42):
    """Generates the fixed subset of training quartets for the model."""
    np.random.seed(seed)
    sampled_quartets = []
    while len(sampled_quartets) < B:
        q = np.random.choice(N, size=4, replace=False)
        q.sort()
        sampled_quartets.append(torch.tensor(q, dtype=torch.long))
    return torch.stack(sampled_quartets)



def get_fresh_test_quartets(N: int, B_test: int = 1000, train_quartets: torch.Tensor = None, seed: int = 123):
    """
    Generates a fresh set of unique quartets for the validation step, ensuring
    there is zero overlap with the quartets used during training.

    Args:
        N (int): Number of taxa
        B_test (int): Number of test quartets to be generated
        train_quartets (Tensor, optional): The (B, 4) tensor of training quartets 
                                           stored in model.fixed_indices
        seed (int): Random seed for reproducibility

    Returns:
        Tensor: The filtered test quartets tensor of shape (B_test, 4)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Convert training quartets to a set of ordered tuples for O(1) lookups
    train_set = set()
    if train_quartets is not None:
        # We loop through the tensor rows and save them as tuples
        train_set = {tuple(q.tolist()) for q in train_quartets}

    test_quartets = []
    while len(test_quartets) < B_test:
        # 1. Randomly sample 4 distinct taxa indices
        q = np.random.choice(N, size=4, replace=False)
        # 2. Sort them to ensure strict ascending order: a < b < c < d
        q.sort()
        q_tuple = tuple(q.tolist())
        
        # 3. Strict Guard Condition: Only keep if it was NOT used in training
        if q_tuple not in train_set:
            test_quartets.append(torch.tensor(q, dtype=torch.long))
            
    # Stack them back into a single clean tensor of shape (B_test, 4)
    return torch.stack(test_quartets)


import torch
import numpy as np

def soft_four_point_penalty(D: torch.Tensor, sampled_quartets: torch.Tensor, tau=0.1):
    """
    Computes the differentiable soft four-point tree relaxation penalty vector.
    Aligns exactly to the manuscript product loop formulation without allowing
    inactive channels to zero out active backpropagation gradients.
    """
    a, b, c, d = sampled_quartets[:, 0], sampled_quartets[:, 1], sampled_quartets[:, 2], sampled_quartets[:, 3]

    # Compute the three path-sum tracks
    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]

    sums = torch.stack([s1, s2, s3], dim=-1) # (B, 3)

    # Compute differences matrix (B, 3, 3) where diff[i, k, j] = s_k - s_j
    diff = sums.unsqueeze(2) - sums.unsqueeze(1)
    
    # Apply temperature-scaled softplus relaxation: (B, 3, 3)
    z = tau * torch.log(1.0 + torch.exp(diff / tau))

    # Identity mask to handle the diagonal (when j == k, s_k - s_k = 0)
    mask = torch.eye(3, device=D.device).unsqueeze(0)  
    
    z_stable = torch.clamp(z, min=1e-8)
    z_masked = z_stable * (1.0 - mask) + mask

    # Aligned directly with Equation 62: multiply across columns, then sum across rows
    penalty = torch.sum(torch.prod(z_masked, dim=2), dim=1)  # Shape: (B,)
    
    return penalty