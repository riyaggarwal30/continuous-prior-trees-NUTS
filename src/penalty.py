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


def soft_four_point_penalty(D: torch.Tensor, sampled_quartets: torch.Tensor, tau=0.1):
    """
    Computes the differentiable soft four-point tree relaxation penalty vector.

    Args:
    D: computed hyperbolic distance matrix (N,N)
    sampled_quartets: the (B,4) tensor containing sampled quartets to be used for training
    tau: scalar value controlling temperature/ smoothness of relaxation (larger values make gradient smoother for NUTS)

    Returns:
    1D tensor of shape (B,) containing penalty for each of B quartets

    """
    #splits (B,4) tensor into 4 separate tensors of length B.
    a, b, c, d = sampled_quartets[:, 0], sampled_quartets[:, 1], sampled_quartets[:, 2], sampled_quartets[:, 3]

    s1 = D[a, b] + D[c, d]
    s2 = D[a, c] + D[b, d]
    s3 = D[a, d] + D[b, c]

    sums = torch.stack([s1, s2, s3], dim=-1) #(B,3)

    # Advanced trick: Use broadcasting to compare all channels simultaneously
    # sums.unsqueeze(2) shape: (B, 3, 1)
    # sums.unsqueeze(1) shape: (B, 1, 3)
    # diff shape: (B, 3, 3) -> diff[i, k, j] = s_k - s_j
    diff = sums.unsqueeze(2) - sums.unsqueeze(1)
    
    # Apply temperature-scaled softplus relaxation: (B, 3, 3)
    z = tau * torch.log(1 + torch.exp(diff / tau))
    
    # Zero out the diagonal terms where k == j (since s_k - s_k = 0, softplus(0) isn't 0)
    # We want the product only over j != k
    mask = 1.0 - torch.eye(3, device=D.device).unsqueeze(0) # Shape: (1, 3, 3)
    z_masked = z * mask + (1.0 - mask) # Fills diagonals with 1.0 so product ignores them
    
    # Multiply across the j axis (dim=2), then sum across the k axis (dim=1)
    # This precisely evaluates the product loop from your paper 
    penalty = torch.sum(torch.prod(z_masked, dim=2), dim=1) # Shape: (B,)
    
    return penalty