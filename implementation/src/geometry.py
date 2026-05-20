import torch

def compute_distance_matrix(x, eps=1e-7):
    """
    Computes pairwise Poincaré distances supporting batched and unbatched coordinates.

    Args:
    x : latent positions , could be unbatched - (N,K) or batched - (S,N,K)
    eps: stability tolerance, prevents dividing by 0

    Returns:
    single tensor containing computed pairwise distances between all taxa

    """
    is_batched = (x.dim() == 3) #checks if x is batched or unbatched, if batched TRUE
    x_sq_norm = torch.sum(x**2, dim=-1, keepdim=True) #adds squared values of every element across the columns
    #keepdim puts 1 as placeholder - (N, 1) or (S, N, 1)
    

    #Tensor broadcasting used to calculate distances between all pairs of latent points
    if is_batched: #(S,N,K)
        dist_sq = torch.sum((x.unsqueeze(2) - x.unsqueeze(1)) ** 2, dim=-1) #resulting difference - (S,N,N,K) --> (S,N,N)
        denom = (1 - x_sq_norm.unsqueeze(2)) * (1 - x_sq_norm.unsqueeze(1))
        denom = denom.squeeze(-1) #(N,N,1) ---> (N,N)

    else:
        #(N,K)
        dist_sq = torch.sum((x.unsqueeze(1) - x.unsqueeze(0)) ** 2, dim=-1) #(N,N)
        denom = (1 - x_sq_norm.unsqueeze(1)) * (1 - x_sq_norm.unsqueeze(0))
        denom = denom.squeeze(-1)

    arg = 1 + 2 * dist_sq / torch.clamp(denom, min=eps) #torch.clamp replaces the denominator with eps if it is less than eps,
    #this is done because if divided by zero it would lead to NaN values and break the NUTS gradient
    return torch.acosh(torch.clamp(arg, min=1.0 + eps))


