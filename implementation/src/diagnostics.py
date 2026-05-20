import torch
import numpy as np

def get_fresh_test_quartets(N: int, B_test: int = 1000,train_quartets: torch.Tensor = None, seed: int = 123):
    """This function generates fresh pair of quartets for validation step.

    Args:
    N: Number of taxa
    B_test : Number of test quartets to be generated

    Returns:
    The test quartets tensor of shape (B_test,4)
    """

    torch.manual_seed(seed)
    np.random.seed(seed)
    test_quartets = []
    while len(test_quartets) < B_test:
        q = np.random.choice(N, size=4, replace=False)
        q.sort()
        test_quartets.append(torch.tensor(q, dtype=torch.long))
    return torch.stack(test_quartets)


#diagnostics for training quartets automatically calculating when NUTS is run
#we will create a function for diagnostics on test quartets

def evaluate_test_diagnostics(D_samples, test_quartets, epsilon=0.05, gamma=0.05):
    """
    Computes geometric and structural diagnostics on held-out test quartets
    and evaluates the global distance scale of the generated matrices.
    
    Args:
        D_samples (Tensor): Batch of MCMC distance matrices, shape ( S, N, N) , we get this after running NUTS
                            where S is number of samples, N is number of leaves.
        test_quartets (Tensor): Held-out quartet indices, shape (B_test, 4)
        epsilon (float): Tolerance threshold for approximate tree-consistency.
        gamma (float): Resolution threshold for star-tree detection.
        """
    
    q = test_quartets.long()  # Shape: (B_test, 4) #ensures int64 datatype
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]


    #1. Extracting the pairwise distances from MCMC samples for test quartets 
    #D_samples shape:  (S,N,N)
    d_ab = D_samples[:, a, b]
    d_cd = D_samples[:, c, d]
    d_ac = D_samples[:, a, c]
    d_bd = D_samples[:, b, d]
    d_ad = D_samples[:, a, d]
    d_bc = D_samples[:, b, c]

    #2. Computing three possible sums
    s1 = d_ab + d_cd  #  Shape: (S, B_test)
    s2 = d_ac + d_bd  # Shape: (S, B_test) 
    s3 = d_ad + d_bc  # Shape: (S, B_test)

    #Stack along a new dimension to easily sort them: Shape (S, B_test, 3)
    s_stacked = torch.stack([s1, s2, s3], dim=-1)
    s_sorted, _ = torch.sort(s_stacked, dim=-1) # Sorts ascending along the last axis

    # Extract sorted sums: s_(1) <= s_(2) <= s_(3)]
    s_1_sorted = s_sorted[..., 0] 
    s_2_sorted = s_sorted[..., 1]
    s_3_sorted = s_sorted[..., 2]


    # 3. Compute Metrics per MCMC sample 
    v_q = s_3_sorted - s_2_sorted  # Hard Four-Point Violation 
    g_q = s_2_sorted - s_1_sorted  # Quartet Gap (Resolution) 


    # 4. Calculate Global Distance Scale Diagnostics (across all pairs i < j) 
    #we check all pairwise distances , if max distance is too low, all points are near the center, no exponential branching happening
    #if max distance too large, points way out of the boundary
    #we calculate this to get the right value for sigma_u , if distance scale metric too small, we increase sigma_u, else decrease sigma_u
    num_nodes = D_samples.shape[1]
    #D_samples -> (S,N,K) where N is at index 1 and represents number of nodes
    triu_indices = torch.triu_indices(num_nodes, num_nodes, offset=1)
    #a standard pairwise distance matrix has 0 in the diagnol and Di,j = Dj,i
    #triu_indices generates coordinates for upper triangle in matrix
    #offset = 1 tells pytorch to skip the main diagnol and starts one row above it
    all_pairwise_distances = D_samples[:, triu_indices[0], triu_indices[1]]
    # : ->  all MCMC samples, triu_indices[0], triu_indices[1] extracts out only top triangle values



    results = {
        "hard_violations": {
            "median": torch.median(v_q).item(), 
            "quantile_95": torch.quantile(v_q, 0.95).item(), 
            "quantile_99": torch.quantile(v_q, 0.99).item() 
        },

        "tree_consistency_rate": torch.mean((v_q <= epsilon).float()).item(),

        "quartet_gap": {
            "median": torch.median(g_q).item(),
            "quantile_05": torch.quantile(g_q, 0.05).item(), 
            "unresolved_fraction": torch.mean((g_q <= gamma).float()).item() 
        },

        "distance_scale": {
            "min": torch.min(all_pairwise_distances).item(), 
            "max": torch.max(all_pairwise_distances).item(), 
            "median": torch.median(all_pairwise_distances).item()
        }
    }
    return results


def print_diagnostic_report(metrics, epsilon, gamma):
    """Prints a beautiful, scannable summary matching paper conventions."""
    print("\n" + "="*55)
    print(f"       HELD-OUT GEOMETRIC DIAGNOSTICS (TEST SET)      ")
    print("="*55)
    
    print("\n[1] Hard Four-Point Violations v_q(D):") 
    print(f"  └─ Median Violation        : {metrics['hard_violations']['median']:.4f}") 
    print(f"  └─ 95th Percentile         : {metrics['hard_violations']['quantile_95']:.4f}") 
    print(f"  └─ 99th Percentile         : {metrics['hard_violations']['quantile_99']:.4f}") 
    
    print(f"\n[2] Approximate Tree-Consistency Rate A(D; ε={epsilon}):")
    print(f"  └─ Fraction Consistent     : {metrics['tree_consistency_rate'] * 100:.2f}%")
    
    print("\n[3] Quartet Gap Diagnostics g_q(D):") 
    print(f"  └─ Median Branch Gap       : {metrics['quartet_gap']['median']:.4f}") 
    print(f"  └─ 5th Percentile Gap      : {metrics['quartet_gap']['quantile_05']:.4f}") 
    print(f"  └─ Star-like Fraction (≤γ) : {metrics['quartet_gap']['unresolved_fraction'] * 100:.2f}%") 
    
    print("\n[4] Global Distance-Scale Diagnostics:")
    print(f"  └─ Minimum Pairwise Dist   : {metrics['distance_scale']['min']:.4f}") 
    print(f"  └─ Median Pairwise Dist    : {metrics['distance_scale']['median']:.4f}") 
    print(f"  └─ Maximum Pairwise Dist   : {metrics['distance_scale']['max']:.4f}") 
    print("="*55 + "\n")

