import torch
import numpy as np
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import ward, cophenet

def evaluate_test_diagnostics(D_samples, test_quartets, epsilon=0.05, gamma=0.05):
    """
    Computes geometric and structural diagnostics on held-out test quartets
    after performing batch scale-normalization across MCMC samples.
    """
    S, N, _ = D_samples.shape
    
    # 1. Scale Normalization per MCMC Sample
    triu_indices = torch.triu_indices(N, N, offset=1)
    mean_D = D_samples[:, triu_indices[0], triu_indices[1]].mean(dim=-1, keepdim=True).unsqueeze(-1)
    mean_D_stable = torch.clamp(mean_D, min=1e-6)
    
    D_normalized = D_samples / mean_D_stable

    # 2. Extracting the pairwise distances from Normalized samples
    q = test_quartets.long()  
    a, b, c, d = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    s_idx = torch.arange(S, device=D_normalized.device).unsqueeze(-1) # (S, 1)
    d_ab = D_normalized[s_idx, a, b] # (S, B_test)
    d_cd = D_normalized[s_idx, c, d]
    d_ac = D_normalized[s_idx, a, c]
    d_bd = D_normalized[s_idx, b, d]
    d_ad = D_normalized[s_idx, a, d]
    d_bc = D_normalized[s_idx, b, c]

    # 3. Computing three possible sums
    s1 = d_ab + d_cd  
    s2 = d_ac + d_bd  
    s3 = d_ad + d_bc  

    s_stacked = torch.stack([s1, s2, s3], dim=-1)
    s_sorted, _ = torch.sort(s_stacked, dim=-1) # Sorts ascending along the last axis

    s_1_sorted = s_sorted[..., 0] 
    s_2_sorted = s_sorted[..., 1] 
    s_3_sorted = s_sorted[..., 2] 

    # 4. Compute Metrics per MCMC sample 
    v_q = s_3_sorted - s_2_sorted  # Hard Four-Point Violation: s_(3) - s_(2)
    g_q = s_2_sorted - s_1_sorted  # Quartet Gap (Resolution): s_(2) - s_(1)

    all_pairwise_distances = D_normalized[:, triu_indices[0], triu_indices[1]]

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
    """Prints a summary matching paper conventions."""

    
    print("\n[1] Hard Four-Point Violations v_q(D~):") 
    print(f"  └─ Median Violation        : {metrics['hard_violations']['median']:.4f}") 
    print(f"  └─ 95th Percentile (q_0.95): {metrics['hard_violations']['quantile_95']:.4f}") 
    print(f"  └─ 99th Percentile         : {metrics['hard_violations']['quantile_99']:.4f}") 
    
    print(f"\n[2] Approx Tree-Consistency Rate A(D~; ε={epsilon}):")
    print(f"  └─ Fraction Consistent     : {metrics['tree_consistency_rate'] * 100:.2f}%")
    
    print("\n[3] Quartet Gap Diagnostics g_q(D~):") 
    print(f"  └─ Median Branch Gap       : {metrics['quartet_gap']['median']:.4f}") 
    print(f"  └─ 5th Percentile Gap      : {metrics['quartet_gap']['quantile_05']:.4f}") 
    print(f"  └─ Star-like Fraction (≤γ) : {metrics['quartet_gap']['unresolved_fraction'] * 100:.2f}%") 
    
    print("\n[4] Normalized Distance-Scale Diagnostics (Expect Mean=1.0):")
    print(f"  └─ Minimum Pairwise Dist   : {metrics['distance_scale']['min']:.4f}") 
    print(f"  └─ Median Pairwise Dist    : {metrics['distance_scale']['median']:.4f}") 
    print(f"  └─ Maximum Pairwise Dist   : {metrics['distance_scale']['max']:.4f}") 
    print("="*55 + "\n")

def compute_nj_residual(D_matrix):
    """
    Computes the post-hoc tree projection residual R_NJ(D~) 
    using stable scipy hierarchical routines.
    """
    if isinstance(D_matrix, torch.Tensor):
        D_np = D_matrix.detach().cpu().numpy()
    else:
        D_np = np.asarray(D_matrix)
        
    D_np = (D_np + D_np.T) / 2.0
    np.fill_diagonal(D_np, 0.0)
    
    D_condensed = squareform(D_np, checks=False)
    Z = ward(D_condensed)
    
    _, D_tree_condensed = cophenet(Z, D_condensed)
    D_tree_square = squareform(D_tree_condensed)
    
    numerator = np.linalg.norm(D_np - D_tree_square, ord='fro')
    denominator = np.linalg.norm(D_np, ord='fro')
    
    if denominator < 1e-7:
        return 0.0
        
    return float(numerator / denominator)