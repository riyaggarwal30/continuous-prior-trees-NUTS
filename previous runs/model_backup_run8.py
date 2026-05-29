import torch
import pyro
import pyro.distributions as dist
from src.geometry import compute_distance_matrix
from src.penalty import get_fixed_quartets, soft_four_point_penalty

class PhylogeneticPrior:
    def __init__(self, N, K, B, sigma_u ,seed=42):
        self.N = N
        self.K = K
        self.B = B
        self.sigma = sigma_u
        self.fixed_indices = get_fixed_quartets(N, B, seed)

    def initialize(self, lmbda_4=1.0, lmbda_g=0.0, tau=0.1, use_scale=False, g0=0.1):
        # 1. Coordinate Base Prior
        u = pyro.sample("u", dist.Normal(0, self.sigma).expand([self.N, self.K]).to_event(1))

        # 2. Hyperbolic Radial Projection
        u_norm = torch.norm(u, p=2, dim=-1, keepdim=True)
        u_norm_stable = torch.clamp(u_norm, min=1e-7)
        x = (u / u_norm_stable) * torch.tanh(u_norm)

        # 3. Pairwise Metric Construction
        D_raw = compute_distance_matrix(x)

        # 4. Scale Normalization
        triu_indices = torch.triu_indices(self.N, self.N, offset=1)
        mean_D = torch.clamp(D_raw[triu_indices[0], triu_indices[1]].mean(), min=1e-6)
        D_tilde = D_raw / mean_D
        pyro.deterministic("D_tilde", D_tilde)

        # 5. Structural Penalization Forces
        if lmbda_4 != 0.0:
            # REMOVED: .mean() here so gradients flow without smoothing out
            p_4pt = soft_four_point_penalty(D_tilde, self.fixed_indices, tau)
            
            # Use a pyro.plate to inform the engine that these are independent quartet constraints
            with pyro.plate("quartet_plate", self.B):
                pyro.factor("four_point_force", -lmbda_4 * p_4pt)
        if lmbda_g != 0.0:
            a, b, c, d = self.fixed_indices[:, 0], self.fixed_indices[:, 1], self.fixed_indices[:, 2], self.fixed_indices[:, 3]
            s1 = D_tilde[a, b] + D_tilde[c, d]
            s2 = D_tilde[a, c] + D_tilde[b, d]
            s3 = D_tilde[a, d] + D_tilde[b, c]
            
            s_stacked = torch.stack([s1, s2, s3], dim=-1)
            s_sorted, _ = torch.sort(s_stacked, dim=-1)
            g_q = s_sorted[..., 1] - s_sorted[..., 0]
            
            # Hinge loss: Only penalize quartets whose individual branch gap is smaller than g0
            individual_penalties = torch.clamp(g0 - g_q, min=0.0) ** 2
            
            # Keep the passive tracker variable for your console logging
            pyro.deterministic("star_loss", individual_penalties.mean())
            
            # CRITICAL FIX: Explicitly apply the structural force array over the quartet plate
            with pyro.plate("anti_star_plate", self.B):
                pyro.factor("anti_star_force", -lmbda_g * individual_penalties)

        if use_scale:
            log_s = pyro.sample("log_s", dist.Normal(0.0, 1.0))
            s = torch.exp(log_s)
            D = pyro.deterministic("D", s * D_tilde)
            return D
            
        return D_tilde