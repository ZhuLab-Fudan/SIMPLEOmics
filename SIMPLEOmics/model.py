import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_geometric.nn import GCNConv   
    
class GCNEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNEncoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = F.relu(x)        
        x = self.conv2(x, edge_index)
        return x

class GCNDecoder(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(GCNDecoder, self).__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.conv1(x, edge_index))
        out = self.conv2(h, edge_index)
        return out

class DecAlignSpatial(nn.Module):
    def __init__(self, args):
        super(DecAlignSpatial, self).__init__()
        
        self.orig_d_omic1 = args.input_dim_omic1
        self.orig_d_omic2 = args.input_dim_omic2
        self.d_h = args.hidden_dim
        self.d_mid = self.orig_d_omic1
        
        # self.d_h = 32
        self.recon_dim_omic1 = args.recon_dim_omic1
        self.recon_dim_omic2 = args.recon_dim_omic2
        
        self.concat_dim = 5 * self.d_h
        self.output_dim = args.output_dim
        
        self.proj_omic1 = nn.Linear(self.orig_d_omic1, self.d_mid)
        self.proj_omic2 = nn.Linear(self.orig_d_omic2, self.d_mid)

        self.encoder_uni_omic1 = GCNEncoder(self.d_mid, self.d_mid, self.d_h)
        self.encoder_uni_omic2 = GCNEncoder(self.d_mid, self.d_mid, self.d_h)
        self.encoder_com = GCNEncoder(self.d_mid, self.d_mid, self.d_h)

        self.decoder_omic1 = GCNDecoder(self.d_h*2, self.d_h*2, self.recon_dim_omic1)
        self.decoder_omic2 = GCNDecoder(self.d_h*2, self.d_h*2, self.recon_dim_omic2)


        self.num_prototypes = args.num_prototypes
        self.gmm_em_iters = getattr(args, 'gmm_em_iters', 5)
        self.gmm_var_floor = getattr(args, 'gmm_var_floor', 1e-4)
        self.ot_reg = getattr(args, 'lambda_ot', 0.1)
        self.ot_num_iters = getattr(args, 'ot_num_iters', 50)
        self.lambda_coral = getattr(args, "lambda_coral", 10)

    def compute_decoupling_loss(self, s, c):
        s_norm = (s - s.mean(dim=0, keepdim=True)) / (s.std(dim=0, keepdim=True) + 1e-6)
        c_norm = (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True) + 1e-6)
        cross_corr = c_norm.T @ s_norm / max(c.size(0) - 1, 1)
        return (cross_corr ** 2).mean()
    
    def torch_standard_scale(self, x, eps=1e-6):
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + eps)

    def compute_variance_loss(self, z, eps=1e-4):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _init_gmm(self, x):
        """Initialize a GMM by sampling equidistantly from the current features."""
        batch_size, dim = x.shape
        k = self.num_prototypes
        if batch_size >= k:
            indices = torch.linspace(0, batch_size - 1, steps=k, device=x.device).round().long()
            mu = x.index_select(0, indices)
        else:
            repeat = math.ceil(k / batch_size)
            mu = x.repeat(repeat, 1)[:k]
        
        var = x.var(dim=0, unbiased=False).clamp_min(self.gmm_var_floor)
        var = var.unsqueeze(0).expand(k, dim).clone()
        pi = torch.full((k,), 1.0 / k, device=x.device, dtype=x.dtype)
        return pi, mu, var
    
    def _gmm_log_prob(self, x, pi, mu, var):
        diff = x.unsqueeze(1) - mu.unsqueeze(0)                       # [N, K, D]
        mahalanobis = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)     # [N, K]
        log_det = var.log().sum(dim=1).unsqueeze(0)                   # [1, K]
        log_norm = x.size(1) * math.log(2.0 * math.pi)
        return pi.clamp_min(1e-8).log().unsqueeze(0) - 0.5 * (mahalanobis + log_det + log_norm)

    def _fit_gmm_em(self, x):
        pi, mu, var = self._init_gmm(x)
        for _ in range(self.gmm_em_iters):
            # E-step
            gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
            # M-step
            nk = gamma.sum(dim=0).clamp_min(1e-6)
            pi = nk / x.size(0)
            mu = gamma.transpose(0, 1).matmul(x) / nk.unsqueeze(1)
            second_moment = gamma.transpose(0, 1).matmul(x.pow(2)) / nk.unsqueeze(1)
            var = (second_moment - mu.pow(2)).clamp_min(self.gmm_var_floor)
        gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
        return {"pi": pi, "mu": mu, "var": var, "posterior": gamma}


    def compute_prototypes(self, features, proto, logvar):
        """Calculate soft assignment weights based on the distance between sample features and prototype means."""
        diff = features.unsqueeze(1) - proto.unsqueeze(0)  # [N, K, d_h]
        dist_sq = (diff ** 2).sum(dim=2)  # [N, K]
        w = F.softmax(-dist_sq, dim=1)
        return w

    def coral_loss(self, x, y):
        mean_loss = F.mse_loss(x.mean(dim=0), y.mean(dim=0))
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        cov_x = x_centered.t().matmul(x_centered) / max(x.size(0) - 1, 1)
        cov_y = y_centered.t().matmul(y_centered) / max(y.size(0) - 1, 1)
        cov_loss = F.mse_loss(cov_x, cov_y)
        return mean_loss + cov_loss

    def pairwise_cost(self, mu1, var1, mu2, var2, eps=1e-9):
        """
        Calculate the cost between two sets of prototypes: 
        the Euclidean distance between their means + the covariance matching cost
        """
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # [K, K, d_h]
        dist_sq = torch.mean(diff ** 2, dim=2)  # [K, K]
        cov_term = torch.mean(var1.unsqueeze(1) + var2.unsqueeze(0)
            - 2 * torch.sqrt(var1.unsqueeze(1) * var2.unsqueeze(0) + eps),dim=2)
        return dist_sq + cov_term

    def sinkhorn_2d(self, C, nu_1, nu_2, reg, num_iters=50, eps=1e-9):
        reg = max(float(reg), eps)
        log_K = -C / reg
        log_nu_1 = torch.log(nu_1.clamp_min(eps))
        log_nu_2 = torch.log(nu_2.clamp_min(eps))
        log_u = torch.zeros_like(nu_1)
        log_v = torch.zeros_like(nu_2)
        for _ in range(num_iters):
            log_u = log_nu_1 - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
            log_v = log_nu_2 - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
        log_T = log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0)
        T = torch.exp(log_T)
        ot_loss = torch.sum(T * C)
        return T, ot_loss

    def compute_homo_loss(self, c_omic1, c_omic2):
        """
        Heterogeneous Alignment Loss: Online GMM -> 
        Marginal Distributions -> Joint Cost -> Sinkhorn OT -> Local Alignment
        """
        g_omic1 = self._fit_gmm_em(c_omic1)
        g_omic2 = self._fit_gmm_em(c_omic2)
        eps = 1e-9
        nu_omic1 = g_omic1["posterior"].mean(dim=0)
        nu_omic1 = nu_omic1 / (nu_omic1.sum() + eps)
        nu_omic2 = g_omic2["posterior"].mean(dim=0)
        nu_omic2 = nu_omic2 / (nu_omic2.sum() + eps)
        C_raw = self.pairwise_cost(g_omic1["mu"], g_omic1["var"], g_omic2["mu"], g_omic2["var"], eps=eps)
        C = C_raw / (C_raw.detach().median() + 1e-8)
        C = torch.clamp(C, max=20.0)
        T, ot_loss = self.sinkhorn_2d(C, nu_omic1, nu_omic2, reg=self.ot_reg, num_iters=self.ot_num_iters)

        coral = self.coral_loss(c_omic1, c_omic2)
        homo_loss = ot_loss+ self.lambda_coral * coral
                
        # prototype_info = {
        #     "pi_1": g_omic1["pi"],
        #     "pi_2": g_omic2["pi"],
        #     "mu_1": g_omic1["mu"],
        #     "mu_2": g_omic2["mu"],
        #     "var_1": g_omic1["var"],
        #     "var_2": g_omic2["var"],
        #     "posterior_1": g_omic1["posterior"],
        #     "posterior_2": g_omic2["posterior"],
        #     "nu_1": nu_omic1,
        #     "nu_2": nu_omic2,
        #     "cost_raw": C_raw,
        #     "cost": C,
        #     "transport": T,
        #     "ot_loss": ot_loss,
        #     "coral_loss": coral,
        # }

        return homo_loss
        
    def forward(self, x_omic1, x_omic2, edge_index_rna, edge_index_adt):
        proj_x_1 = F.relu(self.proj_omic1(x_omic1))
        proj_x_2 = F.relu(self.proj_omic2(x_omic2))

        s_1 = self.encoder_uni_omic1(proj_x_1, edge_index_rna)
        s_2 = self.encoder_uni_omic2(proj_x_2, edge_index_adt)
        c_1 = self.encoder_com(proj_x_1, edge_index_rna)
        c_2 = self.encoder_com(proj_x_2, edge_index_adt)

        dec_loss = self.compute_decoupling_loss(s_1, c_1) + self.compute_decoupling_loss(s_2, c_2)

        homo_loss = self.compute_homo_loss(c_1, c_2)
        
        c_avg = (c_1 + c_2) / 2.0
        
        zeros_c1 = torch.zeros_like(c_1)
        zeros_c2 = torch.zeros_like(c_2)
        zeros_s1 = torch.zeros_like(s_1)
        zeros_s2 = torch.zeros_like(s_2)

        # shared + private full reconstruction
        recon_1 = self.decoder_omic1(torch.cat([c_1, s_1], dim=1),edge_index_rna)
        recon_2 = self.decoder_omic2(torch.cat([c_2, s_2], dim=1),edge_index_adt)
        # shared-only reconstruction
        recon_1_shared = self.decoder_omic1(torch.cat([c_1, zeros_s1], dim=1),edge_index_rna)
        recon_2_shared = self.decoder_omic2(torch.cat([c_2, zeros_s2], dim=1),edge_index_adt)

        # private-only reconstruction
        recon_1_private = self.decoder_omic1(torch.cat([zeros_c1, s_1], dim=1),edge_index_rna)
        recon_2_private = self.decoder_omic2(torch.cat([zeros_c2, s_2], dim=1),edge_index_adt)
        
        final_rep = torch.cat([c_avg, s_1, s_2], dim=1)
     
        return {
            'dec_loss': dec_loss,
            'homo_loss': homo_loss,
            'recon_1': recon_1,
            'recon_2': recon_2,
            'recon_1_shared': recon_1_shared,
            'recon_2_shared': recon_2_shared,
            'recon_1_private': recon_1_private,
            'recon_2_private': recon_2_private,
            'final_rep': final_rep,
            'emb_homo': c_avg,
            'homo_1': c_1,
            'homo_2': c_2,
            'emb_omic1': s_1,
            'emb_omic2': s_2,
        }

class DecAlignSpatial_unpaired(nn.Module):
    def __init__(self, args):
        super(DecAlignSpatial_unpaired, self).__init__()
        
        self.orig_d_omic1 = args.input_dim_omic1
        self.orig_d_omic2 = args.input_dim_omic2
        self.d_h = args.hidden_dim
        self.d_mid = self.orig_d_omic1
        self.num_heads = args.num_heads
        self.pos_1 = args.omic1_pos
        self.pos_2 = args.omic2_pos

        self.recon_dim_omic1 = args.recon_dim_omic1
        self.recon_dim_omic2 = args.recon_dim_omic2
        
        self.concat_dim = 5 * self.d_h
        self.output_dim = args.output_dim
        
        self.proj_omic1 = nn.Linear(self.orig_d_omic1, self.d_mid)
        self.proj_omic2 = nn.Linear(self.orig_d_omic2, self.d_mid)

        self.encoder_uni_omic1 = GCNEncoder(self.d_mid, self.d_mid, self.d_h)
        self.encoder_uni_omic2 = GCNEncoder(self.d_mid, self.d_mid, self.d_h)
        self.encoder_com = GCNEncoder(self.d_mid, self.d_mid, self.d_h) 
        self.decoder_omic1 = GCNDecoder(self.d_h*2, self.d_h*2, self.recon_dim_omic1)
        self.decoder_omic2 = GCNDecoder(self.d_h*2, self.d_h*2, self.recon_dim_omic2)

        self.num_prototypes = args.num_prototypes
        self.gmm_em_iters = getattr(args, 'gmm_em_iters', 5)
        self.gmm_var_floor = getattr(args, 'gmm_var_floor', 1e-4)
        self.ot_reg = getattr(args, 'lambda_ot', 0.1)
        self.ot_num_iters = getattr(args, 'ot_num_iters', 50)
        self.lambda_coral = getattr(args, "lambda_coral", 1)
        
        #unbalanced OT
        self.ot_mode = getattr(args, 'ot_mode', 'unbalanced')
        self.uot_tau = getattr(args, 'uot_tau', 1.0)
        self.uot_num_iters = getattr(args, 'uot_num_iters', 300)

    def compute_decoupling_loss(self, s, c):
        s_norm = (s - s.mean(dim=0, keepdim=True)) / (s.std(dim=0, keepdim=True) + 1e-6)
        c_norm = (c - c.mean(dim=0, keepdim=True)) / (c.std(dim=0, keepdim=True) + 1e-6)
        cross_corr = c_norm.T @ s_norm / max(c.size(0) - 1, 1)
        return (cross_corr ** 2).mean()
    
        
    def torch_standard_scale(self, x, eps=1e-6):
        return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + eps)

    def compute_variance_loss(self, z, eps=1e-4):
        std = torch.sqrt(z.var(dim=0) + eps)
        return torch.mean(F.relu(1.0 - std))

    def _init_gmm(self, x):
        """Initialize a GMM by sampling equidistantly from the current features."""
        batch_size, dim = x.shape
        k = self.num_prototypes
        if batch_size >= k:
            indices = torch.linspace(0, batch_size - 1, steps=k, device=x.device).round().long()
            mu = x.index_select(0, indices)
        else:
            repeat = math.ceil(k / batch_size)
            mu = x.repeat(repeat, 1)[:k]
        
        var = x.var(dim=0, unbiased=False).clamp_min(self.gmm_var_floor)
        var = var.unsqueeze(0).expand(k, dim).clone()
        pi = torch.full((k,), 1.0 / k, device=x.device, dtype=x.dtype)
        return pi, mu, var
    
    def _gmm_log_prob(self, x, pi, mu, var):
        diff = x.unsqueeze(1) - mu.unsqueeze(0)                       # [N, K, D]
        mahalanobis = (diff.pow(2) / var.unsqueeze(0)).sum(dim=2)     # [N, K]
        log_det = var.log().sum(dim=1).unsqueeze(0)                   # [1, K]
        log_norm = x.size(1) * math.log(2.0 * math.pi)
        return pi.clamp_min(1e-8).log().unsqueeze(0) - 0.5 * (mahalanobis + log_det + log_norm)

    def _fit_gmm_em(self, x):
        pi, mu, var = self._init_gmm(x)
        for _ in range(self.gmm_em_iters):
            # E-step
            gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
            # M-step
            nk = gamma.sum(dim=0).clamp_min(1e-6)
            pi = nk / x.size(0)
            mu = gamma.transpose(0, 1).matmul(x) / nk.unsqueeze(1)
            second_moment = gamma.transpose(0, 1).matmul(x.pow(2)) / nk.unsqueeze(1)
            var = (second_moment - mu.pow(2)).clamp_min(self.gmm_var_floor)
        gamma = F.softmax(self._gmm_log_prob(x, pi, mu, var), dim=1)
        return {"pi": pi, "mu": mu, "var": var, "posterior": gamma}


    def compute_prototypes(self, features, proto, logvar):
        """Calculate soft assignment weights based on the distance between sample features and prototype means."""
        diff = features.unsqueeze(1) - proto.unsqueeze(0)  # [N, K, d_h]
        dist_sq = (diff ** 2).sum(dim=2)  # [N, K]
        w = F.softmax(-dist_sq, dim=1)
        return w

    def coral_loss(self, x, y):
        mean_loss = F.mse_loss(x.mean(dim=0), y.mean(dim=0))
        x_centered = x - x.mean(dim=0, keepdim=True)
        y_centered = y - y.mean(dim=0, keepdim=True)
        cov_x = x_centered.t().matmul(x_centered) / max(x.size(0) - 1, 1)
        cov_y = y_centered.t().matmul(y_centered) / max(y.size(0) - 1, 1)
        cov_loss = F.mse_loss(cov_x, cov_y)
        return mean_loss + cov_loss

    def pairwise_cost(self, mu1, var1, mu2, var2, eps=1e-9):
        """
        Calculate the cost between two sets of prototypes: 
        the Euclidean distance between their means + the covariance matching cost
        """
        diff = mu1.unsqueeze(1) - mu2.unsqueeze(0)  # [K, K, d_h]
        dist_sq = torch.mean(diff ** 2, dim=2)  # [K, K]
        cov_term = torch.mean(var1.unsqueeze(1) + var2.unsqueeze(0)
            - 2 * torch.sqrt(var1.unsqueeze(1) * var2.unsqueeze(0) + eps),dim=2)
        return dist_sq + cov_term

    def sinkhorn_2d(self, C, nu_1, nu_2, reg, num_iters=50, eps=1e-9):
        """Balanced legacy solver or fully differentiable KL-reference UOT.

        UOT objective:
          <T,C> + reg*KL(T | a b^T)
                + tau*KL(T 1 | a) + tau*KL(T^T 1 | b).
        Generalized KL includes -x+y. No detaching of EM, C, marginals or T.
        """
        if self.ot_mode == 'balanced':
            # Exactly the original balanced path, including its linear-only loss.
            reg = max(float(reg), eps)
            log_K = -C / reg
            log_nu_1 = torch.log(nu_1.clamp_min(eps))
            log_nu_2 = torch.log(nu_2.clamp_min(eps))
            log_u = torch.zeros_like(nu_1)
            log_v = torch.zeros_like(nu_2)
            for _ in range(num_iters):
                log_u = log_nu_1 - torch.logsumexp(log_K + log_v.unsqueeze(0), dim=1)
                log_v = log_nu_2 - torch.logsumexp(log_K + log_u.unsqueeze(1), dim=0)
            T = torch.exp(log_K + log_u.unsqueeze(1) + log_v.unsqueeze(0))
            return T, torch.sum(T * C)

        if self.ot_mode != 'unbalanced':
            raise ValueError("ot_mode must be 'balanced' or 'unbalanced'")
        reg, tau = float(reg), float(self.uot_tau)
        if reg <= 0 or tau <= 0 or self.uot_num_iters < 1:
            raise ValueError('reg, uot_tau and uot_num_iters must be positive')

        # Casting preserves autograd; the 14x14 solver runs in float64 for stability.
        dtype = C.dtype
        cost = C.to(torch.float64)
        a = nu_1.to(torch.float64).clamp_min(1e-12)
        b = nu_2.to(torch.float64).clamp_min(1e-12)
        log_a, log_b = a.log(), b.log()
        log_K = log_a[:, None] + log_b[None, :] - cost / reg
        rho = tau / (tau + reg)
        log_u, log_v = torch.zeros_like(a), torch.zeros_like(b)
        # Fixed iterations preserve the original unrolled-backprop strategy.
        for _ in range(self.uot_num_iters):
            old_u, old_v = log_u, log_v
            log_u = rho * (log_a - torch.logsumexp(log_K + log_v[None, :], dim=1))
            log_v = rho * (log_b - torch.logsumexp(log_K + log_u[:, None], dim=0))
        log_T = log_K + log_u[:, None] + log_v[None, :]
        T = log_T.exp()

        tiny = torch.finfo(T.dtype).tiny
        def kl(x, reference):
            return (x * (x.clamp_min(tiny).log() - reference.clamp_min(tiny).log())
                    - x + reference).sum()

        linear = (T * cost).sum()
        entropy_kl = kl(T, a[:, None] * b[None, :])
        row_kl = kl(T.sum(1), a)
        col_kl = kl(T.sum(0), b)
        # Full UOT objective, not only <T,C>; do NOT normalize T back to mass=1.
        loss = linear + reg * entropy_kl + tau * (row_kl + col_kl)
        if not bool(torch.isfinite(loss)) or not bool(torch.isfinite(T).all()):
            raise FloatingPointError('Non-finite UOT plan/loss')

        # Only logging is detached. It does not change the backward path.
        with torch.no_grad():
            delta = torch.maximum((log_u - old_u).abs().max(),(log_v - old_v).abs().max())
            mass = T.sum()
            self.ot_diagnostics = {
                'mode': 'unbalanced', 'tau': tau,
                'uot_objective': float(loss), 'linear_cost': float(linear),
                'transport_mass': float(mass),
                'cost_per_mass': float(linear / mass.clamp_min(1e-12)),
                'row_mass_difference': float((T.sum(1) - a).abs().max()),
                'col_mass_difference': float((T.sum(0) - b).abs().max()),
                'log_scaling_delta': float(delta),
                'solver_converged': bool(delta < 1e-6),
                'solver_iterations': self.uot_num_iters,
            }
            self.last_uot_transport = T.detach()
            if not self.ot_diagnostics['solver_converged'] and not getattr(self, '_uot_warned', False):
                import warnings
                warnings.warn('UOT scaling has not converged; inspect log_scaling_delta or increase uot_num_iters')
                self._uot_warned = True
        return T.to(dtype), loss.to(dtype)

    def compute_homo_loss(self, c_omic1, c_omic2):
        """
        Heterogeneous Alignment Loss: Online GMM -> 
        Marginal Distributions -> Joint Cost -> Sinkhorn OT -> Local Alignment
        """
        g_omic1 = self._fit_gmm_em(c_omic1)
        g_omic2 = self._fit_gmm_em(c_omic2)
        eps = 1e-9
        nu_omic1 = g_omic1["posterior"].mean(dim=0)
        nu_omic1 = nu_omic1 / (nu_omic1.sum() + eps)
        nu_omic2 = g_omic2["posterior"].mean(dim=0)
        nu_omic2 = nu_omic2 / (nu_omic2.sum() + eps)
        C_raw = self.pairwise_cost(g_omic1["mu"], g_omic1["var"], g_omic2["mu"], g_omic2["var"], eps=eps)
        C = C_raw / (C_raw.detach().median() + 1e-8)
        C = torch.clamp(C, max=20.0)
        _, ot_loss = self.sinkhorn_2d(C, nu_omic1, nu_omic2, reg=self.ot_reg, num_iters=self.ot_num_iters)
        coral = self.coral_loss(c_omic1, c_omic2)
        homo_loss = ot_loss + self.lambda_coral * coral

        return homo_loss

    def forward(self, x_omic1, x_omic2, edge_index_rna,edge_index_adt):
        proj_x_1 = F.relu(self.proj_omic1(x_omic1))
        proj_x_2 = F.relu(self.proj_omic2(x_omic2))

        s_1 = self.encoder_uni_omic1(proj_x_1, edge_index_rna)
        s_2 = self.encoder_uni_omic2(proj_x_2, edge_index_adt)
        c_1 = self.encoder_com(proj_x_1, edge_index_rna)
        c_2 = self.encoder_com(proj_x_2, edge_index_adt)

        dec_loss = self.compute_decoupling_loss(s_1, c_1) + self.compute_decoupling_loss(s_2, c_2)
        homo_loss = self.compute_homo_loss(c_1, c_2)

        zeros_c1 = torch.zeros_like(c_1)
        zeros_c2 = torch.zeros_like(c_2)
        zeros_s1 = torch.zeros_like(s_1)
        zeros_s2 = torch.zeros_like(s_2)

        # shared + private full reconstruction,用s+c作为decoder输入
        recon_1 = self.decoder_omic1(torch.cat([c_1, s_1], dim=1),edge_index_rna)
        recon_2 = self.decoder_omic2(torch.cat([c_2, s_2], dim=1),edge_index_adt)
        # shared-only reconstruction
        recon_1_shared = self.decoder_omic1(torch.cat([c_1, zeros_s1], dim=1),edge_index_rna)
        recon_2_shared = self.decoder_omic2(torch.cat([c_2, zeros_s2], dim=1),edge_index_adt)

        # private-only reconstruction
        recon_1_private = self.decoder_omic1(torch.cat([zeros_c1, s_1], dim=1),edge_index_rna)
        recon_2_private = self.decoder_omic2(torch.cat([zeros_c2, s_2], dim=1),edge_index_adt)

        final_rep_1 = torch.cat([c_1, s_1], dim=1) # [N, 3*d_h]
        final_rep_2 = torch.cat([c_2, s_2], dim=1) # [N, 3*d_h]

        return {
            'dec_loss': dec_loss,
            'homo_loss':homo_loss,
            'recon_1':recon_1,
            'recon_2':recon_2,
            'recon_1_shared': recon_1_shared,
            'recon_2_shared': recon_2_shared,
            'recon_1_private': recon_1_private,
            'recon_2_private': recon_2_private,
            'final_rep_1': final_rep_1,
            'final_rep_2': final_rep_2,
            'homo_1':c_1,
            'homo_2':c_2,
            'emb_omic1': s_1,
            'emb_omic2': s_2
        }