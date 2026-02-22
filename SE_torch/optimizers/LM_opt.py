import torch
import math as mt
from tqdm import tqdm

from SE_torch.optimizers.base_optimizer import SEOptimizer
from SE_torch.optimizers.se_loss import SELoss


class LMOpt(SEOptimizer):
    def __init__(self, **kwargs):
        super(LMOpt, self).__init__(**kwargs)
        self.xtol = kwargs.get('xtol', 1e-7)
        self.ftol = kwargs.get('ftol', 1e-7)
        self.max_iter = int(kwargs.get('max_iter', 1000))
        self.verbose = kwargs.get('verbose', True)
        self.loss_func = kwargs.get('loss_func', SELoss(**kwargs))
        self.latent = kwargs.get('latent', None)
        self.cart = kwargs.get('cart', False)
        self.prefix = f'-{self.loss_func.__class__.__name__}'

        self.sigma = kwargs.get('sigma', 0.1)
        self.delta_init = kwargs.get('delta_init', None)
        self.rho_tol_min = kwargs.get('rho_tol_min', 0.25)
        self.rho_tol_max = kwargs.get('rho_tol_max', 0.75)
        self.sb_inc = kwargs.get('sb_inc', 2.0)

    def process_x(self, x, decode=False):
        x = x.clone().detach()
        # x = self.loss_func.encode(x)
        if self.latent is not None or self.cart:
            if decode:
                x = self.loss_func.decode(x)
            else:
                x = self.loss_func.encode(x)
                if self.latent != 'ss':
                    x = torch.zeros_like(x)
        return x

    def compute_D(self, J, D_prev=None):
        d = torch.norm(J, dim=0)
        if D_prev is not None:
            d_prev = torch.diag(D_prev)
            d = torch.maximum(d_prev, d)
        return torch.diag(d)

    def update_step_bound(self, rho, f, fp, Jp, Dp, lam, delta_prev=None):
        if delta_prev is None:
            delta_prev = torch.sqrt(Dp)

        if ((self.rho_tol_min < rho) and (rho < self.rho_tol_max) and (lam == 0)) or (rho >= self.rho_tol_max):
            return self.sb_inc * torch.sqrt(Dp)

        gamma = -(Jp + (lam * Dp))
        mu = (0.5 * gamma / (gamma + (0.5 * (f - fp)))).item()
        mu = max(0.1, min(0.5, mu))
        return mu * delta_prev

    def update_step(self, res, J, grad, D, delta=None):
        p, R, Q = self.compute_step(0.0, res, J, grad, D)
        if (delta is None) or (torch.norm(D @ p) <= (1.0 + self.sigma) * delta):
            return p, 0.0
        lam = self.update_damping_factor(res, J, grad, D, delta)
        p, _, _ = self.compute_step(lam, res, J, grad, D)
        return p, lam

    @staticmethod
    def compute_step(lam, res, J, grad, D, R=None, Q=None):
        if R is None and Q is None:
            Q, R = torch.linalg.qr(J, mode='reduced')

        Q_res = torch.matmul(Q.T, res.reshape(-1))
        b = torch.matmul(R.T, Q_res) + grad

        if lam > 0:
            D_lam = mt.sqrt(lam) * D
            R_D_lam = torch.concat([R, D_lam], dim=0)  # (n+k, n)
            _, R_lam = torch.linalg.qr(R_D_lam, mode='reduced')  # (n,n) upper

            u = torch.linalg.solve_triangular(R_lam.T, b.unsqueeze(1), upper=False).squeeze(1)
            p = torch.linalg.solve_triangular(R_lam, -u.unsqueeze(1), upper=True).squeeze(1)
            return p, R_lam, u

        u = torch.linalg.solve_triangular(R.T, b.unsqueeze(1), upper=False).squeeze(1)
        p = torch.linalg.solve_triangular(R, -u.unsqueeze(1), upper=True).squeeze(1)
        return p, R, Q

    def _phi_alpha(self, res, J, grad, D, delta, alpha, R=None, Q=None):
        p_alpha, R_alpha, WQ_alpha = self.compute_step(alpha, res, J, grad, D, R, Q)
        Dp_alpha = D @ p_alpha
        return torch.norm(Dp_alpha) - delta, Dp_alpha, R_alpha, WQ_alpha

    @staticmethod
    def _grad_phi_alpha(D, Dp_alpha, R_alpha):
        norm_Dp = torch.norm(Dp_alpha).clamp_min(1e-16)
        DT_Dp_alpha = D.T @ Dp_alpha
        partial_grad = torch.linalg.solve_triangular(
            R_alpha.T, (DT_Dp_alpha / norm_Dp).unsqueeze(1), upper=False
        ).squeeze(1)
        return -norm_Dp * torch.norm(partial_grad).pow(2)

    def update_damping_factor(self, res, J, grad, D, delta, tol=1e-10, max_iter=50):
        alpha = 0.0
        phi, Dp, R, Q = self._phi_alpha(res, J, grad, D, delta, alpha)

        if phi < tol:
            return alpha

        b = (J.T @ res) + grad

        Q_D_T, R_D_T = torch.linalg.qr(D.T, mode='reduced')
        Q_D_b = Q_D_T.T @ b
        upper_vec = torch.linalg.solve_triangular(R_D_T, Q_D_b.unsqueeze(1), upper=True).squeeze(1)
        upper = (torch.norm(upper_vec) / max(delta, 1e-16)).item()


        try:
            if torch.linalg.matrix_rank(J) == min(J.shape):
                grad_phi = self._grad_phi_alpha(D, Dp, R).item()
                lower = (-phi.item() / grad_phi) if grad_phi < 0 else 0.0
            else:
                lower = 0.0
        except torch.linalg.LinAlgError:
            lower = 0.0

        for _ in range(max_iter):
            if alpha <= lower or alpha >= upper:
                alpha = max(1e-3 * upper, mt.sqrt(max(lower, 0.0) * max(upper, 1e-16)))

            phi, Dp, R_alpha, _ = self._phi_alpha(res, J, grad, D, delta, alpha, R, Q)

            grad_phi_t = self._grad_phi_alpha(D, Dp, R_alpha)
            grad_phi = grad_phi_t.item()
            if not mt.isfinite(grad_phi) or grad_phi == 0.0:
                grad_phi = -1e-16

            phi_val = phi.item()

            upper = alpha if phi_val < 0 else upper
            lower = max(lower, alpha - (phi_val / grad_phi))

            alpha = (alpha - (((phi_val + delta) / max(delta, 1e-16)) * (phi_val / grad_phi))).item()

            if abs(phi_val) < self.sigma * delta:
                break

        return alpha

    def compute_rho(self, x, D, step_size, alpha):
        f = self.loss_func.compute_f(x)
        fp = self.loss_func.compute_f(x, step_size)
        Jp = torch.norm(self.loss_func.compute_J(x) @ step_size).pow(2)
        Dp = torch.norm(D @ step_size).pow(2)
        if fp > f:
            return 0.0, f, fp, Jp, Dp
        num = (f - fp) / f.abs()
        denom = (.5 * Jp + (alpha * Dp)) / f.abs()
        return (num / denom).item(), f, fp, Jp, Dp

    def _compute_ftol(self, f, Jp, Dp, alpha):
        denom = (.5 * Jp + (alpha * Dp)) / f.abs()
        return denom.item()

    def _compute_xtol(self, x, D, delta):
        if delta is None:
            return 1
        x = self.loss_func.reshape_x(x)
        return delta / torch.norm(D @ x).item()

    def __call__(self, x0, z, v, slk_bus, h_ac, nb, norm_H=None):
        self.loss_func.update_params(z, v, slk_bus, h_ac, nb, norm_H)

        all_x = [x0.clone().detach()]
        x = self.process_x(x0)

        J = self.loss_func.compute_J(x)
        f = self.loss_func.compute_f(x)
        res = self.loss_func.compute_residuals(x)
        grad = self.loss_func.compute_grad(x)
        D = self.compute_D(J)

        delta = self.delta_init
        converged = False
        it = 0
        pbar = tqdm(range(self.max_iter), desc=f"Optimizing with LM{self.prefix}", disable=not self.verbose, leave=True, colour='green',
                    postfix={'loss': f"{f.item():.4f}"})
        for it in pbar:
            step, lam = self.update_step(res, J, grad, D, delta)
            rho, f, fp, Jp, Dp = self.compute_rho(x, D, step, lam)

            if rho > 1e-4:
                x = self.loss_func.update_x(x, step)
                J = self.loss_func.compute_J(x)
                res = self.loss_func.compute_residuals(x)
                grad = self.loss_func.compute_grad(x)
                D = self.compute_D(J, D_prev=D)

                all_x.append(self.process_x(x, decode=True))

            ftol = self._compute_ftol(f, Jp, Dp, lam)
            xtol = self._compute_xtol(x, D, delta)
            if rho >= 0 and ftol <= self.ftol and xtol <= self.xtol:
                converged = True
                break
            delta = self.update_step_bound(rho, f, fp, Jp, Dp, lam, delta_prev=delta)

            pbar.set_postfix(ftol=f"{ftol:.4e}", xtol=f"{xtol:.4e}", loss=f"{f.item():.4f}")

        x = self.process_x(x, decode=True)
        T, V = x[:nb], x[nb:]
        return x, T, V, converged, it, f.item(), all_x