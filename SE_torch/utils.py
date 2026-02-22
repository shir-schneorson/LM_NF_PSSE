import torch
import torch.nn.functional as F

def normalize_measurements(H: torch.Tensor, z: torch.Tensor):
    norms = torch.linalg.norm(H, dim=(1, 2)) + 1e-12

    H_normalized = H / norms.view(-1, 1, 1)
    z_normalized = z / norms

    return z_normalized, H_normalized, norms


def square_mag(z: torch.Tensor, var: torch.Tensor, agg_meas_idx: dict[str, torch.Tensor]):
    device = z.device
    dtype = z.dtype

    z_square = z.clone()
    var_square = var.clone()

    mag_idx = torch.cat([
        agg_meas_idx.get('Cm', torch.tensor([], device=device, dtype=torch.long)),
        agg_meas_idx.get('Vm', torch.tensor([], device=device, dtype=torch.long))
    ])

    if mag_idx.numel() > 0:
        z_square[mag_idx] = z[mag_idx] ** 2
        var_square[mag_idx] = 2.0 * (var[mag_idx] ** 2)

    return z_square.to(dtype=dtype), var_square.to(dtype=dtype)


def RMSE(T_true: torch.Tensor, V_true: torch.Tensor, T_est: torch.Tensor, V_est: torch.Tensor):
    u_true = V_true.to(dtype=torch.get_default_dtype()) * torch.exp(1j * T_true.to(torch.get_default_dtype()))
    u_est  = V_est.to(dtype=torch.get_default_dtype())  * torch.exp(1j * T_est.to(torch.get_default_dtype()))

    num = torch.linalg.norm(u_est - u_true)
    den = torch.linalg.norm(u_true) + 1e-12

    err = (num / den).real
    return err

def RMSE_polar(T_true: torch.Tensor, V_true: torch.Tensor, T_est: torch.Tensor, V_est: torch.Tensor):
    err_T = T_true - T_est
    err_V = V_true - V_est
    err = torch.linalg.norm(torch.concat([err_T, err_V]))
    return err


def init_start_point(sys, data=None, how='flat',
                     flat_init=(0, 1), random_init=(0.3, 1, 1e-2)):
    if how == 'flat':
        T = torch.deg2rad(torch.full((sys.nb,), flat_init[0], dtype=torch.get_default_dtype()))
        T[sys.slk_bus[0]] = sys.slk_bus[1]
        V = torch.full((sys.nb,), flat_init[1], dtype=torch.get_default_dtype())

    elif how == 'exact':
        pmu = data['data'].get('pmu')
        if pmu is None:
            T = sys.bus['To']
            V = sys.bus['Vo']
        else:
            T = torch.tensor(pmu['voltage'][:, -1])
            V = torch.tensor(pmu['voltage'][:, -2])

    elif how == 'warm':
        # T = sys.bus['To']
        # V = sys.bus['Vo']
        m_NF = torch.load("../learn_prior/datasets/mean_NF_polar.pt").to(torch.get_default_dtype())
        T = m_NF[:sys.nb]
        V = m_NF[sys.nb:]

    elif how == 'random':
        theta = torch.pi * random_init[0]
        T = torch.empty(sys.nb).uniform_(-theta, theta)
        T[sys.slk_bus[0]] = 0

        mu, sigma = random_init[1], torch.sqrt(torch.tensor(random_init[2]))
        V = torch.normal(mu, sigma, size=(sys.nb,))
        V[sys.slk_bus[0]] = 1

    else:
        T = torch.rand(sys.nb) - 0.5
        T[sys.slk_bus[0]] = sys.slk_bus[1]

        V = (1.05 - 0.95) * torch.rand(sys.nb) + 0.95

    return T, V
