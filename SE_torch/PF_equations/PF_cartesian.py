# h_ac_torch.py
import torch
from SE_torch.net_preprocess.compose_measurement import Pi, Qi, Pf, Qf, Vm, Cm

def complex_to_real_block(H: torch.Tensor) -> torch.Tensor:
    if not torch.is_complex(H):
        raise TypeError("H must be a complex tensor")

    Hr = H.real
    Hi = H.imag
    top = torch.cat([Hr, -Hi], dim=-1)
    bot = torch.cat([Hi,  Hr], dim=-1)
    return torch.cat([top, bot], dim=-2)


def Hinj(p, q, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.complex128)

    # ----- P injections -----
    IPi = I[p.idx]
    outer_eii_Pi = torch.einsum('ik,il->ikl', IPi, IPi)  # (Lp, nb, nb)
    YPi = torch.matmul(outer_eii_Pi, p.Ybu)   # (Lp, nb, nb)
    HPi = 0.5 * (YPi + torch.conj(YPi.transpose(1, 2)))

    # ----- Q injections -----
    IQi = I[q.idx]
    outer_eii_Qi = torch.einsum('ik,il->ikl', IQi, IQi)
    YQi = torch.matmul(outer_eii_Qi, q.Ybu)
    HQi = 0.5j * (YQi - torch.conj(YQi.transpose(1, 2)))

    return torch.cat([HPi, HQi], dim=0)  # (Lp+Lq, nb, nb)


def Hflow(pf, qf, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.complex128)

    # ----- P flows -----
    IPi = I[pf.i]
    IPj = I[pf.j]
    eii_P = torch.einsum('ik,il->ikl', IPi, IPi)  # (Lf, nb, nb)
    eij_P = torch.einsum('ik,il->ikl', IPi, IPj)

    ysi_p = pf.ysi.unsqueeze(1).unsqueeze(2)
    yij_p = pf.yij.unsqueeze(1).unsqueeze(2)
    YPij = eii_P * ysi_p + eij_P * yij_p
    HPij = 0.5 * (YPij + torch.conj(YPij.transpose(1, 2)))

    # ----- Q flows -----
    IQi = I[qf.i]
    IQj = I[qf.j]
    eii_Q = torch.einsum('ik,il->ikl', IQi, IQi)
    eij_Q = torch.einsum('ik,il->ikl', IQi, IQj)

    ysi_q = qf.ysi.unsqueeze(1).unsqueeze(2)
    yij_q = qf.yij.unsqueeze(1).unsqueeze(2)
    YQij = eii_Q * ysi_q + eij_Q * yij_q
    HQij = 0.5j * (YQij - torch.conj(YQij.transpose(1, 2)))

    return torch.cat([HPij, HQij], dim=0)


def Hcm(cm, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.complex128)

    Ii = I[cm.i]
    Ij = I[cm.j]

    eii = torch.einsum('ik,il->ikl', Ii, Ii)  # (L, nb, nb)
    eij = torch.einsum('ik,il->ikl', Ii, Ij)
    eji = torch.einsum('ik,il->ikl', Ij, Ii)
    ejj = torch.einsum('ik,il->ikl', Ij, Ij)

    ysi = cm.ysi.unsqueeze(1).unsqueeze(2)
    yij = cm.yij.unsqueeze(1).unsqueeze(2)

    Hij = (
        ysi * torch.conj(ysi) * eii
        + torch.conj(ysi) * yij * eij
        + ysi * torch.conj(yij) * eji
        + yij * torch.conj(yij) * ejj
    )
    return Hij  # (L, nb, nb)


def Hvm(vm, nb, device=None):
    I = torch.eye(nb, device=device, dtype=torch.complex128)
    IVmi = I[vm.i]
    eii_V = torch.einsum('ik,il->ikl', IVmi, IVmi)  # (Lv, nb, nb)
    return eii_V


class H_AC:
    def __init__(self, sys, branch, meas_idx, device='cpu', dtype=torch.float32):
        # default masks
        Pi_idx = Qi_idx = Vm_idx = torch.zeros(sys.nb, dtype=torch.bool)
        Pf_idx = Qf_idx = Cm_idx = torch.zeros_like(torch.as_tensor(branch.i, dtype=torch.long), dtype=torch.bool)

        # accept external masks (numpy/torch/bool array)
        if meas_idx.get('Pi_idx') is not None: Pi_idx = torch.as_tensor(meas_idx['Pi_idx'], dtype=torch.bool)
        if meas_idx.get('Qi_idx') is not None: Qi_idx = torch.as_tensor(meas_idx['Qi_idx'], dtype=torch.bool)
        if meas_idx.get('Pf_idx') is not None: Pf_idx = torch.as_tensor(meas_idx['Pf_idx'], dtype=torch.bool)
        if meas_idx.get('Qf_idx') is not None: Qf_idx = torch.as_tensor(meas_idx['Qf_idx'], dtype=torch.bool)
        if meas_idx.get('Cm_idx') is not None: Cm_idx = torch.as_tensor(meas_idx['Cm_idx'], dtype=torch.bool)
        if meas_idx.get('Vm_idx') is not None: Vm_idx = torch.as_tensor(meas_idx['Vm_idx'], dtype=torch.bool)

        # build measurement objects (they may hold numpy; we'll convert inside H* funcs)
        self.pi = Pi(Pi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.qi = Qi(Qi_idx, sys.bus, sys.Ybus, sys.Yij, sys.Yii)
        self.pf = Pf(Pf_idx, branch)
        self.qf = Qf(Qf_idx, branch)
        self.cm = Cm(Cm_idx, sys.bus, branch)
        self.vm = Vm(Vm_idx, sys.bus)

        self.nb = sys.nb
        self.device = device
        self.dtype = dtype

        # assemble H (complex128)
        self.Hf = Hflow(self.pf, self.qf, self.nb, device=self.device)
        self.Hc = Hcm(self.cm, self.nb, device=self.device)
        self.Hi = Hinj(self.pi, self.qi, self.nb, device=self.device)
        self.Hv = Hvm(self.vm, self.nb, device=self.device)
        self.H = torch.cat([self.Hf, self.Hc, self.Hi, self.Hv], dim=0)
        self.H_complex = self.H.clone()
        self.H  = complex_to_real_block(self.H).to(dtype=torch.float64)

    def _update_device_dtype(self, device, dtype):
        self.device = device
        self.dtype = dtype
        self.H = self.H.to(device, dtype)

    def estimate(self, x):
        x = x.to(device=self.device, dtype=torch.float64)
        z_est = (torch.tensordot(self.H, x, dims=([2], [0])) * x).sum(dim=-1)
        return z_est.to(dtype=self.dtype, device=self.device)

    def jacobian(self, x):
        x = x.to(device=self.device, dtype=torch.float64)
        J = 2 * torch.tensordot(self.H, x, dims=([2], [0]))

        return J.to(dtype=self.dtype, device=self.device)