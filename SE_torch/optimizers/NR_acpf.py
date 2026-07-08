# torch_nr_pf.py
import time
import torch

# ---------- small helpers ----------

def _to_tensor(x, device=None, dtype=None):
    if isinstance(x, torch.Tensor):
        return x.to(device=device, dtype=dtype) if (device or dtype) else x
    return torch.as_tensor(x, device=device, dtype=dtype)

def _remove_rowcol(M: torch.Tensor, row=None, col=None):
    """Remove a single row and/or single column from a 2D tensor."""
    rmask = torch.ones(M.size(0), dtype=torch.bool, device=M.device)
    cmask = torch.ones(M.size(1), dtype=torch.bool, device=M.device)
    if row is not None:
        rmask[row] = False
    if col is not None:
        cmask[col] = False
    return M[rmask][:, cmask]

def _coo_mv(vals, rows, cols, shape, v):
    """
    Sparse COO matrix-vector multiplication: y = A v
    vals, rows, cols: 1D tensors
    shape: (m, n)
    v: (n,) dense
    returns y: (m,)
    """
    idx = torch.stack([rows.long(), cols.long()], dim=0)
    A = torch.sparse_coo_tensor(idx, vals, size=shape, device=v.device, dtype=v.dtype).coalesce()
    device = v.device
    y = torch.sparse.mm(A.to('cpu'), v.view(-1,1).to('cpu')).view(-1).to(device=device)
    return y

# ---------- constraints ----------

def qv_limits(user, sys, Ql, device=None, dtype=torch.float64):
    nb = sys.nb
    on  = torch.zeros((nb, 1), device=device, dtype=dtype)
    Qcon = torch.zeros((nb, 3), device=device, dtype=dtype)
    Vcon = torch.zeros((nb, 3), device=device, dtype=dtype)

    if 'reactive' in user['list']:
        pv_mask = (sys.bus.bus_type.values == 2) & (sys.bus.Qmin.values < sys.bus.Qmax.values)
        pv_mask = _to_tensor(pv_mask, device=device, dtype=torch.bool)
        on[pv_mask] = 1

        q_min = _to_tensor(sys.bus.Qmin.values, device=device, dtype=dtype).view(-1,1)
        q_max = _to_tensor(sys.bus.Qmax.values, device=device, dtype=dtype).view(-1,1)
        q     = _to_tensor(Ql, device=device, dtype=dtype).view(-1,1)
        Qcon = torch.hstack((q_min - q, q_max - q, on))

    elif 'voltage' in user['list']:
        pq_mask = (sys.bus.bus_type.values == 1) & (sys.bus.Vmin.values < sys.bus.Vmax.values)
        pq_mask = _to_tensor(pq_mask, device=device, dtype=torch.bool)
        on[pq_mask] = 1

        vmm = _to_tensor(sys.bus.loc[:, ['Vmin','Vmax']].values, device=device, dtype=dtype)
        Vcon = torch.hstack((vmm, on))

    return Qcon, Vcon

# ---------- indices & algebra precompute ----------

def idx_par1(sys, device=None, dtype=None):
    Yte =  sys.Yij.clone().to(device=device)
    Yij_mod = sys.Yij.clone().to(device=device)
    slk = int(sys.slk_bus[0])

    # zero slack-bus row
    Yij_mod[slk, :] = torch.zeros(1)

    alg, idx = {}, {}

    ii, jj = torch.nonzero(Yij_mod, as_tuple=True)
    alg['i'], alg['j'] = ii.long(), jj.long()

    GijBij = sys.Ybus[alg['i'], alg['j']]
    alg['Gij'] = torch.real(GijBij).to(device=device, dtype=dtype)
    alg['Bij'] = torch.imag(GijBij).to(device=device, dtype=dtype)

    Yte2 = Yte.clone()
    Yte2 = torch.concat((Yte2[:slk], Yte2[slk + 1:]), dim=0).to(device=device)
    # Yte2 = _remove_rowcol(Yte2, row=slk, col=None)
    fd1i, _ = torch.nonzero(Yte2, as_tuple=True)
    alg['fd1i'] = fd1i.long().to(device=device)

    alg['ii'] = _to_tensor(sys.bus.idx_bus.values, device=device, dtype=torch.long)
    # drop slack index in 'ii'
    keep = torch.arange(sys.nb, device=device) != slk
    alg['ii'] = alg['ii'][keep]

    idx['j11'] = {}
    idx['j11']['ij'] = (alg['j'] != slk)

    # Yte3 = _remove_rowcol(Yte2, row=None, col=slk)
    Yte3 = torch.concat((Yte2[:, :slk], Yte2[:, slk + 1:]), dim=1).to(device=device)
    q, w = torch.nonzero(Yte3, as_tuple=True)

    bu = torch.arange(sys.nb - 1, device=device, dtype=torch.long)
    idx['j11']['jci'] = torch.cat((bu, q.long()))
    idx['j11']['jcj'] = torch.cat((bu, w.long()))

    return alg, idx

def idx_par2(sys, alg, idx, device=None, dtype=torch.float64):
    bus_type = _to_tensor(sys.bus.bus_type.values, device=device, dtype=torch.long)
    pq = torch.nonzero(bus_type == 1, as_tuple=True)[0]  # PQ buses
    alg['pq'] = pq.long()
    alg['Npq'] = int(pq.numel())

    Yij = sys.Yij.clone().to(device=device)
    Ybus = sys.Ybus.clone().to(device=device)
    slk = int(sys.slk_bus[0])

    fdi, fdj = torch.nonzero(Yij[pq, :], as_tuple=True)
    alg['fdi'], alg['fdj'] = fdi.long(), fdj.long()
    alg['fij'] = torch.isin(alg['i'], pq)

    GiiBii = Ybus[pq, pq]
    alg['Gii'] = torch.real(GiiBii)  # diagonal block
    alg['Bii'] = torch.imag(GiiBii)

    idx['j12'] = {}
    idx['j12']['ij'] = torch.isin(alg['j'], pq)
    idx['j12']['i']  = alg['i'][idx['j12']['ij']]

    Yii = sys.Yii[:, pq].clone()
    Yii = torch.concat((Yii[:slk], Yii[slk + 1:]), dim=0)
    # Yii = _remove_rowcol(Yii, row=slk, col=None)
    c, d = torch.nonzero(Yii, as_tuple=True)

    Yij1 = Yij[:, pq].clone()
    Yij1 = torch.concat((Yij1[:slk], Yij1[slk + 1:]), dim=0)
    # Yij1 = _remove_rowcol(Yij1, row=slk, col=None)
    q, w = torch.nonzero(Yij1, as_tuple=True)

    idx['j12']['jci'] = torch.cat([c.long(), q.long()])
    idx['j12']['jcj'] = torch.cat([d.long(), w.long()])

    idx['j21'] = {}
    mn = torch.isin(alg['i'], pq)
    idx['j21']['ij'] = torch.logical_and(idx['j11']['ij'], mn)

    Yii2 = sys.Yii[pq, :].clone()
    Yii2 = torch.concat((Yii2[:, :slk], Yii2[:, slk + 1:]), dim=1)
    # Yii2 = _remove_rowcol(Yii2, row=None, col=slk)
    c2, d2 = torch.nonzero(Yii2, as_tuple=True)

    Yij2 = Yij[pq, :].clone()
    Yij2 = torch.concat((Yij2[:, :slk], Yij2[:, slk + 1:]), dim=1)
    # Yij2 = _remove_rowcol(Yij2, row=None, col=slk)
    q2, w2 = torch.nonzero(Yij2, as_tuple=True)

    if alg['Npq'] == 1:
        idx['j21']['jci'] = torch.cat([c2.long(), q2.view(-1).long()])
        idx['j21']['jcj'] = torch.cat([d2.long(), w2.view(-1).long()])
    else:
        idx['j21']['jci'] = torch.cat([c2.long(), q2.long()])
        idx['j21']['jcj'] = torch.cat([d2.long(), w2.long()])

    idx['j22'] = {}
    idx['j22']['ij'] = torch.logical_and(idx['j12']['ij'], mn)
    idx['j22']['i'] = alg['i'][idx['j22']['ij']]

    Yii_pq = sys.Yii[pq, :][:, pq].clone()
    c3, d3 = torch.nonzero(Yii_pq, as_tuple=True)

    Yij_pq = sys.Yij[pq, :][:, pq].clone()
    q3, w3 = torch.nonzero(Yij_pq, as_tuple=True)

    idx['j22']['jci'] = torch.cat([c3.long(), q3.long()])
    idx['j22']['jcj'] = torch.cat([d3.long(), w3.long()])

    return alg, idx

# ---------- limit handling (Q-V switching) ----------

def cq(sys, alg, idx, pf, Qcon, V, T, Qg, Ql, Qgl, Pgl, DelPQ, device=None):
    cmx_dtype = torch.complex64 if device =='mps' else torch.complex128
    real_dtype = torch.float32 if device =='mps' else torch.float64

    Ybus = _to_tensor(sys.Ybus, device=device, dtype=cmx_dtype)
    Yij  = _to_tensor(sys.Yij,  device=device, dtype=cmx_dtype)
    slk  = int(sys.slk_bus[0])

    Vc = torch.polar(V, T)  # V * exp(jT)
    S  = Vc * torch.conj(torch.matmul(Ybus, Vc))
    Q  = -torch.imag(S)

    active_mask = (Qcon[:,2] == 1)
    Qmin_violated = torch.nonzero((Q < Qcon[:,0]) & active_mask, as_tuple=True)[0]
    Qmax_violated = torch.nonzero((Q > Qcon[:,1]) & active_mask, as_tuple=True)[0]

    if Qmin_violated.numel() > 0:
        idx_np = Qmin_violated.cpu().numpy()
        sys.bus.loc[idx_np, 'bus_type'] = 1
        Qcon[Qmin_violated, 2] = 0

        Qg[Qmin_violated] = Qcon[Qmin_violated, 0] + Ql[Qmin_violated]

        Y_diag  = Ybus[Qmin_violated, Qmin_violated]
        conj_Vc = torch.conj(Vc[Qmin_violated])
        rhs = (Pgl[Qmin_violated] - 1j * Qcon[Qmin_violated, 0]) / conj_Vc
        Vc[Qmin_violated] = (1.0 / Y_diag) * (rhs - torch.matmul(Yij[Qmin_violated, :], Vc))

    if Qmax_violated.numel() > 0:
        idx_np = Qmax_violated.cpu().numpy()
        sys.bus.loc[idx_np, 'bus_type'] = 1
        Qcon[Qmax_violated, 2] = 0

        Qg[Qmax_violated] = Qcon[Qmax_violated, 1] + Ql[Qmax_violated]

        Y_diag  = Ybus[Qmax_violated, Qmax_violated]
        conj_Vc = torch.conj(Vc[Qmax_violated])
        rhs = (Pgl[Qmax_violated] - 1j * Qcon[Qmax_violated, 1]) / conj_Vc  # fixed small bug vs original
        Vc[Qmax_violated] = (1.0 / Y_diag) * (rhs - torch.matmul(Yij[Qmax_violated, :], Vc))

    if (Qmin_violated.numel() + Qmax_violated.numel()) > 0:
        alg, idx = idx_par2(sys, alg, idx, device=device)
        T = torch.angle(Vc)
        V = torch.abs(Vc)
        Qgl = Qg - Ql

        DelS = Vc * torch.conj(torch.matmul(Ybus, Vc)) - (Pgl + 1j * Qgl)
        DelPQ = torch.cat([torch.real(DelS[alg['ii']]), torch.imag(DelS[alg['pq']])])

        pf['limit'][Qmin_violated.cpu().numpy(), 0] = 1
        pf['limit'][Qmax_violated.cpu().numpy(), 1] = 1

    return sys, alg, idx, pf, Qcon, V, T, Qg, Qgl, DelPQ

def cv(sys, alg, idx, pf, Vcon, DelPQ, V, T, Pgl, Qgl, device=None):
    cmx_dtype = torch.complex64 if device == 'mps' else torch.complex128
    real_dtype = torch.float32 if device == 'mps' else torch.float64
    Ybus = _to_tensor(sys.Ybus, device=device, dtype=cmx_dtype)
    Yij  = _to_tensor(sys.Yij,  device=device, dtype=cmx_dtype)

    Vmin_violated = torch.nonzero((V < Vcon[:,0]) & (Vcon[:,2] == 1), as_tuple=True)[0]
    Vmax_violated = torch.nonzero((V > Vcon[:,1]) & (Vcon[:,2] == 1), as_tuple=True)[0]

    if Vmin_violated.numel() > 0:
        sys.bus.loc[Vmin_violated.cpu().numpy(), 'bus_type'] = 2
        Vcon[Vmin_violated, 2] = 0

        V[Vmin_violated] = Vcon[Vmin_violated, 0]
        Vp = torch.polar(V, T)

        Q = -torch.imag(torch.conj(Vp[Vmin_violated]) * (torch.matmul(Ybus[Vmin_violated, :], Vp)))

        Y_diag = Ybus[Vmin_violated, Vmin_violated]
        rhs = (Pgl[Vmin_violated] - 1j * Q) / torch.conj(Vp[Vmin_violated])
        T[Vmin_violated] = torch.angle((1.0 / Y_diag) * (rhs - torch.matmul(Yij[Vmin_violated, :], Vp)))

    if Vmax_violated.numel() > 0:
        sys.bus.loc[Vmax_violated.cpu().numpy(), 'bus_type'] = 2
        Vcon[Vmax_violated, 2] = 0

        V[Vmax_violated] = Vcon[Vmax_violated, 1]
        Vp = torch.polar(V, T)

        Q = -torch.imag(torch.conj(Vp[Vmax_violated]) * (torch.matmul(Ybus[Vmax_violated, :], Vp)))

        Y_diag = Ybus[Vmax_violated, Vmax_violated]
        rhs = (Pgl[Vmax_violated] - 1j * Q) / torch.conj(Vp[Vmax_violated])
        T[Vmax_violated] = torch.angle((1.0 / Y_diag) * (rhs - torch.matmul(Yij[Vmax_violated, :], Vp)))

    if (Vmin_violated.numel() + Vmax_violated.numel()) > 0:
        alg, idx = idx_par2(sys, alg, idx, device=device)

        Vp = torch.polar(V, T)
        DelS = Vp * torch.conj(torch.matmul(Ybus, Vp)) - (Pgl + 1j * Qgl)
        DelPQ = torch.cat([torch.real(DelS[alg['ii']]), torch.imag(DelS[alg['pq']])])

        pf['limit'][Vmin_violated.cpu().numpy(), 0] = 1
        pf['limit'][Vmax_violated.cpu().numpy(), 1] = 1

    return sys, alg, idx, pf, Vcon, DelPQ, V, T

# ---------- Jacobian pieces ----------

def data_jacobian(T, V, alg, Nbu, device=None, dtype=torch.float64):
    Tij = T[alg['i']] - T[alg['j']]

    Te1 = (alg['Gij'] * torch.sin(Tij)) - (alg['Bij'] * torch.cos(Tij))
    Te2 = (alg['Gij'] * torch.cos(Tij)) + (alg['Bij'] * torch.sin(Tij))

    alg['Te1'] = Te1
    alg['Te2'] = Te2

    # fD1 = coo((-Te1, (fd1i, j)), shape=(Nbu-1,Nbu)) @ V
    rows = alg['fd1i']
    cols = alg['j']
    # fD1 = torch.sparse_coo_tensor(
    #     torch.vstack((rows, cols)),
    #     -Te1,
    #     (Nbu-1, Nbu)
    # ).mm(V)
    vals = -Te1
    fD1 = _coo_mv(vals, rows, cols, (Nbu-1, Nbu), V)
    alg['fD1'] = fD1

    # fD2: Te2_pq on (fdi, fdj)
    Te2_pq = Te2[alg['fij']]
    rows = alg['fdi']
    cols = alg['fdj']
    # fD2 = torch.sparse_coo_tensor(
    #     torch.vstack((rows, cols)),
    #     Te2_pq,
    #     (alg['Npq'], Nbu)
    # ).dot(V)
    fD2 = _coo_mv(Te2_pq, alg['fdi'], alg['fdj'], (alg['Npq'], Nbu), V)
    alg['fD2'] = fD2

    # fD3: Te1_pq on (fdi, fdj)
    Te1_pq = Te1[alg['fij']]
    # fD3 = torch.sparse_coo_tensor(
    #     torch.vstack((rows, cols)),
    #     Te1_pq,
    #     (alg['Npq'], Nbu)
    # ).dot(V)
    fD3 = _coo_mv(Te1_pq, alg['fdi'], alg['fdj'], (alg['Npq'], Nbu), V)
    alg['fD3'] = fD3

    alg['Vpq'] = V[alg['pq']]
    alg['Vij'] = V[alg['i']] * V[alg['j']]

    return alg

def jacobian11(V, alg, idx, Nbu, device=None, dtype=torch.float64):
    D = V[alg['ii']] * alg['fD1']  # (Nbu-1,)
    N = alg['Vij'][idx['ij']] * alg['Te1'][idx['ij']]  # (nnz,)

    data = torch.cat([D, N])
    row  = idx['jci']
    col  = idx['jcj']
    J = torch.sparse_coo_tensor(torch.vstack((row, col)), data, (Nbu-1, Nbu-1))
    # J = torch.zeros((Nbu-1, Nbu-1), device=V.device, dtype=dtype)
    # J[row, col] = data
    return J.to_dense()

def jacobian12(V, alg, idx, Nbu, device=None, dtype=torch.float64):
    D = alg['fD2'] + 2.0 * alg['Gii'] * alg['Vpq']
    N = V[idx['i']] * alg['Te2'][idx['ij']]

    data = torch.cat([D, N])
    row  = idx['jci']
    col  = idx['jcj']
    J = torch.sparse_coo_tensor(torch.vstack((row, col)), data, (Nbu-1, alg['Npq']))
    # J = torch.zeros((Nbu-1, alg['Npq']), device=V.device, dtype=dtype)
    # J[row, col] = data
    return J.to_dense()

def jacobian21(alg, idx, Nbu, device=None, dtype=torch.float64):
    D = alg['Vpq'] * alg['fD2']
    N = -alg['Vij'][idx['ij']] * alg['Te2'][idx['ij']]

    data = torch.cat([D, N])
    row  = idx['jci']
    col  = idx['jcj']
    J = torch.sparse_coo_tensor(torch.vstack((row, col)), data, (alg['Npq'], Nbu-1))
    # J = torch.zeros((alg['Npq'], Nbu-1), device=alg['Vpq'].device, dtype=dtype)
    # J[row, col] = data
    return J.to_dense()

def jacobian22(V, alg, idx, device=None, dtype=torch.float64):
    D = alg['fD3'] - 2.0 * alg['Bii'] * alg['Vpq']
    N = V[idx['i']] * alg['Te1'][idx['ij']]

    data = torch.cat([D, N])
    row  = idx['jci']
    col  = idx['jcj']
    J = torch.sparse_coo_tensor(torch.vstack((row, col)), data, (alg['Npq'], alg['Npq']))
    # J = torch.zeros((alg['Npq'], alg['Npq']), device=V.device, dtype=dtype)
    # J[row, col] = data
    return J.to_dense()

# ---------- Newton–Raphson PF ----------

def NR_PF(sys, loads, gens, x_init, user, device=None, rtol=None):
    """
    Torch refactor of your AC NR power flow.
    Arrays in sys.* can be NumPy; they’re converted to torch on-the-fly.
    """
    # choose dtypes
    real_dtype = torch.float32 if device =='mps' else torch.float64
    cmx_dtype = torch.complex64 if device =='mps' else torch.complex128

    pf = {'method': 'AC Power Flow using Newton-Raphson Algorithm',
          'limit':  torch.zeros((sys.nb, 2), dtype=real_dtype).numpy()}  # keep this as numpy-compatible

    x_init = _to_tensor(x_init, device=device, dtype=real_dtype)
    T = x_init[:, 0].clone()
    V = x_init[:, 1].clone()

    loads = _to_tensor(loads, device=device, dtype=real_dtype)
    gens  = _to_tensor(gens,  device=device, dtype=real_dtype)

    Pl, Ql = loads[:, 0], loads[:, 1]
    Pg, Qg = gens[:, 0],  gens[:, 1]

    No = 0

    Qcon, Vcon = qv_limits(user, sys, Ql, device=device, dtype=real_dtype)
    alg, idx = idx_par1(sys, device=device, dtype=real_dtype)
    alg, idx = idx_par2(sys, alg, idx, device=device, dtype=real_dtype)

    Vini = torch.polar(V, T)
    Pgl = Pg - Pl
    Qgl = Qg - Ql

    Ybus = _to_tensor(sys.Ybus, device=device, dtype=cmx_dtype)

    DelS = Vini * torch.conj(torch.matmul(Ybus, Vini)) - (Pgl + 1j * Qgl)
    DelPQ = torch.cat((torch.real(DelS[alg['ii']]), torch.imag(DelS[alg['pq']])))

    pf['time'] = {}
    start_pre = time.time()
    pf['time']['pre'] = time.time() - start_pre
    start_con = time.time()

    stop_tol = user['stop']
    maxIter  = user['maxIter']

    while torch.max(torch.abs(DelPQ)).item() > stop_tol and No < maxIter:
        No += 1

        if 'reactive' in user['list']:
            sys, alg, idx, pf, Qcon, V, T, Qg, Qgl, DelPQ = cq(
                sys, alg, idx, pf, Qcon, V, T, Qg, Ql, Qgl, Pgl, DelPQ, device=device
            )

        if 'voltage' in user['list']:
            sys, alg, idx, pf, Vcon, DelPQ, V, T = cv(
                sys, alg, idx, pf, Vcon, DelPQ, V, T, Pgl, Qgl, device=device
            )

        alg = data_jacobian(T, V, alg, sys.nb, device=device, dtype=real_dtype)
        J11 = jacobian11(V, alg, idx['j11'], sys.nb, device=device, dtype=real_dtype)
        J12 = jacobian12(V, alg, idx['j12'], sys.nb, device=device, dtype=real_dtype)
        J21 = jacobian21(alg, idx['j21'], sys.nb, device=device, dtype=real_dtype)
        J22 = jacobian22(V, alg, idx['j22'], device=device, dtype=real_dtype)

        J = torch.block_diag(J11, J22)
        J[:J11.size(0), J11.size(1):] = J12
        J[J11.size(0):, :J11.size(1)] = J21

        dTV = -(torch.linalg.pinv(J) @ DelPQ)
        TV = torch.concat((T, V[alg['pq']]))

        slk = int(sys.slk_bus[0])
        dTV_full = torch.zeros_like(TV, device=device, dtype=real_dtype)
        dTV_full[:slk] = dTV[:slk]
        dTV_full[slk + 1:] = dTV[slk:]

        TV += dTV_full

        T = TV[:sys.nb]
        V[alg['pq']] = TV[sys.nb:]

        Vc = torch.polar(V, T)
        DelS = Vc * torch.conj(torch.matmul(Ybus, Vc)) - (Pgl + 1j * Qgl)
        DelPQ = torch.cat((torch.real(DelS[alg['ii']]), torch.imag(DelS[alg['pq']])))

    pf['Vc'] = Vc
    pf['time']['con'] = time.time() - start_con
    pf['iteration'] = No

    return pf