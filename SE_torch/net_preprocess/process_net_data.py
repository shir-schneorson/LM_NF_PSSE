import torch
import pandas as pd
from scipy.io import loadmat
from scipy.sparse import coo_array


def parse_ieee_mat(file):
    data = loadmat(file)  # Ensure this file contains 'data.system.line'
    data['data'] = {n: data['data'][n][0, 0] for n in data['data'].dtype.names}
    data['data']['system'] = {n: data['data']['system'][n][0, 0] for n in data['data']['system'].dtype.names}
    data['data']['legacy'] = {n: data['data']['legacy'][n][0, 0] for n in data['data']['legacy'].dtype.names}
    data['data']['pmu'] = {n: data['data']['pmu'][n][0, 0] for n in data['data']['pmu'].dtype.names}
    return data


def process_branch_data(line_data, in_transformer_data, shift_transformer_data):
    cols = ['idx_from', 'idx_to', 'rij', 'xij', 'bsi', 'tij', 'fij']

    line_data = line_data[line_data[:, -1] == 1, :-1]
    nbl = len(line_data)
    tij = torch.ones(nbl)
    fij = torch.zeros(nbl)

    branch_data = torch.cat([torch.tensor(line_data), tij.unsqueeze(1), fij.unsqueeze(1)], dim=1)

    if in_transformer_data is not None:
        in_transformer_data = in_transformer_data[in_transformer_data[:, -1] == 1, :-1]
        nbit = len(in_transformer_data)
        fij = torch.zeros(nbit)
        in_transformer_data = torch.cat([torch.tensor(in_transformer_data), fij.unsqueeze(1)], dim=1)
        branch_data = torch.cat([branch_data, in_transformer_data], dim=0)
    if shift_transformer_data is not None:
        branch_data = torch.cat([branch_data, torch.tensor(shift_transformer_data)], dim=0)

    branch_data[:, [0, 1]] -= 1
    branch_data[:, -1] = torch.deg2rad(branch_data[:, -1])
    branch_data_df = pd.DataFrame(branch_data.numpy(), columns=cols)

    return branch_data_df


def process_bus_data(bus_data, generator_data, base_MVA):
    cols = ['idx_bus', 'bus_type', 'Vo', 'To', 'Pl', 'Ql', 'rsh', 'xsh', 'Vmin', 'Vmax']

    nb = len(bus_data)
    bus_data = torch.tensor(bus_data)
    bus_data[:, 0] -= 1
    bus_data[:, 4:8] /= base_MVA
    bus_data[:, 3] = torch.deg2rad(bus_data[:, 3])
    bus_data_df = pd.DataFrame(bus_data.numpy(), columns=cols)
    bus_data_df.sort_values(by=['idx_bus'], inplace=True, ignore_index=True)

    slack_bus = bus_data_df.loc[bus_data_df['bus_type'] == 3, 'idx_bus'].iloc[0].astype(int)
    slack_bus = (slack_bus, bus_data_df.loc[slack_bus, 'To'], bus_data_df.loc[slack_bus, 'Vo'])

    gen_cols = ['Pg', 'Qg', 'Qmin', 'Qmax']
    if generator_data is not None:
        generator_data = torch.tensor(generator_data, dtype=torch.float32)  # specify dtype
        generator_data = generator_data[generator_data[:, -1] == 1, :-1]
        idx_bus_gen = (generator_data[:, 0] - 1).to(torch.int64)
        generator_data[:, 1:5] /= base_MVA

        nge = len(generator_data)
        A = coo_array((torch.ones(nge).numpy(), [idx_bus_gen.numpy(), torch.arange(nge).numpy()]), (nb, nge)).toarray()
        # Convert A to torch.float32 before multiplication
        bus_data_df.loc[:, gen_cols] = torch.tensor(A, dtype=torch.float32) @ generator_data[:, 1:5]
        bus_data_df.loc[idx_bus_gen.numpy(), 'Vo'] = generator_data[:, -1].numpy()
    else:
        bus_data_df.loc[:, gen_cols] = 0

    return bus_data_df, slack_bus


def generate_Ybus(bus_data, branch_data):
    yij = 1 / (torch.tensor(branch_data['rij'].values) + 1j * torch.tensor(branch_data['xij'].values))
    ysi = 1j * torch.tensor(branch_data['bsi'].values) / 2
    aij = torch.tensor(branch_data['tij'].values) * torch.exp(1j * torch.tensor(branch_data['fij'].values))

    yij_p_bsi = yij + ysi
    yij_p_bsi_d_aij_p2 = yij_p_bsi / (torch.conj(aij) * aij)
    m_yij_d_conj_aij = -yij / torch.conj(aij)
    m_yij_d_aij = -yij / aij

    nb = len(bus_data)
    nbr = len(branch_data)

    idx_branch = torch.tensor(branch_data.index)
    idx_from, idx_to = torch.tensor(branch_data[['idx_from', 'idx_to']].values).T

    Ai = coo_array((torch.ones_like(idx_branch).numpy(), [idx_branch.numpy(), idx_from.numpy()]), (nbr, nb)).toarray()
    Yi = coo_array((yij_p_bsi_d_aij_p2.numpy(), [idx_branch.numpy(), idx_from.numpy()]), (nbr, nb)).toarray()
    Yi += coo_array((m_yij_d_conj_aij.numpy(), [idx_branch.numpy(), idx_to.numpy()]), (nbr, nb)).toarray()

    Aj = coo_array((torch.ones_like(idx_branch).numpy(), [idx_branch.numpy(), idx_to.numpy()]), (nbr, nb)).toarray()
    Yj = coo_array((m_yij_d_aij.numpy(), [idx_branch.numpy(), idx_from.numpy()]), (nbr, nb)).toarray()
    Yj += coo_array((yij_p_bsi.numpy(), [idx_branch.numpy(), idx_to.numpy()]), (nbr, nb)).toarray()

    ysh = torch.tensor(bus_data['rsh'].values) + 1j * torch.tensor(bus_data['xsh'].values)
    Ysh = torch.diag(ysh)

    Ybus = ((torch.tensor(Ai.T, dtype=torch.complex128) @ torch.tensor(Yi)) +
            (torch.tensor(Aj.T, dtype=torch.complex128) @ torch.tensor(Yj)) +
            Ysh)

    branch_data.loc[:, 'yij'] = yij.numpy()
    branch_data.loc[:, 'ysi'] = ysi.numpy()
    branch_data.loc[:, 'aij'] = aij.numpy()
    branch_data.loc[:, 'yij_p_bsi'] = yij_p_bsi.numpy()
    branch_data.loc[:, 'yij_p_bsi_d_aij_p2'] = yij_p_bsi_d_aij_p2.numpy()
    branch_data.loc[:, 'm_yij_d_conj_aij'] = m_yij_d_conj_aij.numpy()
    branch_data.loc[:, 'm_yij_d_aij'] = m_yij_d_aij.numpy()

    Yii = torch.diag(torch.diag(Ybus))
    Yij = Ybus - Yii

    return Ybus, Yii, Yij, branch_data


class Branch:
    def __init__(self, branch_data):
        self.no = torch.arange(len(branch_data) * 2)
        self.i = torch.cat([torch.tensor(branch_data.idx_from.values), torch.tensor(branch_data.idx_to.values)])
        self.j = torch.cat([torch.tensor(branch_data.idx_to.values), torch.tensor(branch_data.idx_from.values)])
        self.yij = torch.cat(
            [torch.tensor(branch_data.m_yij_d_conj_aij.values), torch.tensor(branch_data.m_yij_d_aij.values)])
        self.ysi = torch.cat(
            [torch.tensor(branch_data.yij_p_bsi_d_aij_p2.values), torch.tensor(branch_data.yij_p_bsi.values)])
        self.gij = torch.cat(
            [torch.real(torch.tensor(branch_data.yij.values)), torch.real(torch.tensor(branch_data.yij.values))])
        self.bij = torch.cat(
            [torch.imag(torch.tensor(branch_data.yij.values)), torch.imag(torch.tensor(branch_data.yij.values))])
        self.bsi = torch.cat([torch.tensor(branch_data.bsi.values) / 2, torch.tensor(branch_data.bsi.values) / 2])
        self.tij = torch.cat([1 / torch.tensor(branch_data.tij.values), torch.ones(len(branch_data))])
        self.pij = torch.cat([1 / torch.tensor(branch_data.tij.values), 1 / torch.tensor(branch_data.tij.values)])
        self.fij = torch.cat([torch.tensor(branch_data.fij.values), -torch.tensor(branch_data.fij.values)])


class System:
    def __init__(self, system_data=None, **kwargs):
        if system_data is not None:
            self.baseMVA = system_data['baseMVA'][0, 0]
            bus_data, generator_data = system_data['bus'], system_data.get('generator')
            line_data = system_data['line']
            in_transformer_data = system_data.get('inTransformer')
            shift_transformer_data = system_data.get('shiftTransformer')
            self.bus, self.slk_bus = process_bus_data(bus_data, generator_data, self.baseMVA)
            self.branch = process_branch_data(line_data, in_transformer_data, shift_transformer_data)
            self.nbr = len(self.branch)
            self.nb = len(self.bus)
            self.Ybus, self.Yii, self.Yij, self.branch = generate_Ybus(self.bus, self.branch)

        else:
            self.baseMVA = kwargs.get('baseMVA')
            self.bus = kwargs.get('bus')
            self.branch = kwargs.get('branch')
            self.nbr = kwargs.get('nbr')
            self.nb = kwargs.get('nb')
            self.slk_bus = kwargs.get('slk_bus')
            self.Ybus = kwargs.get('Ybus')
            self.Yii = kwargs.get('Yii')
            self.Yij = kwargs.get('Yij')

    def copy(self, device=torch.device, dtype=torch.float32):
        complex_dtype = torch.complex64 if device.type=='mps' else torch.complex128
        baseMVA = self.baseMVA.copy()
        bus = self.bus.copy()
        branch = self.branch.copy()
        nbr = self.nbr
        nb = self.nb
        slk_bus = self.slk_bus
        Ybus = self.Ybus.clone().to(device=device.type, dtype=complex_dtype)
        Yii = self.Yii.clone().to(device=device.type, dtype=complex_dtype)
        Yij = self.Yij.clone().to(device=device.type, dtype=complex_dtype)
        return System(**{'baseMVA': baseMVA, 'bus': bus, 'branch': branch,
                       'nbr': nbr, 'nb': nb, 'slk_bus':slk_bus, 'Ybus': Ybus,
                       'Yii': Yii, 'Yij': Yij})