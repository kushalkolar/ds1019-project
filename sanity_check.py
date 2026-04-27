from pathlib import Path

from scipy import sparse
import numpy as np
from tqdm import tqdm

import masknmf
import fastplotlib as fpl

from project import SpMVImage

adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)

parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-18"

session_path = parent_path.joinpath(subject, session)

dmr_path = session_path.joinpath(f"demix.hdf5")
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)

V = dmr.pmd_array.v.cpu().numpy()
U = dmr.pmd_array.u
U_csr_torch = U.to_sparse_csr().cpu()
U_csr = sparse.csr_matrix(
    (U_csr_torch.values(), U_csr_torch.col_indices(), U_csr_torch.crow_indices()),
    shape=U_csr_torch.size(),
)

C = dmr.c.numpy().T
A = dmr.a
A_torch_csr = dmr.a.to_sparse_csr().cpu()
A_csr = sparse.csr_matrix(
    (A_torch_csr.values(), A_torch_csr.col_indices(), A_torch_csr.crow_indices()),
    shape=A_torch_csr.size(),
)

m, n = tuple(map(int, dmr.fov_shape))
p = m * n
k = int(V.shape[0])
T = int(V.shape[1])

scale_factor = dmr.pmd_array.var_img.ravel().cpu().numpy()
scale_add = dmr.pmd_array.mean_img.ravel().cpu().numpy()

pmd_spmv = SpMVImage(
    U_csr,
    V,
    scale_factor=scale_factor,
    scale_add=scale_add,
    shape=(m, n),
    benchmark=True,
    spmv_mode="vector",
)

ac_spmv = SpMVImage(
    A_csr,
    C,
    shape=(m, n),
    benchmark=True,
    spmv_mode="vector",
)

dmr.to("cuda")


err_pmd = np.zeros(T)
err_ac = np.zeros(T)
# check that results match pytorch
for i in tqdm(np.random.randint(0, T, size=100)):
    pmd_spmv.t = i
    ac_spmv.t = i
    pmd_frame_torch = dmr.pmd_array[i].cpu().numpy().squeeze()
    ac_frame_torch = dmr.ac_array[i].cpu().numpy().squeeze()
    diff_pmd = pmd_spmv.to_numpy() - pmd_frame_torch
    diff_ac = ac_spmv.to_numpy() - ac_frame_torch

    err_pmd[i] = np.linalg.norm(diff_pmd, ord="fro") / np.linalg.norm(pmd_frame_torch, ord="fro")
    err_ac[i] = np.linalg.norm(diff_ac, ord="fro") / np.linalg.norm(ac_frame_torch, ord="fro")

print(err_pmd.max(), err_ac.max())
