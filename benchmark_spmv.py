from scipy import sparse
import fastplotlib as fpl

from project import SparseDenseImage

from pathlib import Path
import masknmf

adapter = fpl.enumerate_adapters()[0]
print(adapter.info)
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

pmd = SparseDenseImage(
    U_csr,
    V,
    scale_factor=scale_factor,
    scale_add=scale_add,
    shape=(m, n),
    benchmark=True,
)

ac = SparseDenseImage(
    A_csr,
    C,
    shape=(m, n),
    benchmark=True,
)

ac.clear_timings()
for i in range(1_000):
    ac.t = i

timings = ac.get_timings()

print(timings.mean())
print(timings.std())
