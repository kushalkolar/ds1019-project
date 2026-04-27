from itertools import product
from pathlib import Path
import time

from scipy import sparse
import numpy as np
import torch
import pandas as pd

import masknmf
import fastplotlib as fpl

from project import SpMVImage

adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)

parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-23"

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

# I just run the file multiple times to benchmark different workgroup sizes for the wgsl kernel
# forced garbage collection in fastplotlib and pygfx can't be done due to render caching
# so I just ran the file multiple times with different workgroup sizes
WORKGROUP_SIZE = 32


pmd_spmv = SpMVImage(
    U_csr,
    V,
    scale_factor=scale_factor,
    scale_add=scale_add,
    shape=(m, n),
    benchmark=True,
    workgroup_size=WORKGROUP_SIZE,
)

ac_spmv = SpMVImage(
    A_csr,
    C,
    shape=(m, n),
    benchmark=True,
    workgroup_size=WORKGROUP_SIZE,
)


def benchmark_wgsl(obj: SpMVImage, n: int):
    obj.clear_timings()
    for i in range(n):
        obj.t = i

    timings = obj.get_timings()

    return {
        "mean": timings.mean(),
        "median": np.median(timings),
        "std": timings.std(),
        "min": timings.min(),
        "max": timings.max(),
    }


def benchmark_torch(obj, n, cpu: bool):
    timings = np.zeros(n)
    if not cpu:
        for i in range(n):
            t0 = time.perf_counter()
            _ = obj[i]
            torch.cuda.synchronize()
            timings[i] = (time.perf_counter() - t0) * 1000.0
    else:
        for i in range(n):
            t0 = time.perf_counter()
            obj[i].cpu().numpy()
            timings[i] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean": timings.mean(),
        "median": np.median(timings),
        "std": timings.std(),
        "min": timings.min(),
        "max": timings.max(),
    }


df = pd.DataFrame(
    columns=[
        "device",
        "backend",
        "computation",
        "mean",
        "median",
        "std",
        "min",
        "max",
        "torch-download",
        "wgsl-workgroup-size",
        "session"
    ]
)

# for the torch benchmarks
dmr.to("cuda")
dmr.pmd_array.to("cuda")

n = 10_000

params = [
    [benchmark_wgsl, {"denoise": pmd_spmv, "demix": ac_spmv}, [tuple()]],
]

if WORKGROUP_SIZE == 32:
    params.append(
        [
            benchmark_torch,
            {"denoise": dmr.pmd_array, "demix": dmr.ac_array},
            [(True,), (False,)],
        ]
    )

for benchmark_func, objs, torch_cpu_download in params:
    if benchmark_func is benchmark_torch and "NVIDIA" not in adapter.info.device:
        continue

    print(benchmark_func)

    for args, comp in product(torch_cpu_download, ["denoise", "demix"]):
        print(comp)

        result = benchmark_func(objs[comp], n, *args)

        df.loc[df.index.size] = {
            "device": adapter.info.device,
            "backend": "wgsl" if benchmark_func is benchmark_wgsl else "torch",
            "computation": comp,
            "torch-download": args[0] if benchmark_func is benchmark_torch else None,
            "wgsl-workgroup-size": (
                WORKGROUP_SIZE if benchmark_func is benchmark_wgsl else None
            ),
            "session": session,
            **result,
        }


if not Path(__file__).parent.joinpath("benchmarks.csv").is_file():
    # create new dataframe
    df.to_csv("benchmarks.csv", index=False)
else:
    df.to_csv("benchmarks.csv", index=False, header=False, mode="a")
