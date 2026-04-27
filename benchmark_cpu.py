from pathlib import Path
import time

import numpy as np
import pandas as pd

import masknmf
import fastplotlib as fpl


adapter = fpl.enumerate_adapters()[1]
print(adapter.info.device)
fpl.select_adapter(adapter)

parent_path = Path("/home/kushal/data/alyx/cortexlab/Subjects/")

subject = "SP058"
session = "2024-07-18"

session_path = parent_path.joinpath(subject, session)

dmr_path = session_path.joinpath(f"demix.hdf5")
dmr = masknmf.DemixingResults.from_hdf5(dmr_path)


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
        "dataset",
        "kernel",
    ]
)

dmr.to("cpu")

n = 200
def cpu(comp):
    if comp == "denoise":
        timings = np.zeros(n - 100)
        for i in range(n):
            t0 = time.perf_counter()
            _ = dmr.pmd_array[i]

            # don't store the first 100, wait until the CPU freq has stabilized
            if i > 99:
                timings[i - 100] = (time.perf_counter() - t0) * 1000.0
    else:
        timings = np.zeros(n - 100)
        for i in range(n):
            t0 = time.perf_counter()
            _ = dmr.ac_array[i]
            
            if i > 99:
                timings[i - 100] = (time.perf_counter() - t0) * 1000.0

    return {
        "mean": timings.mean(),
        "median": np.median(timings),
        "std": timings.std(),
        "min": timings.min(),
        "max": timings.max(),
    }

for comp in ["denoise", "demix"]:
    print(comp)
    result = cpu(comp)

    df.loc[df.index.size] = {
        "device": "cpu",
        "backend": "torch-cpu",
        "computation": comp,
        "torch-download":  None,
        "wgsl-workgroup-size": None,
        "dataset": session,
        "kernel": None,
        **result,
    }


if not Path(__file__).parent.joinpath("benchmark_spmv.csv").is_file():
    # create new dataframe
    df.to_csv("benchmark_spmv.csv", index=False)
else:
    df.to_csv("benchmark_spmv.csv", index=False, header=False, mode="a")
