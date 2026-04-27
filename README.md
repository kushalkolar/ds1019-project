# Sparse mat-vec on WGPU

## Installation

Installing this currently requires you to clone and install fastplotlib 
from source using the `masknmf-compute` branch. You can try out 
the kernels with random matrices using `sanity_check_gen_data.py`.

Running the benchmarks and viz requires the datasets and `masknmf`, 
contact me if you really need it.

```bash
git clone https://github.com/fastplotlib/fastplotlib.git
git checkout masknmf-compute

cd fastplotlib

# install with extras in place
pip install -e ".[imgui]"

# install latest pygfx
pip install git+https://github.com/pygfx/pygfx.git@main
```

## Layout

project dir

### `_spmv.py`

 tell the GPU to render 
specific data, and then have a Texture so we can view it. The classes in this module handle this.

`ComputeShader` - helper class adopted from pygfx, modified to be compatible with basic uniform buffers. 
This class manages bindings (i.e. describe data that gets binded to the GPU), and the pipeline. We can 
update the uniform buffer and re-run the compute pipeline to compute and render a specific frame of data.

`create_storage_buffer` - helper class to create storage buffers for our data on the GPU

`MatrixCSR` - dataclass to organize the buffers that represent a CSR matrix

`SpMVImage` - the main class that I wrote. This takes a sparse CSR matrix `A` representing spatial components
of neurons, and a dense matrix `C` representing temporal components. IT uses `ComputeShader` to send the data to 
the GPU, run the CSR scalar or CSR vector compute shader (kernel), based on a `kwarg`, and it also manages a 
fastplotlib `TextureArray` which allows us to visualize the results as new frames are reconstructed from `A` and `C`.

### shaders (compute kernels)

`matvec.wgsl` - not used, just a basic naive mat-vec kernel that operates on dense arrays
`matvec_coalesced.wgsl` - basic mat-vec that uses coalesced as opposed to strided memory access for more efficient reads. See: https://developer.nvidia.com/blog/unlock-gpu-performance-global-memory-access-in-cuda/
`spmv_csr_scalar.wgsl` - basic naive sparse CSR mat-vec implementation
`spmv_csr_vector.wgsl` - vector form of sparse CSR mat-vec implementation that performs more efficient reads by parallelizing across local invocations (a.k.a. threads) within a row

### Example files

`sanity_check_gen_data.py` - sanity check the `spmv_csr_scalar.wgsl` and `spmv_csr_vector.wgsl` kernels against scipy sparse 
results by generating random data. This also ensure the entire pipeline from the `SpMVImage` and `ComputeShader` classes are also working correctly. We 
expect the diff between the scipy.sparse result and my kernel to be 0.0 or within machine precision for float32, 2^-23 or ~1.19 x 10^-7

`masknmf_arrays.py` visualizes a reconstruction as fast as it can possibly be computed & rendered using `SpMVImage`.

The rest of the file are various benchmarking files.
