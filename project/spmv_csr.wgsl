// sparse matvec, conceptually using scatter-add
// (A @ c) * scale_factor + scale_add

// implementation is adopted from this which describes various ways to perform sparse mat-vec on GPUS
// https://tschmidt23.github.io/cse599i/CSE%20599%20I%20Accelerated%20Computing%20-%20Programming%20GPUs%20Lecture%2010.pdf
// I use CSR instead of COO because wgsl does not support atomicAdd for float32 values (not all GPUs support it)
//
//  A: CSR matrix of shape [p, k]
//  C: dense matrix of shape [k, T]
//  c: column of C at index `t`
//  output is written into a 2D texture at the corresponding un-flattened 2D index

// A, sparse CSR, [p, K]
// wgsl does not support structs with multiple runtime length arrays
// so we must use 3 separate storage buffers to represent A
// https://www.w3.org/TR/WGSL/#struct-types
@group(0) @binding(0) var<storage, read> indptr:  array<u32>;  // row start indices
@group(0) @binding(1) var<storage, read> indices: array<u32>;  // col indices
@group(0) @binding(2) var<storage, read> values:  array<f32>;  // matrix values

// C, [k, T]
@group(0) @binding(3) var<storage, read> C:       array<f32>;

// t, the index of T to render, i.e. the desired column of C
@group(0) @binding(4) var<uniform> t: u32;

// used to scale the final values
@group(0) @binding(6) var<storage, read> scale_factor: array<f32>;
@group(0) @binding(7) var<storage, read> scale_add:    array<f32>;

// final result is written into a texture so we can visualize it!
@group(0) @binding(5) var out_tex: texture_storage_2d<r32float, write>;

// constants
override wg_size: u32;
override T: u32;
override n_cols: u32;

@compute @workgroup_size(wg_size)
fn spmv_csr(@builtin(global_invocation_id) gid: vec3u) {
    let row = gid.x;  // global index across all workgroups
    let p = arrayLength(&indptr) - 1;
    if (row >= p) {  // if p is not a multiple of wg_size
        return;
    }

    let row_start = indptr[row];
    let row_end   = indptr[row + 1];

    var sum: f32 = 0.0;
    for (var j: u32 = row_start; j < row_end; j = j + 1) {
        let col = indices[j];
        let C_row_index = col * T;
        sum = sum + values[j] * C[C_row_index + t];
    }

    // apply `scale_factor` and `scale_add`
    // we can directly use some SIMD instrutions like FMA
    let val = fma(scale_factor[row], sum, scale_add[row]);

    // write final value to texture which can be visualized using a fastplotlib ImageGraphic
    textureStore(out_tex, vec2u(row % n_cols, row / n_cols), vec4f(val, 0.0, 0.0, 0.0));
}
