// vector form of sparse CSR mat-vec
// ideas from https://www.nvidia.com/docs/io/66889/nvr-2008-004.pdf
// A, sparse CSR, [p, K]
// wgsl does not support structs with multiple runtime length arrays
// so we must use 3 separate storage buffers to represent A
@group(0) @binding(0)
var<storage, read> indptr: array<u32>;  // row start indices
@group(0) @binding(1)
var<storage, read> indices: array<u32>;  // col indices
@group(0) @binding(2)
var<storage, read> values: array<f32>;  // matrix values

// C, [k, T]
@group(0) @binding(3)
var<storage, read> C: array<f32>;

// t, the index of T to render, i.e. the desired column of C
@group(0) @binding(4)
var<uniform> t: u32;

// used to scale the final values
@group(0) @binding(6)
var<storage, read> scale_factor: array<f32>;
@group(0) @binding(7)
var<storage, read> scale_add: array<f32>;

// final result is written into a texture so we can visualize it!
@group(0) @binding(5) var out_tex: texture_storage_2d<r32float, write>;

override wg_size: u32;
override T: u32;
override n_cols: u32;

var<workgroup> sdata: array<f32, wg_size>;

@compute @workgroup_size(wg_size)
fn spmv_csr(@builtin(workgroup_id) wgid: vec3u, @builtin(num_workgroups) nwg: vec3u, @builtin(local_invocation_id) lid: vec3u) {
    let row = wgid.y * nwg.x + wgid.x;
    let p = arrayLength(&indptr) - 1;
    if (row >= p) {  // if p is not a multiple of wg_size
        return;
    }

    let row_start = indptr[row];
    let row_end   = indptr[row + 1];

    // intermediate sum in a given local invocation
    var sum_intermediate: f32 = 0.0;
    var j: u32 = row_start + lid.x;
    while (j < row_end) {
        sum_intermediate = fma(values[j], C[indices[j] * T + t], sum_intermediate);
        j = j + wg_size;
    }
    sdata[lid.x] = sum_intermediate;
    workgroupBarrier();

    // sum reduce when all local invocation intermediate sums are ready
    if (lid.x < 16) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x + 16]; }
    workgroupBarrier();

    if (lid.x <  8) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x +  8]; }
    workgroupBarrier();

    if (lid.x <  4) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x +  4]; }
    workgroupBarrier();

    if (lid.x <  2) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x +  2]; }
    workgroupBarrier();

    if (lid.x <  1) { sdata[lid.x] = sdata[lid.x] + sdata[lid.x +  1]; }
    workgroupBarrier();

    if (lid.x == 0) {
        let val = fma(scale_factor[row], sdata[0], scale_add[row]);
        textureStore(out_tex, vec2u(row % n_cols, row / n_cols), vec4f(val, 0.0, 0.0, 0.0));
    }
}
