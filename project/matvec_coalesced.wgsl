// coalesced mat-vec, non-strided memory access
@group(0) @binding(0)
var<storage, read> A: array<f32>;
@group(0) @binding(1)
var<storage, read> v: array<f32>;
@group(0) @binding(2)
var<storage, read_write> c: array<f32>;

override wg_size: u32;
override m: u32;
override n: u32;

@compute @workgroup_size(wg_size)
fn main(@builtin(global_invocation_id) gid: vec3u) {
    var sum: f32 = 0.0;

    for (var i: u32 = 0; i < n; i++) {
      sum = fma(A[i * m + gid.x], v[i], sum);
    }

    c[gid.x] = sum;

    return;
}
