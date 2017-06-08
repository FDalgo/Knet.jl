# Kernels for Scalar,Array->Array

using Knet: broadcast_ops

function cuda01src(f, j=f, ex="$f(xi,yi)"; BLK=128, THR=128)
  sprint() do s
    for (T,F) in [("float","$(f)_32"),("double","$(f)_64")]
        print(s,
"""
__global__ void _$(F)_01(int n, $T xi, $T *y, $T *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    $T yi = y[i];
    z[i] = $ex;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void $(F)_01(int n, $T xi, $T *y, $T *z) {
    _$(F)_01<<<$BLK,$THR>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
""")
    end
  end
end

for a in broadcast_ops
    if !isa(a,Tuple); a=(a,); end
    print(cuda01src(a...))
end
