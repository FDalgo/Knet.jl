__global__ void _add_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _add_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _add_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _add_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _sub_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _sub_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _mul_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _mul_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _div_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _div_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _pow_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _pow_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _max_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _max_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _min_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _min_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _eq_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _eq_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _ne_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _ne_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _gt_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _gt_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _ge_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _ge_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _lt_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _lt_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _le_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _le_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _invxback_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _invxback_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _reluback_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _reluback_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _sigmback_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _sigmback_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _tanhback_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _tanhback_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_32_12(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _rpow_32_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_64_12(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _rpow_64_12<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
