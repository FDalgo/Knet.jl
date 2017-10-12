__global__ void _add_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_32_01(int n, float xi, float *y, float *z) {
    _add_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _add_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_64_01(int n, double xi, double *y, double *z) {
    _add_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_32_01(int n, float xi, float *y, float *z) {
    _sub_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_64_01(int n, double xi, double *y, double *z) {
    _sub_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_32_01(int n, float xi, float *y, float *z) {
    _mul_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_64_01(int n, double xi, double *y, double *z) {
    _mul_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_32_01(int n, float xi, float *y, float *z) {
    _div_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_64_01(int n, double xi, double *y, double *z) {
    _div_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_32_01(int n, float xi, float *y, float *z) {
    _pow_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_64_01(int n, double xi, double *y, double *z) {
    _pow_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_32_01(int n, float xi, float *y, float *z) {
    _max_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_64_01(int n, double xi, double *y, double *z) {
    _max_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_32_01(int n, float xi, float *y, float *z) {
    _min_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_64_01(int n, double xi, double *y, double *z) {
    _min_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_32_01(int n, float xi, float *y, float *z) {
    _eq_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_64_01(int n, double xi, double *y, double *z) {
    _eq_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_32_01(int n, float xi, float *y, float *z) {
    _ne_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_64_01(int n, double xi, double *y, double *z) {
    _ne_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_32_01(int n, float xi, float *y, float *z) {
    _gt_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_64_01(int n, double xi, double *y, double *z) {
    _gt_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_32_01(int n, float xi, float *y, float *z) {
    _ge_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_64_01(int n, double xi, double *y, double *z) {
    _ge_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_32_01(int n, float xi, float *y, float *z) {
    _lt_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_64_01(int n, double xi, double *y, double *z) {
    _lt_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_32_01(int n, float xi, float *y, float *z) {
    _le_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_64_01(int n, double xi, double *y, double *z) {
    _le_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_32_01(int n, float xi, float *y, float *z) {
    _invxback_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_64_01(int n, double xi, double *y, double *z) {
    _invxback_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_32_01(int n, float xi, float *y, float *z) {
    _reluback_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_64_01(int n, double xi, double *y, double *z) {
    _reluback_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_32_01(int n, float xi, float *y, float *z) {
    _sigmback_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_64_01(int n, double xi, double *y, double *z) {
    _sigmback_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_32_01(int n, float xi, float *y, float *z) {
    _tanhback_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_64_01(int n, double xi, double *y, double *z) {
    _tanhback_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_32_01(int n, float xi, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float yi = y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_32_01(int n, float xi, float *y, float *z) {
    _rpow_32_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_64_01(int n, double xi, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double yi = y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_64_01(int n, double xi, double *y, double *z) {
    _rpow_64_01<<<128,128>>>(n,xi,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _BGH_32_01(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    float yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    float sign = 1.0;
    if (xi < 0.0)
    {
      xi = -xi;
      yi = -yi;
      sign = -1.0;
    }
    //float part1 = exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142));
    if (xi > 1e6) /*like say infty*/
    {
      z[i] = sign * sqrt(2/3.1415) * (exp(-yi*yi/2)) / erfc(yi/sqrt(2.0));
    }
    else if (xi > 15)
    {

      z[i] = sign * sqrt(2/3.1415) * (exp(-yi*yi/2)) / erfc(yi/sqrt(2.0)) / (1 + exp(-2*xi - log(0.5 * erfc(yi / 1.4142))));
    }
    else
    {
      //float hx = exp(-(yi*yi)/2);
      z[i] =sign * (exp(-yi*yi/2)) / (sqrt(2*3.1415) * (1/expm1f(2*xi) + 0.5*erfc(yi/sqrt(2.0))));
    }
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void BGH_32_01(int n, float *x, int sx, int nx, float *y, int sy, int ny, float *z) {
    _BGH_32_01<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _BGH_64_01(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = (nx==n ? x[i] : sx==1 ? x[i%nx] : nx==1 ? x[0] : x[(i/sx)%nx]);
    double yi = (ny==n ? y[i] : sy==1 ? y[i%ny] : ny==1 ? y[0] : y[(i/sy)%ny]);
    double sign = 1.0;
    if (xi < 0.0)
    {
      xi = -xi;
      yi = -yi;
      sign = -1.0;
    }
    double part1 = exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142));
    if (xi > 1e6) /*like say infty*/
    {
      z[i] = sign * part1;
    }
    else if (xi > 15)
    {
      z[i] = sign * part1 * (1 + exp(-2*xi - log(0.5 * erfc(yi / 1.4142))));
    }
    else
    {
      double hx = exp(-(yi*yi)/2);
      z[i] =sign * hx / (2.5066 * (1/expm1f(2*xi) + hx/ 2.5066));
    }
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void BGH_64_01(int n, double *x, int sx, int nx, double *y, int sy, int ny, double *z) {
    _BGH_64_01<<<128,128>>>(n,x,sx,nx,y,sy,ny,z);
  }
#ifdef __cplusplus
}
#endif
