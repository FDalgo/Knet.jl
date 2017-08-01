__global__ void _abs2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi*xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void abs2_32(int n, float *x, float *y) {
    _abs2_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _abs2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi*xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void abs2_64(int n, double *x, double *y) {
    _abs2_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _abs_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi<0?-xi:xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void abs_32(int n, float *x, float *y) {
    _abs_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _abs_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi<0?-xi:xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void abs_64(int n, double *x, double *y) {
    _abs_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _htanh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi<-1?-1 : (xi > 1 ? 1 :xi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void htanh_32(int n, float *x, float *y) {
    _htanh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _htanh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi<-1?-1 : (xi > 1 ? 1 :xi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void htanh_64(int n, double *x, double *y) {
    _htanh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _acos_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = acos(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void acos_32(int n, float *x, float *y) {
    _acos_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _acos_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = acos(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void acos_64(int n, double *x, double *y) {
    _acos_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _acosh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = acosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void acosh_32(int n, float *x, float *y) {
    _acosh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _acosh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = acosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void acosh_64(int n, double *x, double *y) {
    _acosh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _asin_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = asin(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void asin_32(int n, float *x, float *y) {
    _asin_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _asin_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = asin(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void asin_64(int n, double *x, double *y) {
    _asin_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _asinh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = asinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void asinh_32(int n, float *x, float *y) {
    _asinh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _asinh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = asinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void asinh_64(int n, double *x, double *y) {
    _asinh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _atan_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = atan(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void atan_32(int n, float *x, float *y) {
    _atan_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _atan_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = atan(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void atan_64(int n, double *x, double *y) {
    _atan_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _atanh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = atanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void atanh_32(int n, float *x, float *y) {
    _atanh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _atanh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = atanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void atanh_64(int n, double *x, double *y) {
    _atanh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cbrt_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cbrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cbrt_32(int n, float *x, float *y) {
    _cbrt_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cbrt_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cbrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cbrt_64(int n, double *x, double *y) {
    _cbrt_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ceil_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = ceil(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ceil_32(int n, float *x, float *y) {
    _ceil_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ceil_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = ceil(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ceil_64(int n, double *x, double *y) {
    _ceil_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cos_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cos(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cos_32(int n, float *x, float *y) {
    _cos_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cos_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cos(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cos_64(int n, double *x, double *y) {
    _cos_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cosh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cosh_32(int n, float *x, float *y) {
    _cosh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cosh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cosh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cosh_64(int n, double *x, double *y) {
    _cosh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cospi_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = cospi(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cospi_32(int n, float *x, float *y) {
    _cospi_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _cospi_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = cospi(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void cospi_64(int n, double *x, double *y) {
    _cospi_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erf_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = erf(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erf_32(int n, float *x, float *y) {
    _erf_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erf_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = erf(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erf_64(int n, double *x, double *y) {
    _erf_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfc_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = erfc(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfc_32(int n, float *x, float *y) {
    _erfc_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfc_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = erfc(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfc_64(int n, double *x, double *y) {
    _erfc_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfcinv_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = erfcinv(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfcinv_32(int n, float *x, float *y) {
    _erfcinv_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfcinv_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = erfcinv(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfcinv_64(int n, double *x, double *y) {
    _erfcinv_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfcx_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = erfcx(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfcx_32(int n, float *x, float *y) {
    _erfcx_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfcx_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = erfcx(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfcx_64(int n, double *x, double *y) {
    _erfcx_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfinv_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = erfinv(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfinv_32(int n, float *x, float *y) {
    _erfinv_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _erfinv_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = erfinv(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void erfinv_64(int n, double *x, double *y) {
    _erfinv_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp_32(int n, float *x, float *y) {
    _exp_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp_64(int n, double *x, double *y) {
    _exp_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp10_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp10(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp10_32(int n, float *x, float *y) {
    _exp10_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp10_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp10(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp10_64(int n, double *x, double *y) {
    _exp10_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp2(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp2_32(int n, float *x, float *y) {
    _exp2_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _exp2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp2(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void exp2_64(int n, double *x, double *y) {
    _exp2_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _expm1_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = expm1(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void expm1_32(int n, float *x, float *y) {
    _expm1_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _expm1_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = expm1(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void expm1_64(int n, double *x, double *y) {
    _expm1_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _floor_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = floor(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void floor_32(int n, float *x, float *y) {
    _floor_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _floor_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = floor(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void floor_64(int n, double *x, double *y) {
    _floor_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invx_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = 1/xi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invx_32(int n, float *x, float *y) {
    _invx_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invx_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = 1/xi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invx_64(int n, double *x, double *y) {
    _invx_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log_32(int n, float *x, float *y) {
    _log_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log_64(int n, double *x, double *y) {
    _log_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log10_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log10(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log10_32(int n, float *x, float *y) {
    _log10_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log10_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log10(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log10_64(int n, double *x, double *y) {
    _log10_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log1p_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log1p(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log1p_32(int n, float *x, float *y) {
    _log1p_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log1p_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log1p(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log1p_64(int n, double *x, double *y) {
    _log1p_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log2_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = log2(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log2_32(int n, float *x, float *y) {
    _log2_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _log2_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = log2(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void log2_64(int n, double *x, double *y) {
    _log2_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _neg_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = -xi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void neg_32(int n, float *x, float *y) {
    _neg_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _neg_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = -xi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void neg_64(int n, double *x, double *y) {
    _neg_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _relu_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void relu_32(int n, float *x, float *y) {
    _relu_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _relu_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void relu_64(int n, double *x, double *y) {
    _relu_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lzorelu_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi<1e-6?1e-6:xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lzorelu_32(int n, float *x, float *y) {
    _lzorelu_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lzorelu_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi<1e-6?1e-6:xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lzorelu_64(int n, double *x, double *y) {
    _lzorelu_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _round_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = round(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void round_32(int n, float *x, float *y) {
    _round_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _round_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = round(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void round_64(int n, double *x, double *y) {
    _round_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigm_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigm_32(int n, float *x, float *y) {
    _sigm_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigm_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>=0?1/(1+exp(-xi)):(exp(xi)/(1+exp(xi))));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigm_64(int n, double *x, double *y) {
    _sigm_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sign_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = (xi>0?1:xi<0?-1:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sign_32(int n, float *x, float *y) {
    _sign_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sign_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = (xi>0?1:xi<0?-1:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sign_64(int n, double *x, double *y) {
    _sign_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sin_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sin(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sin_32(int n, float *x, float *y) {
    _sin_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sin_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sin(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sin_64(int n, double *x, double *y) {
    _sin_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sinh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sinh_32(int n, float *x, float *y) {
    _sinh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sinh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sinh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sinh_64(int n, double *x, double *y) {
    _sinh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sinpi_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sinpi(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sinpi_32(int n, float *x, float *y) {
    _sinpi_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sinpi_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sinpi(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sinpi_64(int n, double *x, double *y) {
    _sinpi_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sqrt_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = sqrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sqrt_32(int n, float *x, float *y) {
    _sqrt_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sqrt_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = sqrt(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sqrt_64(int n, double *x, double *y) {
    _sqrt_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tan_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = tan(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tan_32(int n, float *x, float *y) {
    _tan_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tan_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = tan(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tan_64(int n, double *x, double *y) {
    _tan_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanh_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = tanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanh_32(int n, float *x, float *y) {
    _tanh_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanh_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = tanh(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanh_64(int n, double *x, double *y) {
    _tanh_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _trunc_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = trunc(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void trunc_32(int n, float *x, float *y) {
    _trunc_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _trunc_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = trunc(xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void trunc_64(int n, double *x, double *y) {
    _trunc_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _G_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = exp(-(xi*xi)*0.5)*0.3989423;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void G_32(int n, float *x, float *y) {
    _G_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _G_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = exp(-(xi*xi)*0.5) *0.3989422804014327;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void G_64(int n, double *x, double *y) {
    _G_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _H_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = 0.5 * erfc(xi *0.70710677);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void H_32(int n, float *x, float *y) {
    _H_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _H_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = 0.5 * erfc(xi*0.7071067811865475);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void H_64(int n, double *x, double *y) {
    _H_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _GH_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi = x[i];
    y[i] = xi > 30 ? xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi)))) : 0.7978846 * exp(-(xi*xi)*0.5) / (erfc(xi *0.70710677));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void GH_32(int n, float *x, float *y) {
    _GH_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _GH_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi = x[i];
    y[i] = xi > 30 ? xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi)))) : exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void GH_64(int n, double *x, double *y) {
    _GH_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif

__global__ void _atanh2Hm1_32(int n, float *x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) 
  {
    float xi = x[i];
		if (xi > 6)
    {
			float x2 = xi * xi;
      float x4 = x2 * x2;
      float x6 = x4 * x2;
      float x8 = x4 * x4;
      float x10= x6 * x4;
			float antisg = (xi >= 0) ? -1.0 : 1.0;
  		y[i] = antisg * (408.1 / x10 - 44.125 /x8 + 37 /(6*x6) - 1.25 /x4 + 0.5 /x2 + 0.25*x2 + 0.459469 + 0.25*log(x2));
		}
  	else
		{
			float doubH = 0.79788456*exp(-xi*xi/2);
  		y[i] = atanh(doubH-1);
		}
    i += blockDim.x * gridDim.x;
  }
	__syncthreads();
}
#ifdef __cplusplus
extern "C" {
#endif
  void atanh2Hm1_32(int n, float *x, float *y) {
    _atanh2Hm1_32<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _atanh2Hm1_64(int n, double *x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) 
  {
    double xi = x[i];
		if (xi > 6)
    {
			double x2 = xi * xi;
      double x4 = x2 * x2;
      double x6 = x4 * x2;
      double x8 = x4 * x4;
      double x10= x6 * x4;
			double antisg = (xi >= 0) ? -1.0 : 1.0;
  		y[i] = antisg * (408.1 / x10 - 44.125 /x8 + 37 /(6*x6) - 1.25 /x4 + 0.5 /x2 + 0.25*x2 + 0.459469 + 0.25*log(x2));
		}
  	else
		{
			double doubH = 0.79788456*exp(-xi*xi/2);
  		y[i] = atanh(doubH-1);
		}
    i += blockDim.x * gridDim.x;
  }
	__syncthreads();
}
#ifdef __cplusplus
extern "C" {
#endif
  void atanh2Hm1_64(int n, double *x, double *y) {
    _atanh2Hm1_64<<<128,128>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif




__global__ void _fill_32(int n, float x, float *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void fill_32(int n, float x, float *y) {
    _fill_32<<<256,256>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _fill_64(int n, double x, double *y) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    y[i] = x;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void fill_64(int n, double x, double *y) {
    _fill_64<<<256,256>>>(n,x,y);
  }
#ifdef __cplusplus
}
#endif
__global__ void _xfill_32(int nrows, int ncols, float x, float *y, int incy) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * incy;
    y[yidx] = x;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void xfill_32(int nrows, int ncols, float x, float *y, int incy) {
    _xfill_32<<<256,256>>>(nrows, ncols, x, y, incy);
  }
#ifdef __cplusplus
}
#endif
__global__ void _xfill_64(int nrows, int ncols, double x, double *y, int incy) {
  int row, col, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    yidx = row + col * incy;
    y[yidx] = x;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void xfill_64(int nrows, int ncols, double x, double *y, int incy) {
    _xfill_64<<<256,256>>>(nrows, ncols, x, y, incy);
  }
#ifdef __cplusplus
}
#endif
__global__ void _xcopy(int nrows, int ncols, const char *x, int incx, char *y, int incy) {
  int row, col, xidx, yidx;
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (1) {
    row = i % nrows;
    col = i / nrows;
    if (col >= ncols) break;
    xidx = row + col * incx;
    yidx = row + col * incy;
    y[yidx] = x[xidx];
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void xcopy(int nrows, int ncols, const void *x, int incx, void *y, int incy) {
    _xcopy<<<256,256>>>(nrows,ncols,(char*)x,incx,(char*)y,incy);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_1_3_2_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = i + dimx1*k + dimx1*dimx2*j;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_1_3_2_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_1_3_2_32_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_1_3_2_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = i + dimx1*k + dimx1*dimx2*j;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_1_3_2_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_1_3_2_64_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_2_1_3_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = j + dimx1*i + dimx1*dimx2*k;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_2_1_3_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_2_1_3_32_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_2_1_3_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = j + dimx1*i + dimx1*dimx2*k;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_2_1_3_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_2_1_3_64_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_2_3_1_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = k + dimx1*i + dimx1*dimx2*j;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_2_3_1_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_2_3_1_32_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_2_3_1_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = k + dimx1*i + dimx1*dimx2*j;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_2_3_1_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_2_3_1_64_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_3_1_2_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = j + dimx1*k + dimx1*dimx2*i;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_3_1_2_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_3_1_2_32_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_3_1_2_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = j + dimx1*k + dimx1*dimx2*i;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_3_1_2_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_3_1_2_64_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_3_2_1_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = k + dimx1*j + dimx1*dimx2*i;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_3_2_1_32_44(float* x, int dimx1, int dimx2, int dimx3, float* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_3_2_1_32_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
__global__ void _permutedims3D_3_2_1_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
  for (int v = threadIdx.x + blockIdx.x * blockDim.x; v < dimy1*dimy2*dimy3; v += blockDim.x * gridDim.x) {

		int i = v % dimy1;
		int j = ((v - i) / dimy1) % dimy2;
		int k = ((v - j * dimy1 - i) / (dimy1 * dimy2)) % dimy3;

		int srcIndex = k + dimx1*j + dimx1*dimx2*i;
		y[v] = x[srcIndex];
	}
}
#ifdef __cplusplus
extern "C" {
#endif
  void permutedims3D_3_2_1_64_44(double* x, int dimx1, int dimx2, int dimx3, double* y, int dimy1, int dimy2, int dimy3) {
    _permutedims3D_3_2_1_64_44<<<256,256>>>(x,dimx1,dimx2,dimx3,y,dimy1,dimy2,dimy3);
  }
#ifdef __cplusplus
}
#endif
