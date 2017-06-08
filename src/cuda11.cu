__global__ void _add_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_32_11(int n, float *x, float *y, float *z) {
    _add_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _add_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi+yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void add_64_11(int n, double *x, double *y, double *z) {
    _add_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_32_11(int n, float *x, float *y, float *z) {
    _sub_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sub_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi-yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sub_64_11(int n, double *x, double *y, double *z) {
    _sub_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_32_11(int n, float *x, float *y, float *z) {
    _mul_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _mul_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi*yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void mul_64_11(int n, double *x, double *y, double *z) {
    _mul_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_32_11(int n, float *x, float *y, float *z) {
    _div_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _div_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi/yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void div_64_11(int n, double *x, double *y, double *z) {
    _div_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_32_11(int n, float *x, float *y, float *z) {
    _pow_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _pow_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = pow(xi,yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void pow_64_11(int n, double *x, double *y, double *z) {
    _pow_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_32_11(int n, float *x, float *y, float *z) {
    _max_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _max_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (xi>yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void max_64_11(int n, double *x, double *y, double *z) {
    _max_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_32_11(int n, float *x, float *y, float *z) {
    _min_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _min_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (xi<yi?xi:yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void min_64_11(int n, double *x, double *y, double *z) {
    _min_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_32_11(int n, float *x, float *y, float *z) {
    _eq_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _eq_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi==yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void eq_64_11(int n, double *x, double *y, double *z) {
    _eq_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_32_11(int n, float *x, float *y, float *z) {
    _ne_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ne_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi!=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ne_64_11(int n, double *x, double *y, double *z) {
    _ne_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_32_11(int n, float *x, float *y, float *z) {
    _gt_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _gt_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi>yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void gt_64_11(int n, double *x, double *y, double *z) {
    _gt_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_32_11(int n, float *x, float *y, float *z) {
    _ge_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _ge_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi>=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void ge_64_11(int n, double *x, double *y, double *z) {
    _ge_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_32_11(int n, float *x, float *y, float *z) {
    _lt_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _lt_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi<yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void lt_64_11(int n, double *x, double *y, double *z) {
    _lt_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_32_11(int n, float *x, float *y, float *z) {
    _le_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _le_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = xi<=yi;
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void le_64_11(int n, double *x, double *y, double *z) {
    _le_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_32_11(int n, float *x, float *y, float *z) {
    _invxback_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _invxback_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (-xi*yi*yi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void invxback_64_11(int n, double *x, double *y, double *z) {
    _invxback_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_32_11(int n, float *x, float *y, float *z) {
    _reluback_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _reluback_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (yi>0?xi:0);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void reluback_64_11(int n, double *x, double *y, double *z) {
    _reluback_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_32_11(int n, float *x, float *y, float *z) {
    _sigmback_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _sigmback_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (xi*yi*(1-yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void sigmback_64_11(int n, double *x, double *y, double *z) {
    _sigmback_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_32_11(int n, float *x, float *y, float *z) {
    _tanhback_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _tanhback_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = (xi*(1-yi*yi));
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void tanhback_64_11(int n, double *x, double *y, double *z) {
    _tanhback_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float xi=x[i];
    float yi=y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_32_11(int n, float *x, float *y, float *z) {
    _rpow_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _rpow_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double xi=x[i];
    double yi=y[i];
    z[i] = pow(yi,xi);
    i += blockDim.x * gridDim.x;
  }
}
#ifdef __cplusplus
extern "C" {
#endif
  void rpow_64_11(int n, double *x, double *y, double *z) {
    _rpow_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif

__global__ void _BGH_32_11(int n, float *x, float *y, float *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    float B=x[i];
    float xi=y[i];
    float flag = 1.0;
		if (B < 0.0)
		{
			flag = -1.0;
			B = -B;
			xi = -xi;
		}
		if (B > 1e6) /*as saying infty*/
		{
			z[i] = xi > 30 ? flag*(xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi))))) : flag * (exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142)));
		}
		else if (B > 15.0)
		{
			float GH = xi > 30 ? flag*(xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi))))) : flag * (exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142)));
			float H = 0.5 * erfc(xi / 1.4142136);
			z[i] = GH/(1 + exp(-2*B - log(H)));
		}
		else
		{
			float G = 0.398942 * exp(-xi*xi/2);
			float H = 0.5 * erfc(xi / 1.4142136);
			z[i] = flag * G / (H + 1/expm1f(2*B));
		}
    i += blockDim.x * gridDim.x;
  }
	__syncthreads();
}
#ifdef __cplusplus
extern "C" {
#endif
  void BGH_32_11(int n, float *x, float *y, float *z) {
    _BGH_32_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif
__global__ void _BGH_64_11(int n, double *x, double *y, double *z) {
  int i = threadIdx.x + blockIdx.x * blockDim.x;
  while (i < n) {
    double B=x[i];
    double xi=y[i];
    double flag = 1.0;
		if (B < 0.0)
		{
			flag = -1.0;
			B = -B;
			xi = -xi;
		}
		if (B > 1e6) /*as saying infty*/
		{
			z[i] = xi > 30 ? flag*(xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi))))) : flag * (exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142)));
		}
		else if (B > 15.0)
		{
			double GH = xi > 30 ? flag*(xi + 1/xi * (1 - 2/(xi*xi) * (1 - 5/(xi*xi) * (1 - 7.4/(xi*xi))))) : flag * (exp(-(xi*xi)/2) / (1.2533 * erfc(xi / 1.4142)));
			double H = 0.5 * erfc(xi / 1.4142136);
			z[i] = GH/(1 + exp(-2*B - log(H)));
		}
		else
		{
			double G = 0.398942 * exp(-xi*xi/2);
			double H = 0.5 * erfc(xi / 1.4142136);
			z[i] = flag * G / (H + 1/expm1f(2*B));
		}
    i += blockDim.x * gridDim.x;
  }
	__syncthreads();
}
#ifdef __cplusplus
extern "C" {
#endif
  void BGH_64_11(int n, double *x, double *y, double *z) {
    _BGH_64_11<<<128,128>>>(n,x,y,z);
  }
#ifdef __cplusplus
}
#endif

