__device__ void _sum_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sum_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sum_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _sum_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _sum_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sum_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sum_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _sum_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _prod_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 1;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=ai*xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=ai*xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai*xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _prod_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void prod_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _prod_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _prod_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _prod_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai*xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai*xi;
}

__global__ void _prod_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 1;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=ai*xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=ai*xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai*xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _prod_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void prod_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _prod_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _prod_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _maximum_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = (-INFINITY);
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _maximum_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void maximum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _maximum_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _maximum_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _maximum_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai>xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai>xi?ai:xi);
}

__global__ void _maximum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = (-INFINITY);
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=(ai>xi?ai:xi);
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai>xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _maximum_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void maximum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _maximum_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _maximum_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _minimum_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _minimum_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void minimum_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _minimum_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _minimum_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _minimum_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+16]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 8]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 4]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 2]; x[i]=(ai<xi?ai:xi);
  ai=x[i]; xi=x[i+ 1]; x[i]=(ai<xi?ai:xi);
}

__global__ void _minimum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = INFINITY;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=xi; ai=(ai<xi?ai:xi);
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=(ai<xi?ai:xi);
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _minimum_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void minimum_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _minimum_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _minimum_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _sumabs_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sumabs_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sumabs_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sumabs_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _sumabs_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _sumabs_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi<0?-xi:xi); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sumabs_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sumabs_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sumabs_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _sumabs_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _sumabs2_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi*xi); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi*xi); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sumabs2_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sumabs2_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sumabs2_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _sumabs2_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _sumabs2_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _sumabs2_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi*xi); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi*xi); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _sumabs2_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void sumabs2_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _sumabs2_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _sumabs2_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _countnz_32_21_0(volatile float *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  float ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ float buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  float ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi!=0); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi!=0); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _countnz_32_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void countnz_32_21(int nx, float *x, int sy, int ny, float *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _countnz_32_21<<<64,64>>>(nx,x,sy,ny,y);
  _countnz_32_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
__device__ void _countnz_64_21_0(volatile double *x, int i) {
//for optimizing warps, volatile must be used as register optimization will lead to wrong answers
  double ai, xi;
  ai=x[i]; xi=x[i+32]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+16]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 8]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 4]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 2]; x[i]=ai+xi;
  ai=x[i]; xi=x[i+ 1]; x[i]=ai+xi;
}

__global__ void _countnz_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  __shared__ double buffer[64];
  int t = threadIdx.x;
  int b = blockIdx.x;
  double ai, xi;

  // sum the elements assigned to this thread
  ai = 0;
  if (sy == 1) {
     int istep = 64*ny;
     for (int i=b+t*ny; i<nx; i+=istep) {
        xi=x[i]; xi=(xi!=0); ai=ai+xi;
     }
  } else {
    int jstep = sy*ny;
    for (int j=0; j<nx; j+=jstep) {
      int i0 = j+b*sy;
      int i1 = i0+sy;
      for (int i=i0+t; i<i1; i+=64) {
        xi=x[i]; xi=(xi!=0); ai=ai+xi;
      }
    }
  }
  buffer[t] = ai;
  __syncthreads();

  // help sum the entries in the block
  for(int stride=64/2; stride>32; stride>>=1) {
    if(t < stride) {
      ai=buffer[t]; xi=buffer[stride+t]; buffer[t]=ai+xi;
    }
    __syncthreads();   // Q: can this be outside the for loop?
  }

  if(t<32) {
    _countnz_64_21_0(buffer,t);  // This reuses warpSum from 20 scalar reduction.
  }
  __syncthreads();

  if(t==0) {  // the first thread in the block writes the block result to y
    y[blockIdx.x]=buffer[0];
  }
}

#ifdef __cplusplus
extern "C" {
#endif
  void countnz_64_21(int nx, double *x, int sy, int ny, double *y) {
  // x[i] goes into y[(i/sy)%ny]
  //  _countnz_64_21<<<64,64>>>(nx,x,sy,ny,y);
  _countnz_64_21<<<ny,64>>>(nx,x,sy,ny,y);
}
#ifdef __cplusplus
}
#endif
