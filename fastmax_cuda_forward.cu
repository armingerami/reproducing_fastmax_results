#include <iostream>
#include <math.h>
#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>

// // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.166666;
// __device__ float a2 = 0.145833;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 2;

// // kernel = a0 + a1x + a2x^2
// __device__ float a0 = 1.0;
// __device__ float a1 = 1.0;
// __device__ float a2 = 0.5;
// // -lim^2 <= q.k <= lim^2
// __device__ float lim = 1;


namespace {
__global__
void calc_unmasked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // UNMASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    if(p == 2){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk;  ++l){
          s[outer] = k[l][i][outer];
          __syncthreads();
          loc1 = rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[outer]*s[loc1+r];
          }
        }
        for(int l = 0; l < nq;  ++l){
          s[d+outer] = 0;
          s[outer] = q[l][i][outer];
          __syncthreads();
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            s[d+outer] += tr[r]*s[outer]*s[loc2+r];
          }
          o[l][i][outer] += s[d+outer];
        }
        __syncthreads();
        for(int l = 0; l < nq; ++l){
          t = 0;
          s[outer] = o[l][i][outer];
          __syncthreads();
          if(outer == 0){
            for(int r = 0; r < d; ++r) t += s[r];
            o[l][i][d] += a2*t;
          }
        }
        __syncthreads();
      }
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    if(p == 2){
      for(int m = 0; m < d; ++m){
        for(int rr = 0; rr < d/sz; ++rr){
          for(int r = 0; r < sz; ++r) tr[r]= 0;
          for(int l = 0; l < nk;  ++l){
            s[d+outer] = k[l][i][m]*k[l][i][outer];
            tv = v[l][i][outer];
            __syncthreads();
            loc1 = d+rr*sz;
            for(int r = 0; r < sz; ++r){
              tr[r] += s[loc1+r]*tv;
            }      
          }
          for(int l = 0; l < nq;  ++l){
            s[outer] = q[l][i][m]*q[l][i][outer];
            __syncthreads();
            t = 0;
            loc2 = rr*sz;
            for(int r = 0; r < sz; ++r){
              t += tr[r]*s[loc2+r];
            }      
            o[l][i][outer] += a2*t;
          }
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void calc_masked(const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> q, const torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> v, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, int bh, int nq, int nk, int d, float a0, float a1, float a2, int p){

  extern __shared__ float s[];
  const int outer = threadIdx.x;
  const int i = blockIdx.x;
  float tv, t;
  int loc1, loc2;
  float tr[64];
  int sz = min(64,d);
  if(outer < d && i < bh){

    // MASKED PART ////////////////////////////
    // calc cons denum
    for(int l = 0; l < nq; ++l){
      o[l][i][d] = a0*(nk-nq+l+1);
    }

    // calc lin denum
    s[d+outer] = 0;
    __syncthreads();
    for(int l = 0; l < nk-nq; ++l){
      s[d+outer] += a1*k[l][i][outer];
    }
    __syncthreads();
    for(int l = 0; l < nq; ++l){
      s[d+outer] += a1*k[nk-nq+l][i][outer];
      s[outer] = q[l][i][outer];
      __syncthreads();
      if(outer == 0){
        t = 0;
        for(int r = 0; r < d; ++r) t += s[r]*s[d+r];
        o[l][i][d] += t;
      }
    }

    // calc quad denum
    if(p == 2){
      for(int rr = 0; rr < d/sz; ++rr){
        for(int r = 0; r < sz; ++r) tr[r]= 0;
        for(int l = 0; l < nk-nq;  ++l){
          s[outer] = k[l][i][outer];
          __syncthreads();
          loc1 = rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[outer]*s[loc1+r];
          }
        }
        __syncthreads();
        for(int l = 0; l < nq; ++l){
          s[outer] = k[nk-nq+l][i][outer];
          __syncthreads();
          loc1 = rr*sz;
          for(int r = 0; r < sz; ++r){
            tr[r] += s[outer]*s[loc1+r];
          }
          s[d+outer] = 0;
          s[outer] = q[l][i][outer];
          __syncthreads();
          loc2 = rr*sz;
          for(int r = 0; r < sz; ++r){
            s[d+outer] += tr[r]*s[outer]*s[loc2+r];
          }
          o[l][i][outer] += s[d+outer];
        }
        __syncthreads();
        for(int l = 0; l < nq; ++l){
          t = 0;
          s[outer] = o[l][i][outer];
          __syncthreads();
          if(outer == 0){
            for(int r = 0; r < d; ++r) t += s[r];
            o[l][i][d] += a2*t;
          }
        }
      }
    }

    // calc cons
    t = 0;
    for(int l = 0; l < nk-nq;  ++l){
      t += v[l][i][outer];
    }
    for(int l = 0; l < nq;  ++l){
      t += v[nk-nq+l][i][outer];
      o[l][i][outer] = a0*t;
    }

    // calc lin
    for(int m = 0; m < d; ++m){
      t = 0;
      for(int l = 0; l < nk-nq;  ++l){
        t += k[l][i][m]*v[l][i][outer];
      }
      for(int l = 0; l < nq;  ++l){
        t += k[nk-nq+l][i][m]*v[nk-nq+l][i][outer];
        o[l][i][outer] += a1*t*q[l][i][m];
      }
    }

    // calc quad
    if(p == 2){
      for(int m = 0; m < d; ++m){
        for(int rr = 0; rr < d/sz; ++rr){
          for(int r = 0; r < sz; ++r) tr[r]= 0;
          for(int l = 0; l < nk-nq;  ++l){
            s[d+outer] = k[l][i][m]*k[l][i][outer];
            tv = v[l][i][outer];
            __syncthreads();
            loc1 = d+rr*sz;
            for(int r = 0; r < sz; ++r){
              tr[r] += s[loc1+r]*tv;
            }      
          }
          for(int l = 0; l < nq;  ++l){
            s[outer] = q[l][i][m]*q[l][i][outer];
            s[d+outer] = k[nk-nq+l][i][m]*k[nk-nq+l][i][outer];
            tv = v[nk-nq+l][i][outer];
            __syncthreads();
            t = 0;
            loc1 = d+rr*sz;
            loc2 = rr*sz;
            for(int r = 0; r < sz; ++r){
              tr[r] += s[loc1+r]*tv;
              t += tr[r]*s[loc2+r];
            }      
            o[l][i][outer] += a2*t;
          }
        }
      }
    }

    for(int l = 0; l < nq;  ++l) o[l][i][outer] /= o[l][i][d];
  }
}

__global__
void apply_rpe_and_temp(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> k, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> rpe_matrix, int bh, int nk, int d, float temperature){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nk; ++l){
      k[l][i][m] /= temperature;
      // k[l][i][m] += rpe_matrix[(l+nk)%(2*nk-1)][m];
    }
  }
}

__global__
void calc_norms(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, int bh, int n, int d, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  const int l = blockIdx.y;
  float t;
  int i;
  if(l < n && ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    t = 0;
    for(int m = 0; m < d; m++){
      t += a[l][i][m]*a[l][i][m];
    }
    norms[l][i] = t;
  }
}

__global__
void find_max(torch::PackedTensorAccessor32<float,2,torch::RestrictPtrTraits> norms, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int th){
  const int ii = threadIdx.x;
  const int j = blockIdx.x;
  float t = 0;
  int i;
  if(ii < th && j < ((bh-1)/th + 1)){
    i = j*th + ii;
    for(int l = 0; l < n; ++l){
      t = max(t,norms[l][i]);
    }
    maxes[i] = t;
  }
}

__global__
void apply_norm(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,1,torch::RestrictPtrTraits> maxes, int bh, int n, int d, float lim){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  float t;
  if(m < d && i < bh){
    t = maxes[i];
    if(t < 0.1) t = 0.1;
    for(int l = 0; l < n; ++l){
      a[l][i][m]*= lim/t;
    }
  }
}

__global__
void apply_dropout(torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> o, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> drop_noise, float scale, int bh, int nq, int d){
  const int m = threadIdx.x;
  const int i = blockIdx.x;
  if(m < d && i < bh){
    for(int l = 0; l < nq; ++l){
      o[l][i][m] *= (1+scale*drop_noise[l][i][m]);
    }
  }
}

// __global__
// void apply_permute(torch::PackedTensorAccessor32<float,4,torch::RestrictPtrTraits> a, torch::PackedTensorAccessor32<float,3,torch::RestrictPtrTraits> a_p, int b, int h, int n, int d, int dir){
//   const int m = threadIdx.x;
//   const int j = blockIdx.x;
//   const int i = blockIdx.y;
//   if(m < d && i < b && j < h){
//     for(int l = 0; l < n; ++l){
//       if(dir == 0) a_p[l][i*h+j][m] = a[i][l][j][m];
//       else a[i][l][j][m] = a_p[l][i*h+j][m];
//     }
//   }
// }

} // namespace

torch::Tensor forward_cuda(
    torch::Tensor q,
    torch::Tensor k,
    torch::Tensor v,
    // torch::Tensor q_old,
    // torch::Tensor k_old,
    // torch::Tensor v_old,
    // torch::Tensor drop_noise_old,
    torch::Tensor drop_noise,
    torch::Tensor rpe_matrix,
    bool mask,
    float dropout,
    bool normalize,
    float temperature,
    float a0,
    float a1,
    float a2,
    float lim,
    int p){
    // q: (nq,bh,d)
    // k: (nk,bh,d)
    // v: (nk,bh,d)

  // const auto nq = q_old.size(1);
  // const auto nk = k_old.size(1);
  // const auto b = q_old.size(0);
  // const auto h = q_old.size(2);
  // const auto d = q_old.size(3);
  // const auto bh = b*h;

  const auto nq = q.size(0);
  const auto nk = k.size(0);
  const auto bh = q.size(1);
  const auto d = q.size(2);

  const int threads = d; // threads = 256
  const int blocks = bh;
  
  auto opts =  torch::TensorOptions().dtype(torch::kFloat32).layout(torch::kStrided).device(torch::kCUDA, 0);
  // auto q = torch::zeros({nq,bh,d},opts);
  // auto k = torch::zeros({nk,bh,d},opts);
  // auto v = torch::zeros({nk,bh,d},opts);
  // auto drop_noise = torch::zeros({nq,bh,d},opts);
  auto o = torch::zeros({nq,bh,d+1},opts);
  // auto out = torch::zeros({b,nq,h,d+1},opts);
  auto qnorms = torch::zeros({nq,bh},opts);
  auto knorms = torch::zeros({nk,bh},opts);
  auto qmaxes = torch::zeros({bh},opts);
  auto kmaxes = torch::zeros({bh},opts);


  // apply_permute<<<dim3(h,b),threads>>>(q_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(k_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(v_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nk,d,0);
  // apply_permute<<<dim3(h,b),threads>>>(drop_noise_old.packed_accessor32<float,4,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d,0);

  apply_rpe_and_temp<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),rpe_matrix.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d,temperature);
  cudaDeviceSynchronize();
  if(normalize){
    const long th_lim = 1024;
    int th = min(th_lim, bh);
    calc_norms<<<dim3((bh-1)/th + 1, nq),th>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nq,d,th);
    calc_norms<<<dim3((bh-1)/th + 1, nk),th>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),bh,nk,d,th);
    find_max<<<(bh-1)/th + 1,th>>>(qnorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,th);
    find_max<<<(bh-1)/th + 1,th>>>(knorms.packed_accessor32<float,2,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,th);
    apply_norm<<<blocks,threads>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),qmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nq,d,lim);
    apply_norm<<<blocks,threads>>>(k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),kmaxes.packed_accessor32<float,1,torch::RestrictPtrTraits>(),bh,nk,d,lim);
  }

  if(mask){
    calc_masked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2,p);
  }
  else{
    calc_unmasked<<<blocks,threads,2*(d)*sizeof(float)>>>(q.packed_accessor32<float,3,torch::RestrictPtrTraits>(),k.packed_accessor32<float,3,torch::RestrictPtrTraits>(),v.packed_accessor32<float,3,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),bh,nq,nk,d,a0,a1,a2,p);
  }

  cudaDeviceSynchronize();
  apply_dropout<<<blocks,threads>>>(o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),drop_noise.packed_accessor32<float,3,torch::RestrictPtrTraits>(),0.0,bh,nq,d);
  cudaDeviceSynchronize();

  // apply_permute<<<dim3(h,b),threads+1>>>(out.packed_accessor32<float,4,torch::RestrictPtrTraits>(),o.packed_accessor32<float,3,torch::RestrictPtrTraits>(),b,h,nq,d+1,1);

  // delete q;
  // delete k;
  // delete v;
  // delete drop_noise;
  // delete o;

  return o;
  // return out;
}