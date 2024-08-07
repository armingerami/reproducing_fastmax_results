#include <iostream>
#include <vector>
#include <math.h>
#include <torch/extension.h>
#include <iostream>
using namespace std;


torch::Tensor forward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor drop_noise, torch::Tensor rpe_matrix, bool mask,float dropout, bool normalize, float temperature, float a0, float a1, float a2, float lim, int p);

vector<torch::Tensor> backward_cuda(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask, float a0, float a1, float a2, int p);

#define CHECK_CUDA(x) TORCH_CHECK(x.device().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor forwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor drop_noise, torch::Tensor rpe_matrix, bool mask, float dropout, bool normalize, float temperature, float a0, float a1, float a2, float lim, int p){

  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

  return forward_cuda(q, k, v, drop_noise, rpe_matrix,mask,dropout,normalize,temperature,a0,a1,a2,lim,p);
}

vector<torch::Tensor> backwardpass(torch::Tensor q, torch::Tensor k, torch::Tensor v, torch::Tensor o, torch::Tensor grad_output, bool mask, float a0, float a1, float a2, int p){
  CHECK_INPUT(q);
  CHECK_INPUT(k);
  CHECK_INPUT(v);

    return backward_cuda(q, k, v, o, grad_output, mask,a0,a1,a2,p);
}

PYBIND11_MODULE(fastmax_cuda, m) {
  m.def("forwardpass", &forwardpass, "forwardpass");
  m.def("backwardpass", &backwardpass, "backwardpass");
}
