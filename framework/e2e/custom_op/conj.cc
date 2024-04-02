#include <iostream>
#include <vector>

#include "paddle/extension.h"

#define CHECK_INPUT(x) \
    PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

template <typename data_t>
using EnableComplex = typename std::enable_if<
      std::is_same<data_t, paddle::complex64>::value ||
      std::is_same<data_t, paddle::complex128>::value>::type;

template <typename data_t>
using DisableComplex = typename std::enable_if<
      !std::is_same<data_t, paddle::complex64>::value &&
      !std::is_same<data_t, paddle::complex128>::value>::type;

template <typename data_t, typename Enable = void>
struct ConjFunctor;

template <typename data_t>
struct ConjFunctor<data_t, EnableComplex<data_t>> {
    ConjFunctor(const data_t* input, int64_t numel, data_t* output)
        : input_(input), numel_(numel), output_(output) {}

    void operator()(size_t idx) const {
      output_[idx] = data_t(input_[idx].real, -input_[idx].imag);
    }

    const data_t* input_;
    int64_t numel_;
    data_t* output_;
};

template <typename data_t>
struct ConjFunctor<data_t, DisableComplex<data_t>> {
    ConjFunctor(const data_t* input, int64_t numel, data_t* output)
        : input_(input), numel_(numel), output_(output) {}

    void operator()(size_t idx) const { output_[idx] = input_[idx]; }

    const data_t* input_;
    int64_t numel_;
    data_t* output_;
};

template <typename data_t>
void ConjCPUKernel(const data_t* x_data, int64_t numel, data_t* out_data) {
    ConjFunctor<data_t> conj(x_data, numel, out_data);
    for (int64_t i = 0; i < numel; ++i) {
      conj(i);
    }
}

std::vector<paddle::Tensor> ConjFunction(const paddle::Tensor& x) {
    CHECK_INPUT(x);

    paddle::Tensor out(x.place());
    out.reshape(x.shape());

    PD_DISPATCH_FLOATING_AND_COMPLEX_TYPES(
        x.type(), "ConjCPUKernel", ([&] {
          ConjCPUKernel<data_t>(
              x.data<data_t>(), x.size(), out.mutable_data<data_t>());
        }));

    return {out};
}

PD_BUILD_OP(custom_conj)
      .Inputs({"X"})
      .Outputs({"Out"})
      .SetKernelFn(PD_KERNEL(ConjFunction));

PD_BUILD_GRAD_OP(custom_conj)
      .Inputs({paddle::Grad("Out")})
      .Outputs({paddle::Grad("X")})
      .SetKernelFn(PD_KERNEL(ConjFunction));
