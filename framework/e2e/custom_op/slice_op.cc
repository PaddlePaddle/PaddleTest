#include "paddle/extension.h"
#include <vector>
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")

std::vector<paddle::Tensor> SliceCPUForward(const paddle::Tensor& x, int a, int b) {
    // forward
    CHECK_INPUT(x);
    auto out = paddle::Tensor(paddle::PlaceType::kCPU);
    out.reshape(x.shape());
    auto* x_data = x.data<float>();
    auto* out_data = out.mutable_data<float>(x.place());
    auto x_numel = x.size();
        for (int i = 0; i < x_numel; ++i) {
            out_data[i] = x_data[i];
        }

    return {out.slice(a, b)};
}

std::vector<std::vector<int64_t>> SliceInferShape(std::vector<int64_t> x_shape) {
    // infer shape
    return {x_shape};
}

std::vector<paddle::DataType> SliceInferDtype(paddle::DataType x_dtype) {
    // infer type
    return {x_dtype};
}

PD_BUILD_OP(slice_test)
    .Inputs({"X"})
    .Outputs({"Out"})
    .Attrs({"a: int", "b: int"})
    .SetKernelFn(PD_KERNEL(SliceCPUForward))
    .SetInferShapeFn(PD_INFER_SHAPE(SliceInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(SliceInferDtype));
