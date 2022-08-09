#include "paddle/extension.h"
#include <vector>
#define CHECK_INPUT(x) PD_CHECK(x.place() == paddle::PlaceType::kCPU, #x " must be a CPU Tensor.")
std::vector<paddle::Tensor> AddGPUForward(const paddle::Tensor& x, const paddle::Tensor& y);
std::vector<paddle::Tensor> AddGPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& y,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out);
std::vector<paddle::Tensor> AddCPUForward(const paddle::Tensor& x, const paddle::Tensor& y) {
    // forward
    if (x.place() == paddle::PlaceType::kCPU) {
        CHECK_INPUT(x);
        CHECK_INPUT(y);
        auto out = paddle::Tensor(paddle::PlaceType::kCPU);
        out.reshape(x.shape());

        auto x_numel = x.size();
        auto* x_data = x.data<float>();
        auto* y_data = y.data<float>();
        auto* out_data = out.mutable_data<float>(x.place());

        for (int i = 0; i < x_numel; ++i) {
            out_data[i] = x_data[i] + y_data[i];
        }
        return {out};
    } else if (x.place() == paddle::PlaceType::kGPU) {
        return AddGPUForward(x, y);
    } else {
        PD_THROW("Not implemented.");
    }
}

std::vector<paddle::Tensor> AddCPUBackward(const paddle::Tensor& x,
                                            const paddle::Tensor& y,
                                            const paddle::Tensor& out,
                                            const paddle::Tensor& grad_out) {
    // backward
    if (x.place() == paddle::PlaceType::kCPU) {
        CHECK_INPUT(x);
        CHECK_INPUT(out);
        CHECK_INPUT(grad_out);
        auto grad_x = paddle::Tensor(paddle::PlaceType::kCPU);
        grad_x.reshape(x.shape());
        auto grad_y = paddle::Tensor(paddle::PlaceType::kCPU);
        grad_y.reshape(y.shape());

        auto size = x.size();
        auto* out_data = out.data<float>();
        auto* grad_out_data = grad_out.data<float>();
        auto* grad_x_data = grad_x.mutable_data<float>(x.place());
        auto* grad_y_data = grad_y.mutable_data<float>(y.place());

        for (int i = 0; i < size; ++i) {
            grad_x_data[i] =
                grad_out_data[i];
        }
        for (int i = 0; i < size; ++i) {
            grad_y_data[i] =
                grad_out_data[i];
            // VLOG(3) << "num" << grad_out_data[i];
        }
        return {grad_x, grad_y};
    } else if (x.place() == paddle::PlaceType::kGPU) {
        return AddGPUBackward(x, y, out, grad_out);
    } else {
        PD_THROW("Not implemented.");
    }
}


std::vector<std::vector<int64_t>> AddInferShape(std::vector<int64_t> x_shape, std::vector<int64_t> y_shape) {
    // infer shape
    return {x_shape, y_shape};
}

std::vector<paddle::DataType> AddInferDtype(paddle::DataType x_dtype, paddle::DataType y_dtype) {
    // infer type
    return {x_dtype, y_dtype};
}

PD_BUILD_OP(add_test)
    .Inputs({"X", "Y"})
    .Outputs({"Out"})
    .SetKernelFn(PD_KERNEL(AddCPUForward))
    .SetInferShapeFn(PD_INFER_SHAPE(AddInferShape))
    .SetInferDtypeFn(PD_INFER_DTYPE(AddInferDtype));

PD_BUILD_GRAD_OP(add_test)
    .Inputs({"X", "Y", "Out", paddle::Grad("Out")})
    .Outputs({paddle::Grad("X"), paddle::Grad("Y")})
    .SetKernelFn(PD_KERNEL(AddCPUBackward));
