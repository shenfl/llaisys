#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    if (bias) {
        CHECK_SAME_DEVICE(out, bias);
    }
    
    // 验证维度
    ASSERT(in->ndim() == 2, "linear: in must be a 2D tensor");
    ASSERT(weight->ndim() == 2, "linear: weight must be a 2D tensor");
    ASSERT(out->ndim() == 2, "linear: out must be a 2D tensor");
    if (bias) {
        ASSERT(bias->ndim() == 1, "linear: bias must be a 1D tensor");
    }
    
    // 验证数据类型相同
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    if (bias) {
        CHECK_SAME_DTYPE(out->dtype(), bias->dtype());
    }
    
    // 验证形状匹配
    size_t batch = in->shape()[0];
    size_t in_features = in->shape()[1];
    size_t out_features = weight->shape()[0];
    
    ASSERT(weight->shape()[1] == in_features, 
           "linear: weight shape[1] must match in shape[1]");
    ASSERT(out->shape()[0] == batch, 
           "linear: out shape[0] must match in shape[0]");
    ASSERT(out->shape()[1] == out_features, 
           "linear: out shape[1] must match weight shape[0]");
    if (bias) {
        ASSERT(bias->shape()[0] == out_features, 
               "linear: bias shape[0] must match weight shape[0]");
    }

    // CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch, in_features, out_features);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(), 
                          bias ? bias->data() : nullptr,
                          out->dtype(), batch, in_features, out_features);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        TO_BE_IMPLEMENTED();
        return;
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
