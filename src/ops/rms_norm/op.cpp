#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    
    // 验证维度
    ASSERT(in->ndim() == 2, "rms_norm: in must be a 2D tensor");
    ASSERT(out->ndim() == 2, "rms_norm: out must be a 2D tensor");
    ASSERT(weight->ndim() == 1, "rms_norm: weight must be a 1D tensor");
    
    // 验证数据类型相同
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    
    // 验证形状匹配
    size_t batch = in->shape()[0];
    size_t dim = in->shape()[1];
    
    ASSERT(out->shape()[0] == batch, "rms_norm: out shape[0] must match in shape[0]");
    ASSERT(out->shape()[1] == dim, "rms_norm: out shape[1] must match in shape[1]");
    ASSERT(weight->shape()[0] == dim, "rms_norm: weight shape[0] must match in shape[1]");

    // CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps,
                            out->dtype(), batch, dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), eps,
                            out->dtype(), batch, dim);
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
