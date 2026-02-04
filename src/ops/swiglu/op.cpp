#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/swiglu_cpu.hpp"

namespace llaisys::ops {
void swiglu(tensor_t out, tensor_t gate, tensor_t up) {
    CHECK_SAME_DEVICE(out, gate, up);
    
    // 验证维度
    ASSERT(out->ndim() == 2, "swiglu: out must be a 2D tensor");
    ASSERT(gate->ndim() == 2, "swiglu: gate must be a 2D tensor");
    ASSERT(up->ndim() == 2, "swiglu: up must be a 2D tensor");
    
    // 验证数据类型相同
    CHECK_SAME_DTYPE(out->dtype(), gate->dtype());
    CHECK_SAME_DTYPE(out->dtype(), up->dtype());
    
    // 验证形状相同
    ASSERT(out->shape()[0] == gate->shape()[0] && out->shape()[0] == up->shape()[0],
           "swiglu: all tensors must have the same shape[0]");
    ASSERT(out->shape()[1] == gate->shape()[1] && out->shape()[1] == up->shape()[1],
           "swiglu: all tensors must have the same shape[1]");
    
    size_t numel = out->numel();

    // CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                          out->dtype(), numel);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::swiglu(out->data(), gate->data(), up->data(),
                          out->dtype(), numel);
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
