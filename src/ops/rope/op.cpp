#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    
    // 验证维度
    ASSERT(in->ndim() == 3, "rope: in must be a 3D tensor [seq_len, n_heads, head_dim]");
    ASSERT(out->ndim() == 3, "rope: out must be a 3D tensor [seq_len, n_heads, head_dim]");
    ASSERT(pos_ids->ndim() == 1, "rope: pos_ids must be a 1D tensor [seq_len]");
    
    // 验证 pos_ids 是 Int64 类型
    ASSERT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be of type Int64");
    
    // 验证数据类型相同
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    
    // 验证形状匹配
    size_t seq_len = in->shape()[0];
    size_t n_heads = in->shape()[1];
    size_t head_dim = in->shape()[2];
    
    ASSERT(out->shape()[0] == seq_len, "rope: out shape[0] must match in shape[0]");
    ASSERT(out->shape()[1] == n_heads, "rope: out shape[1] must match in shape[1]");
    ASSERT(out->shape()[2] == head_dim, "rope: out shape[2] must match in shape[2]");
    ASSERT(pos_ids->shape()[0] == seq_len, "rope: pos_ids shape[0] must match seq_len");
    
    // head_dim 必须是偶数
    ASSERT(head_dim % 2 == 0, "rope: head_dim must be even");

    // CPU计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        in->dtype(), seq_len, n_heads, head_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), theta,
                        in->dtype(), seq_len, n_heads, head_dim);
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
