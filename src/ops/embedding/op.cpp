#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    
    // 验证维度
    ASSERT(index->ndim() == 1, "embedding: index must be a 1D tensor");
    ASSERT(weight->ndim() == 2, "embedding: weight must be a 2D tensor");
    ASSERT(out->ndim() == 2, "embedding: out must be a 2D tensor");
    
    // 验证index是Int64类型
    ASSERT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be of type Int64");
    
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    
    // 验证形状匹配
    size_t idx_size = index->shape()[0];
    size_t embd_dim = weight->shape()[1];
    ASSERT(out->shape()[0] == idx_size, "embedding: out shape[0] must match index size");
    ASSERT(out->shape()[1] == embd_dim, "embedding: out shape[1] must match weight shape[1]");

    // 始终支持 CPU 计算
    if (out->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             weight->dtype(), idx_size, embd_dim);
    }

    llaisys::core::context().setDevice(out->deviceType(), out->deviceId());

    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), 
                             weight->dtype(), idx_size, embd_dim);
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
