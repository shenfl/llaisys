#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    
    // 验证维度
    ASSERT(q->ndim() == 3, "self_attention: q must be a 3D tensor [qlen, nh, hd]");
    ASSERT(k->ndim() == 3, "self_attention: k must be a 3D tensor [kvlen, nkvh, hd]");
    ASSERT(v->ndim() == 3, "self_attention: v must be a 3D tensor [kvlen, nkvh, hd]");
    ASSERT(attn_val->ndim() == 3, "self_attention: attn_val must be a 3D tensor [qlen, nh, hd]");
    
    // 验证数据类型相同
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), k->dtype());
    CHECK_SAME_DTYPE(attn_val->dtype(), v->dtype());
    
    // 提取形状参数
    size_t qlen = q->shape()[0];
    size_t nh = q->shape()[1];      // number of query heads
    size_t hd = q->shape()[2];      // head dimension
    
    size_t kvlen = k->shape()[0];
    size_t nkvh = k->shape()[1];    // number of key/value heads
    
    // 验证形状匹配
    ASSERT(attn_val->shape()[0] == qlen, "self_attention: attn_val shape[0] must match q shape[0]");
    ASSERT(attn_val->shape()[1] == nh, "self_attention: attn_val shape[1] must match q shape[1]");
    ASSERT(attn_val->shape()[2] == hd, "self_attention: attn_val shape[2] must match q shape[2]");
    
    ASSERT(k->shape()[0] == kvlen, "self_attention: k and v must have the same shape[0]");
    ASSERT(v->shape()[0] == kvlen, "self_attention: k and v must have the same shape[0]");
    ASSERT(k->shape()[1] == nkvh, "self_attention: k shape[1] must match v shape[1]");
    ASSERT(v->shape()[1] == nkvh, "self_attention: k shape[1] must match v shape[1]");
    ASSERT(k->shape()[2] == hd, "self_attention: k shape[2] must match q shape[2]");
    ASSERT(v->shape()[2] == hd, "self_attention: v shape[2] must match q shape[2]");
    
    // 验证 Grouped Query Attention: nh 必须是 nkvh 的倍数
    ASSERT(nh % nkvh == 0, "self_attention: nh must be divisible by nkvh (Grouped Query Attention)");

    // CPU计算
    if (attn_val->deviceType() == LLAISYS_DEVICE_CPU) {
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, q->dtype(), qlen, kvlen, nh, nkvh, hd);
    }

    llaisys::core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());

    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(),
                                  scale, q->dtype(), qlen, kvlen, nh, nkvh, hd);
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
