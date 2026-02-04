#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t idx_size, size_t embd_dim) {
    // 遍历所有索引
    for (size_t i = 0; i < idx_size; i++) {
        int64_t row_idx = index[i];
        
        // 计算源地址和目标地址
        const T *src_row = weight + row_idx * embd_dim;
        T *dst_row = out + i * embd_dim;
        
        // 复制整行数据
        std::memcpy(dst_row, src_row, embd_dim * sizeof(T));
    }
}

namespace llaisys::ops::cpu {
void embedding(std::byte *out, const std::byte *index, const std::byte *weight, 
               llaisysDataType_t type, size_t idx_size, size_t embd_dim) {
    // index 始终是 int64_t 类型
    const int64_t *idx_ptr = reinterpret_cast<const int64_t *>(index);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), idx_ptr,
                         reinterpret_cast<const float *>(weight), idx_size, embd_dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), idx_ptr,
                         reinterpret_cast<const llaisys::bf16_t *>(weight), idx_size, embd_dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), idx_ptr,
                         reinterpret_cast<const llaisys::fp16_t *>(weight), idx_size, embd_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
