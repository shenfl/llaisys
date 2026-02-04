#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta,
           size_t seq_len, size_t n_heads, size_t head_dim) {
    // RoPE: Rotary Position Embedding
    // 输入形状: [seq_len, n_heads, head_dim]
    // head_dim 必须是偶数，前半部分和后半部分配对进行旋转
    
    size_t half_dim = head_dim / 2;
    
    // 对每个序列位置
    for (size_t s = 0; s < seq_len; s++) {
        int64_t pos = pos_ids[s];
        
        // 对每个头
        for (size_t h = 0; h < n_heads; h++) {
            size_t base_idx = s * n_heads * head_dim + h * head_dim;
            
            // 对每个维度对 (前半部分和后半部分)
            for (size_t j = 0; j < half_dim; j++) {
                // 计算角度: φ = pos / theta^(2j/d)
                float exponent = (2.0f * static_cast<float>(j)) / static_cast<float>(head_dim);
                float freq = static_cast<float>(pos) / std::pow(theta, exponent);
                
                float cos_val = std::cos(freq);
                float sin_val = std::sin(freq);
                
                // 获取 a 和 b (前半部分和后半部分)
                float a_val, b_val;
                
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    a_val = llaisys::utils::cast<float>(in[base_idx + j]);
                    b_val = llaisys::utils::cast<float>(in[base_idx + half_dim + j]);
                } else {
                    a_val = in[base_idx + j];
                    b_val = in[base_idx + half_dim + j];
                }
                
                // 应用旋转
                // a' = a * cos - b * sin
                // b' = b * cos + a * sin
                float a_new = a_val * cos_val - b_val * sin_val;
                float b_new = b_val * cos_val + a_val * sin_val;
                
                // 写回输出
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    out[base_idx + j] = llaisys::utils::cast<T>(a_new);
                    out[base_idx + half_dim + j] = llaisys::utils::cast<T>(b_new);
                } else {
                    out[base_idx + j] = a_new;
                    out[base_idx + half_dim + j] = b_new;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, float theta,
          llaisysDataType_t type, size_t seq_len, size_t n_heads, size_t head_dim) {
    // pos_ids 始终是 int64_t 类型
    const int64_t *pos_ptr = reinterpret_cast<const int64_t *>(pos_ids);
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out),
                    reinterpret_cast<const float *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out),
                    reinterpret_cast<const llaisys::bf16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out),
                    reinterpret_cast<const llaisys::fp16_t *>(in),
                    pos_ptr, theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
