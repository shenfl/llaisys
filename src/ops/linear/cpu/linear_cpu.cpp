#include "linear_cpu.hpp"

#include "../../../utils.hpp"

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias,
             size_t batch, size_t in_features, size_t out_features) {
    // Y = X * W^T + b
    // X: [batch, in_features]
    // W: [out_features, in_features] (注意：权重未转置)
    // Y: [batch, out_features]
    // b: [out_features] (可选)
    
    for (size_t b = 0; b < batch; b++) {
        for (size_t o = 0; o < out_features; o++) {
            float sum = 0.0f;
            
            // 计算点积: out[b, o] = sum(in[b, :] * weight[o, :])
            for (size_t i = 0; i < in_features; i++) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    float in_val = llaisys::utils::cast<float>(in[b * in_features + i]);
                    float w_val = llaisys::utils::cast<float>(weight[o * in_features + i]);
                    sum += in_val * w_val;
                } else {
                    sum += in[b * in_features + i] * weight[o * in_features + i];
                }
            }
            
            // 加偏置(如果有)
            if (bias != nullptr) {
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    sum += llaisys::utils::cast<float>(bias[o]);
                } else {
                    sum += bias[o];
                }
            }
            
            // 存储结果
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out[b * out_features + o] = llaisys::utils::cast<T>(sum);
            } else {
                out[b * out_features + o] = sum;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t batch, size_t in_features, size_t out_features) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), 
                      reinterpret_cast<const float *>(in),
                      reinterpret_cast<const float *>(weight),
                      bias ? reinterpret_cast<const float *>(bias) : nullptr,
                      batch, in_features, out_features);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out),
                      reinterpret_cast<const llaisys::bf16_t *>(in),
                      reinterpret_cast<const llaisys::bf16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::bf16_t *>(bias) : nullptr,
                      batch, in_features, out_features);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out),
                      reinterpret_cast<const llaisys::fp16_t *>(in),
                      reinterpret_cast<const llaisys::fp16_t *>(weight),
                      bias ? reinterpret_cast<const llaisys::fp16_t *>(bias) : nullptr,
                      batch, in_features, out_features);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
