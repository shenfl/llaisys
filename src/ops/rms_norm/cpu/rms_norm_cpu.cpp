#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, float eps, size_t batch, size_t dim) {
    // RMS Normalization: Y_i = W_i * X_i / sqrt(mean(X^2) + eps)
    // 对每一行进行归一化
    
    for (size_t b = 0; b < batch; b++) {
        const T *in_row = in + b * dim;
        T *out_row = out + b * dim;
        
        // 步骤1: 计算平方和的均值 mean(X^2)
        float sum_squares = 0.0f;
        for (size_t d = 0; d < dim; d++) {
            float val;
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                val = llaisys::utils::cast<float>(in_row[d]);
            } else {
                val = in_row[d];
            }
            sum_squares += val * val;
        }
        float mean_squares = sum_squares / static_cast<float>(dim);
        
        // 步骤2: 计算 RMS = sqrt(mean(X^2) + eps)
        float rms = std::sqrt(mean_squares + eps);
        
        // 步骤3: 归一化并应用权重 Y_i = W_i * X_i / RMS
        for (size_t d = 0; d < dim; d++) {
            float in_val, w_val;
            
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                in_val = llaisys::utils::cast<float>(in_row[d]);
                w_val = llaisys::utils::cast<float>(weight[d]);
            } else {
                in_val = in_row[d];
                w_val = weight[d];
            }
            
            float normalized = in_val / rms;
            float result = w_val * normalized;
            
            if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                out_row[d] = llaisys::utils::cast<T>(result);
            } else {
                out_row[d] = result;
            }
        }
    }
}

namespace llaisys::ops::cpu {
void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, float eps,
              llaisysDataType_t type, size_t batch, size_t dim) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out),
                        reinterpret_cast<const float *>(in),
                        reinterpret_cast<const float *>(weight),
                        eps, batch, dim);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out),
                        reinterpret_cast<const llaisys::bf16_t *>(in),
                        reinterpret_cast<const llaisys::bf16_t *>(weight),
                        eps, batch, dim);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out),
                        reinterpret_cast<const llaisys::fp16_t *>(in),
                        reinterpret_cast<const llaisys::fp16_t *>(weight),
                        eps, batch, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
