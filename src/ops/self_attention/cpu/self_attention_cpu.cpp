#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

template <typename T>
void self_attention_(T *attn_val, const T *q, const T *k, const T *v, float scale,
                    size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    // Self-Attention with causal mask
    // q: [qlen, nh, hd]
    // k: [kvlen, nkvh, hd]
    // v: [kvlen, nkvh, hd]
    // attn_val: [qlen, nh, hd]
    
    // 计算 head 重复次数（Grouped Query Attention）
    size_t head_repeat = nh / nkvh;
    
    // 对每个 query 位置
    for (size_t q_pos = 0; q_pos < qlen; q_pos++) {
        // 对每个 query head
        for (size_t h = 0; h < nh; h++) {
            // 确定对应的 kv head（GQA: 多个 q head 共享一个 kv head）
            size_t kv_h = h / head_repeat;
            
            // 临时存储注意力分数和权重
            std::vector<float> attn_scores(kvlen);
            
            // 步骤1: 计算 Q·K^T * scale
            for (size_t kv_pos = 0; kv_pos < kvlen; kv_pos++) {
                float score = 0.0f;
                
                // 点积: q[q_pos, h, :] · k[kv_pos, kv_h, :]
                for (size_t d = 0; d < hd; d++) {
                    size_t q_idx = q_pos * nh * hd + h * hd + d;
                    size_t k_idx = kv_pos * nkvh * hd + kv_h * hd + d;
                    
                    float q_val, k_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        q_val = llaisys::utils::cast<float>(q[q_idx]);
                        k_val = llaisys::utils::cast<float>(k[k_idx]);
                    } else {
                        q_val = q[q_idx];
                        k_val = k[k_idx];
                    }
                    
                    score += q_val * k_val;
                }
                
                // 应用 scale
                score *= scale;
                
                // 步骤2: 应用因果掩码（causal mask）
                // 当前 query 位置在整个序列中的绝对位置
                size_t abs_q_pos = (kvlen - qlen) + q_pos;
                
                // 只能看到不晚于当前位置的 key
                if (kv_pos > abs_q_pos) {
                    score = -std::numeric_limits<float>::infinity();
                }
                
                attn_scores[kv_pos] = score;
            }
            
            // 步骤3: Softmax (数值稳定版本)
            // 找到最大值用于数值稳定
            float max_score = -std::numeric_limits<float>::infinity();
            for (size_t kv_pos = 0; kv_pos < kvlen; kv_pos++) {
                if (std::isfinite(attn_scores[kv_pos])) {
                    max_score = std::max(max_score, attn_scores[kv_pos]);
                }
            }
            
            // 计算 exp 和 sum
            std::vector<float> attn_weights(kvlen);
            float sum_exp = 0.0f;
            for (size_t kv_pos = 0; kv_pos < kvlen; kv_pos++) {
                if (std::isfinite(attn_scores[kv_pos])) {
                    attn_weights[kv_pos] = std::exp(attn_scores[kv_pos] - max_score);
                    sum_exp += attn_weights[kv_pos];
                } else {
                    attn_weights[kv_pos] = 0.0f;
                }
            }
            
            // 归一化
            for (size_t kv_pos = 0; kv_pos < kvlen; kv_pos++) {
                attn_weights[kv_pos] /= sum_exp;
            }
            
            // 步骤4: 加权求和 attn_weights · V
            for (size_t d = 0; d < hd; d++) {
                float output = 0.0f;
                
                for (size_t kv_pos = 0; kv_pos < kvlen; kv_pos++) {
                    size_t v_idx = kv_pos * nkvh * hd + kv_h * hd + d;
                    
                    float v_val;
                    if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                        v_val = llaisys::utils::cast<float>(v[v_idx]);
                    } else {
                        v_val = v[v_idx];
                    }
                    
                    output += attn_weights[kv_pos] * v_val;
                }
                
                // 写回输出
                size_t out_idx = q_pos * nh * hd + h * hd + d;
                if constexpr (std::is_same_v<T, llaisys::bf16_t> || std::is_same_v<T, llaisys::fp16_t>) {
                    attn_val[out_idx] = llaisys::utils::cast<T>(output);
                } else {
                    attn_val[out_idx] = output;
                }
            }
        }
    }
}

namespace llaisys::ops::cpu {
void self_attention(std::byte *attn_val, const std::byte *q, const std::byte *k,
                   const std::byte *v, float scale, llaisysDataType_t type,
                   size_t qlen, size_t kvlen, size_t nh, size_t nkvh, size_t hd) {
    
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(attn_val),
                              reinterpret_cast<const float *>(q),
                              reinterpret_cast<const float *>(k),
                              reinterpret_cast<const float *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(attn_val),
                              reinterpret_cast<const llaisys::bf16_t *>(q),
                              reinterpret_cast<const llaisys::bf16_t *>(k),
                              reinterpret_cast<const llaisys::bf16_t *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(attn_val),
                              reinterpret_cast<const llaisys::fp16_t *>(q),
                              reinterpret_cast<const llaisys::fp16_t *>(k),
                              reinterpret_cast<const llaisys::fp16_t *>(v),
                              scale, qlen, kvlen, nh, nkvh, hd);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
