#ifndef ENGINE_H
#define ENGINE_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

float fp32_dot_product(const float *A, const float *B, size_t size);
float fp32_to_int8_quantization(const float *input, int8_t* q_out, size_t size);
int32_t int8_dot_product(const int8_t *A, const int8_t *B, size_t size);

#ifdef __cplusplus
}
#endif

#endif