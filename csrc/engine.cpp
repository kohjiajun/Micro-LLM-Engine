#include "engine.h"
#include <stdint.h>
#include <math.h>
#include <omp.h>
#include <immintrin.h>

float fp32_dot_product(const float *A, const float *B, size_t size){
    float sum = 0;
    for (size_t i = 0; i < size; i++){
        sum += A[i]*B[i];
    }
    return sum;
}

float fp32_to_int8_quantization(const float *input, int8_t* q_out, size_t size){
    float max = 0.0;
    for(size_t i = 0; i < size; i++){
        float magnitude = fabsf(input[i]);
        if(magnitude > max){
            max = magnitude;
        }
    }
    float S = max/127.0f;
    for(size_t i = 0; i <size; i++){
        float value = 0.00;
        if(S != 0.00){
            value = roundf(input[i]/S);
        }
        int8_t clamped = (int8_t)fmaxf(-128.0f ,fminf(value,127.0f));
        q_out[i] = clamped;
    }
    return S;
}

int32_t int8_dot_product(const int8_t *A, const int8_t *B, size_t size){
    int32_t sum = 0;
    for(size_t i = 0; i < size; i++){
        sum += A[i]*B[i];
    }
    return sum;
}

void fp32_gemm(const float *A, const float *B, float *res, size_t M, size_t N, size_t K){
    #pragma omp parallel for
    for (int64_t i = 0; i < M; i++){
        for (size_t j = 0; j < N; j++) {
        res[i * N + j] = 0.0f;
        }
        for(size_t k = 0; k < K; k++){
            float a_ik = A[i * K + k]; //register hoist
            __m256 a_vec = _mm256_set1_ps(a_ik);
            size_t j = 0;
            for(; j + 8 <= N; j += 8){
                __m256 b_vec = _mm256_loadu_ps(&B[k * N + j]);
                __m256 res_vec = _mm256_loadu_ps(&res[i * N +j]);
                res_vec = _mm256_fmadd_ps(a_vec,b_vec,res_vec);
                _mm256_storeu_ps(&res[i*N+j],res_vec);
            }
            for(; j < N; j++){
                res[i*N+j] += a_ik * B[k * N + j];
            }
        }
    }
}