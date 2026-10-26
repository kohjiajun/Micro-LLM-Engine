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

void int8_gemm(const int8_t *A, const int8_t *B, int32_t *res, size_t M, size_t N, size_t K){
    #pragma omp parallel for
    for(int64_t i = 0; i < M; i ++){
        for(size_t j = 0 ; j < N; j++){
            res[i*N+j] = 0;
        }
        for(size_t k = 0; k < K; k++){
            int8_t a_ik = A[i*K+k];
            __m256i a_vec = _mm256_set1_epi32(a_ik);
            size_t j = 0;
            for(; j+8 <= N; j += 8){
                __m128i b_raw = _mm_loadu_si64(&B[k*N+j]); //only load 64bits (8x8 int8) because only 8(256/32) int32 values can be loaded from res
                __m256i b_vec = _mm256_cvtepi8_epi32(b_raw);
                __m256i res_vec = _mm256_loadu_si256(reinterpret_cast<const __m256i*>(&res[i * N + j]));
                __m256i prod = _mm256_mullo_epi32(a_vec,b_vec);
                res_vec = _mm256_add_epi32(res_vec,prod);
                _mm256_storeu_si256(reinterpret_cast<__m256i*>(&res[i * N + j]), res_vec);
            }
            for(; j < N; j ++){
                res[i*N+j] += a_ik * B[k*N+j];
            }
        }
    }
}