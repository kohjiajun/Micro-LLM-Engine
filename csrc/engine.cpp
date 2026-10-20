#include "engine.h"
#include <stdint.h>
#include <math.h>

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