#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <stdlib.h>

float dot_product(const float *A, const float *B, size_t size){
    float sum = 0;
    for (size_t i = 0; i < size; i++){
        sum += A[i]*B[i];
    }
    return sum;
}

float quantize(const float *buffer, int8_t* q_out, size_t size){
    float max = 0.0;
    for(size_t i = 0; i < size; i++){
        float magnitude = fabsf(buffer[i]);
        if(magnitude > max){
            max = magnitude;
        }
    }
    float S = max/127.0f;
    for(size_t i = 0; i <size; i++){
        float value = 0.00;
        if(S != 0.00){
            value = roundf(buffer[i]/S);
        }
        int8_t clamped = (int8_t)fmaxf(-128.0f ,fminf(value,127.0f));
        q_out[i] = clamped;
    }
    return S;
}

int main(void){
    float A[] = {0.25, -1.5, 3.2, -0.75, 2.0, -0.1, 0.85, -2.4};
    float B[] = {1.2, -0.4, 0.5, 2.1, -1.0, 3.5, 0.2, -0.8};
    size_t sizeA = sizeof(A) / sizeof(A[0]);
    size_t sizeB = sizeof(B) / sizeof(B[0]);
    float product = -1;
    if(sizeA == sizeB){
        product = dot_product(A,B,sizeA);
    }
    printf("Rounded Dot Product: %.6f\n", product);

    int8_t qA[sizeof(A) / sizeof(A[0])];
    int8_t qB[sizeof(B) / sizeof(B[0])];
    float sA = quantize(A,qA,sizeA);
    float sB = quantize(B,qB,sizeB);
    printf("sA: %f\n", sA);
    printf("qA: ");
    for(size_t i = 0;i < sizeA;i++){
        printf("%d, ", qA[i]);

    }
    printf("\n");
    printf("sB: %f\n", sB);
    printf("qB: ");
    for(size_t i = 0;i < sizeB;i++){
        printf("%d, ", qB[i]);

    }
    printf("\n");
    return 0;
}