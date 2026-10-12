#include <stdio.h>
float dot_product(const float *A, const float *B, size_t size){
    float sum = 0;
    for (size_t i = 0; i < size; i++){
        sum += A[i]*B[i];
    }
    return sum;
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
    return 0;
}