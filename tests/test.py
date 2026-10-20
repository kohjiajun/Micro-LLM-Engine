import numpy as np
import micro_llm_engine as mle

A = np.array([0.25, -1.5, 3.2, -0.75, 2.0, -0.1, 0.85, -2.4], dtype=np.float32)
B = np.array([1.2, -0.4, 0.5, 2.1, -1.0, 3.5, 0.2, -0.8], dtype=np.float32)
sA,qA = mle.fp32_to_int8_quantization(A)
sB,qB = mle.fp32_to_int8_quantization(B)
assert A.size == B.size

fp32_dp = mle.fp32_dot_product(A,B)
int32_dp = mle.int8_dot_product(qA,qB)
deQ_dp = int32_dp * (sA * sB)
abs_err = abs(fp32_dp - deQ_dp)
relative_err = abs_err / abs(fp32_dp)

print(f"Rounded Dot Product: {fp32_dp:.6f}")
print(f"sA: {sA:.6f}")
print("qA:", qA)

print(f"sB: {sB:.6f}")
print("qB:", qB)
print(
    f"Dequantized DP: {deQ_dp:.6f}\n"
    f"Absolute Error: {abs_err:.6f}\n"
    f"Relative Error {relative_err:.6f}"
)