import time
import numpy as np
import micro_llm_engine as mle
from concurrent.futures import ThreadPoolExecutor

N = 30_000_000 # 30 million elements ~ 120 MB in FP32
A1 = np.random.randn(N).astype(np.float32)
B1 = np.random.randn(N).astype(np.float32)
A2 = np.random.randn(N).astype(np.float32)
B2 = np.random.randn(N).astype(np.float32)

t0 = time.perf_counter()
res1 = mle.fp32_dot_product(A1,B1)
res2 = mle.fp32_dot_product(A2,B2)
t_seq = time.perf_counter() - t0

print(res1,res2,t_seq)

t0 = time.perf_counter()

with ThreadPoolExecutor(max_workers=2) as executor:
    future1 = executor.submit(mle.fp32_dot_product,A1,B1)
    future2 = executor.submit(mle.fp32_dot_product,A2,B2)

    res1 = future1.result()
    res2 = future2.result()

t_par = time.perf_counter() - t0
print(res1,res2,t_par)

print(f"speedup: {t_seq/t_par:.6f}\n")