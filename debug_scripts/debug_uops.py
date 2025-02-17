import math
from tinygrad.tensor import Tensor
from tinygrad.dtype import dtypes
# a, b = -1000, 3
# at, bt = Tensor([a], dtype=dtypes.int), Tensor([b], dtype=dtypes.float)
# print((at // 2).numpy())

for i in range(-1000, 1000):
  for j in range(1, abs(i)):
    if j == 0: continue
    print(f"{i} // {j}")
    at = Tensor([i], dtype=dtypes.int)
    q = (at // j).item()
    func = math.ceil if i / j < 0 else math.floor
    assert q == int(func(i / j)), f"mismatch! {i} // {j}: quotient={q}, expected={i // j}"