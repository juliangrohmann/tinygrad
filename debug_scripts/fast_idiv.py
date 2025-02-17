import math
from tqdm import tqdm

N = 32

def fast_idiv_usig(n, d):
  l = math.ceil(math.log2(abs(d)))
  m_p = math.floor(2**N * (2**l - d) / d) + 1
  sh_1 = min(l, 1)
  sh_2 = max(l - 1, 0)
  t_1 = (m_p * n) >> N
  return (t_1 + ((n - t_1) >> sh_1)) >> sh_2


for i in tqdm(range(-5000, 5000)):
  for j in range(1, abs(i)):
    if j != 0:
      q = fast_idiv_usig(i, j)
      assert q == i // j, f"mismatch! {i} // {j}: quotient={q}, expected={i // j}"

# print(fast_idiv_usig(6, 5))