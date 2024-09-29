from tinygrad import Device
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, colorize_float
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_cuda import SASSCompiler, SASSRenderer, CUDACompiler

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for sass at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["CUDA"]
  sass = SASSRenderer(dev.arch)

  start, end = getenv("START", -1), getenv("END", len(ast_strs)) # NOTE: debug
  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]
  skip_idiv = [491, 922]
  skip_spill = [2983, 3721] # 233 spilled (ordered: 20)
  large = [8, 171, 282, 445, 583, 591, 824, 944, 1030, 1227, 1332, 1378, 1433, 1649, 1723, 1741, 2028, 2057, 2467, 2833, 3116, 3246, 3380, 3391, 3455, 3652, 4127, 4168]
  # ordered: 1.68
  # SU: 1.63x
  # SU + RP-reduction:
  # SU + RP-reduction + clustering:

  average_tm_cuda, average_tm_sass = 0, 0
  for num,ast in enumerate(ast_strs):
    if num in skip_idiv or num in skip_spill or (getenv("LARGE") and not num in large) or not start <= num < end: continue # NOTE: debug

    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    cuda_prg = CompiledRunner(lin.to_program())

    bufs = bufs_from_lin(lin)

    # sass compile
    dev.compiler = SASSCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=sass)
    lin.hand_coded_optimizations()
    lin.linearize()
    sass_prg = CompiledRunner(lin.to_program())

    # warmup
    try:
      cuda_prg(bufs, {}, wait=True)
    except RuntimeError:
      print("cuda failed ast:", num)
      continue
    sass_prg(bufs, {}, wait=True)

    tm_cuda, tm_sass = [], []
    for i in range(5):
      tm_cuda.append(cuda_prg(bufs, {}, wait=True))
      tm_sass.append(sass_prg(bufs, {}, wait=True))
    average_tm_cuda += min(tm_cuda)
    average_tm_sass += min(tm_sass)
    ratio = min(tm_sass) / min(tm_cuda)
    print(f"{average_tm_sass / average_tm_cuda:5.2f}x -- {num:4d} {colorize_float(ratio)}  {min(tm_sass) * 1e6:7.2f} us", lin.name)
