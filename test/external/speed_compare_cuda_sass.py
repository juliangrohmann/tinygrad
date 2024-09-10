import itertools
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

  single = getenv("NUM", -1)
  if single != -1: ast_strs = ast_strs[single:single+1]

  average_tm_cuda, average_tm_sass = 0, 0
  for num,ast in enumerate(ast_strs):
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
