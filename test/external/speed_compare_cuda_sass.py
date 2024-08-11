import json
from collections import defaultdict
from io import BytesIO
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, to_function_name
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.ops_cuda import SASSCompiler, CUDACompiler, CuAsmCompiler
from tinygrad.renderer.sass import SASSRenderer
from CuAsm.CuAsmLogger import CuAsmLogger
from CuAsm import CuAsmParser

def info(bufs_cuda, bufs_sass):
  ret = []
  for cuda, sass in zip(bufs_cuda, bufs_sass):
    b_info = {}
    cuda_vals, sass_vals = [np.frombuffer(buf.as_buffer(), dtype=buf.dtype.np) for buf in [cuda, sass]]
    mask = ~(np.isnan(cuda_vals) & np.isnan(sass_vals))
    b_info["cuda"], b_info["sass"] = cuda_vals.tolist(), sass_vals.tolist()
    if cuda_vals[mask].shape == sass_vals[mask].shape and len(cuda_vals[mask]) > 0:
      b_info["sum_diff"] = np.abs(np.sum(cuda_vals[mask] - sass_vals[mask])).item()
      b_info["max_diff"] = np.max(np.abs(cuda_vals[mask] - sass_vals[mask])).item()
    ret.append(b_info)
  return ret

def allclose(bufs_a, bufs_b):
  for a, b in zip(bufs_a, bufs_b):
    cuda_vals, sass_vals = [np.frombuffer(buf.as_buffer(), dtype=buf.dtype.np) for buf in [a, b]]
    mask = ~(np.isnan(cuda_vals) & np.isnan(sass_vals))
    if not np.allclose(cuda_vals[mask], sass_vals[mask]):
      return False
  return True

if __name__ == "__main__":
  CuAsmLogger.initLogger("/home/julian/projects/tinycorp/tinygrad/logs/debug.txt", file_level=15, stdout_level=40)

  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for ptx at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["CUDA"]
  sass = SASSRenderer(dev.arch)

  single = getenv("NUM", -1)
  debug_src = getenv("DEBUG_SRC", "")
  if single != -1: ast_strs = ast_strs[single:single+1]

  result = defaultdict(list)
  average_tm_cuda, average_tm_ptx = 0, 0
  impl = [5, 6, 7, 9, 10, 11, 12]
  for num,ast in list(enumerate(ast_strs))[getenv("START", 0):getenv("END", len(ast_strs))]:
    if getenv("TEST", 0) and num not in impl:
      continue
    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    cuda_prg = CompiledRunner(lin.to_program())

    # sass compile # TODO: keep trying
    # try:
    dev.compiler = SASSCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=sass)
    lin.hand_coded_optimizations()
    raw_prg = lin.to_program()
    sass_prg = CompiledRunner(raw_prg)

    # except Exception as e:
    #   print(e)
    #   result["compile_failure"].append(num)
    #   continue

    # init buffers
    np.random.seed(42)
    cuda_bufs = bufs_from_lin(lin)
    for buf in cuda_bufs:
      buf.copyin(memoryview(np.random.rand(buf.size).astype(buf.dtype.np)))
    sass_bufs, debug_bufs = [[Buffer(buf.device, buf.size, buf.dtype, initial_value=bytearray(buf.as_buffer())) for buf in cuda_bufs] for i in range(2)]

    cuda_t = cuda_prg(cuda_bufs, {}, wait=True)

    if debug_src:
      parser = CuAsmParser()
      parser.parse(debug_src)
      cubin = BytesIO()
      parser.saveAsCubin(cubin)
      debug_prg = CompiledRunner(raw_prg, precompiled=bytes(cubin.getbuffer()))
      print(f"debug: {debug_prg(debug_bufs, {}, wait=True)*1e6:7.2f} us")
      if allclose(cuda_bufs, debug_bufs):
        print("success")
      else:
        print("mismatch")
    else:
      # run programs
      try:
        print(f"{num:>5}/{len(ast_strs)} ({(num + 1) * 100.0 / len(ast_strs):.2f}%)\t"
              f"cuda: {cuda_t*1e6:7.2f} us\tsass: {sass_prg(sass_bufs, {}, wait=True)*1e6:7.2f} us\tnodes: {len(lin.uops.uops)}")
      except Exception as e:
        print(e)
        result["runtime_failure"].append(num)

      # check if cuda and sass buffers match
      if allclose(cuda_bufs, sass_bufs):
        print("success")
        result["success"].append(num)
      else:
        print("mismatch")
        result["mismatch"].append((num, info(cuda_bufs, sass_bufs)))

  with open("results.json", "w") as f:
    json.dump(result, f)

  print(f"{len(result["success"])=}")
  print(f"{result["compile_failure"]=}")
  print(f"{result["runtime_failure"]=}")
  print(f"{result["mismatch"]=}")
