import json
from collections import defaultdict
from io import BytesIO
from copy import deepcopy
from pathlib import Path
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, to_function_name, colored
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, CuAsmCompiler, SASSCompiler
from tinygrad.runtime.support.elf import elf_loader, make_elf
import tinygrad.runtime.autogen.libc as libc
from tinygrad.renderer.sass import SASSRenderer
from CuAsm.CuAsmLogger import CuAsmLogger
from CuAsm import CuAsmParser

def info(bufs_cuda, bufs_sass):
  ret = []
  for cuda, sass in zip(bufs_cuda, bufs_sass):
    b_info = {}
    cuda_vals, sass_vals = [np.frombuffer(buf.as_buffer(), dtype=np.dtype(buf.dtype.fmt)) for buf in [cuda, sass]]
    mask = ~(np.isnan(cuda_vals) & np.isnan(sass_vals))
    b_info["cuda"], b_info["sass"] = cuda_vals.tolist(), sass_vals.tolist()
    try:
      b_info["sum_diff"] = np.abs(np.sum(cuda_vals[mask] - sass_vals[mask])).item()
      b_info["max_diff"] = np.max(np.abs(cuda_vals[mask] - sass_vals[mask])).item()
    except Exception as e:
      pass
    ret.append(b_info)
  return ret

def allclose(bufs_a, bufs_b):
  for a, b in zip(bufs_a, bufs_b):
    cuda_vals, sass_vals = [np.frombuffer(buf.as_buffer(), dtype=np.dtype(buf.dtype.fmt)) for buf in [a, b]]
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
  is_debug = (debug_cuasm := getenv("DEBUG_CUASM", "")) or (debug_cubin := getenv("DEBUG_CUBIN", ""))
  if single != -1: ast_strs = ast_strs[single:single+1]

  result = defaultdict(list)
  average_tm_cuda, average_tm_ptx = 0, 0
  impl = [2, 4, 11, 13, 14, 15, 16, 17, 22, 27, 29, 31, 410] # 29 (move 64+ bits), 39 (underflow uint?), 936 (char cast)
  start, end = getenv("START", 0), getenv("END", len(ast_strs))
  for num,ast in enumerate(ast_strs):
    if (getenv("TEST", 0) and num not in impl) or not (start <= num < end):
      continue

    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    cuda_prg = CompiledRunner(lin.to_program())

    dev.compiler = SASSCompiler(dev.arch) if not getenv("CUASM", 0) else CuAsmCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=sass if not is_debug else dev.renderer)
    lin.hand_coded_optimizations()
    raw_prg = lin.to_program()

    # init buffers
    np.random.seed(42)
    cuda_bufs = bufs_from_lin(lin)
    for buf in cuda_bufs:
      buf.copyin(memoryview(np.random.rand(buf.size).astype(np.dtype(buf.dtype.fmt).type)))
    debug_bufs = [Buffer(buf.device, buf.size, buf.dtype, initial_value=bytearray(buf.as_buffer())) for buf in cuda_bufs]

    if is_debug:
      if debug_cuasm:
        parser = CuAsmParser()
        parser.parse(debug_cuasm)
        parser.saveAsCubin(cubin_buf := BytesIO())
        cubin = bytes(cubin_buf.getbuffer())
      else:
        with open(debug_cubin, "rb") as f: cubin = f.read();

      debug_prg = CompiledRunner(raw_prg, precompiled=cubin)
      print(f"debug: {debug_prg(debug_bufs, {}, wait=True)*1e6:7.2f} us")
    else:
      debug_prg = CompiledRunner(raw_prg)

    # run programs
    try:
      cuda_t, debug_t = cuda_prg(cuda_bufs, {}, wait=True), debug_prg(debug_bufs, {}, wait=True)
    except Exception as e:
      print(e)
      print(colored("runtime failure", "red"))
      result["runtime_failure"].append(num)

    # check if cuda and sass buffers match
    if allclose(cuda_bufs, debug_bufs):
      status = colored("success", "green")
      result["success"].append(num)
    else:
      status = colored("mismatch", "red")
      result["mismatch"].append((num, info(cuda_bufs, debug_bufs)))
    print(f"{num:>5}/{len(ast_strs)} ({(num + 1) * 100.0 / len(ast_strs):.2f}%){' '*4}"
          f"cuda: {cuda_t*1e6:7.2f} us{' '*4}sass: {debug_t*1e6:7.2f} us{' '*4}nodes: {len(lin.uops):>3}{' '*4}{status}")

  with open("results.json", "w") as f:
    json.dump(result, f)

  print(f"{len(result["success"])=}")
  print(f"{result["compile_failure"]=}")
  print(f"{result["runtime_failure"]=}")
  print(f"{result["mismatch"]=}")
