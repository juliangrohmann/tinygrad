import json, tempfile, hashlib, subprocess, traceback
from collections import defaultdict
from io import BytesIO
from copy import deepcopy
from pathlib import Path
import numpy as np
from tinygrad import Device
from tinygrad.device import Buffer
from tinygrad.engine.realize import CompiledRunner
from tinygrad.helpers import getenv, to_function_name, colored
from tinygrad.dtype import dtypes
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.engine.search import bufs_from_lin
from tinygrad.runtime.support.compiler_cuda import CUDACompiler, SASSCompiler
from tinygrad.runtime.support.elf import elf_loader, make_elf
from tinygrad.engine.graph import graph_uops
import tinygrad.runtime.autogen.libc as libc
from tinygrad.renderer.sass import SASSRenderer
from CuAsm.CuAsmLogger import CuAsmLogger
from CuAsm import CuAsmParser, CubinFile

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

  is_debug = (debug_sass := getenv("DEBUG_SASS", "")) or (debug_cubin := getenv("DEBUG_CUBIN", ""))

  result = defaultdict(list)
  average_tm_cuda, average_tm_ptx = 0, 0
  impl = [0, 2, 4, 11, 13, 14, 15, 16, 17, 22, 27, 29, 31, 42, 60, 64, 93, 105, 114, 122, 130, 134, 139, 185, 198, 199, 200, 204, 215, 216, 226, 228,
          231, 232, 336, 351, 372, 381, 396, 410, 426, 427, 429, 435, 469]
  single, start, end, max_nodes = getenv("NUM", -1), getenv("START", 0), getenv("END", len(ast_strs)), getenv("MAX_NODES", -1)
  for num,ast in enumerate(ast_strs):
    if (getenv("TEST", 0) and num not in impl) or not (start <= num < end) or (single != -1 and num != single):
      continue

    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    cuda_prg = CompiledRunner(lin.to_program())

    dev.compiler = SASSCompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=sass if not is_debug else dev.renderer)
    lin.hand_coded_optimizations()
    if max_nodes != -1 and len(lin.linearize().uops) > max_nodes: continue
    if getenv("GRAPH_SASS_UOPS", 0): graph_uops(lin.linearize().uops)
    if out_dir := getenv("WRITE_SRC", ""):
      cuda_src = dev.renderer.render(to_function_name(lin.name), lin.linearize().uops)
      with open(fn_cu := Path(out_dir) / "src.cu", "w") as f: f.write(cuda_src)
      with tempfile.NamedTemporaryFile(suffix=".cubin", delete_on_close=False) as tmp:
        tmp.close()
        subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", tmp.name, fn_cu], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        CubinFile(tmp.name).saveAsCuAsm(Path(out_dir) / "nvcc.cuasm")
      sass_src = sass.render(to_function_name(lin.name), lin.uops)
      elf = SASSCompiler(dev.arch).compile(sass_src)
      inst_blob = [section for section in elf_loader(elf)[1] if section.name.startswith(".text")][0].content
      with open(Path(out_dir) / "rendered.sass", "w") as f: f.write(sass_src)
      with open(Path(out_dir) / "rendered.cubin", "wb") as f: f.write(elf)
      with open(Path(out_dir) / "rendered_cuobjdump.sass", "w") as f:
        subprocess.run(["cuobjdump", "-sass", "-arch", "sm_89", (Path(out_dir) / "rendered.cubin").as_posix()], stdout=f)
      with open(Path(out_dir) / "rendered.bin", "wb") as f: f.write(inst_blob)
      # with open(Path(out_dir) / "rendered_nvdisasm.sass", "w") as f: subprocess.run(["nvdisasm", "--cubin", (Path(out_dir) / "rendered.cubin").as_posix()], stdout=f)
      # subprocess.run(["nvdisasm", tmp.name, "--binary", "SM89"])
    try:
      raw_prg = lin.to_program()
    except Exception as e:
      print(colored(f"kernel {num}: renderer failure", "red"))
      print(traceback.format_exc())
      continue

    # init buffers
    np.random.seed(42)
    cuda_bufs = bufs_from_lin(lin)
    for buf in cuda_bufs:
      gen = np.random.uniform if dtypes.is_float(buf.dtype) else np.random.randint
      buf.copyin(memoryview((gen(-1000, 1000, size=buf.size).astype(np.dtype(buf.dtype.fmt).type))))
    debug_bufs = [Buffer(buf.device, buf.size, buf.dtype, initial_value=bytearray(buf.as_buffer())) for buf in cuda_bufs]

    if is_debug:
      if debug_sass:
        with open(debug_sass) as f: cubin = SASSCompiler(dev.arch).compile(f.read())
        with open(Path(debug_sass).with_name("debug.cubin"), "wb") as f: f.write(cubin)
      else:
        with open(debug_cubin, "rb") as f: cubin = f.read();
      debug_prg = CompiledRunner(raw_prg, precompiled=cubin)
      print(f"debug: {debug_prg(debug_bufs, {}, wait=True)*1e6:7.2f} us")
    else:
      try:
        debug_prg = CompiledRunner(raw_prg)
      except Exception as e:
        print(colored(f"kernel {num}: assembler failure", "red"))
        print(traceback.format_exc())
        continue

    # run programs
    try:
      cuda_t, debug_t = cuda_prg(cuda_bufs, {}, wait=True), debug_prg(debug_bufs, {}, wait=True)
    except Exception as e:
      print(e)
      print(colored(f"kernel {num}: runtime failure", "red"))
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
