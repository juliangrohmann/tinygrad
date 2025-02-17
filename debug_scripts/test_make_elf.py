import subprocess
import tempfile
import hashlib
from copy import deepcopy
from pathlib import Path
from tinygrad.helpers import getenv, colored, to_function_name
import tinygrad.runtime.support.elf as elf
import tinygrad.runtime.autogen.libc as libc
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.runtime.support.compiler_cuda import CUDACompiler
from tinygrad import Device

def print_bytes(data):
  print(labels := ' '.join([f"{i:>02}" for i in range(1, 17)]))
  print('-' * len(labels))
  for row in range(len(data) // 16):
    print(' '.join([colored(f"{hex(b)[2:]:>02}", "red" if b == 0 else "green") for b in data[row*16:min((row+1)*16, len(data))]]))

def c_print(obj):
  for name,_ in obj._fields_:
    print(f"{name}: {obj.__getattribute__(name)}")

def c_equals(a, b):
  for name,_ in a._fields_:
    av, bv = a.__getattribute__(name), b.__getattribute__(name)
    if name == "e_ident": continue
    assert av == bv, f"mismatch in {name}: {av=}, {bv=}"

def info(blob):
  _, sections, _ = elf.elf_loader(blob)
  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  phs = (libc.Elf64_Phdr * header.e_phnum).from_buffer_copy(blob[header.e_phoff:])
  print("HEADER:")
  c_print(header)
  print()
  print("SECTIONS:")
  for s in sections:
    print(f"{s.name}:")
    c_print(s.header)
    print()
  print("PROGRAM HEADERS:")
  for ph in phs:
    c_print(ph)
    print()
  return header, sections, phs,

ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
# no bfloat16 for ptx at the moment
ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
dev = Device["CUDA"]

start, end, single = getenv("START", 0), getenv("END", len(ast_strs)), getenv("NUM", -1)
for num,ast in enumerate(ast_strs):
  if (single != -1 and not num == single) or not (start <= num < end): continue

  print(f"AST {num}")
  # cuda compile
  dev.compiler = CUDACompiler(dev.arch)
  lin = ast_str_to_lin(ast, opts=dev.renderer)
  lin.hand_coded_optimizations()
  lin.linearize()
  code = dev.renderer.render(to_function_name(lin.name), lin.uops.uops)
  fn = (Path(tempfile.gettempdir()) / f"cu_buf_{hashlib.md5(code.encode()).hexdigest()}").as_posix()
  with open(fn + ".cu", "w") as f: f.write(code)
  subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", fn + ".cubin", fn + ".cu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

  # src, out = Path(getenv("IN", "")), Path(getenv("OUT", ""))
  with open(fn + ".cubin", "rb") as f: blob = f.read()
  src_header, src_sections, src_phs = info(blob)
  for s in src_sections: s.header.sh_addr = 0

  built = elf.make_elf(deepcopy(src_header), deepcopy(src_sections), prog_headers=deepcopy(src_phs))
  built_header, built_sections, built_phs = info(bytes(built))
  for s in built_sections: s.header.sh_addr = 0

  # with open(out, "wb") as f: f.write(built)
  # for fn in [src, out]:
  #   with open(fn.with_suffix(".hex"), "w") as f: subprocess.run(["xxd", fn.as_posix()], stdout=f)

  c_equals(src_header, built_header)
  for sa, sb in zip(src_sections, built_sections):
    c_equals(sa.header, sb.header)
    assert sa.content == sb.content, f"mismatch in section content of {sa.name}"
  for pa, pb in zip(src_phs, built_phs):
    c_equals(pa, pb)
  for i,(a,b) in enumerate(zip(built, blob)):
    assert a == b, f"mismatch in blob at address {hex(i)}: {a=}, {b=}"
  print("success")
