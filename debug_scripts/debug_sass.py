import re
import numpy as np
from typing import Optional
from tinygrad.dtype import dtypes, DType
from tinygrad.device import Buffer, Device
from tinygrad.renderer import Program
from tinygrad.helpers import getenv
from tinygrad.runtime.support.compiler_cuda import SASSCompiler
from tinygrad.engine.realize import CompiledRunner
from debug_scripts.compare_cubin import load_cubin
from tinygrad.runtime.support.elf import make_elf
from tinygrad.runtime.support.cubin import build_segments
def _to_np_dtype(dtype:DType) -> Optional[type]: return np.dtype(dtype.fmt).type if dtype.fmt is not None else None

dts = (dtypes.int,)*2
vals = [2]

compiler = SASSCompiler("sm_89")
if cubin_fn := getenv("CUBIN", ""):
  with open(cubin_fn, "rb") as f: cubin = f.read()
  name = getenv("NAME", "")
  src = None
elif (hack_a_fn := getenv("HACK_A", "")) and (hack_b_fn := getenv("HACK_B", "")):
  src = ""
  name = "r_16n1"
  header_a, sections_a, prog_headers_a, sec_map_a = load_cubin(hack_a_fn)
  header_b, sections_b, prog_headers_b, sec_map_b = load_cubin(hack_b_fn)
  # ".text.r_16n1", ".nv.shared.r_16n1", ".nv.rel.action", ".nv.info", ".nv.info.r_16n1", ".shstrtab", ".strtab", ".symtab"
  from_b = []
  sections = [s if s.name not in from_b else sec_map_b[s.name] for s in sections_a]
  print(f"replaced: {len([s for s in sections_a if s.name in from_b])}/{len(from_b)}")
  cubin = bytes(make_elf(header_a, sections_a, build_segments(sections_a)))
else:
  with open(getenv("SRC", "")) as f: cubin = compiler.compile(src := f.read())
  with open("debug_src/debug_out.cubin", "wb") as f: f.write(cubin)
  name = next(m.groups()[0] for line in src.split('\n') if (m := re.match(r"\.text\.([^:]*)", line.strip())))
prg = Program(name, src, Device.DEFAULT, global_size=[1,1,1], local_size=[1,1,1])
runner = CompiledRunner(prg, precompiled=cubin)
buf = Buffer(Device.DEFAULT, 1, dts[-1]).allocate()
buf2 = [Buffer(Device.DEFAULT, 1, dtype).allocate().copyin(np.array([a], dtype=_to_np_dtype(dtype)).data) for a,dtype in zip(vals, dts)]
runner.exec([buf]+buf2)
ret = np.empty(1, _to_np_dtype(dts[-1]))
buf.copyout(ret.data)
print(ret[0])

