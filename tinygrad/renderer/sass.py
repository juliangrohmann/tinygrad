import tempfile
import hashlib
import subprocess
import struct
import re
from dataclasses import dataclass, field
from collections import defaultdict
from pathlib import Path
from typing import DefaultDict, Dict, List, Union, Optional, cast, Callable
from tinygrad.helpers import getenv, all_same
from tinygrad.codegen.linearizer import UOps, UOp
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType
from tinygrad.codegen.uops import UOpGraph, PatternMatcher, UPat
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.renderer.cstyle import CUDARenderer
from CuAsm import CubinFile, CuAsmParser

def render_val(x, dtype):
  if dtypes.is_int(dtype):
    return hex(x)
  elif dtypes.is_float(dtype):
    formats = {dtypes.double: "d", dtypes.float: "f", dtypes.half: "e"}
    form = formats[dtype] if dtype in formats else "f"
    return "0x" + ''.join(f"{c:>02x}" for c in struct.pack(f"!{form}", x))
  else:
    raise NotImplementedError

def attach_sections(sass_src:str, cuda_src:str, reg_cnt:int): # TODO remove HACK: ELF headers/sections attached from nvcc compile
  fn = (Path(tempfile.gettempdir()) / f"cu_buf_{hashlib.md5(cuda_src.encode()).hexdigest()}").as_posix()
  with open(fn + ".cu", "w") as f: f.write(cuda_src)
  subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", fn + ".cubin", fn + ".cu"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
  CubinFile(fn + ".cubin").saveAsCuAsm(fn + "_cuda.cuasm")
  if out_dir := getenv("WRITE_SRC", ""):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "nvcc.cuasm", "w") as f:
      with open(fn + "_cuda.cuasm") as tempf: f.write(tempf.read())
    with open(out_dir / "src.cu", "w") as f: f.write(cuda_src)
  cuasm = ""
  skip = False
  with open(fn + "_cuda.cuasm") as f:
    for line in f:
      if not skip:
        cuasm += line
        if line.strip().startswith(".text."):
          cuasm += sass_src + '\n'
          skip = True
      elif line.strip().startswith(".L"):
        cuasm += line
        skip = False
  cuasm = re.sub(r"(SHI_REGISTERS=)\d+", f"SHI_REGISTERS={reg_cnt + 2}", cuasm) # 3 registers used internally on sm_89
  return cuasm

@dataclass
class ControlCode:
  wait: List[int] = field(default_factory=list)
  read: Optional[int] = None
  write: Optional[int] = None
  yield_: bool = False
  stall: int = 15
  def render(self):
    bs = ''.join(['-' if b not in self.wait else str(b) for b in range(6)])
    rs, ws = [v if v is not None else '-' for v in [self.read, self.write]]
    return f"[B{bs}:R{rs}:W{ws}:{'Y' if self.yield_ else '-'}:S{self.stall}]"

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max = [65535, 65535, 2147483647]
  local_max = [64, 1024, 1024]
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(0,2)], thread_local_sizes=[[2,2,2],[2,2],[2,2]], thread_local_aliases=[ [[0],[0],[5],[-2],[0],[-1,1,2,-3],[3,4]], [[3],[4],[0],[0],[5],[-1,1,2,-2],[0]], [[-1],[1],[5],[-2],[2],[0],[3,4]] ], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float)])] # noqa: E501
  is_64_bit = True # TODO: remove
  def __init__(self, arch:str):
    self.tensor_cores = SASSRenderer.tensor_cores if int(arch[3:]) >= 80 else []
    self.cuda_renderer = CUDARenderer(arch) # TODO: remove
  # language
  gid = [f'SR_CTAID.{"XYZ"[i]}' for i in range(3)]
  tid = [f'SR_TID.{"XYZ"[i]}' for i in range(3)]
  plop = {
    "&": ("0x80", "0x0")
  }
  alu = {
    BinaryOps.ADD: {
      dtypes.int: "IMAD",
      dtypes.half: "HADD2",
      dtypes.float32: "FADD",
      dtypes.float64: "DADD",
    },
    BinaryOps.MUL: {
      dtypes.int: "IMAD",
      dtypes.half: "HMUL2",
      dtypes.float32: "FMUL",
      dtypes.float64: "DMUL",
    },
    BinaryOps.MAX: {
      dtypes.int: "IMNMX",
      dtypes.half: "HMNMX2",
      dtypes.float32: "FMNMX",
    }
  }

  def render(self, name:str, uops:UOpGraph) -> str:
    uops.linearize(sass_matcher)
    attach_sections("", self.cuda_renderer.render(name, uops), 0) # TODO: remove (for writing debug src only)

    kernel:List[Tuple[ControlCode, int, str]] = []
    ctrl_codes:Dict[UOp, List[ControlCode]] = defaultdict(list)
    vals:Dict[Any, str] = {}
    param_cbank = int("160", 16) # TODO: make variable
    pred_cap = 251

    reg_cnt = 0
    def new_reg(byte_size:int=4):
      nonlocal reg_cnt
      n = (byte_size + 3) // 4
      reg = reg_cnt + reg_cnt % n
      reg_cnt = reg + n
      assert reg_cnt + pred_cnt <= pred_cap, "trying to assign to new register: all registers filled" # TODO: remove & optim regs after render
      return f"R{reg}"

    pred_cnt = 0
    def new_pred():
      nonlocal pred_cnt
      pred = pred_cnt
      pred_cnt += 1
      assert reg_cnt + pred_cnt <= pred_cap, "trying to assign to new predicate: all registers filled" # TODO: remove & optim regs after render
      return f"P{pred}"

    active_barriers = []
    def new_barrier():
      nonlocal active_barriers
      for i in range(6):
        if not i in active_barriers:
          active_barriers.append(i)
          return i
      return active_barriers[0]

    def unity():
      if not 1 in vals:
        vals[1] = new_reg()
        queue(None, ControlCode(), f"MOV {vals[1]}, 0x1")
      return vals[1]

    addr = 0
    def queue(uop:UOp, ctrl:ControlCode, ins:str):
      nonlocal addr, active_barriers
      if uop:
        for oper in uop.vin:
          if oper in ctrl_codes:
            for c in ctrl_codes[oper]:
              if c.write in active_barriers:
                ctrl.wait.append(c.write)
      for i in ctrl.wait:
        active_barriers.remove(i)
      ctrl.yield_ |= ctrl.stall >= 12 # TODO: is it 12?
      ins += " ;"
      kernel.append((ctrl, addr, ins))
      if uop: ctrl_codes[uop].append(ctrl)
      addr += 16

    def const_addr(uop:UOp, offset=0):
      return f"c[0x0][{hex(param_cbank + uop.arg[0] * 8 + offset)}]"

    def render_glob_addr(idx:UOp, glob:UOp, pred=None):
      if idx.uop is UOps.CONST:
        glob_addr = vals[glob]
        if "R" not in glob_addr:
          glob_addr = new_reg(byte_size=8)
          queue(None, ControlCode(), f"IMAD.MOV.U32 {glob_addr}, RZ, RZ, {const_addr(glob)}")
          addr_ext = f"R{int(glob_addr[1:]) + 1}"
          queue(None, ControlCode(), f"IMAD.MOV.U32 {addr_ext}, RZ, RZ, {const_addr(glob, offset=4)}")
          vals[glob] = glob_addr
        glob_addr += ".64"
        if idx and idx.arg != 0:
          offset = hex(idx.arg * glob.dtype.itemsize) if idx.uop == UOps.CONST else vals[idx]
          glob_addr += f"+{offset}"
        return glob_addr
      else:
        if glob.dtype.itemsize not in vals:
          vals[glob.dtype.itemsize] = dest = new_reg()
          queue(None, ControlCode(), f"MOV {dest}, {hex(glob.dtype.itemsize)}")
        dest = new_reg(byte_size=8)
        prefix = f"@{pred} " if pred else ""
        queue(None, ControlCode(), prefix + f"IMAD.WIDE {dest}, {vals[idx]}, {vals[glob.dtype.itemsize]}, {vals[glob]}")
        dest += ".64"
        return dest

    def to_reg(uop:UOp):
      var = vals[uop]
      if "R" in var: return var
      vals[uop] = dest = new_reg()
      queue(uop, ControlCode(), f"MOV {dest}, {var}")
      return dest

    vals[0] = "RZ"
    vals[float("inf")] = "INF"
    vals[float("-inf")] = "-INF"
    queue(None, ControlCode(), "ULDC.64 UR4, c[0x0][0x118]") # load explicit memory desc
    for u in uops:
      print(u)
      uop, dtype, vin, arg = u.uop, u.dtype, u.vin, u.arg
      if uop is UOps.SPECIAL:
        vals[u] = dest = new_reg()
        src = (self.gid if arg[1][:3] == "gid" else self.tid)[arg[0]]
        queue(u, ControlCode(write=new_barrier()), f"S2R {dest}, {src}")
      elif uop is UOps.CONST:
        if arg in vals:
          vals[u] = vals[arg]
        else:
          vals[u] = render_val(arg, dtype)
      elif uop is UOps.DEFINE_GLOBAL:
        vals[u] = const_addr(u)
      elif uop is UOps.LOAD:
        print(f"{[v.dtype for v in vin]=}")
        if vin[0].uop is UOps.DEFINE_GLOBAL:
          pred = vals[vin[2]] if len(vin) == 4 else None
          glob_addr = render_glob_addr(vin[1], vin[0], pred=pred)
          vals[u] = dest = new_reg(dtype.itemsize)
          queue(u, ControlCode(write=new_barrier()), f"{f'@{pred} ' if pred else ''}LDG.E {dest}, desc[UR4][{glob_addr}]") # explicit memory desc
          if pred:
            queue(u, ControlCode(), f"@!{pred} MOV {dest}, {vals[vin[3]]}")
        else:
          raise NotImplementedError
      elif uop is UOps.STORE:
        if vin[0].uop is UOps.DEFINE_GLOBAL:
          glob_addr = render_glob_addr(vin[1], vin[0])
          mods = [".E"]
          if vin[2].dtype.itemsize > 4: mods.append(f".{vin[2].dtype.itemsize * 8}")
          queue(u, ControlCode(), f"STG{''.join(mods)} desc[UR4][{glob_addr}], {to_reg(vin[2])}")  # explicit memory descriptor
        else:
          raise NotImplementedError
      elif uop is UOps.ALU:
        srcs = [vals[v] for v in vin]
        if arg is BinaryOps.MUL and dtype is dtypes.bool:
          assert 2 <= len(srcs) <= 3, f"too many arguments for bool mul: {len(srcs)}"
          vals[u] = dest = new_pred()
          if len(srcs) == 2: srcs.append("PT")
          queue(u, ControlCode(), f"PLOP3.LUT {dest}, PT, {', '.join(srcs)}, {', '.join(self.plop['&'])}")
        elif arg is BinaryOps.MUL or arg is BinaryOps.ADD:
          assert len(srcs) == 2
          vals[u] = dest = new_reg()
          if dtype is dtypes.int:
            if arg is BinaryOps.MUL:
              srcs.append(vals[0])
            elif arg is BinaryOps.ADD:
              srcs[1:1] = [unity()]
            else:
              raise NotImplementedError
            queue(u, ControlCode(), f"{self.alu[arg][dtype]} {dest}, {', '.join(srcs)}")
          else:
            assert len(srcs) == 2
            vals[u] = dest = new_reg()
            queue(u, ControlCode(), f"{self.alu[arg][dtype]} {dest}, {', '.join(srcs)}")
        elif arg is BinaryOps.MAX:
          assert all_same(dt := [v.dtype for v in vin]), f"dtype mismatch in min/max: {dt}"
          vals[u] = dest = new_reg(vin[0].dtype.itemsize)
          srcs = [vals[v] for v in vin]
          assert len(srcs) == 2, f"too many min/max operands: {len(src)}"
          queue(u, ControlCode(), f"{self.alu[arg][dtype]} {dest}, {', '.join(srcs)}, !PT")
        elif arg is BinaryOps.CMPLT:
          vals[u] = dest = new_pred()
          srcs = [to_reg(v) for v in vin]
          queue(u, ControlCode(), f"ISETP.LT.AND {dest}, PT, {', '.join(srcs)}, PT")
        elif arg is TernaryOps.WHERE:
          vals[u] = dest = new_reg()
          queue(None, ControlCode(), f"MOV {dest}, {vals[vin[2]]}")
          queue(u, ControlCode(), f"@{vals[vin[0]]} MOV {dest}, {vals[vin[1]]}")
        else:
          raise NotImplementedError
      else:
        raise NotImplementedError

    queue(None, ControlCode(), "EXIT ;")
    sass_src = "\n".join([f"{' '*6}{c.render()}{' '*9}/*{hex(a)[2:]:>04}*/{' '*19}{i}" for c,a,i in kernel])
    return attach_sections(sass_src, self.cuda_renderer.render(name, uops), reg_cnt)

sass_matcher = PatternMatcher([

])
