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
        vals[1] = dest = new_reg()
        render_mov(None, dest, "0x1", dtypes.int)
      return vals[1]

    def is_contiguous(srcs:List[str]):
      return all(s[0] == "R" for s in srcs) and all(int(srcs[i][1]) - int(srcs[0][1]) == i for i in range(len(srcs)))

    def wait_on_barriers(uop:UOp, ctrl:ControlCode):
      for oper in uop.vin:
        if not oper in ctrl_codes:
          wait_on_barriers(oper, ctrl)
        else:
          for c in ctrl_codes[oper]:
            if c.write in active_barriers:
              ctrl.wait.append(c.write)
              active_barriers.remove(c.write)
              if not active_barriers:
                return

    addr = 0
    def queue(uop:UOp, ctrl:ControlCode, ins:str):
      nonlocal addr, active_barriers
      if uop and active_barriers:
        wait_on_barriers(uop, ctrl)
      ctrl.yield_ |= ctrl.stall >= 12 # TODO: is it 12?
      ins += " ;"
      kernel.append((ctrl, addr, ins))
      if uop: ctrl_codes[uop].append(ctrl)
      addr += 16

    def const_addr(uop:UOp, offset=0):
      return f"c[0x0][{hex(param_cbank + uop.arg[0] * 8 + offset)}]"

    def to_reg(uop:UOp):
      var = vals[uop]
      if "R" in var: return var
      vals[uop] = dest = new_reg()
      render_mov(uop, dest, var, uop.dtype)
      return dest

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
          n = (glob.dtype.itemsize + 3) // 4
          offset = hex(idx.arg * n * 4) if idx.uop == UOps.CONST else vals[idx]
          glob_addr += f"+{offset}"
        return glob_addr
      else:
        if glob.dtype.itemsize not in vals:
          vals[glob.dtype.itemsize] = dest = new_reg()
          render_mov(None, dest, hex(glob.dtype.itemsize), dtypes.int)
        dest = new_reg(byte_size=8)
        prefix = f"@{pred} " if pred else ""
        queue(None, ControlCode(), prefix + f"IMAD.WIDE {dest}, {vals[idx]}, {vals[glob.dtype.itemsize]}, {vals[glob]}")
        dest += ".64"
        return dest

    def render_mov(uop, dest, src, dtype, pred=None):
      if uop:
        vals[uop] = dest
      if dtypes.is_float(dtype) and not src.startswith("R") and not src.startswith("0x"):
        src = render_val(float(src), dtype)
      n = (dtype.itemsize + 3) // 4
      queue(uop, ControlCode(), f"{f'@{pred} ' if pred else ''}MOV{f'.{n*32}' if n > 1 else ''} {dest}, {src}")

    def render_where(uop, dest, pred, val_t, val_f):
      render_mov(uop, dest, val_f, uop.dtype)
      render_mov(uop, dest, val_t, uop.dtype, pred)

    def render_log2(uop:UOp, dest:str, src:str):
      p0 = new_pred()
      buf0, buf1, buf2, buf3 = [new_reg() for _ in range(4)]
      queue(uop, ControlCode(), f"FSETP.GEU.AND {p0}, PT, {src}, 1.175494350822287508e-38, PT")
      queue(uop, ControlCode(), f"@!{p0} FMUL {src}, {src}, 8388608")
      queue(uop, ControlCode(), f"IADD3 {dest}, {src}, -0x3f3504f3, RZ")
      queue(uop, ControlCode(), f"LOP3.LUT {buf0}, {dest}, 0xff800000, RZ, 0xc0, !PT")
      queue(uop, ControlCode(), f"IADD3 {dest}, {src}, -{buf0}, RZ")
      queue(uop, ControlCode(), f"I2FP.F32.S32 {buf0}, {buf0}")
      queue(uop, ControlCode(), f"FADD {buf1}, {dest}, -1")
      queue(uop, ControlCode(), f"FSEL {dest}, RZ, -23, {p0}")
      queue(uop, ControlCode(), f"ISETP.GE.U32.AND {p0}, PT, {src}, 0x7f800000, PT")
      queue(uop, ControlCode(), f"MOV {buf2}, 0x3dc6b27f")
      queue(uop, ControlCode(), f"FFMA {buf3}, {buf1}, {buf2}, -0.16845393180847167969")
      params = ["0.1716887056827545166", "-0.17900948226451873779", "0.20512372255325317383", "-0.24046532809734344482",
                "0.28857114911079406738", "-0.36067417263984680176", "0.48089820146560668945", "-0.72134751081466674805"]
      for p in params: queue(uop, ControlCode(), f"FFMA {buf3}, {buf1}, {buf3}, {p}")
      for _ in range(2): queue(uop, ControlCode(), f"FMUL {buf3}, {buf1}, {buf3}")
      queue(uop, ControlCode(), f"FFMA {buf3}, {buf1}, 1.4426950216293334961, {buf3}")
      queue(uop, ControlCode(), f"FFMA {dest}, {buf0}, 1.1920928955078125e-07, {dest}")
      queue(uop, ControlCode(), f"@{p0} MOV {buf0}, 0x7f800000")
      queue(uop, ControlCode(), f"FADD {dest}, {dest}, {buf3}")
      queue(uop, ControlCode(), f"@{p0} FFMA {dest}, {src}, {buf0}, +INF")
      queue(uop, ControlCode(), f"FSETP.NEU.AND {p0}, PT, {src}, RZ, PT")
      queue(uop, ControlCode(), f"FSEL {dest}, {dest}, -INF, {p0}")

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
        elif dtypes.is_float(dtype):
          vals[u] = str(arg)
        else:
          vals[u] = render_val(arg, dtype)
      elif uop is UOps.DEFINE_GLOBAL:
        vals[u] = const_addr(u)
      elif uop is UOps.LOAD:
        if vin[0].uop is UOps.DEFINE_GLOBAL:
          pred = vals[vin[2]] if len(vin) == 4 else None
          glob_addr = render_glob_addr(vin[1], vin[0], pred=pred)
          vals[u] = dest = new_reg(dtype.itemsize)
          n = (dtype.itemsize + 3) // 4
          size_mod = f".{n*32}" if n > 1 else ''
          queue(u, ControlCode(write=new_barrier()), f"{f'@{pred} ' if pred else ''}LDG.E{size_mod} {dest}, desc[UR4][{glob_addr}]") # explicit memory desc
          if pred:
            render_mov(u, dest, vals[vin[3]], dtype, "!"+pred)
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
      elif uop is UOps.CAST:
        if dtypes.is_int(vin[0].dtype):
          if dtypes.is_float(dtype):
            assert vin[0].dtype.count == 1, f"can't cast int to {vin[0].dtype}"
            vals[u] = dest = new_reg(dtype.itemsize)
            queue(u, ControlCode(), f"I2F.{'U' if dtypes.is_unsigned(vin[0].dtype) else 'S'}{vin[0].dtype.itemsize * 8} {dest}, {vals[u.vin[0]]}")
          else:
            raise NotImplementedError
        elif all(dtypes.is_float(v.dtype) for v in vin) and dtype.count > 1:
          srcs = [vals[v] for v in vin]
          if not is_contiguous(srcs):
            dest = new_reg(dtype.itemsize)
            n = (vin[0].dtype.itemsize + 3) // 4
            idx = int(dest[1:])
            for s in srcs:
              render_mov(u, f"R{idx}", s, vin[0].dtype)
              idx += n
            vals[u] = dest
          else:
            vals[u] = srcs[0]
        elif vin[0].dtype is dtypes.bool:
          render_where(u, new_reg(dtype.itemsize), vals[vin[0]], render_val(1, dtype), render_val(0, dtype))
        else:
          raise NotImplementedError
      elif uop is UOps.GEP:
        src = vals[vin[0]]
        assert src.startswith("R"), f"GEP only supported on registers. src: {src}"
        vals[u] = f"R{int(src[1:]) + arg}"
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
          render_where(u, new_reg(dtype.itemsize), *[vals[v] for v in vin])
        elif arg is UnaryOps.LOG2:
          assert dtype is dtypes.float, f"log2 not supported for {dtype}"
          vals[u] = dest = new_reg()
          render_log2(u, dest, vals[vin[0]])
        else:
          raise NotImplementedError
      else:
        raise NotImplementedError

    queue(None, ControlCode(), "EXIT ;")
    sass_src = "\n".join([f"{' '*6}{c.render()}{' '*9}/*{hex(a)[2:]:>04}*/{' '*19}{i}" for c,a,i in kernel])
    return attach_sections(sass_src, self.cuda_renderer.render(name, uops), reg_cnt)

sass_matcher = PatternMatcher([

])
