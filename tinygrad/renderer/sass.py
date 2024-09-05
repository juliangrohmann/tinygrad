import tempfile, hashlib, subprocess, struct, re, dataclasses
import hashlib
import subprocess
import struct
import re
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Union, Optional, cast, Callable
from tinygrad.helpers import getenv, all_same, flatten
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import PatternMatcher, UPat, UOps, UOp
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.renderer.cstyle import CUDARenderer
from CuAsm import CubinFile, CuAsmParser

def render_value(x, dtype):
  if dtypes.is_float(dtype):
    return str(x)
  elif dtype is dtypes.bool:
    return "PT" if x else "!PT"
  else:
    return render_binary(x, dtype)

def render_binary(x, dtype): # TODO: simplify
  x = abs(x) if (neg := dtypes.is_unsigned(dtype) and x < 0) else x
  return f"{'-' if neg else ''}0x" + ''.join(f"{c:>02x}" for c in struct.pack(f"!{dtype.fmt}", x))

def const_addr(uop:UOp, offset=0):
  param_cbank = int("160", 16) # TODO: make variable
  return f"c[0x0][{hex(param_cbank + uop.arg * 8 + offset)}]"

def is_contiguous(srcs:List[Any]):
  return all(isinstance(s, Register) and s.size == srcs[0].size and s.idx - srcs[0].idx == i * srcs[0].size for i,s in enumerate(srcs))

def nregs(byte_size):
  return (byte_size + 3) // 4

@dataclass
class Register:
  idx: int
  size: int = 1
  pred: bool = False # TODO: make these flags a str identifier
  uniform: bool = False
  barrier: bool = False
  negated: bool = False
  mod: str = None
  def render(self):
    pre_mod = '-!'[self.pred] if self.negated else ''
    return f"{pre_mod}{self.identity()}{f'.{self.mod}' if self.mod else ''}"
  def offset(self, n): return dataclasses.replace(self, idx=self.idx + n)
  def base(self): return self.offset(-(self.idx % self.size))
  def negate(self): return dataclasses.replace(self, negated=not self.negated)
  def modify(self, mod): return dataclasses.replace(self, mod=mod)
  def identity(self): return f"{'U' if self.uniform else ''}{'RP'[self.pred] if not self.barrier else 'B'}{self.idx if self.idx != -1 else 'Z'}"
  def __hash__(self): return hash(self.identity())
  def __eq__(self, other): return other.identity() == self.identity()

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
    return f"[B{bs}:R{rs}:W{ws}:{'-Y'[self.yield_]}:S{self.stall}]"

@dataclass
class Instruction:
  op: str
  dest: Register
  srcs: List[Union[Register, str]]
  ctrl: ControlCode = field(default_factory=ControlCode)
  mods: List[str] = field(default_factory=list)
  pred: Register = None
  label: bool = False
  addr: int = None
  def render(self):
    if self.label: return f"  {self.op}:"
    operands = ', '.join(([self.dest.render()] if self.dest else []) + [s.render() if isinstance(s, Register) else s for s in self.srcs])
    ins = f"{f'@{self.pred.render()} ' if self.pred else ''}{self.op}{''.join([f'.{m}' for m in self.mods])} {operands}"
    return f"{' '*6}{self.ctrl.render()}{' '*9}/*{hex(self.addr)[2:]:>04}*/{' '*19}{ins} ;"

write_latency_ops = {"MUFU", "LDG", "S2R", "I2F"} # TODO: I2F is only variable latency for cross register (64bit) ops?
read_latency_ops = {"MUFU"}
sass_matcher = PatternMatcher([
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"),UPat(name="k"))), # NOTE: from PTX, TODO: make uchar, TODO: same pattern for STORE
   lambda root,x,y,z,k: UOp(root.op, dtypes.char, (x,y,z.cast(dtypes.uint8),k)).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())), # NOTE: from PTX, TODO: make uchar, TODO: same pattern for STORE
   lambda root: UOp(root.op, dtypes.char, root.src, root.arg).cast(dtypes.bool)),
  # (UPat(UOps.CAST, name="root", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
  #     src=(UPat(UOps.LOAD, name="x", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)])))),
  #  lambda root, x: UOp(x.op, root.dtype, x.src, x.arg)),
  (UPat(UOps.CAST, name="root", dtype=set([dt for dt in dtypes.fields().values() if dt.itemsize != 4]), src=(UPat(name="x", dtype=dtypes.bool))),
   lambda root, x: UOp(root.op, root.dtype, src=(x.cast(dtypes.int),))),
  # Shift left/right for int mul/div by 2
  # A, B -> ADD, C -> ADD               ===     A, B, C -> ADD3
  # A, B -> MUL, C -> ADD               ===     A, B, C -> FMA
  # A, B -> CMPNE, CONST True -> CMPNE  ===     A, B -> CMPEQ
  # A, B -> CMP, C -> MUL               ===     A, B, C -> CMP.AND
  # A -> NEG, B -> OP                   ===     -A, B -> OP
  # bool, bool -> OP, bool -> OP        ===     bool, bool, bool -> PLOP3
  # A -> CAST(bool) -> CAST(num)        ===     A -> CAST(num) (results from LOAD/STORE: bool -> uchar cast)
])

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2),(0,2),(1,2),(1,2),(1,2)], dtype_in=di, dtype_out=do) for (di, do) in ([(dtypes.half, dtypes.float)])] # noqa: E501
  extra_matcher = sass_matcher
  def __init__(self, arch:str):
    self.tensor_cores = SASSRenderer.tensor_cores if int(arch[3:]) >= 80 else []
    self.cuda_renderer = CUDARenderer(arch) # TODO: remove
  # language
  gid = [f'SR_CTAID.{"XYZ"[i]}' for i in range(3)]
  tid = [f'SR_TID.{"XYZ"[i]}' for i in range(3)]
  lop = {
    "&": ("0xc0", "0x0"),   # a & b & (c | ~c)
    "&3": ("0x80", "0x0"),  # a & b & c
    "^&": ("0x28", "0x0",), # (a ^ b) & c
    "~": ("0x0f", "0x0")    # ~a & (b | ~b) & (c | ~c)
  }
  setp_mod = {
    BinaryOps.CMPLT: "LT",
    BinaryOps.CMPNE: "NE"
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

  def render_mov(self, dest:Register, src:Union[Register,str], dtype:DType, pred:Optional[str]=None) -> Instruction:
    if dtypes.is_float(dtype) and not isinstance(src, Register) and not src.startswith("0x"):
      src = render_binary(float(src), dtype)
    srcs = [src.offset(i) if src.size >= i + 1 else "RZ" for i in range(dest.size)] if isinstance(src, Register) else [src] + (dest.size - 1) * ["RZ"]
    return [Instruction("MOV", dest.offset(i), [srcs[i]], pred=pred) for i in range(dest.size)]

  def render_where(self, dest, pred:Union[Register, str], src_t:Register, src_f:Union[Register,str], dtype:DType) -> List[Instruction]:
    return [Instruction("SEL" if dtypes.is_int(dtype) else "FSEL", dest, [src_t, src_f, pred])] # nvcc does float packed SEL here instead of FSEL

  def render_cmp(self, arg:BinaryOps, dest:Register, src_l:Register, src_r:Register, dtype:DType) -> List[Instruction]:
    srcs = [src_l, src_r]
    if dtypes.is_int(dtype) or dtypes.is_float(dtype):
      ret = []
      for i in range(0, nregs(dtype.itemsize)): # 64bit+ (long)
        assert arg is BinaryOps.CMPNE or i == 0, f"64bit+ only supported for CMPNE. {arg=}"
        srcs = [s.offset(i) for s in srcs]
        ret.append(ins := Instruction("ISETP", dest, ["PT"] + srcs + ["PT"], mods=[f"{self.setp_mod[arg]}"]))
        if dtypes.is_unsigned(dtype): ins.mods.append("U32")
        ins.mods.append("AND")
        if i > 0:
          ins.mods.append("EX")
          ins.srcs.append(dest)
      return ret
    elif dtype is dtypes.bool:
      return [Instruction("PLOP3", dest, ["PT"] + srcs + ["PT"] + list(self.lop['^&']), mods=["LUT"])]
    else:
      raise NotImplementedError

  def render_iter(self, label:str, pred:Register, counter:Register, end:Register, dtype:DType) -> List[Instruction]:
    return [*self.render_cmp(BinaryOps.CMPNE, pred, counter, end, dtype), *self.render_bra(label, pred)]

  def render_bra(self, label:str, pred:Register) -> List[Instruction]:
    return [Instruction("BRA", None, [f"`({label})"], pred=pred)]

  def render_recip(self, dest:Register, src:Register, buf:Register, dtype:DType): # TODO: pass reg/pred generator
    return [Instruction("MUFU", dest, [src], mods=["RCP"]),  # TODO: only valid for divisor >= 2^-126. branch to __internal_0_$__cuda_sm20_rcp_rn_f32_slowpath otherwise
            Instruction("FFMA", buf, [dest, src, "-1"]),
            Instruction("FADD", buf, [buf.negate(), "-RZ"], mods=["FTZ"]),
            Instruction("FFMA", dest, [dest, buf, dest])]

  def render_log2(self, dest:Register, src:Register, pred:Register, bufs:List[Register]) -> List[Instruction]:
    assert len(bufs) == 4, f"expected 4 buffers. {len(bufs)=}"
    ins = [Instruction("FSETP", pred, ["PT", src, "1.175494350822287508e-38", "PT"], mods=["GEU", "AND"]),
           Instruction("FMUL", src, [src, "8388608"], pred=pred.negate()),
           Instruction("IADD3", dest, [src, "-0x3f3504f3", "RZ"]),
           Instruction("LOP3", bufs[0], [dest, "0xff800000", "RZ", "0xc0", "!PT"], mods=["LUT"]),
           Instruction("IADD3", dest, [src, bufs[0].negate(), "RZ"]),
           Instruction("I2FP", bufs[0], [bufs[0]], mods=["F32", "S32"]),
           Instruction("FADD", bufs[1], [dest, "-1"]),
           Instruction("FSEL", dest, ["RZ", "-23", pred]),
           Instruction("ISETP", pred, ["PT", src, "0x7f800000", "PT"], mods=["GE", "U32", "AND"]),
           Instruction("MOV", bufs[2], ["0x3dc6b27f"]),
           Instruction("FFMA", bufs[3], [bufs[1], bufs[2], "-0.16845393180847167969"])]
    params = ["0.1716887056827545166", "-0.17900948226451873779", "0.20512372255325317383", "-0.24046532809734344482",
              "0.28857114911079406738", "-0.36067417263984680176", "0.48089820146560668945", "-0.72134751081466674805"]
    for p in params: ins.append(Instruction("FFMA", bufs[3], [bufs[1], bufs[3], p]))
    for _ in range(2): ins.append(Instruction("FMUL", bufs[3], [bufs[1], bufs[3]]))
    ins.extend([Instruction("FFMA", bufs[3], [bufs[1], "1.4426950216293334961", bufs[3]]),
                Instruction("FFMA", dest, [bufs[0], "1.1920928955078125e-07", dest]),
                Instruction("MOV", bufs[0], ["0x7f800000"], pred=pred),
                Instruction("FADD", dest, [dest, bufs[3]]),
                Instruction("FFMA", dest, [src, bufs[0], "+INF"], pred=pred),
                Instruction("FSETP", pred, ["PT", src, "RZ", "PT"], mods=["NEU", "AND"]),
                Instruction("FSEL", dest, [dest, "-INF", pred])])
    return ins

  def render_sqrt(self, dest:Register, src:Register, bar_regs:Register, bar_lab:str, bufs:List[Register]) -> List[Instruction]: # TODO: only valid for input < 2^102. branch to __internal_0_$__cuda_sm20_sqrt_rn_f32_slowpath
    return [Instruction("BSSY", bar_regs, [f"`({bar_lab})"]),
            Instruction("IADD3", bufs[0], [src, "-0xd000000", "RZ"]),
            Instruction("MUFU", dest, [src], mods=["RSQ"]),
            Instruction("FMUL", bufs[1], [src, dest], mods=["FTZ"]),
            Instruction("FMUL", dest, [dest, "0.5"], mods=["FTZ"]),
            Instruction("FFMA", bufs[2], [bufs[1].negate(), bufs[1], src]),
            Instruction("FFMA", dest, [bufs[2], dest, bufs[1]]),
            Instruction("BSYNC", None, [bar_regs]),
            Instruction(bar_lab, None, [], label=True)]

  def render(self, name:str, uops:List[UOp]) -> str:
    attr:Dict[str, int] = {"PARAM_COUNT": 0}
    kernel:List[Instruction] = []
    vals:Dict[Any, Union[Register,str]] = {}

    reg_cap = 251 # TODO: remove
    reg_cnt = 0
    def new_reg(byte_size:int=4, uniform:bool=False) -> Register:
      nonlocal reg_cnt
      n = nregs(byte_size)
      idx = n * ((reg_cnt + n - 1) // n) # ceil
      reg_cnt = idx + n
      assert reg_cnt <= reg_cap, "trying to assign to new register: all registers filled" # TODO: remove & optim regs after render
      return Register(idx, size=n, uniform=uniform)

    pred_cap = 6 # TODO: remove
    pred_cnt = 0
    def new_pred() -> Register:
      nonlocal pred_cnt
      idx = pred_cnt
      pred_cnt += 1
      assert pred_cnt <= pred_cap, "trying to assign to new predicate: all registers filled" # TODO: remove & optim regs after render
      return Register(idx, pred=True)

    bar_reg_cnt = 0
    def new_bar_reg() -> Register:
      nonlocal bar_reg_cnt
      idx = bar_reg_cnt
      bar_reg_cnt += 1
      return Register(idx, barrier=True)

    iter_stack = []
    label_cnt = 0
    def new_label():
      nonlocal label_cnt
      label_cnt += 1
      return label_cnt - 1

    def unity() -> Register:
      if not 1 in vals: vals[1] = queue(None, self.render_mov(new_reg(), "0x1", dtypes.int))
      return vals[1]

    def queue(uop:UOp, instrs:Union[List[Instruction], Instruction]) -> Register:
      if not isinstance(instrs, list): instrs = [instrs]
      kernel.extend(instrs)
      base_regs = [ins.dest for ins in instrs if ins.dest is not None and ins.dest.idx % ins.dest.size == 0]
      return base_regs[-1] if base_regs else instrs[-1].dest

    def to_var(uop:UOp) -> Union[Register, str]:
      return var if isinstance(var := vals[uop], Register) or "P" in var else to_reg(uop)

    def to_reg(uop:UOp) -> Register:
      if isinstance(var := vals[uop], Register):
        return queue(uop, self.render_where(new_reg(), var.negate(), vals[0], "0x1", dtypes.int)) if var.pred else var
      vals[uop] = dest = new_reg(uop.dtype.itemsize)
      return queue(uop, self.render_mov(dest, var, uop.dtype))

    def glob_addr(idx:UOp, glob:UOp, pred=None) -> str:
      if idx.op is UOps.CONST:
        g_addr = vals[glob]
        if not isinstance(g_addr, Register):
          vals[glob] = g_addr = new_reg(byte_size=8)
          queue(glob, Instruction("IMAD", g_addr, ["RZ", "RZ", const_addr(glob)], mods=["U32"]))
          queue(glob, Instruction("IMAD", g_addr.offset(1), ["RZ", "RZ", const_addr(glob, offset=4)], mods=["U32"]))
        addr_str = g_addr.render() + ".64"
        if idx.arg != 0:
          addr_str += f"+{hex(idx.arg * glob.dtype.itemsize)}"
      else:
        if glob.dtype.itemsize not in vals:
          vals[glob.dtype.itemsize] = queue(glob, self.render_mov(new_reg(), hex(glob.dtype.itemsize), dtypes.int))
        g_addr = queue(glob, Instruction("IMAD", new_reg(byte_size=8), ["PT"] + [vals[v] for v in [idx, glob.dtype.itemsize, glob]], mods=["WIDE"], pred=pred)) # TODO: PT = hack, need better isa fuzzing
        addr_str = g_addr.render() + ".64"
      return f"desc[{vals["DESC"].render()}][{addr_str}]" # explicit memory descriptor

    def memory_mods(dtype:DType):
      return [f"{'' if dtype.itemsize > 4 else 'U' if dtypes.is_unsigned(dtype) else 'S'}{dtype.itemsize*8}"] if dtype.itemsize != 4 else []

    def dtype_mods(dtype:DType):
      sig = f"{'F' if dtypes.is_float(dtype) else 'U' if dtypes.is_unsigned(dtype) else 'S'}{dtype.itemsize*8}"
      return [sig] if sig not in ["S32", "F32"] else []

    def render_alu(arg:BinaryOps, dest:Register, src_l:Union[Register, str], src_r:Union[Register, str], dtype:DType):
      srcs = [src_l, src_r]
      if dtypes.is_int(dtype):
        if arg is BinaryOps.MUL: srcs.append(vals[0])
        elif arg is BinaryOps.ADD: srcs[1:1] = [unity()]
        else: raise NotImplementedError
      elif not isinstance(srcs[0], Register):
        if isinstance(srcs[1], Register): srcs = srcs[::-1]
        else: srcs[0] = to_reg(vin[0])
      return Instruction(self.alu[arg][dtypes.int if dtypes.is_int(dtype) else dtype], dest, srcs)

    def render_permute(dest, msb, lsb, byte_size):
      return [Instruction("PRMT", dest.offset(i), [msb, "0x5410" if byte_size == 2 else "0x6420", lsb]) for i in range(dest.size)]

    queue(None, Instruction(f".text.{name}", None, [], label=True))
    vals[0] = Register(-1)
    vals[float("inf")] = "INF"
    vals[float("-inf")] = "-INF"
    vals["DESC"] = queue(None, Instruction("ULDC", Register(idx=4, uniform=True), ["c[0x0][0x118]"], mods=["64"])) # load explicit memory descriptor

    for u in uops:
      op, dtype, vin, arg = u.op, u.dtype, u.src, u.arg
      if getenv("PRINT_UOPS", 0): # TODO: remove
        print(f"{op=}, {arg=}, {dtype=}")
        for v in vin:
          print(f"\t{v.op=}, {v.arg=}, {v.dtype=}")
      if op is UOps.SPECIAL:
        vals[u] = queue(u, ins := Instruction("S2R", new_reg(), [(self.tid if (tid := arg[0][:3] == "lid") else self.gid)[dim := int(arg[0][-1])]]))
        if tid: attr[f"BLOCK_DIM_{dim}"] = arg[1]
      elif op is UOps.CONST:
        vals[u] = vals[arg] if arg in vals else render_value(arg, dtype)
      elif op is UOps.DEFINE_GLOBAL:
        vals[u] = const_addr(u)
        attr["PARAM_COUNT"] += 1
      elif op is UOps.DEFINE_ACC:
        vals[u] = queue(u, self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], dtype))
      elif op is UOps.RANGE:
        vals[u] = queue(u, self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], dtype))
        queue(u, Instruction(label := f".LOOP_{new_label()}", None, [], label=True))
        update = render_alu(BinaryOps.ADD, vals[u], vals[u], unity(), dtype)
        branch = self.render_iter(label, new_pred(), vals[u], to_reg(vin[1]), dtype)
        iter_stack.append([update, *branch])
      elif op is UOps.PHI:
        vals[u] = queue(u, self.render_mov(vals[vin[0]], vals[vin[1]], dtype))
      elif op is UOps.ENDRANGE:
        queue(u, iter_stack.pop(-1))
      elif op is UOps.LOAD:
        if vin[0].op is UOps.DEFINE_GLOBAL:
          pred = vals[vin[3]] if len(vin) > 3 else None
          vals[u] = queue(u, ins := Instruction("LDG", new_reg(dtype.itemsize), [glob_addr(vin[1], vin[0], pred=pred)], mods=["E"], pred=pred))
          ins.mods.extend(memory_mods(dtype))
          if pred: queue(u, self.render_mov(vals[u], vals[vin[2]], dtype, pred=pred.negate()))
        else:
          raise NotImplementedError
      elif op is UOps.STORE:
        if vin[0].op is UOps.DEFINE_GLOBAL:
          queue(u, ins := Instruction("STG", None, [glob_addr(vin[1], vin[0]), to_reg(vin[2])], mods=["E"]))
          ins.mods.extend(memory_mods(vin[2].dtype))
        else:
          raise NotImplementedError
      elif op is UOps.CAST:
        if dtypes.is_int(vin[0].dtype):
          if dtypes.is_float(dtype):
            vals[u] = queue(u, ins := Instruction("I2F", new_reg(dtype.itemsize), [vals[vin[0]]]))
            if vin[0].dtype is not dtypes.char: ins.mods.extend(dtype_mods(vin[0].dtype))
            elif dtype is dtypes.float: ins.mods.extend(["S16"]) # NOTE: special case to match nvcc
            if dtype is dtypes.half: ins.mods.extend(["F16"])
          elif dtypes.is_int(dtype):
            if dtype.itemsize > 4:
              vals[u] = dest = queue(u, self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], vin[0].dtype))
              queue(u, Instruction("SHF", dest.offset(1), ["RZ", "0x1f", dest], mods=["R", "HI", "SU"[dtypes.is_unsigned(vin[0].dtype)] + "32"]))
            else:
              vals[u] = vals[vin[0]]
          elif dtype is dtypes.bool:
            vals[u] = queue(u, ins := Instruction(f"ISETP", dest := new_pred(), ["PT", src := to_reg(vin[0]), "RZ", "PT"], mods=["NE", "AND"]))
            if vin[0].dtype is dtypes.long:
              ins.mods.extend(["U32"])
              vals[u] = queue(u, Instruction(f"ISETP", dest, ["PT", src.offset(1), "RZ", "PT", dest], mods=["NE", "AND", "EX"]))
        elif dtypes.is_float(vin[0].dtype):
          if dtypes.is_int(dtype):
            vals[u] = queue(u, ins := Instruction("F2I", new_reg(dtype.itemsize), [to_reg(vin[0])], mods=["TRUNC"]))
            if dtype.itemsize <= 4: ins.mods.extend(["NTZ"])
            if vin[0].dtype.itemsize != 4 or dtype.itemsize > 4: ins.mods.extend(dtype_mods(dtype)) # NOTE: special case to match nvcc
            elif dtypes.is_unsigned(dtype): ins.mods.extend(["U32"]) # NOTE: special case to match nvcc
            if vin[0].dtype is dtypes.half: ins.mods.extend(["F16"])
          elif vin[0].dtype is dtypes.float and dtype is dtypes.half:
            vals[u] = queue(u, Instruction("F2FP", new_reg(dtype.itemsize), ["RZ", vals[vin[0]]], mods=["F16", "F32", "PACK_AB"]))
          elif vin[0].dtype is dtypes.half and dtype is dtypes.float:
            vals[u] = queue(u, Instruction("HADD2", new_reg(dtype.itemsize), ["-RZ", to_reg(vin[0]).modify("H0_H0")], mods=["F32"]))
          elif dtype is dtypes.bool:
            if vin[0].dtype is dtypes.half:
              vals[u] = queue(u, Instruction(f"LOP3", new_pred(), ["RZ", to_reg(vin[0]), "0x7fff", "RZ", self.lop["&"][0], "!PT"], mods=["LUT"]))
            else:
              vals[u] = queue(u, Instruction(f"FSETP", new_pred(), ["PT", to_reg(vin[0]), "RZ", "PT"], mods=["NEU", "AND"]))
        elif vin[0].dtype is dtypes.bool:
          vals[u] = queue(u, self.render_where(new_reg(dtype.itemsize), vals[vin[0]].negate(), vals[0], render_value(1, dtype), dtype))
        else:
          raise NotImplementedError
      elif op is UOps.VECTORIZE:
        if vin[0].dtype.itemsize < 4:
          for nb in range(2, vin[0].dtype.itemsize - 1, -1):
            vals[u] = dest = new_reg(len(vin)*2)
            queue(u, flatten([render_permute(dest.offset(i), to_reg(vin[i*2]), to_reg(vin[i*2+1]), 2) for i in range(dest.size)]))
            if vin[0].dtype.itemsize == 1:
              vals[u] = dest = new_reg(len(vin))
              queue(u, flatten([render_permute(dest.offset(i), dest.offset(0), dest.offset(1), 1) for i in range(dest.size)]))
        elif not is_contiguous(srcs := [vals[v] for v in vin]):
          vals[u] = dest = new_reg(dtype.itemsize)
          n = nregs(vin[0].dtype.itemsize)
          queue(u, flatten([self.render_mov(dest.offset(i*n), s, vin[0].dtype) for i,s in enumerate(srcs)]))
        else:
          srcs[0].size = nregs(dtype.itemsize)
          vals[u] = srcs[0]
      elif op is UOps.GEP:
        vals[u] = vals[vin[0]].offset(arg)
      elif op is UOps.ALU:
        srcs = [vals[v] for v in vin]
        assert arg is TernaryOps.WHERE or all_same(dt := [v.dtype for v in vin]), f"dtype mismatch in alu: {dt}" # TODO: remove
        if arg is BinaryOps.MUL and dtype is dtypes.bool:
          if len(srcs) == 2: srcs.append("PT")
          vals[u] = queue(u, Instruction("PLOP3", new_pred(), ["PT"] + srcs + list(self.lop['&3']), mods=["LUT"]))
        elif arg in [BinaryOps.MUL, BinaryOps.ADD]:
          assert len(srcs) == 2, f"too many sources for mul/add/sub: f{len(srcs)}" # TODO: remove
          vals[u] = queue(u, render_alu(arg, new_reg(dtype.itemsize), *srcs, dtype))
        elif arg is UnaryOps.RECIP:
          vals[u] = queue(u, self.render_recip(new_reg(dtype.itemsize), vals[vin[0]], new_reg(dtype.itemsize), dtype))
        elif arg is BinaryOps.MAX:
          assert len(srcs) == 2, f"too many min/max operands: {len(src)}" # TODO: remove
          vals[u] = queue(u, Instruction(self.alu[arg][dtype], new_reg(vin[0].dtype.itemsize), srcs + ["!PT"])) # TODO: change
        elif arg in [BinaryOps.CMPLT, BinaryOps.CMPNE]:
          assert len(srcs) == 2, f"too many sources for compare: f{len(srcs)}" # TODO: remove
          vals[u] = queue(u, self.render_cmp(arg, new_pred(), *[to_var(v) for v in vin], vin[0].dtype))
        elif arg is TernaryOps.WHERE:
          vals[u] = queue(u, self.render_where(new_reg(dtype.itemsize), *[vals[v] for v in vin], dtype))
        elif arg is UnaryOps.LOG2:
          assert dtype is dtypes.float, f"log2 not supported for {dtype}" # TODO: remove
          vals[u] = queue(u, self.render_log2(new_reg(dtype.itemsize), to_reg(vin[0]), new_pred(), [new_reg(dtype.itemsize) for _ in range(4)]))
        elif arg is UnaryOps.SQRT:
          assert dtype is dtypes.float, f"sqrt not supported for {dtype}" # TODO: remove
          vals[u] = queue(u, self.render_sqrt(new_reg(dtype.itemsize), to_reg(vin[0]), new_bar_reg(), f".SQRT_{new_label()}", [new_reg(dtype.itemsize) for _ in range(3)]))
        else:
          raise NotImplementedError
      else:
        raise NotImplementedError

    queue(None, Instruction("EXIT", None, []))
    queue(None, Instruction(buf_lab := ".L_BUF", None, [], label=True))
    queue(None, Instruction("BRA", None, [f"`({buf_lab})"]))
    for _ in range(10): queue(None, Instruction("NOP", None, [])) # TODO: pad to multiple of 8
    queue(None, Instruction(".L_END", None, [], label=True))

    attr["SHI_REGISTERS"] = reg_cnt + 3 # two internal registers on sm >= 8x, and RZ
    for i,ins in enumerate([ins for ins in kernel if not ins.label]): ins.addr = i*16
    set_ctrl(kernel)
    if getenv("CUASM", 0):
      return attach_sections(''.join(ins.render()+"\n" for ins in kernel[1:]), CUDARenderer("sm_89").render(name, uops), reg_cnt)
    else:
      return ''.join(f"{k}={v}\n" for k,v in attr.items()) + ''.join(ins.render()+"\n" for ins in kernel)

def set_ctrl(kernel):
  def new_bar():
    return open_bar[0] if (open_bar := [i for i in range(6) if i not in active_bar]) else active_bar[0]
  def set_bar(deps, bar_tab):
    active_bar.append(bar := new_bar())
    bar_tab.update({d.base(): bar for d in deps if isinstance(d, Register)})
    return bar
  def wait_bar(deps, bar_tab):
    bars = {bar_tab[d.base()] for d in deps if isinstance(d, Register) and d.base() in bar_tab}
    inst.ctrl.wait.extend(list(bars))
    for k,v in list(bar_tab.items()):
      if v in bars: bar_tab.pop(k, None)
    for b in bars: active_bar.remove(b)
  active_bar = []
  write_bar, read_bar = {}, {}
  for inst in kernel:
    inst.ctrl.write = set_bar([inst.dest], write_bar) if inst.op in write_latency_ops else None
    inst.ctrl.read = set_bar(inst.srcs, read_bar) if inst.op in read_latency_ops else None
    wait_bar(inst.srcs, write_bar)
    wait_bar([inst.dest], read_bar)
    inst.ctrl.yield_ |= inst.ctrl.stall >= 12 # TODO: is it 12?

def attach_sections(sass_src:str, cuda_src:str, reg_cnt:int):
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
      elif line.strip().startswith("//"):
        cuasm += line
        skip = False
  cuasm = re.sub(r"(SHI_REGISTERS=)\d+", f"SHI_REGISTERS={reg_cnt + 2}", cuasm) # 3 registers used internally on sm_89
  cuasm = re.sub(r"\.size( *)([^ ,]*).*", r".size\1\2,(.L_END - \2)", cuasm)
  return cuasm