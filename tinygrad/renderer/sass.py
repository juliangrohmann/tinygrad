import tempfile, hashlib, subprocess, struct, re, math
from enum import Enum, auto
from dataclasses import dataclass, field, asdict, replace
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Union, Optional, cast, Callable
from tinygrad.runtime.support.elf import elf_loader
from tinygrad.runtime.support.compiler_cuda import SASSCompiler
from tinygrad.helpers import getenv, all_same, flatten, to_function_name
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType
from tinygrad.ops import PatternMatcher, UPat, UOps, UOp
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.renderer.cstyle import CUDARenderer
from tinygrad.engine.graph import graph_uops
from CuAsm import CubinFile, CuAsmParser

class SASSOps(Enum): # NOTE: these need to shadow exec_alu patterns to prioritize const folding
  IABS = auto(); FMA = auto(); HI = auto(); FLUSH = auto(); RECIP_APPROX = auto() # noqa: E702

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

def lop(func:Callable[[int, int, int], int]):
  return hex(func(*[int(v, 16) for v in ["F0", "CC", "AA"]]))

@dataclass
class Register:
  idx: int
  size: int = 1
  type: str = "R"
  negated: bool = False
  mod: str = None
  def render(self): return f"{'-!'[self.type == "P"] if self.negated else ''}{self.identity()}{f'.{self.mod}' if self.mod else ''}"
  def offset(self, n): return replace(self, idx=self.idx + n)
  def base(self): return self.offset(-(self.idx % self.size))
  def negate(self): return replace(self, negated=not self.negated)
  def modify(self, mod): return replace(self, mod=mod)
  def identity(self): return f"{self.type}{self.idx if self.idx != -1 else 'Z'}"
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

def recip(x:UOp) -> UOp:
  src = x.cast(dtypes.float)
  dest = src.alu(SASSOps.RECIP_APPROX)
  buf = (dest * src - 1) * -1
  dest = dest * buf + dest
  return src.ne(0).where(dest, src.const(math.inf)).cast(x.dtype)

def idiv(x:UOp, y:UOp) -> UOp: # from nvcc
  abs_y = abs(x)
  abs_x = abs(y)
  buf = (abs_y.cast(dtypes.float).alu(SASSOps.RECIP_APPROX).bitcast(x.dtype) + int("0xffffffe", 16)).cast(x.dtype)
  buf = buf.alu(SASSOps.HI, -buf * abs_y, UOp(UOps.VECTORIZE, x.dtype.vec(2), (x.const(0), buf))).alu(SASSOps.HI, abs_x, x.const(0))
  fma = buf * -abs_y + abs_x
  pred = fma.lt(abs_y)
  buf = pred.where(buf, buf + 1)
  buf = pred.where(fma, fma - abs_y).ge(abs_y).where(buf + 1, buf)
  dest = (-x.alu(BinaryOps.XOR, y).lt(x.const(0))).where(buf, -buf)
  return y.ne(0).where(dest, x.const(int("0x7f800000", 16)))

write_latency_ops = {"MUFU", "LDG", "S2R", "I2F"} # TODO: I2F is only variable latency for cross register (64bit) ops?
read_latency_ops = {"MUFU"}
sass_matcher = PatternMatcher([
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"),UPat(name="k"))), # NOTE: from PTX
   lambda root,x,y,z,k: UOp(root.op, dtypes.uchar, (x,y,z.cast(dtypes.uint8),k)).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())), # NOTE: from PTX
   lambda root: UOp(root.op, dtypes.uchar, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool), UPat())), # NOTE: from PTX
   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool))), # NOTE: from PTX
   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.MUL, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.AND)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.ADD, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.OR)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.MAX, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.OR)),
  # (UPat(UOps.CAST, name="root", dtype={dt for dt in dtypes.fields().values() if dtypes.is_int(dt)},
  #     src=(UPat(UOps.LOAD, name="x", dtype={dt for dt in dtypes.fields().values() if dtypes.is_int(dt)}))),
  #  lambda root, x: UOp(x.op, root.dtype, x.src, x.arg)),
  (UPat(UOps.CAST, name="root", dtype={dt for dt in dtypes.fields().values() if dt.itemsize != 4}, src=(UPat(name="x", dtype=dtypes.bool))),
   lambda root, x: UOp(root.op, root.dtype, src=(x.cast(dtypes.int),))),
  # Shift left/right for int mul/div by 2
  # A, B -> ADD, C -> ADD               ===     A, B, C -> ADD3
  # A, B -> MUL, C -> ADD               ===     A, B, C -> FMA
  # A, B -> CMPNE, CONST True -> CMPNE  ===     A, B -> CMPEQ
  # A, B -> CMP, C -> MUL               ===     A, B, C -> CMP.AND
  # A -> NEG, B -> OP                   ===     -A, B -> OP
  # bool, bool -> OP, bool -> OP        ===     bool, bool, bool -> PLOP3
  # A -> CAST(bool) -> CAST(num)        ===     A -> CAST(num) (results from LOAD/STORE: bool -> uchar cast)
  (UPat(UOps.ALU, UnaryOps.RECIP, dtype={dtypes.float, dtypes.half}, src=(UPat(name="x"))), recip),
  (UPat(UOps.ALU, BinaryOps.IDIV, src=(UPat(name="x"),UPat(name="y"))), idiv)
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
    self.arch = arch # TODO: remove
  # language
  gid = [f'SR_CTAID.{"XYZ"[i]}' for i in range(3)]
  tid = [f'SR_TID.{"XYZ"[i]}' for i in range(3)]
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

  def render_cmp(self, arg:BinaryOps, dest:Register, src_l:Register, src_r:Register, dtype:DType) -> List[Instruction]: # TODO: refactor
    if dtypes.is_int(dtype) or dtypes.is_float(dtype):
      ret = []
      for i in range(0, nregs(dtype.itemsize)):
        op = "ISETP" if dtypes.is_int(dtype) else "FSETP"
        ret.append(ins := Instruction(op, dest, ["PT", src_l.offset(i), src_r.offset(i), "PT"], mods=[f"{self.setp_mod[arg]}", "AND"]))
        if (ext := dtype.itemsize // dtype.count > 4) and i % 2 == 1: # TODO: remove vector cmp
          ins.mods.append("EX")
          ins.srcs.append(dest)
        if dtypes.is_unsigned(dtype) or (ext and i % 2 == 0):
          ins.mods.append("U32")
      return ret
    else:
      return [Instruction("PLOP3", dest, ["PT", src_l, src_r, "PT"] + [lop(lambda a,b,c: (a^b)&c), "0x0"], mods=["LUT"])]

  def render_iter(self, label:str, pred:Register, counter:Register, end:Register, dtype:DType) -> List[Instruction]:
    return [*self.render_cmp(BinaryOps.CMPNE, pred, counter, end, dtype), *self.render_bra(label, pred)]

  def render_bra(self, label:str, pred:Register) -> List[Instruction]:
    return [Instruction("BRA", None, [f"`({label})"], pred=pred)]

  def render_log2(self, dest:Register, src:Register, pred:Register, bufs:List[Register]) -> List[Instruction]: # TODO: move to pattern matcher
    assert len(bufs) == 4, f"expected 4 buffers. {len(bufs)=}"
    ins = [Instruction("FSETP", pred, ["PT", src, "1.175494350822287508e-38", "PT"], mods=["GEU", "AND"]),
           Instruction("FMUL", src, [src, "8388608"], pred=pred.negate()),
           Instruction("IADD3", dest, [src, "-0x3f3504f3", "RZ"]),
           Instruction("LOP3", bufs[0], [dest, "0xff800000", "RZ", lop(lambda a,b,c: a&b), "!PT"], mods=["LUT"]),
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

  def render_sqrt(self, dest:Register, src:Register, bar_regs:Register, bar_lab:str, bufs:List[Register]) -> List[Instruction]: # TODO: only valid for input < 2^102. branch to __internal_0_$__cuda_sm20_sqrt_rn_f32_slowpath, # TODO: move to pattern matcher
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
    if getenv("GRAPHUOPS"): # TODO: remove
      graph_uops(uops)
    if debug_sass := getenv("DEBUG_SASS", ""):
      with open(debug_sass) as f: return f.read()

    attr:Dict[str, int] = {"PARAM_COUNT": 0}
    kernel:List[Instruction] = []
    vals:Dict[Any, Union[Register,str]] = {}
    iter_stack = []

    reg_cnt = defaultdict(int)
    def new_reg(byte_size:int=4, prefix:str="R") -> Register:
      n = nregs(byte_size) if prefix == "R" else 1
      idx = n * ((reg_cnt[prefix] + n - 1) // n) # ceil
      reg_cnt[prefix] = idx + n
      return Register(idx, size=n, type=prefix)

    def unity() -> Register:
      if not 1 in vals: vals[1] = queue(self.render_mov(new_reg(), "0x1", dtypes.int))
      return vals[1]

    def queue(instrs:Union[List[Instruction], Instruction]) -> Register:
      if not isinstance(instrs, list): instrs = [instrs]
      kernel.extend(instrs)
      base_regs = [ins.dest for ins in instrs if ins.dest is not None and ins.dest.idx % ins.dest.size == 0]
      return base_regs[-1] if base_regs else instrs[-1].dest

    def to_var(uop:UOp) -> Union[Register, str]:
      return var if isinstance(var := vals[uop], Register) or "P" in var else to_reg(uop)

    def to_reg(uop:UOp) -> Register:
      if isinstance(var := vals[uop], Register): # TODO: replace with better graph rewrite rules
        return queue(self.render_where(new_reg(), var.negate(), vals[0], "0x1", dtypes.int)) if var.type == "P" else var
      vals[uop] = dest = queue(self.render_mov(new_reg(uop.dtype.itemsize), var, uop.dtype))
      return dest

    def glob_addr(idx:UOp, glob:UOp, pred=None) -> str:
      if idx.op is UOps.CONST:
        g_addr = vals[glob]
        if not isinstance(g_addr, Register):
          vals[glob] = g_addr = new_reg(byte_size=8)
          queue(Instruction("IMAD", g_addr, ["RZ", "RZ", const_addr(glob)], mods=["U32"]))
          queue(Instruction("IMAD", g_addr.offset(1), ["RZ", "RZ", const_addr(glob, offset=4)], mods=["U32"]))
        addr_str = g_addr.render() + ".64"
        if idx.arg != 0:
          addr_str += f"+{hex(idx.arg * glob.dtype.itemsize)}"
      else:
        if glob.dtype.itemsize not in vals:
          vals[glob.dtype.itemsize] = queue(self.render_mov(new_reg(), hex(glob.dtype.itemsize), dtypes.int))
        g_addr = queue(Instruction("IMAD", new_reg(byte_size=8), ["PT"] + [vals[v] for v in [idx, glob.dtype.itemsize, glob]], mods=["WIDE"], pred=pred)) # TODO: PT = hack, need better isa fuzzing
        addr_str = g_addr.render() + ".64"
      return f"desc[{vals["DESC"].render()}][{addr_str}]" # explicit memory descriptor

    def memory_mods(dtype:DType):
      return [f"{'' if dtype.itemsize > 4 else 'U' if dtypes.is_unsigned(dtype) else 'S'}{dtype.itemsize*8}"] if dtype.itemsize != 4 else []

    def dtype_mods(dtype:DType):
      sig = f"{'F' if dtypes.is_float(dtype) else 'U' if dtypes.is_unsigned(dtype) else 'S'}{dtype.itemsize*8}"
      return [sig] if sig not in ["S32", "F32"] else []

    def render_alu(arg:BinaryOps, dest:Register, src_l:Union[Register, str], src_r:Union[Register, str], dtype:DType):
      srcs = [src_l, src_r]
      if dtypes.is_int(dtype): # TODO: replace with better graph rewrite rules
        if arg is BinaryOps.MUL: srcs.append(vals[0])
        elif arg is BinaryOps.ADD: srcs[1:1] = [unity()]
        else: raise NotImplementedError
      elif not isinstance(srcs[0], Register):
        if isinstance(srcs[1], Register): srcs = srcs[::-1]
        else: srcs[0] = to_reg(vin[0])
      ret = [ins := Instruction(op := self.alu[arg][dtypes.int if dtypes.is_int(dtype) else dtype], dest, srcs)]
      if dtypes.is_int(dtype) and arg is BinaryOps.MUL:
        if dtypes.is_unsigned(dtype) or dtype.itemsize > 4:
          ins.mods.append("U32")
        if dtype.itemsize > 4:
          ins.mods.append("WIDE")
          ins.srcs = ["PT"] + ins.srcs
          ret.append(Instruction(op, dest.offset(1), [srcs[0].offset(1), srcs[1], dest.offset(1)]))
          ret.append(Instruction(op, dest.offset(1), [srcs[0], srcs[1].offset(1), dest.offset(1)]))
      return ret

    def render_permute(dest, msb, lsb, byte_size):
      return [Instruction("PRMT", dest.offset(i), [msb, "0x5410" if byte_size == 2 else "0x6420", lsb]) for i in range(dest.size)]

    queue(Instruction(f".text.{name}", None, [], label=True))
    vals[0] = Register(-1)
    vals[float("inf")] = "INF"
    vals[float("-inf")] = "-INF"
    vals["DESC"] = queue(Instruction("ULDC", Register(idx=4, type="UR"), ["c[0x0][0x118]"], mods=["64"])) # load explicit memory descriptor

    for u in uops:
      op, dtype, vin, arg = u.op, u.dtype, u.src, u.arg
      if getenv("PRINT_UOPS", 0): # TODO: remove
        print(f"{op=}, {arg=}, {dtype=}")
        for v in vin:
          print(f"\t{v.op=}, {v.arg=}, {v.dtype=}")
      if op is UOps.SPECIAL:
        vals[u] = queue(Instruction("S2R", new_reg(), [(self.tid if (tid := arg[0][:3] == "lid") else self.gid)[dim := int(arg[0][-1])]]))
        if tid: attr[f"BLOCK_DIM_{dim}"] = arg[1]
      elif op is UOps.CONST:
        vals[u] = vals[arg] if arg in vals else render_value(arg, dtype)
      elif op is UOps.DEFINE_GLOBAL:
        vals[u] = const_addr(u)
        attr["PARAM_COUNT"] += 1
      elif op is UOps.DEFINE_ACC:
        vals[u] = queue(self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], dtype))
      elif op is UOps.RANGE:
        vals[u] = queue(self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], dtype))
        queue(Instruction(label := new_reg(prefix=".LOOP_"), None, [], label=True))
        update = render_alu(BinaryOps.ADD, vals[u], vals[u], unity(), dtype)
        branch = self.render_iter(label, new_reg(prefix="P"), vals[u], to_reg(vin[1]), dtype)
        iter_stack.append([update, *branch])
      elif op is UOps.PHI:
        vals[u] = queue(elf.render_mov(vals[vin[0]], vals[vin[1]], dtype))
      elif op is UOps.ENDRANGE:
        queue(iter_stack.pop(-1))
      elif op is UOps.LOAD:
        if vin[0].op is UOps.DEFINE_GLOBAL:
          pred = vals[vin[3]] if len(vin) > 3 else None
          vals[u] = queue(ins := Instruction("LDG", new_reg(dtype.itemsize), [glob_addr(vin[1], vin[0], pred=pred)], mods=["E"], pred=pred))
          ins.mods.extend(memory_mods(dtype))
          if pred: queue(self.render_mov(vals[u], vals[vin[2]], dtype, pred=pred.negate()))
        else:
          raise NotImplementedError
      elif op is UOps.STORE:
        if vin[0].op is UOps.DEFINE_GLOBAL:
          queue(ins := Instruction("STG", None, [glob_addr(vin[1], vin[0]), to_reg(vin[2])], mods=["E"]))
          ins.mods.extend(memory_mods(vin[2].dtype))
        else:
          raise NotImplementedError
      elif op is UOps.CAST:
        if dtypes.is_int(vin[0].dtype):
          if dtypes.is_float(dtype):
            vals[u] = queue(ins := Instruction("I2F", new_reg(dtype.itemsize), [vals[vin[0]]]))
            if vin[0].dtype is not dtypes.char: ins.mods.extend(dtype_mods(vin[0].dtype))
            elif dtype is dtypes.float: ins.mods.extend(["S16"]) # NOTE: special case to match nvcc
            if dtype is dtypes.half: ins.mods.extend(["F16"])
          elif dtypes.is_int(dtype):
            if dtype.itemsize > 4:
              vals[u] = dest = queue(self.render_mov(new_reg(dtype.itemsize), vals[vin[0]], vin[0].dtype))
              if not dtypes.is_unsigned(dtype):
                queue(Instruction("SHF", dest.offset(1), ["RZ", "0x1f", dest], mods=["R", "HI", "S32"]))
            else:
              vals[u] = vals[vin[0]]
          elif dtype is dtypes.bool:
            vals[u] = queue(ins := Instruction(f"ISETP", dest := new_reg(prefix="P"), ["PT", src := to_reg(vin[0]), "RZ", "PT"], mods=["NE", "AND"]))
            if vin[0].dtype is dtypes.long:
              ins.mods.extend(["U32"])
              vals[u] = queue(Instruction(f"ISETP", dest, ["PT", src.offset(1), "RZ", "PT", dest], mods=["NE", "AND", "EX"]))
        elif dtypes.is_float(vin[0].dtype):
          if dtypes.is_int(dtype):
            vals[u] = queue(ins := Instruction("F2I", new_reg(dtype.itemsize), [to_reg(vin[0])], mods=["TRUNC"]))
            if dtype.itemsize <= 4: ins.mods.extend(["NTZ"])
            if vin[0].dtype.itemsize != 4 or dtype.itemsize > 4: ins.mods.extend(dtype_mods(dtype)) # NOTE: special case to match nvcc
            elif dtypes.is_unsigned(dtype): ins.mods.extend(["U32"]) # NOTE: special case to match nvcc
            if vin[0].dtype is dtypes.half: ins.mods.extend(["F16"])
          elif vin[0].dtype is dtypes.float and dtype is dtypes.half:
            vals[u] = queue(Instruction("F2FP", new_reg(dtype.itemsize), ["RZ", vals[vin[0]]], mods=["F16", "F32", "PACK_AB"]))
          elif vin[0].dtype is dtypes.half and dtype is dtypes.float:
            vals[u] = queue(Instruction("HADD2", new_reg(dtype.itemsize), ["-RZ", to_reg(vin[0]).modify("H0_H0")], mods=["F32"]))
          elif dtype is dtypes.bool:
            if vin[0].dtype is dtypes.half:
              vals[u] = queue(Instruction(f"LOP3", new_reg(prefix="P"), ["RZ", to_reg(vin[0]), "0x7fff", "RZ", lop(lambda a,b,c: a&b), "!PT"], mods=["LUT"]))
            else:
              vals[u] = queue(Instruction(f"FSETP", new_reg(prefix="P"), ["PT", to_reg(vin[0]), "RZ", "PT"], mods=["NEU", "AND"]))
        elif vin[0].dtype is dtypes.bool:
          vals[u] = queue(self.render_where(new_reg(dtype.itemsize), vals[vin[0]].negate(), vals[0], render_value(1, dtype), dtype))
        else:
          raise NotImplementedError
      elif op is UOps.BITCAST:
        vals[u] = vals[vin[0]]
      elif op is UOps.VECTORIZE:
        if vin[0].dtype.itemsize < 4:
          for nb in range(2, vin[0].dtype.itemsize - 1, -1):
            vals[u] = dest = new_reg(len(vin)*2)
            queue(flatten([render_permute(dest.offset(i), to_reg(vin[i*2]), to_reg(vin[i*2+1]), 2) for i in range(dest.size)]))
            if vin[0].dtype.itemsize == 1:
              vals[u] = dest = new_reg(len(vin))
              queue(flatten([render_permute(dest.offset(i), dest.offset(0), dest.offset(1), 1) for i in range(dest.size)]))
        elif not is_contiguous(srcs := [vals[v] for v in vin]):
          vals[u] = dest = new_reg(dtype.itemsize)
          n = nregs(vin[0].dtype.itemsize)
          queue(flatten([self.render_mov(dest.offset(i*n), s, vin[0].dtype) for i,s in enumerate(srcs)]))
        else:
          srcs[0].size = nregs(dtype.itemsize)
          vals[u] = srcs[0]
      elif op is UOps.GEP:
        vals[u] = vals[vin[0]].offset(arg)
      elif op is UOps.ALU:
        srcs = [vals[v] for v in vin]
        # assert arg is TernaryOps.WHERE or all_same(dt := [v.dtype for v in vin]), f"dtype mismatch in alu: {dt}" # TODO: remove
        if arg in [BinaryOps.AND, BinaryOps.OR, BinaryOps.XOR]:
          lop_val = lop(lambda a,b,c: a|b if arg is BinaryOps.OR else a&b if arg is BinaryOps.AND else a^b)
          if len(srcs) == 2: srcs.append("PT" if (pred := dtype is dtypes.bool) else "RZ")
          params = ["PLOP3", new_reg(prefix="P"), ["PT"] + srcs + [lop_val, "0x0"]] if pred else ["LOP3", new_reg(dtype.itemsize), srcs + [lop_val, "!PT"]]
          vals[u] = queue(Instruction(*params, mods=["LUT"]))
        elif arg in [BinaryOps.MUL, BinaryOps.ADD]:
          assert len(srcs) == 2, f"too many sources for mul/add/sub: f{len(srcs)}" # TODO: remove
          vals[u] = queue(render_alu(arg, new_reg(dtype.itemsize), *srcs, dtype))
        elif arg is SASSOps.RECIP_APPROX:
          vals[u] = queue(Instruction("MUFU", new_reg(dtype.itemsize), [to_reg(vin[0])], mods=["RCP"]))
        elif arg is BinaryOps.MAX:
          assert len(srcs) == 2, f"too many min/max operands: {len(src)}" # TODO: remove
          vals[u] = queue(Instruction(self.alu[arg][dtype], new_reg(vin[0].dtype.itemsize), srcs + ["!PT"])) # TODO: change
        elif arg in [BinaryOps.CMPLT, BinaryOps.CMPNE]:
          assert len(srcs) == 2, f"too many sources for compare: f{len(srcs)}" # TODO: remove
          vals[u] = queue(self.render_cmp(arg, new_reg(prefix="P"), *[to_var(v) for v in vin], vin[0].dtype))
        elif arg is SASSOps.IABS:
          assert dtypes.is_int(dtype), f"abs only supported for int"
          vals[u] = queue(Instruction("IABS", new_reg(dtype.itemsize), [to_reg(vin[0])]))
        elif arg is SASSOps.HI:
          assert dtypes.is_int(dtype), f"hi only supported for int"
          vals[u] = queue(Instruction("IMAD", new_reg(4), [to_reg(v) for v in vin], mods=["HI"]))
        elif arg is TernaryOps.WHERE:
          vals[u] = queue(self.render_where(new_reg(dtype.itemsize), vals[vin[0]], to_reg(vin[1]), vals[vin[2]], dtype))
        elif arg is UnaryOps.LOG2:
          assert dtype is dtypes.float, f"log2 not supported for {dtype}" # TODO: remove
          vals[u] = queue(self.render_log2(new_reg(dtype.itemsize), to_reg(vin[0]), new_reg(prefix="P"), [new_reg(dtype.itemsize) for _ in range(4)]))
        elif arg is UnaryOps.SQRT:
          assert dtype is dtypes.float, f"sqrt not supported for {dtype}" # TODO: remove
          vals[u] = queue(self.render_sqrt(new_reg(dtype.itemsize), to_reg(vin[0]), new_reg(prefix="B"), new_reg(prefix=".SQRT_"), [new_reg(dtype.itemsize) for _ in range(3)]))
        else:
          raise NotImplementedError
      else:
        raise NotImplementedError

    queue(Instruction("EXIT", None, []))
    queue(Instruction(buf_lab := ".L_BUF", None, [], label=True))
    queue(Instruction("BRA", None, [f"`({buf_lab})"]))
    for _ in range(10): queue(Instruction("NOP", None, [])) # TODO: pad to multiple of 8
    queue(Instruction(".L_END", None, [], label=True))

    attr["SHI_REGISTERS"] = reg_cnt["R"] + 3 # two internal registers on sm >= 8x, and RZ
    for i,ins in enumerate([ins for ins in kernel if not ins.label]): ins.addr = i*16
    set_ctrl(kernel)
    sass_src = ''.join(f"{k}={v}\n" for k,v in attr.items()) + ''.join(ins.render()+"\n" for ins in kernel)

    # debug TODO: remove
    if out_dir := getenv("WRITE_SRC", ""):
      try:
        cuda_src = CUDARenderer(self.arch).render(to_function_name(name), uops)
        with open(fn_cu := Path(out_dir) / "nvcc.cu", "w") as f: f.write(cuda_src)
        subprocess.run(["nvcc", "--cubin", "-arch", "sm_89", "-o", (Path(out_dir) / "nvcc.cubin").as_posix(), (Path(out_dir) / "nvcc.cu").as_posix()])
        with open(Path(out_dir) / "nvcc_cuobjdump.sass", "w") as f:
          subprocess.run(["cuobjdump", "-sass", "-arch", "sm_89", (Path(out_dir) / "nvcc.cubin").as_posix()], stdout=f)
        with open(Path(out_dir) / "nvcc.cubin", "rb") as f: cuda_blob = f.read()
        cuda_kernel = [section for section in elf_loader(cuda_blob)[1] if section.name.startswith(".text")][0].content
        with open(Path(out_dir) / "nvcc.bin", "wb") as f: f.write(cuda_kernel)
        with tempfile.NamedTemporaryFile(suffix=".cubin", delete_on_close=False) as tmp:
          tmp.close()
          subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", tmp.name, fn_cu], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
          CubinFile(tmp.name).saveAsCuAsm(Path(out_dir) / "nvcc.cuasm")
      except Exception:
        pass
      with open(Path(out_dir) / "rendered.sass", "w") as f: f.write(sass_src)
      elf = SASSCompiler(self.arch).compile(sass_src)
      with open(Path(out_dir) / "rendered.cubin", "wb") as f: f.write(elf)
      with open(Path(out_dir) / "rendered_cuobjdump.sass", "w") as f:
        subprocess.run(["cuobjdump", "-sass", "-arch", "sm_89", (Path(out_dir) / "rendered.cubin").as_posix()], stdout=f)
    return sass_src

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
