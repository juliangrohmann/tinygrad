import struct, re, math
from enum import Enum, auto
from dataclasses import dataclass, field, replace
from collections import defaultdict
from typing import Any, Dict, List, Tuple, Set, Sequence, Union, Optional, Callable, cast
from tinygrad.helpers import data64_le
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType, to_dtype
from tinygrad.ops import PatternMatcher, UPat, UOps, UOp
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.codegen.transcendental import xlog2

class SASSOps(Enum): # NOTE: for secondary graph rewrite: match subgraphs to SASS ops
  IABS = auto(); FMA = auto(); IMAD_HI = auto(); SHR_LO = auto(); SHR_HI = auto(); SHL_HI = auto(); WIDE = auto(); DMAX = auto() # noqa: E702
  SET_BITS = auto(); RECIP_APPROX = auto(); RECIP_HI = auto(); EXP2_APPROX = auto(); SQRT_APPROX = auto() # noqa: E702

ext_to_word_dt = {dtypes.long:dtypes.int, dtypes.ulong:dtypes.uint, dtypes.double:dtypes.float}
inf = {2: 0x7c00, 4: 0x7f800000, 8: 0x7ff0000000000000}

def raw(x:UOp) -> UOp: return x.bitcast(to_dtype(f"uint{8*x.dtype.itemsize}")) if x.dtype else x
def is_nan(x:UOp) -> UOp: return -raw(x).lt(inf[x.dtype.itemsize] + 1) if x.dtype else UOp.const(dtypes.bool, False)
def is_inf(x:UOp) -> UOp: return (raw(x) & 2**(x.dtype.itemsize*8 - 1) - 1).eq(inf[x.dtype.itemsize]) if x.dtype else UOp.const(dtypes.bool, False)
def geu(x:UOp, v:ConstType) -> UOp: return (-x.lt(x.const(v))) | is_nan(x)
def ext_to_vec(x:UOp) -> UOp:
  assert x.dtype in ext_to_word_dt, f"cannot bitcast {x.dtype} to vector: expected double width dtype"
  return x.bitcast(ext_to_word_dt[x.dtype].vec(2))

def exp2(x:UOp) -> UOp:
  valid = geu(x, -126.0)
  dest = valid.where(x, x * x.const(0.5)).alu(SASSOps.EXP2_APPROX)
  return valid.where(dest, dest * dest)

def log2(x:UOp) -> UOp: # TODO: faster than transcendental.py for int operands
  denorm = geu(x, 1.175494350822287508e-38)
  src = denorm.where(x, x * 8388608).bitcast(dtypes.uint)
  high = (src - 0x3f3504f3) & 0xff800000
  coeff = (src - high).bitcast(x.dtype) - 1
  buf = coeff + x.const(0.0970201) - x.const(0.16845393180847167969)
  params = [0.1716887056827545166, -0.17900948226451873779, 0.20512372255325317383, -0.24046532809734344482,
            0.28857114911079406738, -0.36067417263984680176, 0.48089820146560668945, -0.72134751081466674805]
  for p in params: buf *= coeff + p
  for _ in range(2): buf *= coeff
  result = high.cast(x.dtype) * 1.1920928955078125e-07 + denorm.where(x.const(0), x.const(-23)) + buf + coeff * 1.4426950216293334961
  return (-x.bitcast(dtypes.uint).lt(0x7f800000)).where(x.const(float("nan")), x.eq(0).where(x.const(-float("inf")), result))

def sqrt(x:UOp) -> UOp:
  assert x.dtype in numeric - {dtypes.half}, f"unsupported dtype for SQRT: {x.dtype}"
  buf = (root := x.alu(SASSOps.SQRT_APPROX)) * x
  return x.bitcast(dtypes.uint).eq(inf[x.dtype.itemsize]).where(x.const(float("inf")), x.eq(0).where(x.const(0), 0.5*(-buf*buf + x)*root + buf))

def recip_single(x:UOp) -> UOp:
  assert x.dtype is dtypes.float, f"unsupported dtype for single width SQRT: {x.dtype}"
  approx = x.alu(SASSOps.RECIP_APPROX)
  dest = -approx*(approx*x - 1) + approx
  sinf, zero = [(r:=raw(x)).alu(SASSOps.SET_BITS, r.const(v), r.const(2**31 - 1)).bitcast(x.dtype) for v in [inf[x.dtype.itemsize], 0]]
  return is_inf(x).where(zero, x.ne(0).where(dest, sinf))

def recip_double(x:UOp) -> UOp:
  xv = ext_to_vec(x)
  assert x.dtype is dtypes.double and xv.dtype, f"unsupported dtype for double width SQRT: {x.dtype}"
  rcp_base = (xv.gep(1).bitcast(dtypes.int) + UOp.const(dtypes.int, 0x300402)).bitcast(xv.dtype.scalar())
  rcp_ext = xv.gep(1).alu(SASSOps.RECIP_HI)
  approx = UOp(UOps.VECTORIZE, xv.dtype, (rcp_base, rcp_ext)).bitcast(x.dtype)
  buf = -x*approx + 1
  it = approx*(buf*buf + buf) + approx
  return it*(-x*it + 1) + it

def idiv(x:UOp, y:UOp) -> UOp:
  bits = raw(x.cast(fdt := dtypes.double) / y.cast(fdt))
  assert x.dtype in ints and y.dtype in ints and bits.dtype, f"unsupported dtypes for IDIV: x={x.dtype}, y={y.dtype}"
  exp = (bits & inf[bits.dtype.itemsize]).alu(BinaryOps.SHR, bits.const(52)) - (2**10 - 1)
  mask = bits.const(1).alu(BinaryOps.SHL, -exp + 52) - 1
  return exp.lt(0).where(x.const(0), bits.alu(SASSOps.SET_BITS, bits.const(0), mask).bitcast(fdt).cast(x.dtype))

def where_ext(p:UOp, x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  return UOp(UOps.VECTORIZE, xv.dtype, (p.where(xv.gep(0), yv.gep(0)), p.where(xv.gep(1), yv.gep(1)))).bitcast(x.dtype)

def mul_long(x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  base = UOp(UOps.ALU, x.dtype, (raw(xv.gep(0)), raw(yv.gep(0))), SASSOps.WIDE).bitcast(xv.dtype)
  return UOp(UOps.VECTORIZE, xv.dtype, (base.gep(0), xv.gep(0) * yv.gep(1) + xv.gep(1) * yv.gep(0) + base.gep(1))).bitcast(x.dtype)

def add_long(x:UOp, y:UOp) -> UOp:
  xv = ext_to_vec(x)
  base = (fac := raw(xv.gep(0))).alu(SASSOps.WIDE, fac.const(1), y).bitcast(xv.dtype)
  return UOp(UOps.VECTORIZE, xv.dtype, (base.gep(0), xv.gep(1) + base.gep(1))).bitcast(x.dtype)

def shf_long(root:UOp, x:UOp, y:UOp) -> UOp:
  xv = ext_to_vec(x)
  assert x.dtype in longs and y.dtype in ints, f"unsupported dtypes for double width shift funnel: root={root.dtype}, x={x.dtype}, y={y.dtype}"
  ext = UOp(UOps.ALU, wdt := ext_to_word_dt[x.dtype], (x, y), SASSOps.SHR_HI if (shr := root.arg == BinaryOps.SHR) else SASSOps.SHL_HI)
  base = UOp(UOps.ALU, wdt, (xv.gep(0), ext_to_vec(y).gep(0) if y.dtype.itemsize > 4 else y), SASSOps.SHR_LO if shr else root.arg)
  return UOp(UOps.VECTORIZE, cast(DType, base.dtype).vec(2), (base, ext)).bitcast(x.dtype)

def set_bits_long(x:UOp, y:UOp, z:UOp) -> UOp:
  xv, yv, zv = ext_to_vec(x), ext_to_vec(y), ext_to_vec(z)
  return UOp(UOps.VECTORIZE, xv.dtype, tuple(xv.gep(i).alu(SASSOps.SET_BITS, yv.gep(i), zv.gep(i)) for i in range(2))).bitcast(x.dtype)

def bitwise_long(root:UOp, x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  base, ext = (b := xv.gep(0)).bitcast(dtypes.uint).alu(root.arg, yv.gep(0).bitcast(dtypes.uint)).bitcast(b.dtype), xv.gep(1).alu(root.arg, yv.gep(1))
  return UOp(UOps.VECTORIZE, xv.dtype, (base, ext)).bitcast(x.dtype)

shift_consts = set(2 ** i for i in range(64))
r_shift = set(1/i for i in range(2, 64))
half_not_supported = [UnaryOps.RECIP, UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.SQRT]
not_half = {dt for dt in dtypes.fields().values() if dt is not dtypes.half}
ints, floats = set(dt for dt in dtypes.fields().values() if dtypes.is_int(dt)), set(dt for dt in dtypes.fields().values() if dtypes.is_float(dt))
usig, longs = set(dt for dt in dtypes.fields().values() if dtypes.is_unsigned(dt)), {dtypes.long, dtypes.ulong}
numeric = ints|floats

sass_matcher = PatternMatcher([
  (UPat(UOps.ALU, BinaryOps.MUL, name="root", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
        src=[UPat(UOps.CONST,  name="const"), UPat(name="mul")]),
   lambda root, mul, const: UOp(UOps.ALU, root.dtype,
                                (mul, UOp.const(dtypes.int, int(math.log2(const.arg)))), BinaryOps.SHL) if const.arg in shift_consts else None),
  (UPat(UOps.ALU, BinaryOps.IDIV, name="root", dtype=set([dt for dt in dtypes.fields().values() if dtypes.is_int(dt)]),
        src=[UPat(UOps.CONST, name="const"), UPat(name="div")]),
   lambda root, div, const: UOp(UOps.ALU, root.dtype,
                                (div, UOp.const(dtypes.int, int(abs(math.log2(const.arg))))), BinaryOps.SHR) if const.arg in shift_consts else None),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"),UPat(name="k"))),
   lambda root,x,y,z,k: UOp(root.op, dtypes.uchar, (x,y,z.cast(dtypes.uint8),k)).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())),
   lambda root: UOp(root.op, dtypes.uchar, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool), UPat())),
   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat(name="z",dtype=dtypes.bool))),
   lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.MUL, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.AND)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.ADD, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.OR)),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.MAX, dtype=dtypes.bool),
   lambda root: UOp(root.op, root.dtype, root.src, BinaryOps.OR)),
  (UPat(UOps.CAST, name="root", dtype={dt for dt in dtypes.fields().values() if dt.itemsize != 4}, src=(UPat(name="x", dtype=dtypes.bool))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.int),))),
  (UPat(UOps.CAST, name="root", dtype={dtypes.double, dtypes.half}, src=(UPat(name="x", dtype={dtypes.double, dtypes.half}))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.float),))),
  (UPat(UOps.CAST, name="root", dtype={dtypes.double, dtypes.float, dtypes.half}, src=(UPat(name="x", dtype={dtypes.char, dtypes.uchar}))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.short),))),
  (UPat(UOps.CAST, name="root", dtype={dt for dt in ints if dt.itemsize < 4}, src=(UPat(name="x", dtype={dtypes.double, dtypes.float}))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.uint if dtypes.is_unsigned(root.dtype) else dtypes.int),))),
  (UPat(UOps.ALU, BinaryOps.MAX, dtype=dtypes.double, src=(UPat(name="x"),UPat(name="y"))),
   lambda x,y: UOp(UOps.ALU, dtypes.bool.vec(2), (x, y), SASSOps.DMAX).gep(0).where(x, y)),
  (UPat(UOps.ALU, BinaryOps.MUL, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"))), mul_long),  # TODO: refactor
  (UPat(UOps.ALU, BinaryOps.ADD, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"))), add_long),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.SHL, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"))), shf_long),
  (UPat(UOps.ALU, name="root", arg=BinaryOps.SHR, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"))), shf_long),
  (UPat(UOps.ALU, arg=SASSOps.SET_BITS, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"))), set_bits_long),
  (UPat(UOps.ALU, TernaryOps.WHERE, dtype=set(ext_to_word_dt.keys()), src=(UPat(name="p"),UPat(name="x"),UPat(name="y"))), where_ext),
  (UPat(UOps.ALU, TernaryOps.WHERE, dtype=dtypes.bool, src=(UPat(name="x"),UPat(name="y"),UPat(name="z"))),
   lambda x,y,z: (x & y) | (x.ne(True) & z)),
  (UPat(UOps.ALU, UnaryOps.RECIP, dtype={dtypes.float}, src=(UPat(name="x"))), recip_single),
  (UPat(UOps.ALU, UnaryOps.RECIP, dtype={dtypes.double}, src=(UPat(name="x"))), recip_double),
  (UPat(UOps.ALU, UnaryOps.RECIP, dtype={dt for dt in ints if dt.itemsize <= 4}, src=(UPat(name="x"))),
   lambda x: (UOp(x.op, dtypes.float, tuple([vv.cast(dtypes.float) for vv in x.src]), x.arg).cast(dtypes.half))),
  (UPat(UOps.ALU, UnaryOps.EXP2, dtype=not_half, src=(UPat(name="x"),)), exp2),
  (UPat(UOps.ALU, UnaryOps.LOG2, dtype=not_half, src=(UPat(name="d"),)), xlog2),
  (UPat(UOps.ALU, UnaryOps.SQRT, dtype=not_half, src=(UPat(name="x"),)), sqrt),
  (UPat(UOps.ALU, BinaryOps.IDIV, src=(UPat(name="x"),UPat(name="y"))), idiv),
  (UPat(UOps.ALU, BinaryOps.MOD, src=(UPat(name="x"),UPat(name="y"))), lambda x,y: x - idiv(x, y)),
  *[(UPat(UOps.ALU, op, dtype=dtypes.half, name="x"),
     lambda x: (UOp(x.op, dtypes.float, tuple([vv.cast(dtypes.float) for vv in x.src]), x.arg).cast(dtypes.half))) for op in half_not_supported],
  *[(UPat(UOps.ALU, name="root", arg=op, dtype={dtypes.long, dtypes.ulong}, src=(UPat(name="x"),UPat(name="y"))),
     bitwise_long) for op in {BinaryOps.AND, BinaryOps.OR, BinaryOps.XOR}],
])

@dataclass
class Register:
  idx:int; size:int=1; type:str="R"; negated:bool=False; mem_type:Optional[str]=None; postfix:str=""; mod:Optional[str]=None # noqa: E702
  def render(self) -> str:
    infix = f"{self.identity()}{f'.{self.mod}' if self.mod else ''}{self.postfix}"
    return f"{'-!'[self.type == "P"] if self.negated else ''}{f'{self.mem_type}[{infix}]' if self.mem_type is not None else infix}"
  def identity(self) -> str: return f"{self.type}{self.idx if self.idx != -1 else 'ZT'[self.type == 'P']}"
  def offset(self, n): return replace(self, idx=self.idx + n)
  def base(self): return self.offset(-(self.idx % self.size))
  def negate(self): return replace(self, negated=not self.negated)
  def __hash__(self): return id(self)

@dataclass
class ControlCode:
  wait:List[int]=field(default_factory=list); read:Optional[int]=None; write:Optional[int]=None; yield_:bool=False; stall:int=15 # noqa: E702
  def render(self) -> str:
    bs = ''.join(['-' if b not in self.wait else str(b) for b in range(6)])
    rs, ws = [v if v is not None else '-' for v in [self.read, self.write]]
    return f"[B{bs}:R{rs}:W{ws}:{'-Y'[self.yield_]}:S{self.stall}]"

@dataclass
class Instruction:
  op:str; dest:Optional[Register]; srcs:List[Union[Register, str]]; ctrl:ControlCode=field(default_factory=ControlCode) # noqa: E702
  mods:List[str]=field(default_factory=list); pred:Optional[Register]=None; label:bool=False; addr:int=-1 # noqa: E702
  def render(self) -> str:
    if self.label: return f"  {self.op}:"
    operands = ', '.join(([self.dest.render()] if self.dest else []) + [s.render() if isinstance(s, Register) else s for s in self.srcs])
    ins = f"{f'@{self.pred.render()} ' if self.pred else ''}{self.op}{''.join([f'.{m}' for m in self.mods])} {operands}"
    return f"{' '*6}{self.ctrl.render()}{' '*9}/*{hex(self.addr)[2:]:>04}*/{' '*19}{ins} ;"

def render_binary(x, dtype) -> str:
  x = abs(x) if (neg := dtypes.is_unsigned(dtype) and x < 0) else x
  return f"{'-' if neg else ''}0x{struct.pack('>' + {dtypes.long: 'q', dtypes.ulong: 'Q'}.get(dtype, dtype.fmt), dtypes.as_const(x, dtype)).hex()}"

def render_value(x, dtype, allow_reg=False) -> str:
  if dtype is dtypes.bool: return "PT" if x else "!PT"
  if x == 0 and allow_reg: return "RZ"
  if dtypes.is_float(dtype): return str(x).upper()
  return render_binary(x, dtype)

def render_mov(dest:Register, src:Union[Register,str], dtype:DType, pred:Optional[Register]=None) -> List[Instruction]:
  if isinstance(src, Register):
    srcs = [src.offset(i) if i < src.size - src.idx % src.size else "RZ" for i in range(nregs(dtype.itemsize))]
  else:
    val = int(render_binary(float(src), dtype) if not re.match(r"-?0x", src) else src, 16)
    srcs = [render_binary(v, dtypes.uint) if v != 0 else "RZ" for v in ([val] if dtype.itemsize <= 4 else data64_le(val))]
  return [Instruction("MOV", dest.offset(i), [s], pred=pred) for i,s in enumerate(srcs)]

def render_cmp(arg:BinaryOps, dest:Register, srcs:List[Register], dtype:DType) -> List[Instruction]:
  if dtypes.is_int(dtype) or dtypes.is_float(dtype):
    ret = []
    op = dtype_op("SETP", dtype)
    ret.append(ins := Instruction(op, dest, ["PT", *srcs, "PT"], mods=["AND"]))
    ins.mods.append(cmp_op := (("NEU" if dtypes.is_float(dtype) else "NE") if arg is BinaryOps.CMPNE else "LT"))
    if dtypes.is_unsigned(dtype) or dtype in [dtypes.long, dtypes.ulong]:
      ins.mods.append("U32")
    if dtype in [dtypes.long, dtypes.ulong]:
      ret.append(ins := Instruction(op, dest, ["PT", srcs[0].offset(1), srcs[1].offset(1), "PT", dest], mods=[cmp_op, "AND", "EX"]))
      if dtypes.is_unsigned(dtype): ins.mods.append("U32")
    return ret
  func = (lambda a,b,c: (a^b)&c) if arg == BinaryOps.CMPNE else lambda a,b,c: (~a)&b&c
  return [Instruction("PLOP3", dest, ["PT", *srcs, "PT", lop_code(func), "0x0"], mods=["LUT"])]

def render_lop(d, s, dt, code) -> List[Instruction]:
  srcs = fill(s, 3, dt, val=True if dt is dtypes.bool else 0)
  return [Instruction("PLOP3", d, ["PT", *srcs, code, "0x0"]) if dt is dtypes.bool else Instruction("LOP3", d, [*srcs, code, "!PT"])]

def render_iter(label:str, pred:Register, counter:Register, end:Register, dtype:DType) -> List[Instruction]:
  return [*render_cmp(BinaryOps.CMPNE, pred, [counter, end], dtype), Instruction("BRA", None, [f"`({label})"], pred=pred)]

def nregs(byte_size) -> int: return (byte_size + 3) // 4
def const_addr(uop:UOp, offset=0) -> str: return f"c[0x0][{hex(int("160", 16) + 8*uop.arg + offset)}]"
def is_contiguous(srcs:List[Register]): return all(s.size == srcs[0].size and s.idx - srcs[0].idx == i * srcs[0].size for i,s in enumerate(srcs))
def is_aligned(src:Register, dtype) -> bool: return src.idx % nregs(dtype.itemsize) == 0
def fill(srcs, count, dtype, val=0): return [srcs[i] if len(srcs) > i else render_value(val, dtype, allow_reg=True) for i in range(count)]
def dtype_op(op, dt) -> str: return (dt.name[0].upper() if dtypes.is_float(dt) else 'I') + op + ('2' if dt is dtypes.half else '')
def dtype_mods(dt:DType): return [sig] if (sig:=f"{'F' if dt in floats else 'U' if dt in usig else 'S'}{dt.itemsize*8}") not in ["S32","F32"] else []
def mem_mods(dtype:DType) -> List[str]: return [f"{'' if sz > 4 else 'SU'[dtypes.is_unsigned(dtype)]}{8*sz}"] if (sz := dtype.itemsize) != 4 else []
def prmt_code(a:Register, b:Register) -> str: return "0x"+"".join(str(i+j+2*(r.mod == "H1_H1")) for i in [0,4] for j,r in enumerate([a,b]))[::-1]
def lop_code(func:Callable[[int, int, int], int]) -> str: return hex(func(0xF0, 0xCC, 0xAA))

inst_for_alu: Dict[Union[Op,SASSOps], Callable] = {
  BinaryOps.ADD: lambda d,s,dt,u: Instruction(dtype_op("ADD", dt) + ['', '3'][dt in ints], d, fill(s, 3, dt) if dt in ints else s),
  BinaryOps.MUL: lambda d,s,dt,u: Instruction("IMAD" if dt in ints else dtype_op("MUL", dt), d, fill(s, 3, dt) if dt in ints else s),
  BinaryOps.MAX: lambda d,s,dt,u: Instruction(dtype_op("MNMX", dt), d, [*s, "!PT"]),
  BinaryOps.SHR: lambda d,s,dt,u: Instruction("SHF", d, [*s, "RZ"], mods=["R", "U32"]),
  BinaryOps.SHL: lambda d,s,dt,u: Instruction("SHF", d, [*s, "RZ"], mods=["L", "U32"]),
  TernaryOps.WHERE: lambda d,s,dt,u: Instruction("SEL" if dt in ints else "FSEL", d, s[1:] + s[0:1]),
  SASSOps.IABS: lambda d,s,dt,u: Instruction("IABS", d, s),
  SASSOps.FMA: lambda d,s,dt,u: Instruction("IMAD" if dt in ints else "FMA", d, s),
  SASSOps.IMAD_HI: lambda d, s, dt, u: Instruction("IMAD", d, fill(s, 3, dt), mods=["HI"]),
  SASSOps.WIDE: lambda d,s,dt,u: Instruction("IMAD", d, ["PT", *fill(s, 3, dt)], mods=["WIDE"] + (["U32"] if u and u.src[0].dtype in usig else [])),
  SASSOps.DMAX: lambda d,s,dt,u: Instruction("DSETP", d, [d.offset(1), *s, "PT"], mods=["MAX", "AND"]),
  SASSOps.SHR_LO: lambda d,s,dt,u: Instruction("SHF", d, [*s, s[0].offset(1)], mods=["R", "U64"]),
  SASSOps.SHR_HI: lambda d,s,dt,u: Instruction("SHF", d, ["RZ", s[1], s[0].offset(1)], mods=["R", "U32", "HI"]),
  SASSOps.SHL_HI: lambda d,s,dt,u: Instruction("SHF", d, [*s, s[0].offset(1)], mods=["L", "U64", "HI"]),
  SASSOps.SET_BITS: lambda d,s,dt,u: Instruction("LOP3", d, [*s, lop_code(lambda a,b,c: (c&b)|(~c&a)), "!PT"]),
  SASSOps.RECIP_APPROX: lambda d,s,dt,u: Instruction("MUFU", d, s, mods=["RCP"]),
  SASSOps.RECIP_HI: lambda d,s,dt,u: Instruction("MUFU", d, s, mods=["RCP64H"]),
  SASSOps.EXP2_APPROX: lambda d,s,dt,u: Instruction("MUFU", d, s, mods=["EX2"]),
  SASSOps.SQRT_APPROX: lambda d,s,dt,u: Instruction("MUFU", d, s, mods=["RSQ"]),
  BinaryOps.AND: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a & b if len(s) == 2 else a & b & c)),
  BinaryOps.OR: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a | b if len(s) == 2 else a | b | c)),
  BinaryOps.XOR: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a ^ b if len(s) == 2 else a ^ b ^ c)),
} # TODO: treat lops separately to fuse into arbitrary ternary combinations

inst_for_cast: Tuple[Tuple[Set, Set, Callable],...] = (
  (ints, floats, lambda d,s,di,do: Instruction("I2F", d, [s], mods=[] + dtype_mods(di) + dtype_mods(do))),
  (ints, longs, lambda d,s,di,do: render_mov(d,s,do) + ([Instruction("SHF", d.offset(1), ["RZ","0x1f",d], mods=["R","HI","S32"])], [])[di in usig]),
  (floats, ints, lambda d,s,di,do: Instruction("F2I", d, [s], mods=["TRUNC"] + dtype_mods(di) + dtype_mods(do) +
                                                                      (["NTZ"] if do not in longs and di is not dtypes.double else []))),
  ({dtypes.float}, {dtypes.half}, lambda d,s,di,do: Instruction("F2FP", d, ["RZ", s], mods=["F16","F32","PACK_AB"])),
  ({dtypes.half}, {dtypes.float}, lambda d,s,di,do: Instruction("HADD2", d, ["-RZ", s], mods=["F32"])),
  ({dtypes.double, dtypes.float}, {dtypes.double, dtypes.float}, lambda d,s,di,do: Instruction("F2F", d, [s], mods=[f"F{do.itemsize * 8}"])),
  ({dtypes.half}, {dtypes.bool}, lambda d,s,di,do: Instruction("LOP3", d, ["RZ",s,"0x7fff","RZ",lop_code(lambda a,b,c: a&b),"!PT"], mods=["LUT"])),
  (floats, {dtypes.bool}, lambda d,s,di,do: Instruction(dtype_op("SETP",di), d, ["PT",s,"RZ","PT"], mods=["NEU","AND"])),
  ({dtypes.bool}, ints|floats, lambda d,s,di,do: inst_for_alu[TernaryOps.WHERE](d, [s.negate(),"RZ",render_value(1, do)], do, None)),
  (ints, {dtypes.bool}, lambda d,s,di,do:
    [Instruction("ISETP", d, ["PT",s,"RZ","PT"], mods=["NE","AND"] + (["U32"] if di in usig or di in longs else []))] +
    ([Instruction("ISETP", d, ["PT",s.offset(1),"RZ","PT",d], mods=["NE","AND","EX"] + (["U32"] if di in usig else []))] if di in longs else [])),
)

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2)]*2 + [(1,2)]*3, dtype_in=di, dtype_out=do) for di,do in [(dtypes.half, dtypes.float)]]
  extra_matcher = sass_matcher
  code_for_op = {op: lambda x: "" for op in [UnaryOps.EXP2, UnaryOps.LOG2]} # HACK: transcendental override in sass matcher
  def __init__(self, arch:str, device="CUDA"): self.device, self.tensor_cores = device, SASSRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  def render(self, name:str, uops:List[UOp]) -> str:
    attr: Dict[str, int] = {"PARAM_COUNT": 0}
    kernel: List[Instruction] = []
    iter_stack: List[List[Instruction]] = []

    c:Dict[str, int] = defaultdict(int)
    r:Dict[Any, Union[Register,str]] = {}
    def ssa(uop:Optional[UOp], byte_size:Optional[int]=None, prefix:Optional[str]=None) -> Register: # TODO: bad interface
      n = nregs(byte_size or (uop and uop.dtype and (8 if isinstance(uop.dtype, PtrDType) else uop.dtype.itemsize)))
      idx = n * ((c[p := prefix or ("P" if uop and uop.dtype is dtypes.bool else "R")] + n - 1) // n) # ceil
      c[p] = idx + n
      ret = Register(idx, size=n, type=p)
      if uop: r[uop] = ret
      return ret

    def kk(instrs:Union[List[Instruction],Instruction]):
      kernel.extend(instrs := [instrs] if not isinstance(instrs, list) else instrs)
      return next((inst.dest for inst in instrs[::-1] if inst.dest and inst.dest.idx % inst.dest.size == 0), instrs[-1].dest)

    def to_reg(uop:UOp) -> Register:
      if isinstance(var := r[uop], Register): return var
      assert uop.dtype, f"cannot move untyped uop to register: {uop=}"
      return Register(-1, type="P") if "PT" in var else kk(render_mov(ssa(uop), var, uop.dtype))

    def glob_addr(idx:UOp, glob:UOp, pred=None) -> Register:
      assert glob.dtype, f"cannot render untyped uop as global address: {glob=}"
      if idx.op is UOps.CONST:
        if not isinstance(g_addr := r[glob], Register):
          g_addr = ssa(glob)
          kk([Instruction("IMAD", g_addr.offset(i), ["RZ", "RZ", const_addr(glob, offset=i*4)], mods=["U32"]) for i in range(2)])
        dest = replace(g_addr, postfix=f"+{hex(idx.arg * glob.dtype.itemsize)}" if idx.arg != 0 else "")
      else:
        if glob.dtype.itemsize not in r:
          r[glob.dtype.itemsize] = kk(render_mov(ssa(None, byte_size=4, prefix="R"), hex(glob.dtype.itemsize), dtypes.uint))
        dest = ssa(None, byte_size=8, prefix="R")
        kk(ins := inst_for_alu[SASSOps.WIDE](dest, [r[v] for v in [idx, glob.dtype.itemsize, glob]], dtypes.ulong, None))
        ins.pred=pred
      return replace(dest, mem_type=f"desc[{desc.render()}]", mod="64") # explicit memory descriptor

    kk(Instruction(f".text.{name}", None, [], label=True))
    r[0] = Register(-1)
    r[float("inf")] = "INF"
    r[-float("inf")] = "-INF"
    desc: Register = kk(Instruction("ULDC", Register(idx=4, type="UR"), ["c[0x0][0x118]"], mods=["64"])) # load explicit memory descriptor

    for u in uops:
      op,dtype,vin,arg = u.op,u.dtype,u.src,u.arg
      if op is UOps.STORE:
        assert vin[2].dtype and vin[0].dtype
        if vin[0].op is UOps.DEFINE_LOCAL:
          kk(Instruction("STS", None, [replace(to_reg(vin[1]), mem_type="", mod=f"X{vin[0].dtype.itemsize}"), to_reg(vin[2])]))
        else:
          kk(Instruction("STG", None, [glob_addr(vin[1], vin[0]), to_reg(vin[2])], mods=["E"] + mem_mods(vin[2].dtype)))
      elif op is UOps.ENDRANGE:
        kk(iter_stack.pop(-1))
      elif op is UOps.BARRIER:
        kk(Instruction("BAR", None, ["0x0"], mods=["SYNC", "DEFER_BLOCKING"]))
      else:
        assert dtype, f"None dtype for uop {u}"
        if op is UOps.DEFINE_LOCAL:
          attr["SHM_SIZE"] = arg[1]*dtype.itemsize
        elif op is UOps.SPECIAL:
          kk(Instruction("S2R", ssa(u), [('SR_TID.' if (tid := arg[0][:3] == "lid") else 'SR_CTAID.') + "XYZ"[dim := int(arg[0][-1])]]))
          if tid: attr[f"BLOCK_DIM_{dim}"] = arg[1]
        elif op is UOps.CONST:
          if dtype.itemsize <= 4: r[u] = r[arg] if arg in r else render_value(arg, dtype)
          else: kk(render_mov(ssa(u), render_value(arg, dtype), dtype))
        elif op is UOps.DEFINE_GLOBAL:
          r[u] = const_addr(u)
          attr["PARAM_COUNT"] += 1
        elif op is UOps.DEFINE_ACC:
          kk(render_mov(ssa(u), r[vin[0]], dtype))
        elif op is UOps.RANGE:
          kk([*render_mov(ssa(u), r[vin[0]], dtype), Instruction(label := ssa(None, byte_size=4, prefix=".LOOP_").render(), None, [], label=True)])
          update = inst_for_alu[BinaryOps.ADD](r[u], [r[u], "0x1" if len(vin) < 3 else to_reg(vin[2])], dtype, u)
          branch = render_iter(label, ssa(None, byte_size=4, prefix="P"), to_reg(u), to_reg(vin[1]), dtype)
          iter_stack.append([update, *branch])
        elif op is UOps.PHI:
          r[u] = kk(render_mov(to_reg(vin[0]), r[vin[1]], dtype))
        elif op is UOps.LOAD:
          assert vin[0].dtype
          if vin[0].op is UOps.DEFINE_LOCAL:
            kk(Instruction("LDS", ssa(u), [replace(to_reg(vin[1]), mem_type="", mod=f"X{vin[0].dtype.itemsize}")]))
          elif vin[0].op is UOps.DEFINE_GLOBAL:
            pred = to_reg(vin[3]) if len(vin) > 3 else None
            kk(Instruction("LDG", ssa(u), [glob_addr(vin[1], vin[0], pred=pred)], mods=["E"] + mem_mods(dtype), pred=pred))
            if pred: kk(render_mov(to_reg(u), r[vin[2]], dtype, pred=pred.negate()))
        elif op is UOps.CAST:
          for dti,dto,func in inst_for_cast:
            if vin[0].dtype in dti and dtype in dto:
              kk(func(ssa(u), to_reg(vin[0]), vin[0].dtype, dtype))
              break
          else: r[u] = r[vin[0]]
        elif op is UOps.BITCAST:
          r[u] = f"0f{vr[2:]}" if isinstance(vr := r[vin[0]], str) and vr.startswith("0x") and dtypes.is_float(dtype) else vr
        elif op is UOps.VECTORIZE:
          if vin[0].dtype is dtypes.half:
            dest, srcs = ssa(u), [to_reg(v) for v in vin]
            kk([Instruction("PRMT", dest.offset(i // 2), [srcs[i], prmt_code(srcs[i], srcs[i+1]), srcs[i+1]]) for i in range(0, len(srcs), 2)])
          elif not all(isinstance(r[v],Register) for v in vin) or not is_contiguous([to_reg(v) for v in vin]) or not is_aligned(to_reg(vin[0]),dtype):
            assert vin[0].dtype
            dest, n = ssa(u), nregs(vin[0].dtype.itemsize)
            kk([inst for i,s in enumerate([r[v] for v in vin]) for inst in render_mov(dest.offset(i*n), s, vin[0].dtype)])
          else:
            r[u] = r[vin[0]]
            for v in vin: to_reg(v).size = nregs(dtype.itemsize)
        elif op is UOps.GEP:
          r[u] = replace(to_reg(vin[0]).offset((b := dtype.itemsize*arg)//4), mod='_'.join([f"H{int(b % 4 != 0)}"]*2) if dtype.itemsize < 4 else "")
        elif op is UOps.ALU:
          if arg in inst_for_alu: kk(inst_for_alu[arg](ssa(u), [to_reg(v) for v in vin], dtype, u))
          elif arg in [BinaryOps.CMPLT, BinaryOps.CMPNE] and vin[0].dtype: kk(render_cmp(arg, ssa(u), [to_reg(v) for v in vin], vin[0].dtype))
          else: raise NotImplementedError
        else: raise NotImplementedError

    kk(Instruction("EXIT", None, []))
    kk(Instruction(buf_lab := ".L_BUF", None, [], label=True))
    kk(Instruction("BRA", None, [f"`({buf_lab})"]))
    for _ in range(10): kk(Instruction("NOP", None, []))
    kk(Instruction(".L_END", None, [], label=True))

    rewrite_registers(kernel, "P")
    spill_to_flags(kernel)
    attr["SHI_REGISTERS"] = {k: rewrite_registers(kernel, k) for k in ["R", "P", "B"]}["R"] + 3 # two internal registers on sm >= 8x, and RZ
    set_ctrl(kernel)
    for i,ins in enumerate([ins for ins in kernel if not ins.label]): ins.addr = 16*i
    return ''.join(f"{k}={v}\n" for k,v in attr.items()) + ''.join(ins.render()+"\n" for ins in kernel)

def rewrite_registers(kernel:List[Instruction], reg_type:str) -> int:
  def alloc(size): return next((i for i in range(reg_type == "P", 256) if i % size == 0 and all(i+j not in allocated for j in range(size))), None)
  unrolled = []
  loops: Dict[str, List[Instruction]] = {}
  for inst in kernel:
    if inst.label and "LOOP" in inst.op:
      loops[inst.op] = []
    unrolled.append(inst)
    for v in loops.values(): v.append(inst)
    if inst.srcs and isinstance(inst.srcs[0], str) and (label := next((k for k in loops.keys() if k in inst.srcs[0]), None)):
      loop_inst = loops.pop(label)
      unrolled.extend(loop_inst)
      for v in loops.values(): v.extend(loop_inst)
  locs, all_reg = defaultdict(list), set()
  for i,r in enumerate([src for inst in unrolled for src in [inst.pred, *inst.srcs, inst.dest]]):
    if isinstance(r, Register) and r.idx != -1 and r.type == reg_type:
      locs[r.base().idx, r.size].append(i)
      all_reg.add(r)
  events = sorted([(k, max(v), False) for k,v in locs.items()] + [(k, min(v), True) for k,v in locs.items()], key=lambda x: x[1])
  allocated, repl = [], {}
  for (idx, size), _, is_alloc in events:
    if is_alloc:
      repl[idx] = base_idx = alloc(size)
      allocated.extend([base_idx + i for i in range(size)])
    elif idx in repl:
      for i in range(size): allocated.remove(repl[idx] + i)
  for reg in all_reg:
    if (bidx := reg.base().idx) in repl:
      reg.idx = repl[bidx] + reg.idx - bidx
  return max([r.base().idx + r.size for r in all_reg] + [0])

def pred_cap(kernel:Sequence[Instruction]) -> int:
  cap = 6
  while True:
    cnt = max(len([r for r in [*inst.srcs, inst.pred] if isinstance(r, Register) and r.type == "P" and r.idx > cap]) for inst in kernel)
    if cnt + cap <= 6: return cap
    cap = 6 - cnt

def spill_to_flags(kernel:List[Instruction]):
  def is_spill(r): return isinstance(r, Register) and r.type == "P" and r.idx > cap
  def flag_idx(p): return p.idx - cap - 1
  cap, flags, buf, bit_src = pred_cap(kernel), Register(-2), Register(-3), Register(-4)
  for inst in kernel: # decouple registers
    if isinstance(inst.dest, Register): inst.dest = replace(inst.dest)
    if isinstance(inst.pred, Register): inst.pred = replace(inst.pred)
    inst.srcs = [replace(s) if isinstance(s, Register) else s for s in inst.srcs]
  if cap >= 6: return
  kernel[1:1] = [Instruction("MOV", bit_src, [render_value(2**32 - 1, dtypes.uint)]), Instruction("MOV", flags, ["RZ"])]
  for i in range(len(kernel) - 1, -1, -1):
    if (dp := kernel[i].dest) and is_spill(dp):
      kernel[i+1:i+1] = [Instruction("SEL", buf, [bit_src, "RZ", dp]),
                         Instruction("LOP3", flags, [flags,render_value(1<<flag_idx(dp),dtypes.uint),buf,lop_code(lambda a,b,c: (b&c)|(~b&a)),"!PT"])]
      dp.idx = cap + 1
    reads = [r for r in [*kernel[i].srcs, kernel[i].pred] if is_spill(r)]
    for j,sp in enumerate(reads):
      kernel[i:i] = [*inst_for_alu[BinaryOps.AND](buf, [flags, render_value(1 << flag_idx(sp), dtypes.uint)], dtypes.uint, None),
                     Instruction("ISETP", Register(cap + j + 1, type="P"), ["PT", buf, "RZ", "PT"], mods=["NE", "AND", "U32"])]
      cast(Register, sp).idx = cap + j + 1

write_latency_ops = {"MUFU", "LDG", "S2R", "I2F", "F2I", "F2F", "DSETP", "DADD", "DMUL", "LDS"} # TODO: casts are only variable lat for double width
read_latency_ops = {"MUFU", "DSETP", "STS"}

def set_ctrl(kernel:List[Instruction]):
  def new_bar():
    return open_bar[0] if (open_bar := [i for i in range(6) if i not in active_bar]) else active_bar[0]
  def set_bar(deps, bar_tab):
    active_bar.append(bar := new_bar())
    bar_tab.update({d.base().identity(): bar for d in deps if isinstance(d, Register)})
    return bar
  def wait_bar(deps, bar_tab):
    bars = {bar_tab[d.base().identity()] for d in deps if isinstance(d, Register) and d.base().identity() in bar_tab}
    inst.ctrl.wait.extend(list(bars))
    for k,v in list(bar_tab.items()):
      if v in bars: bar_tab.pop(k, None)
    for b in bars: active_bar.remove(b)
  active_bar: List[int] = []
  write_bar: Dict[str,int] = {}
  read_bar: Dict[str,int] = {}
  for inst in kernel:
    wait_bar(inst.srcs, write_bar), wait_bar([inst.dest], read_bar)
    inst.ctrl.read = set_bar(inst.srcs, read_bar) if inst.op in read_latency_ops else None
    inst.ctrl.write = set_bar([inst.dest], write_bar) if inst.op in write_latency_ops else None
    inst.ctrl.yield_ |= inst.ctrl.stall >= 12
