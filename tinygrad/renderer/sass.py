import struct, re, itertools, math
from functools import partial
from dataclasses import dataclass, field, replace
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, DefaultDict, List, Tuple, Set, Union, Optional, Callable
from tinygrad.helpers import data64_le
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType, ConstType, to_dtype
from tinygrad.ops import PatternMatcher, UPat, UOps, UOp
from tinygrad.renderer import Renderer, TensorCore
from tinygrad.codegen.transcendental import xlog2

class SASSOps(Enum): # NOTE: for secondary graph rewrite: match subgraphs to SASS ops
  IABS = auto(); FMA = auto(); IMAD_HI = auto(); SHR_LO = auto(); SHR_HI = auto(); SHL_HI = auto(); WIDE = auto(); DMAX = auto() # noqa: E702
  SET_BITS = auto(); NOT = auto(); RECIP_APPROX = auto(); RECIP_HI = auto(); EXP2_APPROX = auto(); SQRT_APPROX = auto() # noqa: E702

ext_to_word_dt = {dtypes.int64:dtypes.int32, dtypes.uint64:dtypes.uint32, dtypes.float64:dtypes.float32}
inf = {2: 0x7c00, 4: 0x7f800000, 8: 0x7ff0000000000000}

def raw(x:UOp) -> UOp: return x.bitcast(to_dtype(f"uint{8*x.dtype.itemsize}"))
def is_nan(x:UOp) -> UOp: return -raw(x).lt(inf[x.dtype.itemsize] + 1)
def is_inf(x:UOp) -> UOp: return (raw(x) & 2**(x.dtype.itemsize*8 - 1) - 1).eq(inf[x.dtype.itemsize])
def geu(x:UOp, v:ConstType) -> UOp: return (-x.lt(x.const_like(v))) | is_nan(x)
def ext_to_vec(x:UOp) -> UOp:
  assert x.dtype in ext_to_word_dt, f"cannot bitcast {x.dtype} to vector: expected double width dtype"
  return x.bitcast(ext_to_word_dt[x.dtype].vec(2))

def exp2(d:UOp) -> UOp:
  valid = geu(d, -126.0)
  dest = valid.where(d, d * d.const_like(0.5)).alu(SASSOps.EXP2_APPROX)
  return valid.where(dest, dest * dest)

def sqrt(d:UOp) -> UOp: # TODO: valid for double/long/int?
  assert d.dtype in set(numeric) - {dtypes.float16}, f"unsupported dtype for SQRT: {d.dtype}"
  buf = (root := d.alu(SASSOps.SQRT_APPROX)) * d
  return raw(d).eq(inf[d.dtype.itemsize]).where(d.const_like(float("inf")), d.eq(0).where(d.const_like(0), 0.5 * (-buf * buf + d) * root + buf))

def recip_single(x:UOp) -> UOp:
  assert x.dtype is dtypes.float32, f"unsupported dtype for single width RECIP: {x.dtype}"
  approx = x.alu(SASSOps.RECIP_APPROX)
  dest = -approx*(approx*x - 1) + approx
  sinf, zero = [(r:=raw(x)).alu(SASSOps.SET_BITS, r.const_like(v), r.const_like(2**31 - 1)).bitcast(x.dtype) for v in [inf[x.dtype.itemsize], 0]]
  return is_inf(x).where(zero, x.ne(0).where(dest, sinf))

def recip_double(x:UOp) -> UOp:
  xv = ext_to_vec(x)
  assert x.dtype is dtypes.float64 and xv.dtype, f"unsupported dtype for double width RECIP: {x.dtype}"
  rcp_base = (xv.gep(1).bitcast(dtypes.int32) + UOp.const(dtypes.int32, 0x300402)).bitcast(xv.dtype.scalar())
  rcp_ext = xv.gep(1).alu(SASSOps.RECIP_HI)
  approx = UOp(UOps.VECTORIZE, xv.dtype, (rcp_base, rcp_ext)).bitcast(x.dtype)
  buf = -x*approx + 1
  it = approx*(buf*buf + buf) + approx
  return it*(-x*it + 1) + it

def idiv(x:UOp, y:UOp) -> UOp:
  bits = raw(x.cast(fdt := dtypes.float64) / y.cast(fdt))
  assert x.dtype in ints and y.dtype in ints and bits.dtype, f"unsupported dtypes for IDIV: x={x.dtype}, y={y.dtype}"
  exp = (bits & inf[bits.dtype.itemsize]).alu(BinaryOps.SHR, bits.const_like(52)) - (2**10 - 1)
  mask = bits.const_like(1).alu(BinaryOps.SHL, -exp + 52) - 1
  return exp.lt(0).where(x.const_like(0), bits.alu(SASSOps.SET_BITS, bits.const_like(0), mask).bitcast(fdt).cast(x.dtype))

def where_ext(p:UOp, x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  return UOp(UOps.VECTORIZE, xv.dtype, (p.where(xv.gep(0), yv.gep(0)), p.where(xv.gep(1), yv.gep(1)))).bitcast(x.dtype)

def mul_long(x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  base = UOp(UOps.ALU, x.dtype, (raw(xv.gep(0)), raw(yv.gep(0))), SASSOps.WIDE).bitcast(xv.dtype)
  return UOp(UOps.VECTORIZE, xv.dtype, (base.gep(0), xv.gep(0) * yv.gep(1) + xv.gep(1) * yv.gep(0) + base.gep(1))).bitcast(x.dtype)

def add_long(x:UOp, y:UOp) -> UOp:
  xv = ext_to_vec(x)
  base = (fac := raw(xv.gep(0))).alu(SASSOps.WIDE, fac.const_like(1), y).bitcast(xv.dtype)
  return UOp(UOps.VECTORIZE, xv.dtype, (base.gep(0), xv.gep(1) + base.gep(1))).bitcast(x.dtype)

def shf_long(root:UOp, x:UOp, y:UOp) -> UOp:
  xv = ext_to_vec(x)
  assert x.dtype in longs and y.dtype in ints, f"unsupported dtypes for double width shift funnel: root={root.dtype}, x={x.dtype}, y={y.dtype}"
  ext = UOp(UOps.ALU, wdt := ext_to_word_dt[x.dtype], (x, y), SASSOps.SHR_HI if (shr := root.arg == BinaryOps.SHR) else SASSOps.SHL_HI)
  base = UOp(UOps.ALU, wdt, (xv.gep(0), ext_to_vec(y).gep(0) if y.dtype.itemsize > 4 else y), SASSOps.SHR_LO if shr else root.arg)
  return UOp(UOps.VECTORIZE, base.dtype.vec(2), (base, ext)).bitcast(x.dtype)

def set_bits_long(x:UOp, y:UOp, z:UOp) -> UOp:
  xv, yv, zv = ext_to_vec(x), ext_to_vec(y), ext_to_vec(z)
  return UOp(UOps.VECTORIZE, xv.dtype, tuple(xv.gep(i).alu(SASSOps.SET_BITS, yv.gep(i), zv.gep(i)) for i in range(2))).bitcast(x.dtype)

def bitwise_long(root:UOp, x:UOp, y:UOp) -> UOp:
  xv, yv = ext_to_vec(x), ext_to_vec(y)
  base, ext = raw(b := xv.gep(0)).alu(root.arg, raw(yv.gep(0))).bitcast(b.dtype), xv.gep(1).alu(root.arg, yv.gep(1))
  return UOp(UOps.VECTORIZE, xv.dtype, (base, ext)).bitcast(x.dtype)

def mem_offset(root:UOp, idx:UOp, off:UOp):
  sz, glob = UOp.const(dtypes.int32, root.src[0].dtype.itemsize), root.src[0].op is UOps.DEFINE_GLOBAL
  base = idx.cast(dtypes.int32).alu(SASSOps.WIDE, sz, root.src[0].bitcast(dtypes.int64)) if glob else idx
  return UOp(root.op, root.dtype, (base,off*sz)+root.src[2:], None if glob else root.src[0].arg)

shift_consts = set(2 ** i for i in range(64))
half_not_supported = (UnaryOps.RECIP, UnaryOps.EXP2, UnaryOps.LOG2, UnaryOps.SIN, UnaryOps.SQRT)
not_half = tuple(dt for dt in dtypes.fields().values() if dt is not dtypes.half)
ints, floats = tuple(dt for dt in dtypes.fields().values() if dtypes.is_int(dt)), tuple(dt for dt in dtypes.fields().values() if dtypes.is_float(dt))
usig, longs = tuple(dt for dt in dtypes.fields().values() if dtypes.is_unsigned(dt)), (dtypes.int64, dtypes.uint64)
numeric = ints + floats

sass_matcher = PatternMatcher([
  (UPat(UOps.ALU, arg=BinaryOps.MUL, name="root", dtype=ints, src=[UPat.var("x"),UPat.cvar(name="c")]),
   lambda root,x,c: UOp(UOps.ALU, root.dtype, (x, x.const_like(int(math.log2(c.arg)))), BinaryOps.SHL) if c.arg in shift_consts else None),
  (UPat(UOps.ALU, arg=BinaryOps.IDIV, name="root", dtype=ints, src=(UPat.var("x"),UPat.cvar(name="c"))),
   lambda root,x,c: UOp(UOps.ALU, root.dtype, (x, x.const_like(int(math.log2(c.arg)))), BinaryOps.SHR) if c.arg in shift_consts else None),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("z"),UPat.var("k"))),
    lambda root,x,y,z,k: UOp(root.op, dtypes.uint8, (x,y,z.cast(dtypes.uint8),k)).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())),
    lambda root: UOp(root.op, dtypes.uint8, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat.var("z",dtypes.bool), UPat())),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat.var("z",dtypes.bool))),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat((UOps.LOAD, UOps.STORE), name="root", allow_any_len=True,
    src=(UPat((UOps.DEFINE_GLOBAL,UOps.DEFINE_LOCAL)), UPat(UOps.ALU, arg=BinaryOps.ADD, src=[UPat.var("idx"),UPat.cvar("off")]))), mem_offset),
  (UPat((UOps.LOAD, UOps.STORE), name="root", allow_any_len=True, src=(UPat((UOps.DEFINE_GLOBAL,UOps.DEFINE_LOCAL)),UPat.var("x"))),
    lambda root,x: mem_offset(root,*[UOp.const(dtypes.uint32, 0), x][::1 if x.op is UOps.CONST else -1])),
  (UPat(UOps.CAST, name="root", dtype=tuple(dt for dt in dtypes.fields().values() if dt.itemsize != 4), src=(UPat.var("x", dtypes.bool))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.int32),))),
  (UPat(UOps.CAST, name="root", dtype=(dtypes.float64,dtypes.float16), src=(UPat(name="x", dtype=(dtypes.float64, dtypes.float16)))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.float32),))),
  (UPat(UOps.CAST, name="root", dtype=floats, src=(UPat(name="x", dtype=(dtypes.int8,dtypes.uint8)))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.int16),))),
  (UPat(UOps.CAST, name="root", dtype=tuple(dt for dt in ints if dt.itemsize < 4), src=(UPat(name="x", dtype=(dtypes.float64,dtypes.float32)))),
   lambda root,x: UOp(root.op, root.dtype, src=(x.cast(dtypes.uint32 if dtypes.is_unsigned(root.dtype) else dtypes.int32),))),
  (UPat(UOps.ALU, arg=BinaryOps.MAX, dtype=dtypes.float64, src=(UPat.var("x"),UPat.var("y"))),
   lambda x,y: UOp(UOps.ALU, dtypes.bool.vec(2), (x, y), SASSOps.DMAX).gep(0).where(x, y)),
  (UPat(UOps.ALU, name="root", arg=UnaryOps.RECIP, dtype=tuple(dt for dt in ints if dt.itemsize <= 4), src=(UPat.var("x"))),
   lambda root,x: UOp(x.op, dtypes.float32, tuple(vv.cast(dtypes.float32) for vv in x.src), x.arg).cast(root.dtype)),
  (UPat(UOps.ALU, arg=SASSOps.SET_BITS, dtype=longs, src=(UPat.var("x"),UPat.var("y"),UPat.var("z"))), set_bits_long),
  (UPat(UOps.ALU, arg=TernaryOps.WHERE, dtype=tuple(ext_to_word_dt.keys()), src=(UPat.var("p"),UPat.var("x"),UPat.var("y"))), where_ext),
  (UPat(UOps.ALU, arg=TernaryOps.WHERE, dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("z"))), lambda x,y,z: (x&y)|(-x&z)),
  (UPat(UOps.ALU, arg=BinaryOps.CMPLT, dtype=dtypes.bool, src=(UPat.var("x",dtypes.bool),UPat.var("y",dtypes.bool))), lambda x,y: -x&y),
  (UPat(UOps.ALU, arg=BinaryOps.IDIV, src=(UPat.var("x"),UPat.var("y"))), idiv),
  (UPat(UOps.ALU, arg=BinaryOps.MOD, src=(UPat.var("x"),UPat.var("y"))), lambda x,y: x - idiv(x,y)),
  (UPat(UOps.ALU, arg=BinaryOps.CMPNE, dtype=dtypes.bool, src=[UPat.var("x"),UPat.cvar("c",dtypes.bool)]),
   lambda x,c: x.alu(SASSOps.NOT) if c.arg else x),
  *[(UPat(UOps.ALU, name="root", arg=iop, dtype=dtypes.bool, src=UPat(dtype=dtypes.bool)),
    partial(lambda root, oop: UOp(root.op, root.dtype, root.src, oop), oop=oop))
    for (iop,oop) in [(BinaryOps.ADD,BinaryOps.OR), (BinaryOps.MAX,BinaryOps.OR), (BinaryOps.MUL,BinaryOps.AND), (BinaryOps.CMPNE,BinaryOps.XOR)]],
  *[(UPat(UOps.ALU, arg=UnaryOps.RECIP, dtype=dt, src=(UPat.var("x"))), func)
    for dt,func in [(dtypes.float32,recip_single), (dtypes.float64,recip_double)]],
  *[(UPat(UOps.ALU, arg=op, dtype=longs, src=(UPat.var("x"),UPat.var("y"))), func)
    for op,func in [(BinaryOps.MUL,mul_long), (BinaryOps.ADD,add_long)]],
  *[(UPat(UOps.ALU, arg=op, dtype=not_half, src=(UPat.var("d"),)), func)
    for op,func in [(UnaryOps.EXP2,exp2), (UnaryOps.LOG2,xlog2), (UnaryOps.SQRT,sqrt)]],
  *[(UPat(UOps.ALU, name="root", arg=op, dtype=longs, src=(UPat.var("x"),UPat.var("y"))), shf_long)
    for op in [BinaryOps.SHL, BinaryOps.SHR]],
  *[(UPat(UOps.ALU, name="root", arg=op, dtype=longs, src=(UPat.var("x"),UPat.var("y"))), bitwise_long)
    for op in [BinaryOps.AND, BinaryOps.OR, BinaryOps.XOR]],
  *[(UPat(UOps.ALU, arg=op, dtype=dtypes.float16, name="x"),
     lambda x: (UOp(x.op, dtypes.float32, tuple(vv.cast(dtypes.float32) for vv in x.src), x.arg).cast(dtypes.float16))) for op in half_not_supported],
  (UPat(UOps.ALU, name="root", arg=BinaryOps.ADD, src=[UPat(UOps.ALU, arg=BinaryOps.MUL, src=(UPat.var("x"),UPat.var("y"))), UPat.var("z")]),
   lambda root,x,y,z: UOp(root.op, root.dtype, src=(x,y,z), arg=SASSOps.FMA))
])

@dataclass
class Register:
  idx:int; size:int=1; type:str="R"; negated:bool=False; mem_type:Optional[str]=None; postfix:str=""; mod:Optional[str]=None # noqa: E702
  phys:Optional[int]=None; spill:Optional[bool]=None # noqa: E702
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

def nregs(byte_size:int) -> int: return (byte_size + 3) // 4
def const_addr(uop:UOp, offset:int=0) -> str: return f"c[0x0][{hex(int("160", 16) + 8*uop.arg + offset)}]"
def shm_addr(idx:Register, offset:str, dtype:DType): return replace(idx, mem_type="", mod=f"X{dtype.itemsize}", postfix="+"+offset)
def is_contiguous(srcs:List[Register]): return all(s.size == srcs[0].size and s.idx - srcs[0].idx == i * srcs[0].size for i,s in enumerate(srcs))
def is_aligned(src:Register, dtype:DType) -> bool: return src.idx % nregs(dtype.itemsize) == 0
def fill(srcs, count, dtype, val=0): return [srcs[i] if len(srcs) > i else render_value(val, dtype) for i in range(count)]
def dtype_op(op:str, dt:DType) -> str: return (dt.name[0].upper() if dtypes.is_float(dt) else 'I') + op + ('2' if dt is dtypes.float16 else '')
def dtype_mods(dt:DType): return [sig] if (sig:=f"{'F' if dt in floats else 'U' if dt in usig else 'S'}{dt.itemsize*8}") not in ["S32","F32"] else []
def mem_mods(dtype:DType) -> List[str]: return [f"{'' if sz > 4 else 'SU'[dtypes.is_unsigned(dtype)]}{8*sz}"] if (sz := dtype.itemsize) != 4 else []
def prmt_code(a:Register, b:Register) -> str: return "0x"+"".join(str(i+j+2*(r.mod == "H1_H1")) for i in [0,4] for j,r in enumerate([a,b]))[::-1]
def lop_code(func:Callable[[int, int, int], int]) -> str: return hex(func(0xF0, 0xCC, 0xAA))

def render_binary(x, dtype) -> str:
  x = dtypes.as_const(x, dtype) * (-1 if (neg := dtype in ints and x < 0) else 1)
  return f"{'-' if neg else ''}0{'xf'[dtypes.is_float(dtype)]}{struct.pack('>'+{dtypes.int64:'q', dtypes.uint64:'Q'}.get(dtype, dtype.fmt), x).hex()}"

def render_value(x, dtype, allow_reg=True) -> str:
  if dtype is dtypes.bool: return "PT" if x else "!PT"
  if x == 0 and allow_reg: return "RZ"
  return str(x).upper() if dtype in [dtypes.float32, dtypes.float64] else render_binary(x, dtype)

def render_mov(dest:Register, src:Union[Register,str], dtype:DType, pred:Optional[Register]=None) -> List[Instruction]:
  def enc_to_hex(s:str): return ''.join(m.groups()) if (m := re.match(r"(-?)0[xf](.*)", s)) else None
  if isinstance(src, Register):
    srcs = [src.offset(i) if i < src.size - src.idx % src.size else "RZ" for i in range(nregs(dtype.itemsize))]
  else:
    val = int(enc_to_hex(render_binary(float(src), dtype)) if not (h := enc_to_hex(src)) else h, 16)
    srcs = [render_binary(v, dtypes.uint32) if v != 0 else "RZ" for v in ([val] if dtype.itemsize <= 4 else data64_le(val))]
  return [Instruction("MOV", dest.offset(i), [s], pred=pred) for i,s in enumerate(srcs)]

def render_lop(d, s, dt, code) -> List[Instruction]:
  srcs = fill(s, 3, dt, val=True if dt is dtypes.bool else 0)
  return [Instruction("PLOP3", d, ["PT", *srcs, code, "0x0"]) if dt is dtypes.bool else Instruction("LOP3", d, [*srcs, code, "!PT"])]

def render_cmp(d, s, dt, op_mod) -> List[Instruction]:
  return ([Instruction(op := dtype_op("SETP", dt), d, ["PT",*s,"PT"], mods=(m := ["AND", op_mod]) + (["U32"] if dt in usig or dt in longs else []))] +
         ([Instruction(op, d, ["PT",*[v.offset(1) for v in s],"PT",d], mods=m+["EX"]+(["U32"] if dt in usig else []))] if dt in longs else []))

def render_bra(label:str, pred:Optional[Register]=None):
  return Instruction("BRA", None, [f"`({label})"], pred=pred)

def render_shm(idx:Register, offset:str, src:Register, dt:DType, load:bool, pred:Optional[Register]=None):
  return [Instruction(*("LDS",src) if load else ("STS",None), [shm_addr(idx,offset,dt)] + ([src] if not load else []), mods=mem_mods(dt), pred=pred)]

inst_for_cast: Tuple[Tuple[Tuple, Tuple, Callable],...] = (
  (ints, floats, lambda d,s,di,do: Instruction("I2F", d, [s], mods=[] + dtype_mods(di) + dtype_mods(do))),
  (ints, longs, lambda d,s,di,do: render_mov(d,s,do) + ([Instruction("SHF", d.offset(1), ["RZ","0x1f",d], mods=["R","HI","S32"])], [])[di in usig]),
  (floats, ints, lambda d,s,di,do: Instruction("F2I", d, [s], mods=["TRUNC"] + dtype_mods(di) + dtype_mods(do) +
                                                                   (["NTZ"] if do not in longs and di is not dtypes.float64 else []))),
  ((dtypes.float32,), (dtypes.float16,), lambda d,s,di,do: Instruction("F2FP", d, ["RZ", s], mods=["F16","F32","PACK_AB"])),
  ((dtypes.float16,), (dtypes.float32,), lambda d,s,di,do: Instruction("HADD2", d, ["-RZ", s], mods=["F32"])),
  ((dtypes.float64, dtypes.float32), (dtypes.float64, dtypes.float32), lambda d,s,di,do: Instruction("F2F", d, [s], mods=[f"F{do.itemsize * 8}"])),
  ((dtypes.float16,), (dtypes.bool,), lambda d,s,di,do: Instruction("LOP3",d,["RZ",s,"0x7fff","RZ",lop_code(lambda a,b,c: a&b),"!PT"], mods=["LUT"])),
  (floats, (dtypes.bool,), lambda d,s,di,do: Instruction(dtype_op("SETP",di), d, ["PT",s,"RZ","PT"], mods=["NEU","AND"])),
  ((dtypes.bool,), ints+floats, lambda d,s,di,do: inst_for_alu[TernaryOps.WHERE](d, [s.negate(),"RZ",render_value(1, do)], do, None)),
  (ints, (dtypes.bool,), lambda d,s,di,do:
  [Instruction("ISETP", d, ["PT",s,"RZ","PT"], mods=["NE","AND"] + (["U32"] if di in usig or di in longs else []))] +
  ([Instruction("ISETP", d, ["PT",s.offset(1),"RZ","PT",d], mods=["NE","AND","EX"] + (["U32"] if di in usig else []))] if di in longs else [])),
)

inst_for_alu: Dict[Union[Op, SASSOps], Callable] = {
  BinaryOps.ADD: lambda d,s,dt,u: Instruction(dtype_op("ADD", dt) + ['', '3'][dt in ints], d, fill(s, 3, dt) if dt in ints else s),
  BinaryOps.MUL: lambda d,s,dt,u: Instruction("IMAD" if dt in ints else dtype_op("MUL", dt), d, fill(s, 3, dt) if dt in ints else s),
  BinaryOps.MAX: lambda d,s,dt,u: Instruction(dtype_op("MNMX", dt), d, [*s, "!PT"]),
  BinaryOps.SHR: lambda d,s,dt,u: Instruction("SHF", d, [*s, "RZ"], mods=["R", "U32"]),
  BinaryOps.SHL: lambda d,s,dt,u: Instruction("SHF", d, [*s, "RZ"], mods=["L", "U32"]),
  BinaryOps.AND: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a&b if len(s) == 2 else a&b&c)),
  BinaryOps.OR: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a|b if len(s) == 2 else a|b|c)),
  BinaryOps.XOR: lambda d,s,dt,u: render_lop(d, s, dt, lop_code(lambda a,b,c: a^b if len(s) == 2 else a^b^c)),
  BinaryOps.CMPLT: lambda d,s,dt,u: render_cmp(d, s, u.src[0].dtype, "LT"),
  BinaryOps.CMPNE: lambda d,s,dt,u: render_cmp(d, s, sdt := u.src[0].dtype, "NEU" if sdt in floats else "NE"),
  TernaryOps.WHERE: lambda d,s,dt,u: Instruction("SEL" if dt in ints else "FSEL", d, s[1:] + s[0:1]),
  SASSOps.IABS: lambda d,s,dt,u: Instruction("IABS", d, s),
  SASSOps.FMA: lambda d,s,dt,u: Instruction("IMAD" if dt in ints else dtype_op("FMA", dt), d, s),
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
} # TODO: treat lops separately to fuse into arbitrary ternary combinations

commutative = {BinaryOps.ADD, BinaryOps.MUL, BinaryOps.MAX, BinaryOps.AND, BinaryOps.OR, BinaryOps.XOR, BinaryOps.CMPNE, SASSOps.FMA, SASSOps.WIDE,
               SASSOps.DMAX}

class Allocator:
  def __init__(self, vals:Dict[Any, Union[Register,str]]):
    self.vals = vals
    self.counts: Dict[str, int] = defaultdict(int)
  def __call__(self, uop:Optional[UOp], byte_size:Optional[int]=None, prefix:Optional[str]=None): # TODO: bad interface
    n = nregs(byte_size or (8 if isinstance(uop.dtype, PtrDType) else max(uop.dtype.itemsize, 4)) if uop else 1)
    idx = n * ((self.counts[p := prefix or ("P" if uop and uop.dtype is dtypes.bool else "R")] + n - 1) // n) # ceil
    self.counts[p] = idx + n
    ret = Register(idx, size=n, type=p)
    if uop: self.vals[uop] = ret
    return ret

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max = (2147483647, 65535, 65535)
  local_max = (1024, 1024, 64)
  shared_max = 49152
  tensor_cores = [TensorCore(dims=(8,16,16), threads=[(0,2)]*2 + [(1,2)]*3, dtype_in=di, dtype_out=do) for di,do in [(dtypes.float16,dtypes.float32)]]
  extra_matcher = sass_matcher
  code_for_op = {op: lambda x: "" for op in [UnaryOps.EXP2, UnaryOps.LOG2]} # HACK: transcendental override in sass matcher
  def __init__(self, arch:str, device="CUDA"): self.device, self.tensor_cores = device, SASSRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  def render(self, name:str, uops:List[UOp]) -> str:
    attr: Dict[str, int] = defaultdict(int)
    iter_stack: List[List[Instruction]] = []
    kernel: List[Instruction] = []
    r:Dict[Any, Union[Register,str]] = {}
    ssa = Allocator(r)

    def kk(instrs:Union[List[Instruction],Instruction]):
      kernel.extend(instrs := [instrs] if not isinstance(instrs, list) else instrs)
      return next((inst.dest for inst in instrs[::-1] if inst.dest and inst.dest.idx % inst.dest.size == 0), instrs[-1].dest)

    def to_reg(uop:UOp):
      if isinstance(var := r[uop], Register): return var
      if "PT" in var or "RZ" in var: return Register(-1, type="P" if "PT" in var else "R")
      if var.startswith("c["): return kk(inst_for_alu[SASSOps.WIDE](ssa(uop), ["RZ", "RZ", var], uop.dtype, None))
      return kk(render_mov(ssa(uop), var, uop.dtype))

    def fit_consts(op:Any, srcs:Tuple[UOp,...]):
      if op in [SASSOps.WIDE]: return [to_reg(s) for s in srcs]
      allowed = {SASSOps.FMA: (1,2), SASSOps.WIDE: (1,2), TernaryOps.WHERE: (2,)}.get(op, (1,) if len(srcs) > 1 else (0,))
      consts = [i for i,s in enumerate(srcs) if isinstance(v:=r[s], str) and "RZ" not in v and "PT" not in v]
      cidx, swap = next(((i, i) for i in consts if i in allowed), (consts[0], allowed[0]) if consts and op in commutative else (-1, -1))
      regs = [to_reg(s) for i,s in enumerate(srcs) if i != cidx]
      ret = regs if cidx == -1 else [*regs[:swap], r[srcs[cidx]], *regs[swap:]]
      if op is TernaryOps.WHERE and cidx != swap: ret[0] = ret[0].negate()
      if srcs[-1].dtype is dtypes.float16 and op in [BinaryOps.ADD, BinaryOps.MUL, BinaryOps.CMPLT, BinaryOps.CMPNE, SASSOps.FMA] and swap != -1:
        ret[swap:swap] = ["0.0"]
      return ret

    def glob_addr(idx:UOp, offset:Optional[UOp]=None) -> Register:
      return replace(to_reg(idx), mem_type="", mod="64", postfix=f"{'+-'[offset.arg < 0]}{hex(abs(offset.arg))}" if offset and offset.arg else "")

    kk(Instruction(f".text.{name}", None, [], label=True))
    r[0] = Register(-1)
    for u in uops:
      op,dtype,vin,arg = u.op,u.dtype,u.src,u.arg
      if op is UOps.IF:
        kk(render_bra(ssa(u, prefix=".IF_").render(), to_reg(vin[0])))
      elif op is UOps.ENDIF:
        kk(Instruction(label.render() if isinstance(label := r[vin[0]], Register) else label, None, [], label=True))
      elif op is UOps.STORE:
        if arg:
          kk(render_shm(to_reg(vin[0]), render_value(vin[1].arg, dtypes.uint32, False), to_reg(vin[2]), vin[2].dtype, False))
          attr["SHM_SIZE"] = arg[1]*vin[0].dtype.itemsize
        elif any(p.op is UOps.DEFINE_GLOBAL for p in vin[0].sparents):
          assert len(vin) == 3, f"unexpected STORE src count: {u}"
          kk(Instruction("STG", None, [glob_addr(*vin[:2]), to_reg(vin[2])], mods=["E"] + mem_mods(vin[2].dtype)))
        else: raise NotImplementedError
      elif op is UOps.ENDRANGE:
        kk(iter_stack.pop(-1))
      elif op is UOps.BARRIER:
        kk(Instruction("BAR", None, ["0x0"], mods=["SYNC", "DEFER_BLOCKING"]))
      elif op is UOps.SPECIAL:
        kk(Instruction("S2R", ssa(u), [('SR_TID.' if (tid := arg[0][:3] == "lid") else 'SR_CTAID.') + "XYZ"[dim := int(arg[0][-1])]]))
        if tid: attr[f"BLOCK_DIM_{dim}"] = arg[1]
      elif op is UOps.CONST:
        if dtype.itemsize <= 4: r[u] = r[arg] if arg in r else render_value(arg, dtype)
        else: kk(render_mov(ssa(u), render_value(arg, dtype, allow_reg=False), dtype))
      elif op is UOps.DEFINE_GLOBAL:
        r[u] = const_addr(u)
        attr["PARAM_COUNT"] += 1
      elif op is UOps.DEFINE_ACC:
        kk(render_mov(ssa(u), r[vin[0]], dtype))
      elif op is UOps.RANGE:
        kk([*render_mov(ssa(u), r[vin[0]], dtype), Instruction(rng_label := ssa(None, byte_size=4, prefix=".RANGE_").render(), None, [], label=True)])
        pred, counter, end = ssa(None, byte_size=4, prefix="P"), to_reg(u), to_reg(vin[1])
        update = inst_for_alu[BinaryOps.ADD](r[u], [r[u], "0x1" if len(vin) < 3 else to_reg(vin[2])], dtype, u)
        branch = [*inst_for_alu[BinaryOps.CMPNE](pred, [counter, end], dtype, u), render_bra(rng_label, pred)]
        iter_stack.append([update, *branch])
      elif op is UOps.ASSIGN:
        r[u] = kk(render_mov(to_reg(vin[0]), r[vin[1]], dtype))
      elif op is UOps.LOAD:
        gate = to_reg(vin[3]) if len(vin) > 3 else None
        if arg:
          kk(render_shm(to_reg(vin[0]), render_value(vin[1].arg, dtypes.uint32, False), ssa(u), dtype, True))
          attr["SHM_SIZE"] = arg[1]*dtype.itemsize
        elif any(p.op is UOps.DEFINE_GLOBAL for p in vin[0].parents):
          kk(Instruction("LDG", ssa(u), [glob_addr(*vin[:2])], mods=["E"] + mem_mods(dtype), pred=gate))
        else: raise NotImplementedError
        if gate: kk(render_mov(to_reg(u), r[vin[2]], dtype, pred=gate.negate()))
      elif op is UOps.CAST:
        for dti,dto,func in inst_for_cast:
          if vin[0].dtype in dti and dtype in dto:
            kk(func(ssa(u), to_reg(vin[0]), vin[0].dtype, dtype))
            break
        else: r[u] = r[vin[0]]
      elif op is UOps.BITCAST:
        r[u] = vr.replace("0x", "0f") if isinstance(vr := r[vin[0]], str) and dtypes.is_float(dtype) and re.match(r"-?0x", vr) else vr
      elif op is UOps.VECTORIZE:
        if vin[0].dtype is dtypes.float16:
          dest, srcs = ssa(u), [to_reg(v) for v in vin]
          kk([Instruction("PRMT", dest.offset(i // 2), [srcs[i], prmt_code(srcs[i], srcs[i+1]), srcs[i+1]]) for i in range(0, len(srcs), 2)])
        elif not all(isinstance(r[v],Register) for v in vin) or not is_contiguous([to_reg(v) for v in vin]) or not is_aligned(to_reg(vin[0]),dtype):
          dest, n = ssa(u), nregs(vin[0].dtype.itemsize)
          kk([inst for i,s in enumerate([r[v] for v in vin]) for inst in render_mov(dest.offset(i*n), s, vin[0].dtype)])
        else:
          r[u] = r[vin[0]]
          for v in vin: to_reg(v).size = nregs(dtype.itemsize)
      elif op is UOps.GEP:
        assert len(arg) == 1, f"unexpected gep arg: {arg}"
        r[u] = replace(to_reg(vin[0]).offset((b := dtype.itemsize*arg[0])//4), mod='_'.join([f"H{int(b%4 != 0)}"]*2) if dtype.itemsize < 4 else "")
      elif op is UOps.ALU:
        if arg in inst_for_alu: kk(inst_for_alu[arg](ssa(u), fit_consts(arg, vin), dtype, u))
        elif arg is SASSOps.NOT: r[u] = to_reg(vin[0]).negate()
        else: raise NotImplementedError(f"ALU op not implemented: {arg}")
      else: r[u] = "0x0"

    kk(Instruction("EXIT", None, []))
    kk(Instruction(buf_lab := ".L_BUF", None, [], label=True))
    kk(Instruction("BRA", None, [f"`({buf_lab})"]))
    for _ in range(10): kk(Instruction("NOP", None, []))
    kk(Instruction(".L_END", None, [], label=True))

    rewrite_registers(kernel, "P")
    spill_to_flags(kernel, ssa)
    rewrite_registers(kernel, "R")
    attr["SHI_REGISTERS"] = {k: rewrite_registers(kernel, k) for k in ["R", "P", "B"]}["R"] + 3 # two internal registers on sm >= 8x, and RZ
    for x in [s for inst in kernel for s in [inst.pred, inst.dest, *inst.srcs] if isinstance(s, Register)]:
      if x.phys is not None: x.idx = x.phys
    for i,ins in enumerate([ins for ins in kernel if not ins.label]): ins.addr = 16*i
    set_ctrl(kernel)
    return ''.join(f"{k}={v}\n" for k,v in attr.items()) + ''.join(ins.render()+"\n" for ins in kernel)

reg_cap, buf_size = 240, 6

def rewrite_registers(kernel:List[Instruction], reg_type:str) -> int:
  def alloc(size): return next(i for i in itertools.count(reg_type == "P") if i % size == 0 and all(i + j not in allocated for j in range(size)))
  locs: DefaultDict[Tuple[int,int], List[int]] = defaultdict(list)
  loop_deps: Dict[str, Set[Tuple[int,int]]] = {}
  ext_deps, all_reg = {}, set()
  for i,inst in enumerate(kernel):
    if inst.label and "RANGE" in inst.op:
      ext_deps[inst.op], loop_deps[inst.op] = set(locs.keys()), set()
    for r in [src for src in [inst.pred, *inst.srcs, inst.dest] if isinstance(src, Register) and src.idx != -1 and src.type == reg_type]:
      locs[lk := (r.base().idx, r.size)].append(i)
      all_reg.add(r)
      for k,v in loop_deps.items(): v.add(lk)
    if inst.srcs and isinstance(inst.srcs[0], str) and (label := next((k for k in loop_deps.keys() if k in inst.srcs[0]), None)):
      for d in loop_deps.pop(label) & ext_deps.pop(label):
        locs[d].append(i + 1) # keep external loop dependencies live until end of loop
  events = sorted([(k, max(v), False) for k,v in locs.items()] + [(k, min(v) + 0.5, True) for k,v in locs.items()], key=lambda x: x[1])
  allocated, repl = [], {}
  for (idx, size), _, is_alloc in events:
    if is_alloc:
      repl[idx] = base_idx = alloc(size)
      allocated.extend([base_idx + i for i in range(size)])
    elif idx in repl:
      for i in range(size): allocated.remove(repl[idx] + i)
  for reg in all_reg:
    bidx = reg.base().idx
    reg.phys = repl[bidx] + reg.idx - bidx
    if reg_type == "R" and reg.phys is not None:
      reg.spill = reg.phys + reg.size - 1 > reg_cap
  return max([r.phys - (r.phys % r.size) + r.size for r in all_reg if r.phys is not None] + [0])

def pred_cap(kernel:List[Instruction]) -> int:
  cap = 6
  while True: # TODO: refactor
    cnt = max(len([r for r in [*inst.srcs,inst.pred] if isinstance(r,Register) and r.type=="P" and r.idx!=-1 and r.phys is not None and r.phys > cap])
              for inst in kernel)
    if cnt + cap <= 6: return cap
    cap = 6 - cnt

def decouple(inst:Instruction):
  if isinstance(inst.dest, Register): inst.dest = replace(inst.dest)
  if isinstance(inst.pred, Register): inst.pred = replace(inst.pred)
  inst.srcs = [replace(s) if isinstance(s, Register) else s for s in inst.srcs]

def spill_to_flags(kernel:List[Instruction], ssa:Allocator):
  def is_spill(r):
    assert r.idx == -1 or not r.type == "P" or r.phys, f"missing phys: {r}"
    return r.type == "P" and r.idx != -1 and r.phys and r.phys > cap
  def spill_idx(p): return p.phys - cap - 1
  cap, flags, buf, bit_src = pred_cap(kernel), ssa(None, 4), ssa(None, 4), ssa(None, 4)
  if cap >= 6: return
  for inst in kernel: decouple(inst) # TODO: clean up
  kernel[1:1] = [Instruction("MOV", bit_src, [render_value(2**32 - 1, dtypes.uint32)]), Instruction("MOV", flags, ["RZ"])]
  for i in range(len(kernel) - 1, -1, -1):
    if (dp := kernel[i].dest) and isinstance(dp, Register) and is_spill(dp):
      kernel[i+1:i+1] = [Instruction("SEL", buf, [bit_src, "RZ", dp]),
        Instruction("LOP3", flags, [flags, render_value(1 << (spill_idx(dp)), dtypes.uint32), buf, lop_code(lambda a,b,c: (b&c)|(~b&a)), "!PT"])]
      dp.idx = ssa(None, 4, "P").idx
    reads = [r for r in [*kernel[i].srcs, kernel[i].pred] if isinstance(r, Register) and is_spill(r)]
    for sp in reads:
      kernel[i:i] = [*inst_for_alu[BinaryOps.AND](buf, [flags, render_value(1 << (spill_idx(sp)), dtypes.uint32)], dtypes.uint32, None),
                     Instruction("ISETP", sp, ["PT", buf, "RZ", "PT"], mods=["NE", "AND", "U32"])]
      sp.idx = ssa(None, 4, "P").idx

write_latency_ops = {"MUFU", "LDG", "S2R", "I2F", "F2I", "F2F", "DSETP", "DADD", "DMUL", "LDS", "DFMA"} # TODO: casts only var lat for double width
read_latency_ops = {"MUFU", "LDG", "DSETP", "STS", "STG", "F2I", "F2F", "I2F", "DMUL", "DADD", "DFMA"}

def set_ctrl(kernel:List[Instruction]):
  def new_bar(): return open_bar[0] if (open_bar := [i for i in range(6) if i not in active_bar]) else active_bar[0]
  def all_ids(dep): return [dep.base().offset(i).identity() for i in range(dep.size if isinstance(dep, Register) else 0)]
  def set_bar(deps, bar_tab):
    active_bar.append(bar := new_bar())
    bar_tab.update({rid: bar for d in deps for rid in all_ids(d)})
    return bar
  def wait_bar(deps, bar_tab):
    bars = {bar_tab[rid] for d in deps for rid in all_ids(d) if rid in bar_tab}
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
