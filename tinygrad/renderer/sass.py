import struct, re
from dataclasses import dataclass, field, replace
from collections import defaultdict
from enum import Enum, auto
from typing import Any, Dict, List, Tuple, Union, Optional, Callable
from tinygrad.helpers import data64_le
from tinygrad.ops import BinaryOps, UnaryOps, TernaryOps, Op
from tinygrad.dtype import dtypes, DType, PtrDType
from tinygrad.ops import PatternMatcher, UPat, UOps, UOp
from tinygrad.renderer import Renderer, TensorCore

class SASSOps(Enum): # NOTE: for secondary graph rewrite: match subgraphs to SASS ops
  IABS = auto(); FMA = auto(); IMAD_HI = auto(); SHR_LO = auto(); SHR_HI = auto(); SHL_HI = auto(); WIDE = auto(); DMAX = auto() # noqa: E702
  SET_BITS = auto(); NOT = auto() # noqa: E702

def mem_offset(root:UOp, idx:UOp, off:UOp):
  sz, glob = UOp.const(dtypes.uint32, root.src[0].dtype.itemsize), root.src[0].op is UOps.DEFINE_GLOBAL
  base = idx.cast(dtypes.uint32).alu(SASSOps.WIDE, sz, root.src[0].bitcast(dtypes.uint64)) if glob else idx
  return UOp(root.op, root.dtype, (base,off*sz)+root.src[2:], None if glob else root.src[0].arg)

ints, floats = tuple(dt for dt in dtypes.fields().values() if dtypes.is_int(dt)), tuple(dt for dt in dtypes.fields().values() if dtypes.is_float(dt))
usig, longs = tuple(dt for dt in dtypes.fields().values() if dtypes.is_unsigned(dt)), (dtypes.int64, dtypes.uint64)
numeric = ints + floats

sass_matcher = PatternMatcher([
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat.var("x"),UPat.var("y"),UPat.var("z"),UPat.var("k"))),
    lambda root,x,y,z,k: UOp(root.op, dtypes.uint8, (x,y,z.cast(dtypes.uint8),k)).cast(dtypes.bool)),
  (UPat(UOps.LOAD, name="root", dtype=dtypes.bool, src=(UPat(),UPat())),
    lambda root: UOp(root.op, dtypes.uint8, root.src, root.arg).cast(dtypes.bool)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat.var("z",dtypes.bool), UPat())),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat(UOps.STORE, name="root", src=(UPat(),UPat(),UPat.var("z",dtypes.bool))),
    lambda root,z: UOp(root.op, root.dtype, root.src[:2] + (z.cast(dtypes.uint8),), root.arg)),
  (UPat((UOps.LOAD, UOps.STORE), name="root", allow_any_len=True,
      src=(UPat((UOps.DEFINE_GLOBAL,UOps.DEFINE_LOCAL)), UPat(UOps.ALU, arg=BinaryOps.ADD, src=[UPat.var("x"),UPat.cvar("c")]))), mem_offset),
  (UPat((UOps.LOAD, UOps.STORE), name="root", allow_any_len=True, src=(UPat((UOps.DEFINE_GLOBAL,UOps.DEFINE_LOCAL)),UPat.var("x"))),
    lambda root,x: mem_offset(root,*[UOp.const(dtypes.uint32, 0), x][::1 if x.op is UOps.CONST else -1])),
  (UPat(UOps.ALU, arg=BinaryOps.MAX, dtype=dtypes.float64, src=(UPat.var("x"),UPat.var("y"))),
   lambda x,y: UOp(UOps.ALU, dtypes.bool.vec(2), (x, y), SASSOps.DMAX).gep(0).where(x, y)),
  (UPat(UOps.ALU, arg=BinaryOps.CMPNE, dtype=dtypes.bool, src=[UPat.var("x"),UPat.cvar("c",dtypes.bool)]),
   lambda x,c: x.alu(SASSOps.NOT) if c.arg else x),
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

def nregs(byte_size:int) -> int: return (byte_size + 3) // 4
def const_addr(uop:UOp, offset:int=0) -> str: return f"c[0x0][{hex(int("160", 16) + 8*uop.arg + offset)}]"
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
} # TODO: treat lops separately to fuse into arbitrary ternary combinations

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
    kernel: List[Instruction] = []
    r:Dict[Any, Union[Register,str]] = {}
    c:Dict[str, int] = {}

    def ssa(uop:Optional[UOp], byte_size:Optional[int]=None, prefix:Optional[str]=None): # TODO: bad interface
      n = nregs(byte_size or (8 if isinstance(uop.dtype, PtrDType) else max(uop.dtype.itemsize, 4)) if uop else 1)
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
      if "PT" in var or "RZ" in var: return Register(-1, type="P" if "PT" in var else "R")
      if var.startswith("c["): return kk(inst_for_alu[SASSOps.WIDE](ssa(uop), ["RZ", "RZ", var], uop.dtype, None))
      return kk(render_mov(ssa(uop), var, uop.dtype))

    def glob_addr(idx:UOp, offset:Optional[UOp]=None) -> Register:
      return replace(to_reg(idx), mem_type="", mod="64", postfix=f"+{hex(offset.arg)}" if offset else "")

    kk(Instruction(f".text.{name}", None, [], label=True))
    r[0] = Register(-1)
    for u in uops:
      op,dtype,vin,arg = u.op,u.dtype,u.src,u.arg
      if op is UOps.STORE:
        if any(p.op is UOps.DEFINE_GLOBAL for p in vin[0].sparents):
          assert len(vin) == 3, f"unexpected STORE src count: {u}"
          kk(Instruction("STG", None, [glob_addr(*vin[:2]), to_reg(vin[2])], mods=["E"] + mem_mods(vin[2].dtype)))
        else: raise NotImplementedError
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
      elif op is UOps.LOAD:
        gate = to_reg(vin[3]) if len(vin) > 3 else None
        if any(p.op is UOps.DEFINE_GLOBAL for p in vin[0].parents):
          kk(Instruction("LDG", ssa(u), [glob_addr(*vin[:2])], mods=["E"] + mem_mods(dtype), pred=gate))
        else: raise NotImplementedError
        if gate: kk(render_mov(to_reg(u), r[vin[2]], dtype, pred=gate.negate()))
      elif op is UOps.CAST:
        for dti,dto,func in inst_for_cast:
          if vin[0].dtype in dti and dtype in dto:
            kk(func(ssa(u), to_reg(vin[0]), vin[0].dtype, dtype))
            break
        else: r[u] = r[vin[0]]
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
        if arg in inst_for_alu: kk(inst_for_alu[arg](ssa(u), [to_reg(v) for v in vin], dtype, u))
        elif arg is SASSOps.NOT: r[u] = to_reg(vin[0]).negate()
        else: raise NotImplementedError(f"ALU op not implemented: {arg}")
      else: r[u] = "0x0"

    kk(Instruction("EXIT", None, []))
    kk(Instruction(buf_lab := ".L_BUF", None, [], label=True))
    kk(Instruction("BRA", None, [f"`({buf_lab})"]))
    for _ in range(10): kk(Instruction("NOP", None, []))
    kk(Instruction(".L_END", None, [], label=True))

    for i,ins in enumerate([ins for ins in kernel if not ins.label]): ins.addr = 16*i
    return ''.join(f"{k}={v}\n" for k,v in attr.items()) + ''.join(ins.render()+"\n" for ins in kernel)
