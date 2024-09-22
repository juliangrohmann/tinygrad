import json, struct, re
from enum import StrEnum, auto
from typing import List, Dict, DefaultDict, Sequence, Tuple, FrozenSet, Callable, Pattern, Union, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
from tinygrad.dtype import ConstType
from tinygrad.helpers import flatten

addr_ops = {'BRA', 'BRX', 'BRXU', 'CALL', 'JMP', 'JMX', 'JMXU', 'RET', 'BSSY', 'SSY', 'CAL', 'PRET', 'PBK'}
cast_ops = {'I2F', 'I2I', 'F2I', 'F2F', 'I2FP', 'I2IP', 'F2IP', 'F2FP', 'FRND'}
pos_dep_ops = {'HMMA', 'IMMA', 'I2IP', 'F2FP', 'I2I', 'F2F', 'IDP', 'TLD4', 'VADD',
               'VMAD', 'VSHL', 'VSHR', 'VSET', 'VSETP', 'VMNMX', 'VABSDIFF', 'VABSDIFF4'}
const_tr = {r'(?<!\.)\bRZ\b': 'R255', r'\bURZ\b': 'UR63', r'\bPT\b': 'P7', r'\bUPT\b': 'UP7', r'\bQNAN\b': 'NAN'}
pre_mod = {'!': 'cNOT', '-': 'cNEG', '|': 'cABS', '~': 'cINV'}
float_fmt = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}
sr_vals = {"SR_TID": 32, "SR_TID.X": 33, "SR_TID.Y": 34, "SR_TID.Z": 35, "SR_CTAID.X": 37, "SR_CTAID.Y": 38, "SR_CTAID.Z": 39, "SR_NTID": 40}
ins_pat = re.compile(r'(?P<Pred>@!?U?P\w\s+)?\s*(?P<Op>[\w\.\?]+)(?P<Operands>.*)')
text_pat = re.compile(r'\[([\w:-]+)\](.*)')
ins_addr = re.compile(r'/\*([\da-fA-F]*)\*/')
reg_imme_addr = re.compile(r'(?P<R>R\d+)\s*(?P<II>-?0x[0-9a-fA-F]+)')
const_mem = re.compile(r'c\[(?P<Bank>0x\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')
insig_whitespace = re.compile(r'((?<=[\w\?]) (?![\w\?]))|((?<![\w\?]) (?=[\w\?]))|((?<![\w\?]) (?![\w\?]))')
whitespace = re.compile(r'\s+')
cpp_comment = re.compile(r'//.*$') # cpp style line comments TODO: needed?
cc_comment = re.compile(r'\/\*.*?\*\/') # c style line comments TODO: needed?
ins_label = re.compile(r'`\(([^\)]+)\)')
def_label = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')

class EncodingType(StrEnum): CONSTANT = auto(); OPERAND = auto(); MODIFIER = auto(); OPERAND_MODIFIER = auto() # noqa: E702
@dataclass(frozen=True)
class Encoding: type:str; key:str; start:int; length:int; value:int; idx:int; shift:int; inverse:bool # noqa: E702
@dataclass(frozen=True)
class OpCodeSpec:
  enc:List[Encoding]; cmods:List[str]; all_mods:FrozenSet[str]; op_mods:List[Dict[str,int]]; vmods:Dict[int,List[Dict[str,int]]] # noqa: E702
  @classmethod
  def from_json(cls, code, enc:List[Dict], cmods:List[str], op_mods:List[Dict[str,int]], vmods:Dict[int,List[Dict[str,int]]]):
    return cls([Encoding(**p) for p in enc],cmods,frozenset([m for g in op_mods for m in g.keys()]+cmods),op_mods,{int(k):v for k,v in vmods.items()})

class InstructionSpec:
  def __init__(self, specs:Sequence[OpCodeSpec]):
    self.specs = {key: [s for s in specs if frozenset(s.cmods) == key] for key in {frozenset(k.cmods) for k in specs}}
    self.code_mods = frozenset({mod for s in specs for mod in s.cmods}) # if mod and not "INVALID" in mod and not "??" in mod})

class SASSAssembler:
  def __init__(self, json_obj:Dict[str,Any]):
    self.isa = {k: InstructionSpec([OpCodeSpec.from_json(**spec) for spec in v]) for k,v in json_obj.items()}

  def assemble(self, ctrl:str, key:str, values:List[Union[int,float]], op_mods:Sequence[str]=(), operand_mods:Optional[Dict[int,Sequence[str]]]=None):
    ctrl_code, inst_code = self.encode_control_code(*parse_ctrl(ctrl)), self.encode_instruction(key, values, op_mods, operand_mods)
    return ((ctrl_code << 105) | inst_code).to_bytes(16, "little")

  def encode_instruction(self, key:str, values:List[ConstType], op_mods:Sequence[str]=(), operand_mods:Optional[Dict[int,Sequence[str]]]=None) -> int:
    def set_bits(value, start, length): return (value & (2 ** length - 1)) << start
    def choose_mod(explicit_mods, spec_mods):
      valid_mods = [m for m in explicit_mods if m in spec_mods]
      return valid_mods[0] if len(valid_mods) else ''

    inst, predicate, values = self.isa[key], values[0], values[1:]
    seen:DefaultDict[int,int] = defaultdict(int)
    spec_group = list(inst.specs.values())[0] if len(inst.specs) == 1 else inst.specs[frozenset(mod for mod in op_mods if mod in inst.code_mods)]
    valid_specs = [s for s in spec_group if all(m in s.all_mods for m in op_mods)]
    assert len(valid_specs), (f"invalid SASS instruction spec:\n{key=}, {values=}, {op_mods=}\n"
                              f"spec group:\n{'\n'.join([f"{i}: {spec.all_mods}" for i,spec in enumerate(spec_group)])}")
    spec, code = valid_specs[0], set_bits(predicate, 12, 4)
    for enc in spec.enc:
      if enc.type == EncodingType.CONSTANT:
        code += set_bits(enc.value, enc.start, enc.length)
      elif enc.type == EncodingType.OPERAND:
        value = encode_float(v, key, op_mods) if isinstance(v := values[enc.idx], float) else v
        if value < 0: value += 2 ** sum(e.length for e in spec.enc if e.type == EncodingType.OPERAND and e.idx == enc.idx)
        if enc.inverse: value ^= 2 ** enc.length - 1
        value = value >> (seen[enc.idx] + enc.shift)
        code += set_bits(value, enc.start, enc.length)
        seen[enc.idx] += enc.length
      elif enc.type == EncodingType.MODIFIER:
        if not len(spec.op_mods[enc.value]): continue # TODO: remove empty mod entries from isa
        chosen_mod = choose_mod(op_mods, list(spec.op_mods[enc.value].keys()))
        code += set_bits(spec.op_mods[enc.value][chosen_mod], enc.start, enc.length)
      elif enc.type == EncodingType.OPERAND_MODIFIER:
        mod_tab = spec.vmods[enc.idx][enc.value]
        explicit_mods = operand_mods[enc.idx] if operand_mods and enc.idx in operand_mods else []
        code += set_bits(spec.vmods[enc.idx][enc.value][choose_mod(explicit_mods, list(mod_tab.keys()))], enc.start, enc.length)
      else:
        raise ValueError(f"Unknown encoding type: {enc.type}")
    return code

  def encode_control_code(self, wait:int, read:int, write:int, yield_:int, stall:int) -> int:
    return sum(c << i for c,i in zip([wait, read, write, yield_, stall], [11, 8, 5, 4, 0]))

  def to_json(self) -> str:
    return json.dumps({key: [asdict(spec) for spec_group in inst.specs.values() for spec in spec_group] for key,inst in self.isa.items()})

class SASSParser:
  def __init__(self, src:str):
    self.eiattr: DefaultDict[str,List] = defaultdict(list)
    self.labels: Dict[str,str] = dict()
    self.parse_labels(src)
    self.parse_attributes(src)

  def parse(self, line:str):
    r = text_pat.match(strip_comments(line).strip())
    assert r, f"invalid SASS line: {line}"
    ctrl, ins = r.groups()
    ins = self.labels_to_addr(ins.strip())
    addr = parse_inst_addr(line)
    key, vals, op_mods, vmods = parse_inst(ins, addr=addr)
    return ctrl, key, vals, op_mods, vmods

  def labels_to_addr(self, s:str):
    r = ins_label.search(s)
    return ins_label.sub(self.labels[r.groups()[0]], s) if r else s

  def parse_labels(self, src:str):
    addr = None
    for line in src.split('\n'):
      if a := parse_inst_addr(line):
        addr = a
      if r := def_label.match(line.strip()):
        if addr: self.labels[r.groups()[0]] = hex(addr + 16)
        elif r.groups()[0].startswith(".text."): self.function_name = r.groups()[0].replace(".text.", "", 1)
        else: raise ValueError

  def parse_attributes(self, src:str):
    block_dims = [1, 1, 1]
    for line in src.split('\n'):
      if dim_match := re.match(r"BLOCK_DIM_(\d)=(\d+)", line):
        block_dims[int(dim_match.groups()[0])] = int(dim_match.groups()[1])
      if reg_match := re.match(r"SHI_REGISTERS=(\d+)", line):
        self.eiattr["EIATTR_REGCOUNT"].append(["FUNC", int(reg_match.groups()[0])])
      if reg_match := re.match(r"SHM_SIZE=(\d+)", line):
        self.eiattr["SHM_SIZE"].append([int(reg_match.groups()[0])])
      if param_match := re.match(r"PARAM_COUNT=(\d+)", line):
        n = int(param_match.groups()[0])
        for i in range(n-1, -1, -1):
          self.eiattr["EIATTR_KPARAM_INFO"].append([0, (8*i << 16) + i, 0x21f000])
        self.eiattr["EIATTR_PARAM_CBANK"].append([".nv.constant0.FUNC", (8*n << 16) + 0x160])
        self.eiattr["EIATTR_CBANK_PARAM_SIZE"].append(8*n)
      if not (text_match := text_pat.match(strip_comments(line).strip())):
        continue
      ins = text_match.groups()[1].strip()
      if ins.startswith("EXIT"):
        self.eiattr["EIATTR_EXIT_INSTR_OFFSETS"].append([parse_inst_addr(line)])
      elif ins.startswith("BAR"):
        self.eiattr["SHF_BARRIERS"].append([parse_inst_addr(line)])
    self.eiattr["EIATTR_MAX_THREADS"].append(block_dims)
    self.eiattr["EIATTR_MAXREG_COUNT"].append(255)

def parse_inst(ins:str, addr:Optional[int]=None):
  for k,v in const_tr.items():
    ins = re.sub(k, v, ins)
  ins = insig_whitespace.sub('', whitespace.sub(' ', ins)).strip(' {};')
  r = ins_pat.match(ins)
  assert r, f"invalid SASS instruction: {ins}"
  op_toks = r.group('Op').split('.')
  op, op_mods = op_toks[0], op_toks[1:]
  keys, vals, vmods = parse_operands(split_operands(r.group('Operands'), op))
  if addr is not None and op in addr_ops and keys[-1] == "I" and 'ABS' not in op:
    vals[-1] -= addr + 16
  return '_'.join([op] + keys), [parse_pred(r.group('Pred'))] + vals, op_mods, dict(vmods)

def parse_ctrl(ctrl:str):
  s_wait, s_read, s_write, s_yield, s_stall = tuple(ctrl.split(':'))
  wait = int(''.join('1' if c != '-' else '0' for c in s_wait[:0:-1]), 2)
  read = int(s_read[1].replace('-', '7'))
  write = int(s_write[1].replace('-','7'))
  yield_ = int(s_yield != 'Y')
  stall = int(s_stall[1:])
  return wait, read, write, yield_, stall

def parse_inst_addr(s:str):
  return int(m.groups()[0], 16) if (m := ins_addr.search(s)) else None

def parse_operands(operands:Sequence[str]):
  parsed = [parse_token(tok) for tok in operands]
  idx_mods = defaultdict(list)
  for i, (_, _, mods) in enumerate(parsed):
    for j, v in mods.items(): idx_mods[i + j].extend(v)
  return flatten(p[0] for p in parsed), flatten(p[1] for p in parsed), idx_mods

def parse_token(token:str):
  for parser, regex in token_formats:
    if r := regex.match(token): return parser(*r.groupdict().values())
  raise ValueError(f"Unexpected token: \"{token}\"")

def parse_const_memory(prefix, bank, addr):
  bk, bv, mods = parse_addr(prefix, bank)
  ak, av, am = parse_addr('', addr)
  mods.update({len(bv) + i: v for i,v in am.items()})
  return [''.join(bk + ak)], bv + av, mods

def parse_addr(prefix:str, addr:str):
  operands = re.split(r'[-+]', addr.strip('[]'))
  keys, vals, mods = parse_operands(operands)
  return [f"{prefix}[{''.join(keys)}]"], vals, mods

def parse_int(value:str):
  return ["I"], [int(value, 16)], {}

def parse_float(value:str):
  value = "0f7fffffff" if value == "NAN" else "0f7f800000" if value == "INF" else value
  return ["FI"], [int(value[2:], 16) if value.startswith("0f") else float(value)], {}

def parse_special_reg(label:str):
  return ["SR"], [sr_vals[label]], {}

def parse_indexed_token(prefix:str, label:str, index:str, post_mods:str): # TODO: how to encode negative registers?
  mods = [c for c in post_mods.split('.') if len(c)] + [pre_mod[c] for c in prefix]
  return [label], [int(index)], {0: mods} if mods else {}

def parse_pred(s:str) -> int:
  if not s: return 7
  v = parse_token(s.strip('@ '))[1]
  return v[0] + 8 if '!' in s else v[0]

def strip_comments(s:str) -> str:
  return re.subn(r'\s+', ' ', cc_comment.subn(' ', cpp_comment.subn(' ', s)[0])[0])[0].strip()

def split_operands(s:str, op:str) -> List[str]:
  s = s.strip()
  if op in addr_ops and (res := reg_imme_addr.search(s)):
    s = s.replace(res.group(), res.group('R')+','+res.group('II'))
  return s.split(',') if s else []

def encode_float(val:float, op:str, mods:Sequence[str]):
  if op[0] in {'H', 'F', 'D'}: prec = op[0]
  else: prec = 'D' if '64' in mods else 'H' if '16' in mods else 'F'
  nbits = 32 if '32I' in mods else -1
  ifmt, ofmt, fullbits, keepbits = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}[prec]
  fb = struct.pack(ifmt, val)
  ival = struct.unpack(ofmt, fb)[0]
  trunc_bits = fullbits - max(nbits, keepbits)
  return ival >> trunc_bits if trunc_bits > 0 else ival

token_formats: Tuple[Tuple[Callable,Pattern],...] = (
  (parse_special_reg, re.compile(r'(?P<Label>SR_[\w\.]+)')),
  (parse_indexed_token, re.compile(r'(?P<Prefix>[!\-|~]?)(?P<Label>R|UR|P|UP|B|SB|SBSET|SR)(?P<Index>\d+)(?P<PostMod>(\.\w+)*)')),
  (parse_const_memory, re.compile(r'(?P<Prefix>\w*)\[(?P<Bank>[\w\.]+)\]\[(?P<Addr>[+-?\w\.]+)\]')),
  (parse_addr, re.compile(r'(?P<Prefix>\w*)\[(?P<Addr>[^\]]+)\]$')),
  (parse_int, re.compile(r'(?P<Value>[-+]?0x[0-9a-fA-F]+)')),
  (parse_float, re.compile(r'^(?P<Value>((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]?NAN)|-?(0[fF][0-9a-fA-F]+))'
                           r'(\.[a-zA-Z]\w*)*$')),
)
