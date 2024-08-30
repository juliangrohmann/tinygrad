import json, pathlib, struct, re
from enum import Enum, auto
from typing import List, Dict, Set, Sequence, FrozenSet, Union, Any
from dataclasses import dataclass, asdict
from collections import defaultdict
from tinygrad.helpers import flatten
from instruction_solver import ISASpec, EncodingRangeType
from parser import InstructionParser

addr_ops = {'BRA', 'BRX', 'BRXU', 'CALL', 'JMP', 'JMX', 'JMXU', 'RET', 'BSSY', 'SSY', 'CAL', 'PRET', 'PBK'}
cast_ops = {'I2F', 'I2I', 'F2I', 'F2F', 'I2FP', 'I2IP', 'F2IP', 'F2FP', 'FRND'}
pos_dep_ops = {'HMMA', 'IMMA', 'I2IP', 'F2FP', 'I2I', 'F2F', 'IDP', 'TLD4', 'VADD',
               'VMAD', 'VSHL', 'VSHR', 'VSET', 'VSETP', 'VMNMX', 'VABSDIFF', 'VABSDIFF4'}
const_tr = {r'(?<!\.)\bRZ\b': 'R255', r'\bURZ\b': 'UR63', r'\bPT\b': 'P7', r'\bUPT\b': 'UP7', r'\bQNAN\b': 'NAN'}
pre_mod = {'!': 'cNOT', '-': 'cNEG', '|': 'cABS', '~': 'cINV'}
float_fmt = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}
ins_pat = re.compile(r'(?P<Pred>@!?U?P\w\s+)?\s*(?P<Op>[\w\.\?]+)(?P<Operands>.*)')
text_pat = re.compile(r'\[([\w:-]+)\](.*)')
ins_addr = re.compile(r'/\*([\da-fA-F]{4})\*/')
reg_imme_addr = re.compile(r'(?P<R>R\d+)\s*(?P<II>-?0x[0-9a-fA-F]+)')
const_mem = re.compile(r'c\[(?P<Bank>0x\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')
insig_whitespace = re.compile(r'((?<=[\w\?]) (?![\w\?]))|((?<![\w\?]) (?=[\w\?]))|((?<![\w\?]) (?![\w\?]))')
whitespace = re.compile(r'\s+')
cpp_comment = re.compile(r'//.*$') # cpp style line comments TODO: needed?
cc_comment = re.compile(r'\/\*.*?\*\/') # c style line comments TODO: needed?
ins_label = re.compile(r'`\(([^\)]+)\)')
def_label = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')

class EncodingType(str, Enum): CONSTANT = auto(); OPERAND = auto(); MODIFIER = auto(); OPERAND_MODIFIER = auto() # noqa: E702
@dataclass(frozen=True)
class Encoding: type:str; key:str; start:int; length:int; value:int; shift:int; offset:int; inverse:bool # noqa: E702
@dataclass(frozen=True)
class OpCodeSpec:
  code:int; enc:List[Dict]; cmods:List[str]; op_mods:List[Dict[str,int]]; vmods:Dict[int,Dict[str,int]] # noqa: E702
  @classmethod
  def from_json(cls, code:int, enc:List[Dict], cmods:List[str], op_mods:List[Dict[str,int]], vmods:Dict[int,Dict[str,int]]):
    return cls(code, [Encoding(**p) for p in enc], cmods, op_mods, {int(k):v for k,v in vmods.items()})

class InstructionSpec:
  def __init__(self, specs:Sequence[OpCodeSpec]):
    self.specs = {frozenset(s.cmods): s for s in specs}
    self.code_mods = frozenset({mod for s in specs for mod in s.cmods}) # if mod and not "INVALID" in mod and not "??" in mod})

class SASSAssembler:
  def __init__(self, json_obj:Dict[str, Any]):
    self.isa = {k: InstructionSpec([OpCodeSpec.from_json(**spec) for spec in v]) for k,v in json_obj.items()}

  def assemble(self, ctrl:str, key:str, values:List[Union[int, float]], op_mods:Sequence[str]=(), operand_mods:Dict[int, Sequence[str]]=None):
    ctrl_code, inst_code = self.encode_control_code(*parse_ctrl(ctrl)), self.encode_instruction(key, values, op_mods, operand_mods)
    return (((ctrl_code << 105) | inst_code) | 1 << 101).to_bytes(16, "little")

  def encode_instruction(self, key:str, values:List[Union[int, float]], op_mods:Sequence[str]=(), operand_mods:Dict[int, Sequence[str]]=None) -> int:
    def set_bits(value, start, length): return (value & (2 ** length - 1)) << start
    predicate, values = values[0], values[1:]
    inst, seen = self.isa[key], defaultdict(int)
    spec = list(inst.specs.values())[0] if len(inst.specs) == 1 else inst.specs[frozenset(mod for mod in op_mods if mod in inst.code_mods)]
    code = set_bits(predicate, 12, 4)
    for enc in spec.enc:
      if enc.type == EncodingType.CONSTANT:
        code += set_bits(enc.value, enc.start, enc.length)
      elif enc.type == EncodingType.OPERAND:
        value = encode_float(v, key, op_mods) if isinstance(v := values[enc.value], float) else v
        if enc.offset: value -= enc.offset
        if value < 0: value += 2 ** sum(e.length for e in spec.enc if e.type == EncodingType.OPERAND and e.value == enc.value)
        if enc.inverse: value ^= 2 ** enc.length - 1
        value = value >> (seen[enc.value] + enc.shift)
        code += set_bits(value, enc.start + seen[enc.value], enc.length)
        seen[enc.value] += enc.length
      elif enc.type == EncodingType.MODIFIER:
        mod_key = valid_mods[0] if (valid_mods := [m for m in op_mods if m in spec.op_mods[enc.value]]) else ''
        if mod_key in spec.op_mods[enc.value]:
          code += set_bits(spec.op_mods[enc.value][mod_key], enc.start, enc.length)
      elif enc.type == EncodingType.OPERAND_MODIFIER:
        if operand_mods and enc.value in operand_mods:
          code += sum(set_bits(spec.vmods[enc.value][mod], enc.start, enc.length) for mod in operand_mods[enc.value])
      else:
        raise ValueError(f"Unknown encoding type: {enc.type}")
    return code

  def encode_control_code(self, wait:int, read:int, write:int, yield_:int, stall:int) -> int:
    return sum(c << i for c,i in zip([wait, read, write, yield_, stall], [11, 8, 5, 4, 0]))

  def to_json(self) -> str:
    return json.dumps({key: [asdict(spec) for spec in inst.specs.values()] for key,inst in self.isa.items()})

class SASSParser:
  def __init__(self, src:str):
    self.function_name, self.param_cnt, self.c_mem_sz = None, None, None
    self.labels = {}
    self.eiattr = defaultdict(list)
    self.parse_labels(src)
    self.parse_attributes(src)

  def parse(self, line):
    r = text_pat.match(strip_comments(line).strip())
    ctrl, ins = r.groups()
    ins = self.labels_to_addr(ins.strip())
    addr = parse_inst_addr(line)
    key, vals, op_mods, vmods = parse_inst(ins, addr=addr)
    assert key == (ref_key := InstructionParser.parseInstruction(ins).get_key()), f"key mismatch: parsed={key}\tkuter={ref_key}" # TODO: remove
    return ctrl, key, vals, op_mods, vmods

  def labels_to_addr(self, s):
    r = ins_label.search(s)
    return ins_label.sub(self.labels[r.groups()[0]], s) if r else s

  def parse_labels(self, src):
    addr = None
    for line in src.split('\n'):
      if a := parse_inst_addr(line):
        addr = a
      if r := def_label.match(line.strip()):
        if addr: self.labels[r.groups()[0]] = hex(addr + 16)
        elif not self.function_name and r.groups()[0].startswith(".text."): self.function_name = r.groups()[0].replace(".text.", "", 1)
        else: raise ValueError

  def parse_attributes(self, src):
    c_size, c_base = 0, 0x160
    block_dims = [1, 1, 1]
    for line in src.split('\n'):
      if dim_match := re.match(r"BLOCK_DIM_(\d)=(\d+)", line):
        block_dims[int(dim_match.groups()[0])] = int(dim_match.groups()[1])
      if reg_match := re.match(r"SHI_REGISTERS=(\d+)", line):
        self.eiattr["EIATTR_REGCOUNT"].append(["FUNC", int(reg_match.groups()[0])])
      if param_match := re.match(r"PARAM_COUNT=(\d+)", line):
        n = int(param_match.groups()[0])
        for i in range(n-1, -1, -1):
          self.eiattr["EIATTR_KPARAM_INFO"].append([0, (8*i << 16) + i, 0x21f000])
      text_match = text_pat.match(strip_comments(line).strip())
      if not text_match: continue
      ins = text_match.groups()[1].strip()
      if c_match := const_mem.search(ins):
        offset = 8 if "IMAD.WIDE" in ins else 4 # HACK TODO: write addr range from renderer instead
        c_size = max(int(c_match.group("Addr"), 16) + offset - c_base, c_size) # TODO: remove and write attribute from renderer like SHI_REGISTERS
      if ins.startswith("EXIT"):
        self.eiattr["EIATTR_EXIT_INSTR_OFFSETS"].append([parse_inst_addr(line)])
    self.eiattr["EIATTR_PARAM_CBANK"].append([".nv.constant0.FUNC", (c_size << 16) + c_base])
    self.eiattr["EIATTR_CBANK_PARAM_SIZE"].append(c_size)
    self.eiattr["EIATTR_MAX_THREADS"].append(block_dims)
    self.eiattr["EIATTR_MAXREG_COUNT"].append(255)
    self.c_mem_sz = c_base + c_size

def parse_inst(ins:str, addr:int=None):
  for k,v in const_tr.items():
    ins = re.sub(k, v, ins)
  # TODO: translate scoreboard for DEPBAR?
  ins = insig_whitespace.sub('', whitespace.sub(' ', ins)).strip(' {};') # TODO: needed?
  r = ins_pat.match(ins)
  op_toks = r.group('Op').split('.')
  op, op_mods = op_toks[0], op_toks[1:]
  keys, vals, vmods = parse_operands(split_operands(r.group('Operands'), op))
  if addr is not None and op in addr_ops and keys[-1] == "I" and 'ABS' not in op:
    vals[-1] -= addr
  return '_'.join([op] + keys), [parse_pred(r.group('Pred')) if not op.startswith("NOP") else 0] + vals, op_mods, dict(vmods)

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
  for i, (keys, vals, mods) in enumerate(parsed):
    for j, v in mods.items(): idx_mods[i + j].extend(v)
  return flatten(p[0] for p in parsed), flatten(p[1] for p in parsed), idx_mods

def parse_token(token):
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

def parse_int(value):
  return ["I"], [int(value, 16)], {}

def parse_float(value):
  return ["FI"], [int(val[2:], 16) if value.startswith('0f') else float(value)], {}

def parse_indexed_token(prefix, label, index, post_mods): # TODO: how to encode negative registers?
  mods = [c for c in post_mods.split('.') if len(c)] + [pre_mod[c] for c in prefix]
  return [label], [int(index)], {0: mods} if mods else {}

def parse_pred(s):
  if not s: return 7
  v = parse_token(s.strip('@ '))[1]
  return v[0] + 8 if '!' in s else v[0]

def strip_comments(s):
  s = cpp_comment.subn(' ', s)[0] # TODO: needed?
  s = cc_comment.subn(' ', s)[0]
  s = re.subn(r'\s+', ' ', s)[0]
  return s.strip()

def split_operands(s, op):
  s = s.strip()
  if op in addr_ops and (res := reg_imme_addr.search(s)):
    s = s.replace(res.group(), res.group('R')+','+res.group('II'))
  return s.split(',') if s else []

def encode_float(val, op, mods):
  if op[0] in {'H', 'F', 'D'}: prec = op[0]
  else: prec = 'D' if '64' in mods else 'H' if '16' in mods else 'F'
  nbits = 32 if '32I' in mods else -1
  ifmt, ofmt, fullbits, keepbits = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}[prec]
  fb = struct.pack(ifmt, float(val))
  ival = struct.unpack(ofmt, fb)[0]
  trunc_bits = fullbits - max(nbits, keepbits)
  return ival >> trunc_bits if trunc_bits > 0 else ival

token_formats = (
  (parse_indexed_token, re.compile(r'(?P<Prefix>[!\-|~]?)(?P<Label>R|UR|P|UP|B|SB|SBSET|SR)(?P<Index>\d+)(?P<PostMod>(\.\w+)*)')),
  (parse_const_memory, re.compile(r'(?P<Prefix>\w*)\[(?P<Bank>[\w\.]+)\]\[(?P<Addr>[+-?\w\.]+)\]')),
  (parse_addr, re.compile(r'(?P<Prefix>\w*)\[(?P<Addr>[^\]]+)\]$')),
  (parse_int, re.compile(f'(?P<Value>[-+]?0x[0-9a-fA-F]+)')),
  (parse_float, re.compile(r'^(?P<Value>((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]NAN)|-?(0[fF][0-9a-fA-F]+))'
                           r'(\.[a-zA-Z]\w*)*$')),
)
