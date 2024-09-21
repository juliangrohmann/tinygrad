import json, re
from enum import StrEnum, auto
from typing import List, Dict, Sequence, Tuple, FrozenSet, Callable, Pattern, Any, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict
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
ins_addr = re.compile(r'/\*([\da-fA-F]{4})\*/')
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

  def to_json(self) -> str:
    return json.dumps({key: [asdict(spec) for spec_group in inst.specs.values() for spec in spec_group] for key,inst in self.isa.items()})

def parse(line:str):
  r = text_pat.match(strip_comments(line).strip())
  assert r, f"invalid SASS line: {line}"
  ctrl, ins = r.groups()
  addr = parse_inst_addr(line)
  key, vals, op_mods, vmods = parse_inst(ins, addr=addr)
  return ctrl, key, vals, op_mods, vmods

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

token_formats: Tuple[Tuple[Callable,Pattern],...] = (
  (parse_special_reg, re.compile(r'(?P<Label>SR_[\w\.]+)')),
  (parse_indexed_token, re.compile(r'(?P<Prefix>[!\-|~]?)(?P<Label>R|UR|P|UP|B|SB|SBSET|SR)(?P<Index>\d+)(?P<PostMod>(\.\w+)*)')),
  (parse_const_memory, re.compile(r'(?P<Prefix>\w*)\[(?P<Bank>[\w\.]+)\]\[(?P<Addr>[+-?\w\.]+)\]')),
  (parse_addr, re.compile(r'(?P<Prefix>\w*)\[(?P<Addr>[^\]]+)\]$')),
  (parse_int, re.compile(r'(?P<Value>[-+]?0x[0-9a-fA-F]+)')),
  (parse_float, re.compile(r'^(?P<Value>((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]?NAN)|-?(0[fF][0-9a-fA-F]+))'
                           r'(\.[a-zA-Z]\w*)*$')),
)
