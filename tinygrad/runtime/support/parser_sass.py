import re, struct
from collections import defaultdict

addr_ops = {'BRA', 'BRX', 'BRXU', 'CALL', 'JMP', 'JMX', 'JMXU', 'RET', 'BSSY', 'SSY', 'CAL', 'PRET', 'PBK'}
cast_ops = {'I2F', 'I2I', 'F2I', 'F2F', 'I2FP', 'I2IP', 'F2IP', 'F2FP', 'FRND'}
pos_dep_ops = {'HMMA', 'IMMA', 'I2IP', 'F2FP', 'I2I', 'F2F', 'IDP', 'TLD4', 'VADD',
               'VMAD', 'VSHL', 'VSHR', 'VSET', 'VSETP', 'VMNMX', 'VABSDIFF', 'VABSDIFF4'}
const_tr = {r'(?<!\.)\bRZ\b': 'R255', r'\bURZ\b': 'UR63', r'\bPT\b': 'P7', r'\bUPT\b': 'UP7', r'\bQNAN\b': 'NAN'}
pre_mod = {'!': 'cNOT', '-': 'cNEG', '|': 'cABS', '~': 'cINV'}
float_fmt = {'H':('e','H', 16, 16), 'F':('f','I', 32, 32), 'D':('d','Q', 64, 32)}
ins_pat = re.compile(r'(?P<Pred>@!?U?P\w\s+)?\s*(?P<Op>[\w\.\?]+)(?P<Operands>.*)')
idx_pat = re.compile(r'\b(?P<Label>R|UR|P|UP|B|SB|SBSET|SR)(?P<Index>\d+)$')
mod_pat = re.compile(r'^(?P<PreModi>[~\-\|!]*)(?P<Main>.*?)\|?(?P<PostModi>(\.\w+)*)\|?$')
text_pat = re.compile(r'\[([\w:-]+)\](.*)')
ins_addr = re.compile(r'/\*([\da-fA-F]{4})\*/')
reg_imme_addr = re.compile(r'(?P<R>R\d+)\s*(?P<II>-?0x[0-9a-fA-F]+)')
desc_addr = re.compile(r'desc\[(?P<URIndex>UR\d+)\](?P<Addr>\[.*\])$')
float_imme = re.compile(r'^(?P<Value>((-?\d+)(\.\d*)?((e|E)[-+]?\d+)?)|([+-]?INF)|([+-]NAN)|-?(0[fF][0-9a-fA-F]+))(?P<ModiSet>(\.[a-zA-Z]\w*)*)$')
const_mem = re.compile(r'c\[(?P<Bank>0x\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')
ur_const_mem = re.compile(r'cx\[(?P<URBank>UR\w+)\]\[(?P<Addr>[+-?\w\.]+)\]')
imme_modi = re.compile(r'^(?P<Value>.*?)(?P<ModiSet>(\.[a-zA-Z]\w*)*)$')
insig_whitespace = re.compile(r'((?<=[\w\?]) (?![\w\?]))|((?<![\w\?]) (?=[\w\?]))|((?<![\w\?]) (?![\w\?]))')
whitespace = re.compile(r'\s+')
cpp_comment = re.compile(r'//.*$') # cpp style line comments TODO: needed?
cc_comment = re.compile(r'\/\*.*?\*\/') # c style line comments TODO: needed?
ins_label = re.compile(r'`\(([^\)]+)\)')
def_label = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')

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
    addr = parse_ins_addr(line)
    return ctrl, *self.parse_ins(ins, addr)

  def parse_ins(self, ins:str, addr:str):
    ins = self.labels_to_addr(ins.strip())
    for k,v in const_tr.items():
      ins = re.sub(k, v, ins)

    # TODO: translate scoreboard for DEPBAR?
    ins = whitespace.sub(' ', ins)
    ins = insig_whitespace.sub('', ins) # TODO: needed?
    ins = ins.strip(' {};')
    r = ins_pat.match(ins)

    vals = [parse_pred(r.group('Pred'))]
    op_full = r.group('Op')
    op_toks = op_full.split('.')
    key = op_toks[0]
    modi = ['0_' + m for m in op_toks]
    if len(operands := preprocess_operands(r.group('Operands'), op_toks[0])):
      for i, operand in enumerate(operands.split(',')):
        optype, opval, opmodi = parse_operands(operand, op_full)
        key += '_' + optype
        vals.extend(opval)
        modi.extend([('%d_' % (i + 1)) + m for m in opmodi])
      special_fixup(op_full, key, vals, modi, addr)
    return key, vals, modi

  def labels_to_addr(self, s):
    r = ins_label.search(s)
    return ins_label.sub(self.labels[r.groups()[0]], s) if r else s

  def parse_labels(self, src):
    addr = None
    for line in src.split('\n'):
      if a := parse_ins_addr(line):
        addr = a
      if r := def_label.match(line.strip()):
        if addr: self.labels[r.groups()[0]] = hex(addr + 16)
        elif not self.function_name and r.groups()[0].startswith(".text."): self.function_name = r.groups()[0].replace(".text.", "", 1)
        else: raise ValueError
    self.ins_size = addr + 16

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
        self.eiattr["EIATTR_EXIT_INSTR_OFFSETS"].append([parse_ins_addr(line)])
    self.eiattr["EIATTR_PARAM_CBANK"].append([".nv.constant0.FUNC", (c_size << 16) + c_base])
    self.eiattr["EIATTR_CBANK_PARAM_SIZE"].append(c_size)
    self.eiattr["EIATTR_MAX_THREADS"].append(block_dims)
    self.eiattr["EIATTR_MAXREG_COUNT"].append(255)
    self.c_mem_sz = c_base + c_size

def parse_ins_addr(s:str):
  if m := ins_addr.search(s):
    return int(m.groups()[0], 16)
  return None

def parse_operands(operand, op_full): # TODO: refactor
  op, modi = strip_modi(operand)
  if op[0] == '[':
    return parse_addr(op)
  if idx_pat.match(op) is not None:
    optype, opval, tmodi = parse_indexed_tok(op)
    opmodi = modi
    opmodi.extend(tmodi)
  elif op.startswith('c['):
    optype, opval, opmodi = parse_const_memory(op)
    opmodi.extend(modi)
  elif op.startswith('0x'):
    optype = 'II'
    op, modi = strip_imme_modi(operand)
    opval, opmodi = parse_int_imme(op)
    opmodi.extend(modi)
  elif float_imme.match(operand) is not None:
    optype = 'FI'
    op, modi = strip_imme_modi(operand)
    opval, opmodi = parse_float_imme(op, op_full)
    opmodi.extend(modi)
  elif op.startswith('desc'):
    optype, opval, opmodi = parse_desc_addr(op)
    opmodi.extend(modi)
  elif op.startswith('cx['):
    optype, opval, opmodi = parse_ur_const_mem(op)
    opmodi.extend(modi)
  else:
    return 'L', [], [operand]
  return optype, opval, opmodi

def parse_const_memory(s):
  opmain, opmodi = strip_modi(s)
  r = const_mem.match(opmain)
  atype, aval, amodi = parse_addr(r.group('Addr'))
  return 'c' + atype, [int(r.group('Bank'), 16)] + aval, opmodi + amodi

def parse_addr(s): # TODO: refactor
  ss = re.sub(r'(?<![\[\+])-0x', '+-0x', s)
  ss = ss.strip('[]').split('+')

  pdict = {}
  for ts in ss:
    if not len(ts):
      continue
    if '0x' in ts:
      pdict['I'] = ('I', *parse_int_imme(ts))
    else:
      ttype, tval, tmodi = parse_indexed_tok(ts)
      tmodi = [(ttype + '.' + m) for m in tmodi]
      pdict[ttype] = (ttype, tval, tmodi)

  optype = 'A'
  opval = []
  opmodi = []
  for key in ['R', 'UR', 'I']:
    if key in pdict:
      optype += key
      opval.extend(pdict[key][1])
      opmodi.extend(pdict[key][2])
  if not 'I' in pdict:
    optype += 'I'
    opval.append(0)
  return optype, opval, opmodi

def parse_desc_addr(s):
  r = desc_addr.match(s)
  opval = parse_indexed_tok(r.group('URIndex'))[1]
  atype, aval, amodi = parse_addr(r.group('Addr'))
  return 'd' + atype, opval + aval, amodi

def parse_ur_const_mem(s):
  opmain = strip_modi(s)[0]
  r = ur_const_mem.match(opmain)
  btype, opval, opmodi = parse_indexed_tok(r.group('URBank'))
  atype, aval, amodi = parse_addr(r.group('Addr'))
  return 'cx' + atype, opval + aval, [btype + '_' + m for m in opmodi] + amodi

def parse_int_imme(s):
  return [i := int(s, 16)], ([] if i >= 0 else ['NegIntImme'])

def parse_float_imme(s, op_full):
  if op_full[0] in {'H', 'F', 'D'}: prec = op_full[0]
  else: prec = 'D' if '64' in op_full else 'H' if '16' in op_full else 'F'
  nbits = 32 if op_full.split('.')[0].endswith('32I') else -1
  return [convert_float_imme(s, prec, nbits)], [] # TODO: return first val only

def parse_pred(s):
  if not s: return 7
  v = parse_indexed_tok(s.strip('@! '))[1]
  return v[0] + 8 if '!' in s else v[0]

def parse_indexed_tok(s):
  tmain, modi = strip_modi(s)
  r = idx_pat.match(tmain)
  return r.group('Label'), [int(r.group('Index'))], modi

def strip_modi(s):
  r = mod_pat.match(s)
  return r.group('Main'), [pre_mod[c] for c in r.group('PreModi')] + [c for c in r.group('PostModi').split('.') if len(c)]

def strip_imme_modi(s):
  r = imme_modi.match(s)
  modis = r.group('ModiSet')
  return r.group('Value'), [] if not modis else modis.lstrip('.').split('.')

def strip_comments(s):
  s = cpp_comment.subn(' ', s)[0] # TODO: needed?
  s = cc_comment.subn(' ', s)[0]
  s = re.subn(r'\s+', ' ', s)[0]
  return s.strip()

def preprocess_operands(s, op):
  s = s.strip()
  if op in addr_ops:
    res = reg_imme_addr.search(s)
    if res is not None:
      s = s.replace(res.group(), res.group('R')+','+res.group('II'))
  return s

def convert_float_imme(fval, prec, nbits=-1):
  val = fval.lower().strip()
  if val.startswith('0f'): # TODO: remove if not used
    v = int(val[2:], 16)
    return v
  else:
    ifmt, ofmt, fullbits, keepbits = float_fmt[prec]
    fb = struct.pack(ifmt, float(val))
    ival = struct.unpack(ofmt, fb)[0]
    trunc_bits = fullbits - max(nbits, keepbits)
    if trunc_bits > 0:
      ival >>= trunc_bits
    return ival

def special_fixup(op_full, key, vals, modi, addr):
  op = op_full.split('.')[0]
  if op in {'PLOP3', 'UPLOP3'}: # immLut for PLOP3 is encoded with seperating 5+3 bits
    vals[-2] = (vals[-2] & 7) + ((vals[-2] & 0xf8) << 5)
  elif op in cast_ops:
    if '64' in op_full:
      modi.append('0_CVT64')
  elif op in addr_ops:
    if key.endswith('_II'):
      if 'ABS' not in op_full:
        vals[-1] = vals[-1] - addr - 16
        if vals[-1] < 0:
          modi.append('0_NegAddrOffset')
  if op in pos_dep_ops:
    counter = 0
    for i,m in enumerate(modi):
      if m.startswith('0_') and m[2:] in c_PosDepModis[op]:
        modi[i] += '@%d' % counter
        counter += 1
