import hashlib, tempfile, re, pathlib, io, json, ctypes
from collections import defaultdict
from typing import Sequence, List, Tuple, Dict, Union
from tinygrad.helpers import getenv
from tinygrad.device import Compiler
from tinygrad.runtime.support.elf import ElfSection, ElfSegment
from tinygrad.runtime.support.compiler_cuda import nvrtc, nvrtc_check
import tinygrad.runtime.autogen.libc as libc
from CuAsm import CubinFile, CuAsmParser, CuAsmLogger

strtab_common_pre = (".shstrtab", ".strtab", ".symtab", ".symtab_shndx", ".nv.info", ".text.FUNC", ".nv.info.FUNC", ".nv.shared.FUNC")
strtab_common_post = (".debug_frame", ".rel.debug_frame", ".rela.debug_frame", ".nv.callgraph", ".nv.prototype", ".nv.rel.action")
strtab_names = strtab_common_pre + (".rel.nv.constant0.FUNC", ".nv.constant0.FUNC") + strtab_common_post + ("FUNC",)
shstrtab_names = strtab_common_pre + (".nv.constant0.FUNC", ".rel.nv.constant0.FUNC") + strtab_common_post
sym_names = (".text.FUNC", ".nv.constant0.FUNC", ".debug_frame", ".nv.callgraph", ".nv.rel.action", "FUNC")
section_names = (".shstrtab", ".strtab", ".symtab", ".debug_frame", ".nv.info", ".nv.info.FUNC", ".nv.callgraph",
         ".nv.rel.action", ".rel.debug_frame", ".nv.constant0.FUNC", ".text.FUNC")
nv_info_attrs = ("EIATTR_REGCOUNT", "EIATTR_MIN_STACK_SIZE", "EIATTR_FRAME_SIZE", "EIATTR_MIN_STACK_SIZE")
nv_info_func_attrs = ("EIATTR_CUDA_API_VERSION", "EIATTR_PARAM_CBANK", "EIATTR_CBANK_PARAM_SIZE", "EIATTR_KPARAM_INFO",
                      "EIATTR_MAXREG_COUNT", "EIATTR_EXIT_INSTR_OFFSETS", "EIATTR_MAX_THREADS")
eiattr = {'EIATTR_CTAIDZ_USED': 0x0401, # TODO: remove unneeded attrs # TODO: frozendict
          'EIATTR_MAX_THREADS': 0x0504,
          'EIATTR_PARAM_CBANK': 0x0a04,
          'EIATTR_EXTERNS': 0x0f04,
          'EIATTR_REQNTID': 0x1004,
          'EIATTR_FRAME_SIZE': 0x1104,
          'EIATTR_MIN_STACK_SIZE': 0x1204,
          'EIATTR_BINDLESS_TEXTURE_BANK': 0x1502,
          'EIATTR_BINDLESS_SURFACE_BANK': 0x1602,
          'EIATTR_KPARAM_INFO': 0x1704,
          'EIATTR_CBANK_PARAM_SIZE': 0x1903,
          'EIATTR_MAXREG_COUNT': 0x1b03,
          'EIATTR_EXIT_INSTR_OFFSETS': 0x1c04,
          'EIATTR_S2RCTAID_INSTR_OFFSETS': 0x1d04,
          'EIATTR_CRS_STACK_SIZE': 0x1e04,
          'EIATTR_NEED_CNP_WRAPPER': 0x1f01,
          'EIATTR_NEED_CNP_PATCH': 0x2001,
          'EIATTR_EXPLICIT_CACHING': 0x2101,
          'EIATTR_MAX_STACK_SIZE': 0x2304,
          'EIATTR_LD_CACHEMOD_INSTR_OFFSETS': 0x2504,
          'EIATTR_ATOM_SYS_INSTR_OFFSETS': 0x2704,
          'EIATTR_COOP_GROUP_INSTR_OFFSETS': 0x2804,
          'EIATTR_SW1850030_WAR': 0x2a01,
          'EIATTR_WMMA_USED': 0x2b01,
          'EIATTR_ATOM16_EMUL_INSTR_REG_MAP': 0x2e04,
          'EIATTR_REGCOUNT': 0x2f04,
          'EIATTR_SW2393858_WAR': 0x3001,
          'EIATTR_INT_WARP_WIDE_INSTR_OFFSETS': 0x3104,
          'EIATTR_INDIRECT_BRANCH_TARGETS': 0x3404,
          'EIATTR_SW2861232_WAR': 0x3501,
          'EIATTR_SW_WAR': 0x3604,
          'EIATTR_CUDA_API_VERSION': 0x3704}
nv_rel_action = b'\x73\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x25\x00\x05\x36' # TODO: figure out dynsym d_un type
debug_frame = (b'\xff\xff\xff\xff\x24\x00\x00\x00\x00\x00\x00\x00\xff\xff\xff\xff' # sm_89, mostly independent of kernel # TODO: needed?
               b'\xff\xff\xff\xff\x03\x00\x04\x7c\xff\xff\xff\xff\x0f\x0c\x81\x80'
               b'\x80\x28\x00\x08\xff\x81\x80\x28\x08\x81\x80\x80\x28\x00\x00\x00'
               b'\xff\xff\xff\xff\x34\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00'
               b'\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00'
               b'\x00\x00\x00\x00\x04\x04\x00\x00\x00\x04\x00\x00\x00\x00\x0c\x81'
               b'\x80\x80\x28\x00\x04\xfc\xff\xff\x3f\x00\x00\x00\x00\x00\x00\x00')
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
cpp_comment = re.compile(r'//.*$')      # cpp style line comments # TODO: needed?
cc_comment = re.compile(r'\/\*.*?\*\/')  # c style line comments # TODO: needed?
ins_label = re.compile(r'`\(([^\)]+)\)')
def_label = re.compile(r'([a-zA-Z0-9._$@#]+?)\s*:\s*(.*)')

class SASSParser:
  def __init__(self, src:str):
    self.function_name, self.param_cnt, self.c_mem_sz = None, None, None
    self.labels = {}
    self.eiattr = defaultdict(list)
    self._parse_labels(src)
    self._parse_features(src)

  def parse_ins(self, ins:str, addr:str):
    ins = self._labels_to_addr(ins.strip())
    for k,v in const_tr.items():
      ins = re.sub(k, v, ins)

    # TODO: translate scoreboard for DEPBAR?
    ins = whitespace.sub(' ', ins)
    ins = insig_whitespace.sub('', ins) # TODO: needed?
    ins = ins.strip(' {};')
    r = ins_pat.match(ins)

    vals = [_parse_pred(r.group('Pred'))]
    op_full = r.group('Op')
    op_toks = op_full.split('.')
    key = op_toks[0]
    modi = ['0_' + m for m in op_toks]
    if len(operands := _preprocess_operands(r.group('Operands'), op_toks[0])):
      for i, operand in enumerate(operands.split(',')):
        optype, opval, opmodi = _parse_operand(operand, op_full)
        key += '_' + optype
        vals.extend(opval)
        modi.extend([('%d_' % (i + 1)) + m for m in opmodi])
      _special_fixup(op_full, key, vals, modi, addr)
    return key, vals, modi

  def _labels_to_addr(self, s):
    r = ins_label.search(s)
    return ins_label.sub(self.labels[r.groups()[0]], s) if r else s

  def _parse_labels(self, src):
    addr = None
    for line in src.split('\n'):
      if a := _parse_ins_addr(line):
        addr = a
      if r := def_label.match(line.strip()):
        if addr: self.labels[r.groups()[0]] = hex(addr + 16)
        elif not self.function_name and r.groups()[0].startswith(".text."): self.function_name = r.groups()[0].replace(".text.", "", 1)
        else: raise ValueError
    self.ins_size = addr + 16

  def _parse_features(self, src):
    c_size, c_base = 0, 0x160
    block_dims = [1, 1, 1]
    for line in src.split('\n'):
      if dim_match := re.match(r"BLOCK_DIM_(\d)=(\d+)", line):
        block_dims[int(dim_match.groups()[0])] = int(dim_match.groups()[1])
      if reg_match := re.match(r"SHI_REGISTERS=(\d+)", line):
        self.eiattr["EIATTR_REGCOUNT"].append([sym_names.index("FUNC") + 1, int(reg_match.groups()[0])])
      if param_match := re.match(r"PARAM_COUNT=(\d+)", line):
        n = int(param_match.groups()[0])
        for i in range(n-1, -1, -1):
          self.eiattr["EIATTR_KPARAM_INFO"].append([0, (8*i << 16) + i, 0x21f000])
      text_match = text_pat.match(_strip_comments(line).strip())
      if not text_match: continue
      ins = text_match.groups()[1].strip()
      if c_match := const_mem.search(ins):
        offset = 8 if "IMAD.WIDE" in ins else 4 # HACK: write from renderer instead
        c_size = max(int(c_match.group("Addr"), 16) + offset - c_base, c_size) # TODO: remove and write attribute from renderer like SHI_REGISTERS
      if ins.startswith("EXIT"):
        self.eiattr["EIATTR_EXIT_INSTR_OFFSETS"].append([_parse_ins_addr(line)])
    self.eiattr["EIATTR_PARAM_CBANK"].append([sym_names.index(".nv.constant0.FUNC") + 1, (c_size << 16) + c_base])
    self.eiattr["EIATTR_CBANK_PARAM_SIZE"].append(c_size)
    self.eiattr["EIATTR_MAX_THREADS"].append(block_dims)
    self.c_mem_sz = c_base + c_size

class SASSCompiler(Compiler):
  def __init__(self, arch:str):
    nvrtc_check(nvrtc.nvrtcVersion((nvrtcMajor := ctypes.c_int()), (nvrtcMinor := ctypes.c_int())))
    self.arch = arch
    self.eiattr = defaultdict(list)
    self.eiattr["EIATTR_CUDA_API_VERSION"].append([nvrtcMajor.value*10 + nvrtcMinor.value])
    self.eiattr["EIATTR_MIN_STACK_SIZE"].append([sym_names.index("FUNC") + 1, 0],)
    self.eiattr["EIATTR_FRAME_SIZE"].append([sym_names.index("FUNC") + 1, 0])
    self.eiattr["EIATTR_MAXREG_COUNT"].append(255)
    with open(pathlib.Path(__file__).parent / f"sass.{self.arch}.json") as f:
      self.ins_repo = json.load(f)
    super().__init__(f"compile_sass_{self.arch}")

  def compile(self, src:str) -> Tuple[libc.Elf64_Ehdr, List[ElfSection], List[ElfSegment]]:
    parser = SASSParser(src)
    for k,v in parser.eiattr.items(): self.eiattr[k].extend(v)
    sec_tab = {}
    sec_tab[".shstrtab"] = self._compile_shstrtab(parser.function_name)
    sec_tab[".strtab"] = self._compile_strtab(parser.function_name)
    sec_tab[".text.FUNC"] = self._compile_text(src, parser, self.eiattr)
    sec_tab[".symtab"] = self._compile_symtab(parser.function_name, sec_tab[".strtab"], sec_tab[".text.FUNC"])
    sec_tab[".debug_frame"] = self._compile_debug_frame(parser)
    sec_tab[".nv.info"] = self._compile_nv_info(self.eiattr)
    sec_tab[".nv.info.FUNC"] = self._compile_nv_info_func(parser.function_name, self.eiattr)
    sec_tab[".nv.callgraph"] = self._compile_nv_callgraph()
    sec_tab[".nv.rel.action"] = self._compile_nv_rel_action()
    sec_tab[".rel.debug_frame"] = self._compile_rel_debug_frame()
    sec_tab[".nv.constant0.FUNC"] = self._compile_constant_memory(parser.function_name, parser.c_mem_sz)
    for s in sec_tab.values():
      s.header.sh_name = _sh_name(s.name, sec_tab[".shstrtab"].content)
      s.header.sh_size = len(s.content)
    sections = [ElfSection("", libc.Elf64_Shdr(sh_type=libc.SHT_NULL), content=b'')] + [sec_tab[name] for name in section_names]
    segments = self._compile_segments(sections)
    return self._compile_file_header(len(sections), len(segments)), sections, segments

  def _compile_file_header(self, shnum:int, phnum:int):
    header = libc.Elf64_Ehdr()
    header.e_ident[:] = b'\0'*16
    header.e_ident[:4] = b"\x7fELF"
    # header.e_ident[4:9] = [libc.ELFCLASS64, libc.ELFDATA2LSB, libc.EV_CURRENT, libc.ELFOSABI_LINUX, libc.ELFOSABI_AIX]
    header.e_ident[4:9] = [libc.ELFCLASS64, libc.ELFDATA2LSB, libc.EV_CURRENT, 51, libc.ELFOSABI_AIX]
    header.e_type = libc.ET_EXEC
    header.e_machine = 190 # TODO: sm_89?
    header.e_version = self.eiattr["EIATTR_CUDA_API_VERSION"][0][0]
    header.e_flags = 0x590559 # TODO: ???
    header.e_ehsize = 0x40
    header.e_phentsize = 0x38
    header.e_phnum = phnum
    header.e_shentsize = 0x40
    header.e_shnum = shnum
    header.e_shstrndx = section_names.index(".shstrtab") + 1
    return header

  def _compile_segments(self, sections):
    allocs = [s for s in sections if s.header.sh_flags & libc.SHF_ALLOC]
    segments = [(libc.PT_PHDR, []), (libc.PT_LOAD, allocs), (libc.PT_LOAD, [])]
    return [ElfSegment(header=libc.Elf64_Phdr(p_type=t, p_flags=libc.PF_R + libc.PF_X, p_align=8), sections=s) for t,s in segments]

  def _compile_shstrtab(self, function_name:str):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_STRTAB, sh_addralign=1)
    shstrtab = [s.replace("FUNC", function_name) for s in shstrtab_names]
    return ElfSection(".shstrtab", sh, b'\0' + b'\0'.join(s.encode() for s in shstrtab) + b'\0')

  def _compile_strtab(self, function_name:str):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_STRTAB, sh_addralign=1)
    strtab = [s.replace("FUNC", function_name) for s in strtab_names]
    return ElfSection(".strtab", sh, b'\0' + b'\0'.join(s.encode() for s in strtab) + b'\0')

  def _compile_symtab(self, function_name:str, strtab:ElfSection, kernel:ElfSection):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_SYMTAB, sh_link=section_names.index(".strtab") + 1, sh_info=sym_names.index("FUNC") + 1,
                         sh_addralign=8, sh_entsize=24)
    symbols = [libc.Elf64_Sym()]
    for name in sym_names:
      symbols.append(sym := libc.Elf64_Sym())
      sym.st_name = strtab.content.index(b'\x00' + name.replace("FUNC", function_name).encode() + b'\x00') + 1
      sym.st_info = (libc.STB_GLOBAL << 4) + libc.STT_FUNC if name == "FUNC" else (libc.STT_NOTYPE << 4) + libc.STT_SECTION
      sym.st_other = 1 << 4 if name == "FUNC" else 0
      sym.st_shndx = section_names.index(name if name != "FUNC" else ".text." + name) + 1
      sym.st_size = 128 * ((len(kernel.content) + 127) // 128) if name == "FUNC" else 0
    return ElfSection(".symtab", sh, b''.join(bytes(sym) for sym in symbols))

  def _compile_debug_frame(self, parser:SASSParser):
    content = bytearray(debug_frame)
    content[0x4c:0x4e] = parser.ins_size.to_bytes(2, "little")
    content[0x5a] = parser.eiattr["EIATTR_EXIT_INSTR_OFFSETS"][0][0] >> 2
    return ElfSection(".debug_frame", libc.Elf64_Shdr(sh_type=libc.SHT_PROGBITS, sh_addralign=1), bytes(content))

  def _compile_nv_info(self, attr:Dict[str, Union[List[int], int]]):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC, sh_link=section_names.index(".symtab") + 1, sh_addralign=4)
    return ElfSection(".nv.info", sh, _compile_eiattr(nv_info_attrs, attr))

  def _compile_nv_info_func(self, function_name:str, attr:Dict[str, Union[List[List[int]], List[int]]]):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC, sh_flags=0x40, sh_link=section_names.index(".symtab") + 1,
                         sh_info=section_names.index(".text.FUNC") + 1, sh_addralign=4) # TODO: sh_info is a guess, not documented
    return ElfSection(f".nv.info.{function_name}", sh, _compile_eiattr(nv_info_func_attrs, attr))

  def _compile_nv_callgraph(self):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC + libc.SHT_PROGBITS, sh_link=section_names.index(".symtab") + 1, sh_addralign=4, sh_entsize=8)
    content = b''.join(b'\0'*4 + ((1 << 8) - 1 - i).to_bytes(1, "little") + b'\xff'*3 for i in range(4))
    return ElfSection(f".nv.callgraph", sh, content)

  def _compile_nv_rel_action(self):
    return ElfSection(f".nv.rel.action", libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC + libc.SHT_DYNSYM, sh_addralign=8, sh_entsize=8), nv_rel_action)

  def _compile_rel_debug_frame(self):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_REL, sh_flags=0x40, sh_link=section_names.index(".symtab") + 1,
                         sh_info=section_names.index(".debug_frame") + 1, sh_addralign=8, sh_entsize=16)
    return ElfSection(f".rel.debug_frame", sh, _compile_rel(68, sym_names.index("FUNC") + 1, 2)) # TODO: addr?

  def _compile_constant_memory(self, function_name:str, c_mem_sz):
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_PROGBITS, sh_flags=0x40 + libc.SHF_ALLOC, sh_info=section_names.index(".text.FUNC") + 1, sh_addralign=4)
    return ElfSection(f".nv.constant0.{function_name}", sh, b'\0'*c_mem_sz)

  def _compile_text(self, src:str, parser:SASSParser, attr:Dict[str, Union[List[List[int]], List[int]]]) -> ElfSection:
    def merge(ctrl:int, ins:int) -> bytes: return ((ctrl << 105) + ins).to_bytes(16, 'little') # sm >= 7x
    content = bytearray()
    for line in src.split('\n'):
      if not line: continue
      r = text_pat.match(_strip_comments(line).strip())
      if not r: continue
      ctrl, ins = r.groups()
      addr = _parse_ins_addr(line)
      content += merge(self._compile_ctrl(ctrl), self._compile_ins(*parser.parse_ins(ins, addr)))
    sh = libc.Elf64_Shdr(sh_type=libc.SHT_PROGBITS, sh_flags=libc.SHF_EXECINSTR + libc.SHF_ALLOC, sh_link=section_names.index(".symtab") + 1,
                         sh_info=(attr["EIATTR_REGCOUNT"][0][1] << 24) + 6, sh_addralign=128)
    return ElfSection(f".text.{parser.function_name}", sh, bytes(content))

  def _compile_ins(self, key:str, vals:Sequence[int], modi:Sequence[str]) -> int:
    repo = self.ins_repo[key]
    code = 0
    for v0, vs in zip(repo["sol"][-len(vals):], vals):
      code += v0 * vs
    for m in modi:
      code += repo["sol"][repo["modi"][m]]
    return code // repo["fac"]

  def _compile_ctrl(self, ctrl:str) -> int:
    s_waitbar, s_readbar, s_writebar, s_yield, s_stall = tuple(ctrl.split(':')) # format: [B------:R-:W-:Y:S15]
    c_waitbar = int(''.join(['1' if c != '-' else '0' for c in s_waitbar[:0:-1]]), 2)
    c_readbar = int(s_readbar[1].replace('-', '7'))
    c_writebar = int(s_writebar[1].replace('-','7'))
    c_yield = int(s_yield != 'Y')
    c_stall = int(s_stall[1:])
    code = sum(c << i for c,i in zip([c_waitbar, c_readbar, c_writebar, c_yield, c_stall], [11, 8, 5, 4, 0]))
    return code

def _sh_name(name:str, shstrtab:bytes):
  return shstrtab.index(b'\0' + name.encode() + b'\0') + 1

def _compile_eiattr(attr_names:Sequence[str], kernel_eiattr:Dict[str, int]) -> bytes:
  content = bytearray()
  for attr_name in attr_names:
    for val in kernel_eiattr[attr_name]:
      attr_code = eiattr[attr_name]
      vfmt = attr_code & 0xff
      content += attr_code.to_bytes(2, 'little') + _pack_eiattr(vfmt, val)
  return bytes(content)

def _compile_rel(addr:int, sym_idx:int, rel_type:int):
  return addr.to_bytes(8, "little") + ((sym_idx << 32) + rel_type).to_bytes(8, "little")

def _pack_eiattr(vfmt, val):
  if vfmt == 1: # EIFMT_NVAL
    return b'\x00\x00'
  elif vfmt == 2: # EIFMT_BVAL
    return int.to_bytes(val, 2, 'little')
  elif vfmt == 3: # EIFMT_HVAL
    return int.to_bytes(val, 2, 'little')
  elif vfmt == 4: # EIFMT_SVAL
    bval = bytearray((len(val)*4).to_bytes(2, 'little'))
    for v in val: bval += v.to_bytes(4, 'little')
    return bval
  else:
    raise ValueError

def _parse_ins_addr(s:str):
  if m := ins_addr.search(s):
    return int(m.groups()[0], 16)
  return None

def _parse_operand(operand, op_full):
  op, modi = _strip_modi(operand)
  if op[0] == '[':
    return _parse_addr(op)
  if idx_pat.match(op) is not None:
    optype, opval, tmodi = _parse_indexed_tok(op)
    opmodi = modi
    opmodi.extend(tmodi)
  elif op.startswith('c['):
    optype, opval, opmodi = _parse_const_memory(op)
    opmodi.extend(modi)
  elif op.startswith('0x'):
    optype = 'II'
    op, modi = _strip_imme_modi(operand)
    opval, opmodi = _parse_int_imme(op)
    opmodi.extend(modi)
  elif float_imme.match(operand) is not None:
    optype = 'FI'
    op, modi = _strip_imme_modi(operand)
    opval, opmodi = _parse_float_imme(op, op_full)
    opmodi.extend(modi)
  elif op.startswith('desc'):
    optype, opval, opmodi = _parse_desc_addr(op)
    opmodi.extend(modi)
  elif op.startswith('cx['):
    optype, opval, opmodi = _parse_ur_const_mem(op)
    opmodi.extend(modi)
  else:
    return 'L', [], [operand]
  return optype, opval, opmodi

def _parse_const_memory(s):
  opmain, opmodi = _strip_modi(s)
  r = const_mem.match(opmain)
  atype, aval, amodi = _parse_addr(r.group('Addr'))
  return 'c' + atype, [int(r.group('Bank'), 16)] + aval, opmodi + amodi

def _parse_addr(s):
  ss = re.sub(r'(?<![\[\+])-0x', '+-0x', s)
  ss = ss.strip('[]').split('+')

  pdict = {}
  for ts in ss:
    if not len(ts):
      continue
    if '0x' in ts:
      pdict['I'] = ('I', *_parse_int_imme(ts))
    else:
      ttype, tval, tmodi = _parse_indexed_tok(ts)
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

def _parse_desc_addr(s):
  r = desc_addr.match(s)
  opval = _parse_indexed_tok(r.group('URIndex'))[1]
  atype, aval, amodi = _parse_addr(r.group('Addr'))
  return 'd' + atype, opval + aval, amodi

def _parse_ur_const_mem(s):
  opmain = _strip_modi(s)[0]
  r = ur_const_mem.match(opmain)
  btype, opval, opmodi = _parse_indexed_tok(r.group('URBank'))
  atype, aval, amodi = _parse_addr(r.group('Addr'))
  return 'cx' + atype, opval + aval, [btype + '_' + m for m in opmodi] + amodi

def _parse_int_imme(s):
  return [i := int(s, 16)], ([] if i >= 0 else ['NegIntImme'])

def _parse_float_imme(s, op_full):
  if op_full[0] in {'H', 'F', 'D'}: prec = op_full[0]
  else: prec = 'D' if '64' in op_full else 'H' if '16' in op_full else 'F'
  nbits = 32 if op_full.split('.')[0].endswith('32I') else -1
  return [_convert_float_imme(s, prec, nbits)], [] # TODO: return first val only

def _parse_pred(s):
  if not s: return 7
  v = _parse_indexed_tok(s.strip('@! '))[1]
  return v[0] + 8 if '!' in s else v[0]

def _parse_indexed_tok(s):
  tmain, modi = _strip_modi(s)
  r = idx_pat.match(tmain)
  return r.group('Label'), [int(r.group('Index'))], modi

def _convert_float_imme(fval, prec, nbits=-1):
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

def _special_fixup(op_full, key, vals, modi, addr):
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

def _strip_modi(s):
  r = mod_pat.match(s)
  return r.group('Main'), [pre_mod[c] for c in r.group('PreModi')] + [c for c in r.group('PostModi').split('.') if len(c)]

def _strip_imme_modi(s):
  r = imme_modi.match(s)
  modis = r.group('ModiSet')
  return r.group('Value'), [] if not modis else modis.lstrip('.').split('.')

def _strip_comments(s):
  s = cpp_comment.subn(' ', s)[0] # TODO: needed?
  s = cc_comment.subn(' ', s)[0]
  s = re.subn(r'\s+', ' ', s)[0]
  return s.strip()

def _preprocess_operands(s, op):
  s = s.strip()
  if op in addr_ops:
    res = reg_imme_addr.search(s)
    if res is not None:
      s = s.replace(res.group(), res.group('R')+','+res.group('II'))     #
  return s
