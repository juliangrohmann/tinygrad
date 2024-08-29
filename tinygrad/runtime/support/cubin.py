from typing import Sequence, List, Dict, Union
import tinygrad.runtime.autogen.libc as libc
from tinygrad.runtime.support.assembler_sass import SASSParser
from tinygrad.runtime.support.elf import ElfSection, ElfSegment, make_elf

strtab_common_pre = (".shstrtab", ".strtab", ".symtab", ".symtab_shndx", ".nv.info", ".text.FUNC", ".nv.info.FUNC", ".nv.shared.FUNC")
strtab_common_post = (".nv.prototype", ".nv.rel.action")
strtab_names = strtab_common_pre + (".rel.nv.constant0.FUNC", ".nv.constant0.FUNC") + strtab_common_post + ("FUNC",)
shstrtab_names = strtab_common_pre + (".nv.constant0.FUNC", ".rel.nv.constant0.FUNC") + strtab_common_post
sym_names = (".text.FUNC", ".nv.constant0.FUNC", ".nv.rel.action", "FUNC")
section_names = (".shstrtab", ".strtab", ".symtab", ".nv.info", ".nv.info.FUNC", ".nv.rel.action", ".nv.constant0.FUNC", ".text.FUNC") # TODO: is .nv.info needed?
eiattr = {'EIATTR_CTAIDZ_USED': 0x0401, 'EIATTR_MAX_THREADS': 0x0504, 'EIATTR_PARAM_CBANK': 0x0a04, 'EIATTR_EXTERNS': 0x0f04, # TODO: make ' and " consistent
          'EIATTR_REQNTID': 0x1004, 'EIATTR_FRAME_SIZE': 0x1104, 'EIATTR_MIN_STACK_SIZE': 0x1204, 'EIATTR_BINDLESS_TEXTURE_BANK': 0x1502,
          'EIATTR_BINDLESS_SURFACE_BANK': 0x1602, 'EIATTR_KPARAM_INFO': 0x1704, 'EIATTR_CBANK_PARAM_SIZE': 0x1903, 'EIATTR_MAXREG_COUNT': 0x1b03,
          'EIATTR_EXIT_INSTR_OFFSETS': 0x1c04, 'EIATTR_S2RCTAID_INSTR_OFFSETS': 0x1d04, 'EIATTR_CRS_STACK_SIZE': 0x1e04,
          'EIATTR_NEED_CNP_WRAPPER': 0x1f01, 'EIATTR_NEED_CNP_PATCH': 0x2001, 'EIATTR_EXPLICIT_CACHING': 0x2101,
          'EIATTR_MAX_STACK_SIZE': 0x2304, 'EIATTR_LD_CACHEMOD_INSTR_OFFSETS': 0x2504, 'EIATTR_ATOM_SYS_INSTR_OFFSETS': 0x2704,
          'EIATTR_COOP_GROUP_INSTR_OFFSETS': 0x2804, 'EIATTR_SW1850030_WAR': 0x2a01, 'EIATTR_WMMA_USED': 0x2b01,
          'EIATTR_ATOM16_EMUL_INSTR_REG_MAP': 0x2e04, 'EIATTR_REGCOUNT': 0x2f04, 'EIATTR_SW2393858_WAR': 0x3001,
          'EIATTR_INT_WARP_WIDE_INSTR_OFFSETS': 0x3104, 'EIATTR_INDIRECT_BRANCH_TARGETS': 0x3404, 'EIATTR_SW2861232_WAR': 0x3501,
          'EIATTR_SW_WAR': 0x3604, 'EIATTR_CUDA_API_VERSION': 0x3704}
indexed_attr = ("EIATTR_PARAM_CBANK", "EIATTR_REGCOUNT")
nv_info_attr = ("EIATTR_REGCOUNT", "EIATTR_MIN_STACK_SIZE", "EIATTR_FRAME_SIZE", "EIATTR_MIN_STACK_SIZE")
nv_info_func_attr = ("EIATTR_CUDA_API_VERSION", "EIATTR_PARAM_CBANK", "EIATTR_CBANK_PARAM_SIZE", "EIATTR_KPARAM_INFO",
                      "EIATTR_MAXREG_COUNT", "EIATTR_EXIT_INSTR_OFFSETS", "EIATTR_MAX_THREADS")
nv_rel_action = b'\x73\x00\x00\x00\x00\x00\x00\x00\x00\x00\x00\x11\x25\x00\x05\x36' # TODO: figure out dynsym d_un type

def make_cubin(kernel:bytes, eiattr, parser:SASSParser, arch:str) -> memoryview:
  assert arch == "sm_89", f"cubin generation not supported for {arch}"
  attr = {k:v for k,v in eiattr.items()}
  attr.update({"EIATTR_MIN_STACK_SIZE": [[sym_names.index("FUNC") + 1, 0]], "EIATTR_FRAME_SIZE": [[sym_names.index("FUNC") + 1, 0]]})
  attr.update({k: [[sym_names.index(v[0]) + 1, v[1]] for v in attr[k]] for k in indexed_attr}) # map sym names to indices
  sec_tab = {".shstrtab": build_shstrtab(parser.function_name),
             ".strtab": build_strtab(parser.function_name),
             ".text.FUNC": build_kernel(kernel, parser.function_name, attr),
             ".nv.info": build_nv_info(attr),
             ".nv.info.FUNC": build_nv_info_func(parser.function_name, attr),
             ".nv.rel.action": build_nv_rel_action(),
             ".nv.constant0.FUNC": build_constant_memory(parser.function_name, parser.c_mem_sz)}
  sec_tab[".symtab"] = build_symtab(parser.function_name, sec_tab[".strtab"], sec_tab[".text.FUNC"])
  for s in sec_tab.values(): s.header.sh_name, s.header.sh_size = sh_name(s.name, sec_tab[".shstrtab"].content), len(s.content)
  sections = [ElfSection("", libc.Elf64_Shdr(sh_type=libc.SHT_NULL), content=b'')] + [sec_tab[name] for name in section_names]
  segments = build_segments(sections)
  header = build_file_header(attr, len(sections), len(segments))
  return make_elf(header, sections, segments)

def build_file_header(attr, shnum:int, phnum:int) -> libc.Elf64_Ehdr:
  header = libc.Elf64_Ehdr()
  header.e_ident[:] = b'\0'*16
  header.e_ident[:4] = b"\x7fELF"
  # header.e_ident[4:9] = [libc.ELFCLASS64, libc.ELFDATA2LSB, libc.EV_CURRENT, libc.ELFOSABI_LINUX, libc.ELFOSABI_AIX]
  header.e_ident[4:9] = [libc.ELFCLASS64, libc.ELFDATA2LSB, libc.EV_CURRENT, 51, libc.ELFOSABI_AIX]
  header.e_type = libc.ET_EXEC
  header.e_machine = 190 # TODO: sm_89?
  header.e_version = attr["EIATTR_CUDA_API_VERSION"][0][0]
  header.e_flags = 0x590559 # TODO: ???
  header.e_ehsize = 0x40
  header.e_phentsize = 0x38
  header.e_phnum = phnum
  header.e_shentsize = 0x40
  header.e_shnum = shnum
  header.e_shstrndx = section_names.index(".shstrtab") + 1
  return header

def build_segments(sections) -> List[ElfSegment]:
  allocs = [s for s in sections if s.header.sh_flags & libc.SHF_ALLOC]
  segments = [(libc.PT_PHDR, []), (libc.PT_LOAD, allocs), (libc.PT_LOAD, [])]
  return [ElfSegment(header=libc.Elf64_Phdr(p_type=t, p_flags=libc.PF_R + libc.PF_X, p_align=8), sections=s) for t,s in segments]

def build_shstrtab(function_name:str) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_STRTAB, sh_addralign=1)
  shstrtab = [s.replace("FUNC", function_name) for s in shstrtab_names]
  return ElfSection(".shstrtab", sh, b'\0' + b'\0'.join(s.encode() for s in shstrtab) + b'\0')

def build_strtab(function_name:str) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_STRTAB, sh_addralign=1)
  strtab = [s.replace("FUNC", function_name) for s in strtab_names]
  return ElfSection(".strtab", sh, b'\0' + b'\0'.join(s.encode() for s in strtab) + b'\0')

def build_kernel(kernel:bytes, function_name:str, attr:Dict[str, Union[List[List[int]], List[int]]]) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_PROGBITS, sh_flags=libc.SHF_EXECINSTR + libc.SHF_ALLOC, sh_link=section_names.index(".symtab") + 1,
                       sh_info=(attr["EIATTR_REGCOUNT"][0][1] << 24) + 6, sh_addralign=128)
  return ElfSection(f".text.{function_name}", sh, kernel)

def build_symtab(function_name:str, strtab:ElfSection, kernel:ElfSection) -> ElfSection:
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

def build_nv_info(attr:Dict[str, Union[List[int], int]]) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC, sh_link=section_names.index(".symtab") + 1, sh_addralign=4)
  return ElfSection(".nv.info", sh, pack_eiattr_tab(nv_info_attr, attr))

def build_nv_info_func(function_name:str, attr:Dict[str, Union[List[List[int]], List[int]]]) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC, sh_flags=0x40, sh_link=section_names.index(".symtab") + 1,
                       sh_info=section_names.index(".text.FUNC") + 1, sh_addralign=4) # TODO: sh_info is a guess, not documented
  return ElfSection(f".nv.info.{function_name}", sh, pack_eiattr_tab(nv_info_func_attr, attr))

def build_nv_rel_action() -> ElfSection:
  return ElfSection(f".nv.rel.action", libc.Elf64_Shdr(sh_type=libc.SHT_LOPROC + libc.SHT_DYNSYM, sh_addralign=8, sh_entsize=8), nv_rel_action)

def build_constant_memory(function_name:str, c_mem_sz) -> ElfSection:
  sh = libc.Elf64_Shdr(sh_type=libc.SHT_PROGBITS, sh_flags=0x40 + libc.SHF_ALLOC, sh_info=section_names.index(".text.FUNC") + 1, sh_addralign=4)
  return ElfSection(f".nv.constant0.{function_name}", sh, b'\0'*c_mem_sz)

def pack_eiattr_tab(attr_names:Sequence[str], kernel_eiattr:Dict[str, int]) -> bytes:
  content = bytearray()
  for attr_name in attr_names:
    for val in kernel_eiattr[attr_name]:
      attr_code = eiattr[attr_name]
      content += attr_code.to_bytes(2, 'little') + pack_eiattr(attr_code & 0xff, val)
  return bytes(content)

def pack_eiattr(vfmt, val):
  if vfmt == 1: return b'\x00\x00' # EIFMT_NVAL
  elif vfmt == 2: return int.to_bytes(val, 2, 'little') # EIFMT_BVAL
  elif vfmt == 3: return int.to_bytes(val, 2, 'little') # EIFMT_HVAL
  elif vfmt == 4: # EIFMT_SVAL
    return bytearray((len(val)*4).to_bytes(2, 'little')) + b''.join(v.to_bytes(4, 'little') for v in val)
    # bval = bytearray((len(val)*4).to_bytes(2, 'little')) # TODO: verify
    # for v in val: bval += v.to_bytes(4, 'little')
    # return bval
  else: raise ValueError

def pack_rel(addr:int, sym_idx:int, rel_type:int):
  return addr.to_bytes(8, "little") + ((sym_idx << 32) + rel_type).to_bytes(8, "little")

def sh_name(name:str, shstrtab:bytes):
  return shstrtab.index(b'\0' + name.encode() + b'\0') + 1
