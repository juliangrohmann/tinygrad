from __future__ import annotations
from typing import Tuple, List, Any
from dataclasses import dataclass
import io
import tinygrad.runtime.autogen.libc as libc

@dataclass(frozen=True)
class ElfSection: name:str; header:libc.Elf64_Shdr; content:bytes # noqa: E702

def elf_loader(blob:bytes, force_section_align:int=1) -> Tuple[memoryview, List[ElfSection], Any]:
  def _strtab(blob: bytes, idx: int) -> str: return blob[idx:blob.find(b'\x00', idx)].decode('utf-8')

  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  section_headers = (libc.Elf64_Shdr * header.e_shnum).from_buffer_copy(blob[header.e_shoff:])
  sh_strtab = blob[(shstrst:=section_headers[header.e_shstrndx].sh_offset):shstrst+section_headers[header.e_shstrndx].sh_size]
  sections = [ElfSection(_strtab(sh_strtab, sh.sh_name), sh, blob[sh.sh_offset:sh.sh_offset+sh.sh_size]) for sh in section_headers]

  def _to_carray(sh, ctype): return (ctype * (sh.header.sh_size // sh.header.sh_entsize)).from_buffer_copy(sh.content)
  rel = [(sh, sh.name[4:], _to_carray(sh, libc.Elf64_Rel)) for sh in sections if sh.header.sh_type == libc.SHT_REL]
  rela = [(sh, sh.name[5:], _to_carray(sh, libc.Elf64_Rela)) for sh in sections if sh.header.sh_type == libc.SHT_RELA]
  symtab = [_to_carray(sh, libc.Elf64_Sym) for sh in sections if sh.header.sh_type == libc.SHT_SYMTAB][0]
  progbits = [sh for sh in sections if sh.header.sh_type == libc.SHT_PROGBITS]

  # Prealloc image for all fixed addresses.
  image = bytearray(max([sh.header.sh_addr + sh.header.sh_size for sh in progbits if sh.header.sh_addr != 0] + [0]))
  for sh in progbits:
    if sh.header.sh_addr != 0: image[sh.header.sh_addr:sh.header.sh_addr+sh.header.sh_size] = sh.content
    else:
      image += b'\0' * (((align:=max(sh.header.sh_addralign, force_section_align)) - len(image) % align) % align) + sh.content
      sh.header.sh_addr = len(image) - len(sh.content)

  # Relocations
  relocs = []
  for sh, trgt_sh_name, c_rels in rel + rela:
    target_image_off = next(tsh for tsh in sections if tsh.name == trgt_sh_name).header.sh_addr
    rels = [(r.r_offset, symtab[libc.ELF64_R_SYM(r.r_info)], libc.ELF64_R_TYPE(r.r_info), getattr(r, "r_addend", 0)) for r in c_rels]
    relocs += [(target_image_off + roff, sections[sym.st_shndx].header.sh_addr + sym.st_value, rtype, raddend) for roff, sym, rtype, raddend in rels]

  return memoryview(image), sections, relocs

def make_elf(header:libc.Elf64_Ehdr, sections:List[ElfSection], prog_headers:Optional[List[libc.Elf64_Phdr]]=None) -> memoryview:
  def _pad(n:int, align:int): return b'\0' * ((align - n % align) % align if align != 0 else 0)

  blob = bytearray()
  for s in sections:
    if s.header.sh_type == libc.SHT_NOBITS: continue
    blob += _pad(len(blob) + header.e_ehsize, s.header.sh_addralign)
    s.header.sh_offset = header.e_ehsize + len(blob) if s.header.sh_name != 0 else 0
    s.header.sh_size = len(s.content)
    blob += s.content

  blob += _pad(len(blob) + header.e_ehsize, 8)
  header.e_shoff = header.e_ehsize + len(blob)
  header.e_shnum = len(sections)
  for s in sections:
    blob += bytearray(s.header)

  if prog_headers:
    prog_seg_st = header.e_phoff
    blob += _pad(len(blob) + header.e_ehsize, 8)
    header.e_phoff = header.e_ehsize + len(blob)
    segments = [(ph, [s.header for s in sections if ph.p_offset <= s.header.sh_offset < ph.p_offset + ph.p_memsz]) for ph in prog_headers]
    for i, (ph, sh) in enumerate(segments):
      if ph.p_offset == prog_seg_st:
        ph.p_offset = header.e_phoff
        ph.p_filesz = ph.p_memsz = header.e_phentsize * header.e_phnum
      else:
        assert sh, f"no sections match program header at index {i}"
        ph.p_offset = sh[0].sh_offset
        ph.p_memsz = (sh[-1].sh_offset + sh[-1].sh_size - sh[0].sh_offset)
        ph.p_filesz = ph.p_memsz - sum(s.sh_size for s in sh if s.sh_type == libc.SHT_NOBITS)
      blob += bytearray(ph)
  return memoryview(bytearray(header) + blob)
