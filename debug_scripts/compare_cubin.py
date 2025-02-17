import subprocess
from pathlib import Path
from tinygrad import Device
from tinygrad.helpers import getenv, colored
from tinygrad.runtime.support.elf import elf_loader
import tinygrad.runtime.autogen.libc as libc

def print_bytes(data, color_func=lambda i,x: "red" if x == 0 else "green", decode=False):
  print(labels := ' '.join([f"{i:>02}" for i in range(1, 17)]))
  print('-' * len(labels))
  for row in range((len(data) + 15) // 16):
    d = data[row*16:min((row+1)*16, len(data))]
    s = ' '.join([colored(f"{hex(b)[2:]:>02}", color_func(row*16+i,b)) for i,b in enumerate(d)])
    if decode: s += ' '*3 + str(d)
    print(s)

def compare_bytes(a, b, decode=False):
  def color_func(i,x): return "green" if i < len(a) and i < len(b) and a[i] == b[i] else "red"
  print_bytes(a, color_func=color_func, decode=decode)
  print()
  print_bytes(b, color_func=color_func, decode=decode)

def to_dict(sections):
  return {s.name: s for s in sections}

def split_blob(blob):
  header = libc.Elf64_Ehdr.from_buffer_copy(blob)
  sections = elf_loader(blob)[1]
  prog_headers = (libc.Elf64_Phdr*header.e_phnum).from_buffer_copy(blob[header.e_phoff:])
  return header, sections, prog_headers

def compare_cobj(attr, tiny_obj, cuasm_obj, prefix=""):
  for a in attr:
    if (tv := tiny_obj.__getattribute__(a)) != (cv := cuasm_obj.__getattribute__(a)):
      print(colored(f"{prefix}{a} mismatch: tiny={tv}, cuasm={cv}", "red"))
  return None

def load_cubin(fn):
  with open(fn, "rb") as f:
    header, sections, prog_headers = split_blob(f.read())
  sec_map = to_dict(sections)
  return header, sections, prog_headers, sec_map

def print_header(header):
  for attr, _ in libc.Elf64_Ehdr._fields_:
    val = header.__getattribute__(attr)
    if attr != "e_ident":
      print(f"{attr:} {hex(val)}")
    else:
      print(f"{attr}:")
      print_bytes(val)

def print_prog_headers(prog_headers):
  for i,phdr in enumerate(prog_headers):
    print(f"program header #{i}:")
    for attr, _ in libc.Elf64_Phdr._fields_:
      print(f"{attr:} {phdr.__getattribute__(attr)}")

if __name__ == "__main__":
  dev = Device["CUDA"]
  out_dir = getenv("OUT", "")
  debug_sec = getenv("SECTION", "")

  tiny_header, tiny_sections, tiny_prog_headers, tiny_sec_map = load_cubin(fn_a := getenv("SRC_A", ""))
  cuasm_header, cuasm_sections, cuasm_prog_headers, cuasm_sec_map = load_cubin(fn_b := getenv("SRC_B", ""))

  if getenv("PRINT_SECTIONS", 0):
    print("\n[A] sections:")
    for s in tiny_sections: print(s.name)
    print("\n[B] sections:")
    for s in cuasm_sections: print(s.name)
  if getenv("PRINT_HEADER", 0):
    print("\n[A] file header:")
    print_header(tiny_header)
    print("\n[B] file header:")
    print_header(cuasm_header)
  if getenv("PRINT_PROG_HEADERS", 0):
    print("\n[A]program header table:")
    print_prog_headers(tiny_prog_headers)
    print("\n[B]program header table:")
    print_prog_headers(cuasm_prog_headers)
  if out_dir:
    with open(Path(out_dir) / "tiny.hex", "w") as f: subprocess.run(["xxd", getenv("SRC_A", "")], stdout=f)
    with open(Path(out_dir) / "cuasm.hex", "w") as f: subprocess.run(["xxd", getenv("SRC_B", "")], stdout=f)

  print("testing file headers...")
  valid_fh_attr = ["e_type", "e_machine", "e_version", "e_entry", "e_flags", "e_ehsize",
                   "e_phentsize", "e_phnum", "e_shentsize", "e_shnum", "e_shstrndx"]
  compare_cobj(valid_fh_attr, tiny_header, cuasm_header)

  print("testing section headers...")
  valid_sh_attr = ["sh_name", "sh_type", "sh_flags", "sh_link", "sh_info", "sh_entsize", "sh_addralign"]
  for s in cuasm_sections:
    if s.name not in tiny_sec_map:
      print(f"missing section: {s.name}")
      continue
    print(f"testing section header: {s.name}")
    compare_cobj(valid_sh_attr, tiny_sec_map[s.name].header, cuasm_sec_map[s.name].header, prefix=f"{s.name}, ")

  print("testing program header table...")
  valid_ph_attr = ["p_type", "p_flags", "p_vaddr", "p_paddr", "p_align"]
  if len(tiny_prog_headers) != len(cuasm_prog_headers):
    print(colored(f"program header table length mismatch: tiny={len(tiny_prog_headers)}, cuasm={len(cuasm_prog_headers)}", "red"))
  else:
    for i, (tp, cp) in enumerate(zip(tiny_prog_headers, cuasm_prog_headers)):
      compare_cobj(valid_ph_attr, tp, cp, prefix=f"entry #{i}, ")

  print("testing section content...")
  for s in cuasm_sections:
    if s.name not in tiny_sec_map:
      print(f"skipping {s.name} content")
      continue
    tiny_blob, cuasm_blob = tiny_sec_map[s.name].content, cuasm_sec_map[s.name].content
    if tiny_blob == cuasm_blob:
      print(colored(f"{s.name}: success", "green"))
    else:
      print(colored(f"{s.name}: mismatch", "red"))
      compare_bytes(tiny_blob, cuasm_blob, decode=getenv("DECODE", False))

  if hex_out := getenv("HEX_OUT", ""):
    for fn in [fn_a, fn_b]:
      with open(fn.with_suffix(".hex"), "w") as f: subprocess.run(["xxd", fn.as_posix()], stdout=f)

  if debug_sec:
    for c,sec_map in [("A",tiny_sec_map), ("B", cuasm_sec_map)]:
      if debug_sec in sec_map:
        sec = sec_map[debug_sec]
        print(f"DEBUG {c}: {sec.name}")
        print_bytes(sec.content, decode=getenv("DECODE", False))
        print(sec.content)
        print(len(sec.content))
  elif debug_sec:
    print(f"unknown section name: {debug_sec}")

  # for i,(a,b) in enumerate(zip(tiny_cubin, cuasm_cubin)):
  #   assert a == b, f"mismatch in blob at address {hex(i)}: a={hex(a)[2:]:>02}, b={hex(b)[2:]:<02}"
  # print(colored("cubin match!", "green"))
