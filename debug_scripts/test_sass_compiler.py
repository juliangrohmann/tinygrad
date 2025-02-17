import subprocess, tempfile, hashlib, re
from pathlib import Path
from tqdm import tqdm
from tinygrad import Device
from tinygrad.helpers import getenv, to_function_name, colored
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.runtime.ops_cuda import CUDACompiler
from tinygrad.codegen.uops import UOps
from tinygrad.runtime.support.compiler_cuda import SASSCompiler
from tinygrad.runtime.support.elf import elf_loader
import tinygrad.runtime.autogen.libc as libc
from CuAsm import CuAsmParser, CubinFile, CuAsmLogger

def print_bytes(data, color_func=lambda i,x: "red" if x == 0 else "green", decode=False):
  print(labels := ' '.join([f"{i:>02}" for i in range(1, 17)]))
  print('-' * len(labels))
  for row in range((len(data) + 15) // 16):
    d = data[row*16:min((row+1)*16, len(data))]
    s = ' '.join([colored(f"{hex(b)[2:]:>02}", color_func(row*16+i,b)) for i,b in enumerate(d)])
    if decode: s += ' '*3 + str(d)
    print(s)

def compare_bytes(a, b, decode=False):
  def color_func(i, x): return "green" if i < len(a) and i < len(b) and a[i] == b[i] else "red"
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

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  # no bfloat16 for ptx at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["CUDA"]

  CuAsmLogger.initLogger("/logs/debug.txt", file_level=15, stdout_level=40)
  parser = CuAsmParser()
  parser.setInsAsmRepos(str(Path(__file__).parent / "DefaultInsAsmRepos.sm_89.txt"), arch="sm_89")
  sass_compiler = SASSCompiler("sm_89")

  out_dir = getenv("OUT", "")
  debug_sec = getenv("SECTION", "")
  start, end, single = getenv("START", 0), getenv("END", len(ast_strs)), getenv("NUM", -1)
  for num,ast in tqdm(enumerate(ast_strs), total=min(end, len(ast_strs))):
    if not (start <= num < end) or (single != -1 and num != single): continue

    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    lin.linearize()

    if cuasm_fn := getenv("CUASM", ""):
      fn = Path(cuasm_fn)
    else:
      if cu_fn := getenv("CU", ""):
        temp_fn = (Path(tempfile.gettempdir()) / f"cu_buf_{hashlib.md5(cu_fn.encode()).hexdigest()}").as_posix()
        subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", temp_fn + ".cubin", cu_fn], stdout=subprocess.DEVNULL)
      else:
        src = dev.renderer.render(to_function_name(lin.name), lin.uops.uops)
        temp_fn = (Path(tempfile.gettempdir()) / f"cu_buf_{hashlib.md5(src.encode()).hexdigest()}").as_posix()
        with open(temp_fn + ".cu", "w") as f: f.write(src)
        subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", temp_fn + ".cubin", temp_fn + ".cu"],
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
      CubinFile(temp_fn + ".cubin").saveAsCuAsm(temp_fn + ".cuasm")
      fn = Path(temp_fn + ".cuasm")

    print(f"source file: {fn}")
    parser.parse(fn.with_suffix(".cuasm").as_posix())
    parser.saveAsCubin(fn.with_name(fn.name + "_cuasm.cubin").as_posix())

    block_dims = [1, 1, 1]
    for uop in lin.uops.uops:
      if uop.op == UOps.SPECIAL and uop.arg[0][:3] == "lid":
        block_dims[int(uop.arg[0][-1])] = uop.arg[1]
    param_cnt = sum(1 for u in lin.uops.uops if u.op is UOps.DEFINE_GLOBAL)

    skip = True
    cuasm = ""
    with open(fn.with_suffix(".cuasm")) as f:
      for line in f:
        r = re.search(r"SHI_REGISTERS=(\d+)", line)
        if r:
          reg_cnt = int(r.groups()[0])
        if not skip:
          if line.strip().startswith("//"):
            break
          cuasm += line
        elif line.strip().startswith(".text."):
          skip = False
          cuasm += line

    cuasm = f"PARAM_COUNT={param_cnt}\n" + cuasm
    cuasm = f"SHI_REGISTERS={reg_cnt}\n" + cuasm
    for i,dim in list(enumerate(block_dims))[::-1]:
      cuasm = f"BLOCK_DIM_{i}={dim}\n" + cuasm

    if getenv("PRINT_SRC", 0): print('\n' + cuasm)
    tiny_header, tiny_sections, tiny_prog_headers = split_blob(tiny_cubin := sass_compiler.compile(cuasm))
    tiny_sec_map = to_dict(tiny_sections)

    with open(fn.with_name(fn.name + "_cuasm.cubin"), "rb") as f:
      cuasm_header, cuasm_sections, cuasm_prog_headers = split_blob(cuasm_cubin := f.read())
    cuasm_sec_map = to_dict(cuasm_sections)

    if getenv("PRINT_SECTIONS", 0):
      print("\nsections:")
      for s in cuasm_sections: print(s.name)
    if getenv("PRINT_HEADER", 0):
      print("\nfile header:")
      for attr, _ in libc.Elf64_Ehdr._fields_:
        val = cuasm_header.__getattribute__(attr)
        if attr != "e_ident":
          print(f"{attr:} {hex(val)}")
        else:
          print(f"{attr}:")
          print_bytes(val)
    if getenv("PRINT_PROG_HEADERS", 0):
      print("\nprogram header table:")
      for i,phdr in enumerate(cuasm_prog_headers):
        print(f"program header #{i}:")
        for attr, _ in libc.Elf64_Phdr._fields_:
          print(f"{attr:} {phdr.__getattribute__(attr)}")

    print("testing file headers...")
    valid_fh_attr = ["e_type", "e_machine", "e_version", "e_entry", "e_flags", "e_ehsize",
                     "e_phentsize", "e_phnum", "e_shentsize", "e_shnum", "e_shstrndx"]
    compare_cobj(valid_fh_attr, tiny_header, cuasm_header)

    print("testing section headers...")
    valid_sh_attr = ["sh_name", "sh_type", "sh_flags", "sh_link", "sh_info", "sh_entsize", "sh_addralign"]
    for s in cuasm_sections:
      if s.name not in tiny_sec_map:
        print(f"missing section: {s.name}")
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
      tiny_blob, cuasm_blob = tiny_sec_map[s.name].content, cuasm_sec_map[s.name].content
      if tiny_blob == cuasm_blob:
        print(colored(f"{s.name}: success", "green"))
      else:
        print(colored(f"{s.name}: mismatch", "red"))
        compare_bytes(tiny_blob, cuasm_blob, decode=getenv("DECODE", False))
        break

    if hex_out := Path(getenv("HEX_OUT", "")):
      with open(tiny_hex := hex_out / "tiny.cubin", "wb") as f: f.write(tiny_cubin)
      with open(cuasm_hex := hex_out / "cuasm.cubin", "wb") as f: f.write(cuasm_cubin)
      for fn in [tiny_hex, cuasm_hex]:
        with open(fn.with_suffix(".hex"), "w") as f: subprocess.run(["xxd", fn.as_posix()], stdout=f)

    if debug_sec and debug_sec in cuasm_sec_map:
      sec = cuasm_sec_map[debug_sec]
      print(f"DEBUG: {sec.name}")
      print_bytes(sec.content, decode=getenv("DECODE", False))
      print(sec.content)
      print(len(sec.content))
    elif debug_sec:
      print(f"unknown section name: {debug_sec}")

    for i,(a,b) in enumerate(zip(tiny_cubin, cuasm_cubin)):
      assert a == b, f"mismatch in blob at address {hex(i)}: a={hex(a)[2:]:>02}, b={hex(b)[2:]:<02}"
    print(colored("cubin match!", "green"))
