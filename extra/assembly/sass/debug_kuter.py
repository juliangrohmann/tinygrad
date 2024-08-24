from pathlib import Path
from collections import defaultdict
from tinygrad.runtime.support.parser_sass import SASSParser
from tinygrad.runtime.support.compiler_cuda import SASSCompiler
from tinygrad.runtime.support.cubin import make_cubin
bin = "00000000 00000000 00000000 00100000 00000000 00000000 00001000 00000000 00000000 00000000 00100011 00000000 00000000 00000100 01111010 10111001"

compiler = SASSCompiler("sm_89")
with open("/home/julian/projects/tinycorp/tinygrad/debug_src/debug.cuasm") as f:
  src = f.read()
blob = int(bin.replace(" ", ""), 2).to_bytes(128 // 8, "little")
print(len(blob))
with open(out := Path(__file__).parent / "test.cubin", "wb") as f:
  f.write(compiler.compile(src, True, inject=blob))
