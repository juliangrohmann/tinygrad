from pathlib import Path
from tqdm import tqdm
import tempfile
import subprocess
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.helpers import getenv
from tinygrad import Device

arch="sm_89"
sass_dir = Path(__file__).parent / "sass_dump"
lib_dir = Path("/usr/local/cuda/lib64")
libs = ['libcublasLt.so']
sass_dir.mkdir(parents=True, exist_ok=True)
ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
dev = Device["CUDA"]
if getenv("DUMP_SASS"):
  for num,ast in tqdm(enumerate(ast_strs), total=len(ast_strs)):
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    src = lin.to_program().src
    with tempfile.NamedTemporaryFile(mode="w", suffix=".cu", delete_on_close=False) as tmp_cu:
      tmp_cu.write(src)
      tmp_cu.close()
      with tempfile.NamedTemporaryFile(delete_on_close=False, suffix=".cubin") as tmp_cubin:
        tmp_cubin.close()
        subprocess.run(["nvcc", "--cubin", f"-arch={arch}", "-o", tmp_cubin.name, tmp_cu.name])
        with open(sass_dir / f"ast_{num}.sass", 'w') as out:
          subprocess.run(["cuobjdump", "-sass", f"-arch={arch}", tmp_cubin.name], stdout=out)

with open(sass_dir / "dump.sass", "w") as out:
  for file in sass_dir.iterdir():
    with open(file) as src:
      out.write(src.read() + "\n\n")
