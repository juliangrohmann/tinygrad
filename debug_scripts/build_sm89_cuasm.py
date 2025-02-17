from pathlib import Path
from tqdm import tqdm
import tempfile, hashlib
import subprocess
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.helpers import getenv
from tinygrad import Device
from CuAsm import CuInsFeeder, CuInsAssemblerRepos
from CuAsm.utils import CubinUtils

arch="sm_89"
sass_dir = Path(__file__).parent / "sass_dump"
lib_dir = Path("/usr/local/cuda/lib64")
out_fn = getenv("OUT_FILE", str(Path(__file__).parent) / "DefaultInsAsmRepos.sm_89.txt")
libs = ['libcublasLt.so']
if getenv("DUMP_SASS", 1):
  sass_dir.mkdir(parents=True, exist_ok=True)
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  dev = Device["CUDA"]
  for num,ast in tqdm(enumerate(ast_strs), total=len(ast_strs)):
    lin = ast_str_to_lin(ast, opts=dev.renderer)
    lin.hand_coded_optimizations()
    src = lin.to_program().src
    fn = (Path(tempfile.gettempdir()) / f"tinycuda_{hashlib.md5(src.encode()).hexdigest()}").as_posix()
    with open(fn + ".cu", "w") as f: f.write(src)
    subprocess.run(["nvcc", "--cubin", f"-arch={arch}", "-o", fn + "_orig.cubin", fn + ".cu"], check=True, capture_output=True)
    CubinUtils.hackCubinDesc(fn + "_orig.cubin", fn + ".cubin")
    with open(sass_dir / f"ast_{num}.sass", 'w') as out:
      subprocess.run(["cuobjdump", "-sass", f"-arch={arch}", fn + ".cubin"], check=True, stdout=out)

repos = CuInsAssemblerRepos(arch=arch)
ncnt = 0
for fn in sass_dir.iterdir():
  feeder = CuInsFeeder(str(fn))
  ncnt += repos.update(feeder)
print(f"processed {ncnt} new entries.")
repos.save2file(out_fn)
