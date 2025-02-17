import subprocess, tempfile, hashlib
from pathlib import Path
from tinygrad.helpers import getenv
from CuAsm import CubinFile

fn = Path(getenv("IN", ""))
buf = Path(tempfile.gettempdir()) / str(hashlib.md5(fn.as_posix().encode()))
subprocess.run(["nvcc", "--cubin", "-arch=sm_89", "-o", buf.with_suffix(".cubin").as_posix(), fn.as_posix()],
               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
CubinFile(buf.with_suffix(".cubin").as_posix()).saveAsCuAsm(Path(fn).with_suffix(".cuasm"))
