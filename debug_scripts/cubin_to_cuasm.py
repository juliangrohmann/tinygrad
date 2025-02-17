from pathlib import Path
from tinygrad.helpers import getenv
from CuAsm import CubinFile

fn = Path(getenv("SRC", ""))
CubinFile(fn).saveAsCuAsm(fn.with_suffix(".cuasm"))