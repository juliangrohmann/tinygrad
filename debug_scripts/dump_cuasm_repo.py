import json
from pathlib import Path
from CuAsm.CuInsAssemblerRepos import CuInsAssemblerRepos

cache = Path(__file__).parents[1] / "tinygrad" / "runtime" / "support" / "DefaultInsAsmRepos.sm_89.txt"
repos = CuInsAssemblerRepos(cache.as_posix())
isa = {}
for key,ins in repos.m_InsAsmDict.items():
  isa[key] = {
    "sol": [int(e) for e in ins.m_PSol],
    "modi": ins.m_InsModiSet,
    "fac": ins.m_PSolFac
  }
with open(cache.with_name("sass.sm_89.json"), "w") as f: json.dump(isa, f)