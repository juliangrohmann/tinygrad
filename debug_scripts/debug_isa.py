import json
from pathlib import Path
from tinygrad.runtime.support.assembler_sass import SASSAssembler

isa_89 = (Path(__file__).parents[1] / "tinygrad" / "runtime" / "support" / "isa.sm_89.json").as_posix()
with open(isa_89) as f: assembler = SASSAssembler(json.load(f))
for k,v in assembler.isa.items():
  if k == "IMAD_R_R_R_c[I][I]":
    print(k)
    print(v.code_mods)
    for inst_key, op_spec in v.specs.items():
      print(f"{inst_key=}")
      for spec in op_spec:
        print(f"{spec.cmods=}, {spec.code}")
  # all_modis = [modi.strip('.') for inst in v.values() for group in inst.op_modis for modi in group.keys()]
  # print(all_modis)
  # for inst in v.values():
  #   for m in inst.code_modis:
  #     assert m.strip('.') not in all_modis
