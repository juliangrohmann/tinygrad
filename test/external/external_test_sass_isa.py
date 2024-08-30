import unittest, tempfile, json, subprocess
from pathlib import Path
from tqdm import tqdm
from tinygrad.runtime.support.assembler_sass import SASSAssembler, EncodingType, parse_inst

def test_values(spec):
  enc = sorted([e for e in spec.enc if e.type == EncodingType.OPERAND], key=lambda x: x.value)
  values = [(i + 1 << e.shift) + e.offset for i,e in enumerate(enc)]
  values = [v if v < 2 ** e.length else (v - 2 ** e.length + 1) for v,e in zip(values, enc)]
  values = [-v if e.key == "I" and e.shift == 0 and e.offset == 0 and e.length >= 8 else v for v,e in zip(values, enc)]
  return [float(v) if e.key == "FI" else v for v,e in zip(values, enc)]

def disassemble(code:bytes):
  with tempfile.NamedTemporaryFile(mode="wb", delete=False) as tmp:
    tmp.write(code)
    tmp.close()
    disasm = subprocess.run(["nvdisasm", tmp.name, "--binary", "SM89"], capture_output=True)
    return "\n".join(line[line.find("*/") + 2 :].strip() for line in disasm.stdout.decode().split("\n")[1:]).strip()

class TestISASpec(unittest.TestCase):
  def setUp(self):
    self.bad_oper_encoding = ["LDG_R_R_RURI_I", "LDG_R_R_RUR_I", "FSWZADD"]
    with open((Path(__file__).parents[2] / "tinygrad" / "runtime" / "support" / "isa.sm_89.json").as_posix()) as f:
      self.assembler = SASSAssembler(json.load(f))

  def helper_test_values(self, key, values, op_mods=(), operand_mods=None):
    inst_code = self.assembler.encode_instruction(key, values, op_mods=op_mods, operand_mods=operand_mods)
    disasm = disassemble(inst_code.to_bytes(16, "little"))
    print(f"{disasm=}")
    disasm_vals = parse_inst(disasm)[1]
    self.assertEqual(disasm_vals, values, msg=f"\n{key=}\n{op_mods=}\n{values=}\n{disasm_vals=}\n{disasm=}\n{inst_code}")

  def test_alu(self):
    ops = [(k,v) for k,v in self.assembler.isa.items()
           if any(s in k for s in ["MAD", "FMA", "MUL", "ADD"]) and not any(s in k for s in self.bad_oper_encoding)]
    for k,inst in tqdm(ops):
      for spec in inst.specs.values():
        self.helper_test_values(k, [7] + test_values(spec), op_mods=spec.cmods)

  def test_bra(self):
    self.helper_test_values("BRA_I", [7, 48])
    self.helper_test_values("BRA_I", [0, 48])

  def test_nop(self):
    self.helper_test_values("NOP", [0])

if __name__ == '__main__':
  unittest.main()
