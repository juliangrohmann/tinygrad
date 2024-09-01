import unittest, tempfile, json, subprocess
from pathlib import Path
from tqdm import tqdm
from tinygrad.runtime.support.assembler_sass import SASSAssembler, EncodingType, parse_inst, sr_vals
from instruction_solver import set_bit_range

def helper_values(spec, neg=False):
  enc = sorted([e for e in spec.enc if e.type == EncodingType.OPERAND], key=lambda x: x.idx)
  values = [i + 1 << e.shift for i,e in enumerate(enc)]
  values = [v if v < 2 ** e.length else (v - 2 ** e.length + 1) for v,e in zip(values, enc)]
  if neg: values = [-v if e.key == "I" and e.shift == 0 and e.length >= 8 else v for v,e in zip(values, enc)]
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
    disasm_vals = parse_inst(disasm)[1]
    self.assertEqual(disasm_vals, values, msg=f"\n{key=}\n{op_mods=}\n{values=}\n{disasm_vals=}\n{disasm=}\n{inst_code=}")

  @unittest.skip
  def test_alu(self):
    ops = [(k,v) for k,v in self.assembler.isa.items()
           if any(s in k for s in ["MAD", "FMA", "MUL", "ADD"]) and not any(s in k for s in self.bad_oper_encoding)]
    for k,inst in tqdm(ops):
      for spec_group in inst.specs.values():
        for spec in spec_group:
          self.helper_test_values(k, [7] + helper_values(spec), op_mods=spec.cmods)
          self.helper_test_values(k, [7] + helper_values(spec, neg=True), op_mods=spec.cmods)

  @unittest.skip
  def test_global_memory(self):
    ops = [(k,v) for k,v in self.assembler.isa.items()
           if any(s in k for s in ["LDG", "STG"]) and not any(s in k for s in self.bad_oper_encoding)]
    for k,inst in tqdm(ops):
      for spec_group in inst.specs.values():
        for spec in spec_group:
          self.helper_test_values(k, [7] + helper_values(spec), op_mods=["E"] + spec.cmods)

  @unittest.skip
  def test_bra(self):
    self.helper_test_values("BRA_I", [7, 48])
    self.helper_test_values("BRA_I", [0, 48])

  @unittest.skip
  def test_nop(self):
    self.helper_test_values("NOP", [0])

  @unittest.skip
  def test_s2r(self):
    for v in sr_vals.values():
      self.helper_test_values("S2R_R_SR", [7, 1, v])

  def test_disasm(self):
    inst_code = self.assembler.encode_instruction("I2FP_R_R", [7, 0, 0])
    bits = bytearray(inst_code.to_bytes(16, "little"))
    for mod_1 in range(4):
      for mod_2 in range(4):
        for mod_3 in range(4):
          set_bit_range(bits, 74, 76, mod_1)
          set_bit_range(bits, 76, 78, mod_2)
          set_bit_range(bits, 84, 86, mod_3)
          disasm = disassemble(bits)
          print(f"{disasm=}")

if __name__ == '__main__':
  unittest.main()
