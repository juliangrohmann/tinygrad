# import re
# import struct
#
# def getI(orig, pos, mask):
#   val = orig
#   neg = val.startswith('-')
#   if neg:
#     val = val[1:]
#
#   # Parse out custom index immediates for addresses
#   m = re.match(r'^(\d+)[xX]<([^>]+)>', val)
#   if m:
#     mul = int(m.group(1))
#     exp = m.group(2)
#     # Strip leading zeros (don't interpret numbers as octal)
#     exp = re.sub(r'(?<!\d)0+(?=[1-9])', '', exp)
#     globals_vars = re.findall(r'\$\w+', exp)
#     globals_dict = {}
#     if globals_vars:
#       # Handle global variables (assuming they are defined in the current scope)
#       for var in globals_vars:
#         var_name = var[1:]  # Remove leading '$'
#         globals_dict[var_name] = globals().get(var_name, 0)
#       # Evaluate the expression safely
#       val = mul * eval(exp, {"__builtins__": None}, globals_dict)
#     else:
#       # No globals, safe to evaluate
#       val = mul * eval(exp, {"__builtins__": None})
#   # Hexadecimal value
#   elif re.match(r'^0x[0-9a-fA-F]+$', val):
#     val = int(val, 16)
#   else:
#     # Simple decimal value
#     val = int(val)
#
#   if neg:
#     # If the mask removes the sign bit, the "neg" flag adds it back elsewhere in the code
#     val = -val
#     val &= mask
#
#   if (val & mask) != val:
#     raise ValueError(f"Immediate value out of range(0x{mask:x}): 0x{val:x} ({orig})")
#
#   return val << pos
#
# def getF(val, pos, type_, trunc=None):
#   # Hexadecimal value
#   if re.match(r'^0x[0-9a-fA-F]+$', val):
#     val = int(val, 16)
#   # Support infinity
#   elif re.search(r'INF', val, re.IGNORECASE):
#     if trunc:
#       val = 0x7f800 if type_ == 'f' else 0x7ff00
#     else:
#       val = 0x7f800000
#   else:
#     if type_ == 'f':
#       val = struct.unpack('I', struct.pack('f', float(val)))[0]
#     else:
#       val = struct.unpack('Q', struct.pack('d', float(val)))[0]
#     # Strip off sign bit if truncating; it will be added elsewhere by the flag capture
#     if trunc:
#       val = (val >> trunc) & 0x7ffff
#
#   return val << pos
#
# def getR(val, pos):
#   m = re.match(r'^R(\d+|Z)$', val)
#   if m:
#     reg = m.group(1)
#     if reg == 'Z':
#       val = 0xff
#     else:
#       reg_num = int(reg)
#       if reg_num < 255:
#         val = reg_num
#       else:
#         raise ValueError(f"Register number out of range: {val}")
#   else:
#     raise ValueError(f"Bad register name found: {val}")
#
#   return val << pos
#
# def getP(val, pos):
#   m = re.match(r'^P(\d|T)$', val)
#   if m:
#     pred = m.group(1)
#     if pred == 'T':
#       val = 7
#     else:
#       pred_num = int(pred)
#       if pred_num < 7:
#         val = pred_num
#       else:
#         raise ValueError(f"Predicate number out of range: {val}")
#   else:
#     raise ValueError(f"Bad predicate name found: {val}")
#
#   return val << pos
#
# def getC(val):
#   return ((int(val, 16) >> 2) & 0x7fff) << 20
#
# # Map operands into their value and position in the opcode
# operands = {
#   'p0'     : lambda val: getP(val, 0),
#   'p3'     : lambda val: getP(val, 3),
#   'p12'    : lambda val: getP(val, 12),
#   'p29'    : lambda val: getP(val, 29),
#   'p39'    : lambda val: getP(val, 39),
#   'p45'    : lambda val: getP(val, 45),
#   'p48'    : lambda val: getP(val, 48),
#   'p58'    : lambda val: getP(val, 58),
#   'r0'     : lambda val: getR(val, 0),
#   'r8'     : lambda val: getR(val, 8),
#   'r20'    : lambda val: getR(val, 20),
#   'r28'    : lambda val: getR(val, 28),
#   'r39s20' : lambda val: getR(val, 39),
#   'r39'    : lambda val: getR(val, 39),
#   'r39a'   : lambda val: getR(val, 39),  # Does not modify opcode; XOR the r39 value again to wipe it out
#   'c20'    : lambda val: getC(val),
#   'c39'    : lambda val: getC(val),
#   'c34'    : lambda val: int(val, 16) << 34,
#   'c36'    : lambda val: int(val, 16) << 36,
#   'f20w32' : lambda val: getF(val, 20, 'f'),
#   'f20'    : lambda val: getF(val, 20, 'f', 12),
#   'd20'    : lambda val: getF(val, 20, 'd', 44),
#   'i8w4'   : lambda val: getI(val, 8,  0xf),
#   'i20'    : lambda val: getI(val, 20, 0x7ffff),
#   'i20w6'  : lambda val: getI(val, 20, 0x3f),
#   'i20w7'  : lambda val: getI(val, 20, 0x7f),
#   'i20w8'  : lambda val: getI(val, 20, 0xff),
#   'i20w12' : lambda val: getI(val, 20, 0xfff),
#   'i20w24' : lambda val: getI(val, 20, 0xffffff),
#   'i20w32' : lambda val: getI(val, 20, 0xffffffff),
#   'i31w4'  : lambda val: getI(val, 31, 0xf),
#   'i34w13' : lambda val: getI(val, 34, 0x1fff),
#   'i36w20' : lambda val: getI(val, 36, 0xfffff),
#   'i39w8'  : lambda val: getI(val, 39, 0xff),
#   'i28w8'  : lambda val: getI(val, 28, 0xff),
#   'i28w20' : lambda val: getI(val, 28, 0xfffff),
#   'i48w8'  : lambda val: getI(val, 48, 0xff),
#   'i51w5'  : lambda val: getI(val, 51, 0x1f),
#   'i53w5'  : lambda val: getI(val, 53, 0x1f),
# }
#
# def parseInstruct(inst, grammar):
#   # Check if the grammar rule is a compiled regex; if not, compile it
#   pattern = grammar.get('rule')
#   if isinstance(pattern, str):
#     pattern = re.compile(pattern)
#     grammar['rule'] = pattern  # Cache the compiled pattern if desired
#
#   match = pattern.match(inst)
#   if not match:
#     return None
#
#   capData = match.groupdict()
#   return capData
