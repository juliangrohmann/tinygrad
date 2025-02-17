# def Scheduler(block, blockNum, regMap, debug):
#   vectors = regMap['__vectors']
#   lineNum = 0
#
#   instructs = []
#   comments = []
#   ordered = None
#   first = 0
#
#   for line in block.split("\n"):
#     # keep track of line nums in the physical file
#     lineNum += 1
#
#     if not preProcessLine(line):
#       if re.search(r'\S', line):
#         comments.append(line)
#       continue
#
#     # match an instruction
#     inst = processAsmLine(line, lineNum)
#     if inst:
#       # if the first instruction in the block is waiting on a dep, it should go first.
#       if first == 0 and (inst['ctrl'] & 0x1f800):
#         inst['first'] = 0
#       else:
#         inst['first'] = 1
#       first += 1
#
#       # if the instruction has a stall of zero set, it's meant to be last (to mesh with next block)
#       # inst['first'] = 1 if inst['ctrl'] & 0x0000f else 2
#       inst['exeTime'] = 0
#       if ordered:
#         inst['order'] = ordered
#         ordered += 1
#       if re.search('FORCE', inst['comment']):
#         inst['force_stall'] = inst['ctrl'] & 0xf
#
#       instructs.append(inst)
#     # match a label
#     elif re.match(r'^([a-zA-Z]\w*):', line):
#       raise Exception(f"SCHEDULE_BLOCK's cannot contain labels. block: {blockNum} line: {lineNum}")
#     # open an ORDERED block
#     elif re.match(r'^<ORDERED>', line):
#       if ordered:
#         raise Exception("you cannot use nested <ORDERED> tags")
#       ordered = 1
#     # close an ORDERED block
#     elif re.match(r'^</ORDERED>', line):
#       if not ordered:
#         raise Exception("missing opening <ORDERED> for closing </ORDERED> tag")
#       ordered = 0
#     else:
#       raise Exception(f"badly formed line at block: {blockNum} line: {lineNum}: {line}")
#
#   writes = {}
#   reads = {}
#   ready = []
#   schedule = []
#   orderedParent = None
#
#   # assemble the instructions to op codes
#   for instruct in instructs:
#     match = False
#     for gram in grammar[instruct['op']]:
#       capData = parseInstruct(instruct['inst'], gram)
#       if not capData:
#         continue
#       dest = []
#       src = []
#
#       # copy over instruction types for easier access
#       for key in itypes:
#         instruct[key] = gram['type'][key]
#
#       instruct['dualCnt'] = 1 if instruct.get('dual') else 0
#
#       # A predicate prefix is treated as a source reg
#       if instruct.get('pred'):
#         src.append(instruct['predReg'])
#
#       # Handle P2R and R2P specially
#       if re.search('P2R|R2P', instruct['op']) and capData.get('i20w7'):
#         lst = dest if instruct['op'] == 'R2P' else src
#         mask = int(capData['i20w7'], 16)
#         for p in range(7):
#           if mask & (1 << p):
#             lst.append(f"P{p}")
#           # make this instruction dependent on any predicates it's not setting
#           # this is to prevent a race condition for any predicate sets that are pending
#           elif instruct['op'] == 'R2P':
#             src.append(f"P{p}")
#         # These instructions can't be dual issued
#         instruct['nodual'] = 1
#
#       # Populate our register source and destination lists, skipping any zero or true values
#       for operand in sorted(capData.keys()):
#         if operand not in regops:
#           continue
#         lst = dest if operand in destReg and instruct['op'] not in noDest else src
#         badVal = 'RZ' if operand.startswith('r') else 'PT'
#
#         if capData[operand] != badVal:
#           if operand == 'r0':
#             lst.extend(getRegNum(regMap, reg) for reg in getVecRegisters(vectors, capData))
#           elif operand == 'r8':
#             lst.extend(getRegNum(regMap, reg) for reg in getAddrVecRegisters(vectors, capData))
#           elif operand in ('CC', 'X'):
#             lst.append('CC')
#           else:
#             lst.append(getRegNum(regMap, capData[operand]))
#
#       if 'c20' in capData or 'c39' in capData:
#         instruct['const'] = 1
#
#       # Find Read-After-Write dependencies
#       for src_reg in [s for s in src if s in writes]:
#         # Memory operations get delayed access to registers but not to the predicate
#         regLatency = 0 if src_reg == instruct['predReg'] else instruct['rlat']
#         # the parent should be the most recently added dest op to the stack
#         for parent in writes[src_reg]:
#           # add this instruction as a child of the parent
#           # set the edge to the total latency of reg source availability
#           # print(f"R {parent['inst']}\n\t\t{instruct['inst']}")
#           latency = 13 if re.match(r'^P\d', src_reg) else parent['lat']
#           parent.setdefault('children', []).append((instruct, latency - regLatency))
#           instruct['parents'] = instruct.get('parents', 0) + 1
#
#           # if the destination was conditionally executed, we also need to keep going back till it wasn't
#           if not parent.get('pred'):
#             break
#
#       # Find Write-After-Read dependencies
#       for dest_reg in [d for d in dest if d in reads]:
#         # Flag this instruction as dependent to any previous read
#         for reader in reads[dest_reg]:
#           # no need to stall for these types of dependencies
#           # print(f"W {reader['inst']} \t\t\t {instruct['inst']}")
#           reader.setdefault('children', []).append((instruct, 0))
#           instruct['parents'] = instruct.get('parents', 0) + 1
#         # Once dependence is marked we can clear out the read list (unless this write was conditional).
#         # The assumption here is that you would never want to write out a register without
#         # subsequently reading it in some way prior to writing it again.
#         if not instruct.get('pred'):
#           del reads[dest_reg]
#
#       # Enforce instruction ordering where requested
#       if instruct.get('order'):
#         if orderedParent and instruct['order'] > orderedParent['order']:
#           orderedParent.setdefault('children', []).append((instruct, 0))
#           instruct['parents'] = instruct.get('parents', 0) + 1
#         orderedParent = instruct
#       elif orderedParent:
#         orderedParent = None
#
#       # For a dest reg, push it onto the write stack
#       for d in dest:
#         writes.setdefault(d, []).insert(0, instruct)
#
#       # For a src reg, push it into the read list
#       for s in src:
#         reads.setdefault(s, []).append(instruct)
#
#       # if this instruction has no dependencies it's ready to go
#       if 'parents' not in instruct:
#         ready.append(instruct)
#
#       match = True
#       break
#     if not match:
#       raise Exception(f"Unable to recognize instruction at block: {blockNum} line: {lineNum}: {instruct['inst']}")
#
#   writes.clear()
#   reads.clear()
#
#   if ready:
#     # update dependent counts for sorting heuristic
#     readyParent = {'children': [(x, 1) for x in ready], 'inst': "root"}
#
#     countUniqueDescendants(readyParent, {})
#     updateDepCounts(readyParent, {})
#
#     # sort the initial ready list
#     ready.sort(key=lambda a: (a['first'], -a['deps'], a['dualCnt'], a['lineNum']))
#
#     if debug:
#       print("0: Initial Ready List State:\n\tf,ext,stl,mix,dep,lin, inst")
#       for instr in ready:
#         print("\t{}, {:3s}, {:3s}, {:3s}, {:3s}, {:3s}, {:3s}, {}".format(
#           instr['first'], instr.get('exeTime'), instr.get('stall'), instr.get('dualCnt'),
#           instr.get('mix'), instr.get('deps'), instr['lineNum'], instr['inst']))
#
#   # Process the ready list, adding new instructions to the list as we go.
#   clock = 0
#   while ready:
#     instruct = ready.pop(0)
#     stall = instruct.get('stall', 0)
#
#     # apply the stall to the previous instruction
#     if schedule and stall < 16:
#       prev = schedule[-1]
#       if prev.get('force_stall', 0) > stall:
#         stall = prev['force_stall']
#
#       # if stall is greater than 4 then also yield
#       # the yield flag is required to get stall counts 12-15 working correctly.
#       if stall > 4:
#         prev['ctrl'] &= 0x1ffe0
#       else:
#         prev['ctrl'] &= 0x1fff0
#       prev['ctrl'] |= stall
#       clock += stall
#     else:
#       instruct['ctrl'] &= 0x1fff0
#       instruct['ctrl'] |= 1
#       clock += 1
#
#     if debug:
#       print(f"{clock}: {instruct['inst']}")
#
#     # add a new instruction to the schedule
#     schedule.append(instruct)
#
#     # update each child with a new earliest execution time
#     if 'children' in instruct:
#       for child, latency in instruct['children']:
#         earliest = clock + latency
#         if child.get('exeTime', 0) < earliest:
#           child['exeTime'] = earliest
#
#         if debug:
#           print(f"\t\t{child['exeTime']},{child.get('parents', 0)} {child['inst']}")
#
#         # decrement parent count and add to ready queue if none remaining.
#         child['parents'] -= 1
#         if child['parents'] < 1:
#           ready.append(child)
#       del instruct['children']
#
#     # update stall and mix values in the ready queue on each iteration
#     for ready_instr in ready:
#       stall = ready_instr.get('exeTime', 0) - clock
#       stall = max(stall, 1)
#
#       # if using the same compute resource as the prior instruction then limit the throughput
#       if ready_instr['class'] == instruct['class']:
#         if stall < ready_instr['tput']:
#           stall = ready_instr['tput']
#       # dual issue with a simple instruction (tput <= 2)
#       # can't dual issue two instructions that both load a constant
#       elif (ready_instr.get('dual') and not instruct.get('dual') and instruct['tput'] <= 2 and
#             not instruct.get('nodual') and stall == 1 and ready_instr.get('exeTime', 0) <= clock and
#             not (ready_instr.get('const') and instruct.get('const'))):
#         stall = 0
#       ready_instr['stall'] = stall
#
#       # add an instruction class mixing heuristic that catches anything not handled by the stall
#       ready_instr['mix'] = 0 if ready_instr['class'] == instruct['class'] else 1
#       if ready_instr['mix'] and ready_instr['op'] == 'R2P':
#         ready_instr['mix'] = 2
#
#     # sort the ready list by stall time, mixing heuristic, dependencies and line number
#     ready.sort(key=lambda x: (x['first'], x['stall'], x['dualCnt'], -x['mix'], -x['deps'], x['lineNum']))
#
#     if debug:
#       print("\tf,ext,stl,duc,mix,dep,lin, inst")
#       for instr in ready:
#         print("\t{}, {:3s}, {:3s}, {:3s}, {:3s}, {:3s}, {:3s}, {}".format(
#           instr['first'], instr.get('exeTime'), instr.get('stall'), instr.get('dualCnt'),
#           instr.get('mix'), instr.get('deps'), instr['lineNum'], instr['inst']))
#
#     for ready_instr in ready:
#       if ready_instr.get('dualCnt') and ready_instr['stall'] == 1:
#         ready_instr['dualCnt'] = 0
#
#   out = ''
#   # out += "\n".join(comments)
#   for instr in schedule:
#     out += ''.join([printCtrl(instr['ctrl']), instr.get('space', ''), instr['inst'], instr.get('comment', ''), "\n"])
#   return out
#
# def countUniqueDescendants(node, edges):
#   """
#   Traverse the graph and count total descendants per node.
#   Only count unique nodes (by lineNum).
#   """
#   # print(f"P:{node['inst']}")
#
#   node_lineNum = node['lineNum']
#   children = node.get('children')
#
#   if children:
#     # Initialize the 'deps' set to store unique descendants
#     node_deps = node.get('deps', set())
#
#     # Process non-WaR (Write-after-Read) dependencies
#     for child in [c for c in children if c[1]]:  # c[1] is True
#       child_node = child[0]
#       edge_key = f"{node_lineNum}^{child_node['lineNum']}"
#       if edges.get(edge_key, 0):
#         edges[edge_key] += 1
#         continue
#       else:
#         edges[edge_key] = 1
#
#       descendants = countUniqueDescendants(child_node, edges)
#       node_deps.update(descendants)
#
#     # Process WaR dependencies
#     for child in [c for c in children if not c[1]]:  # c[1] is False
#       child_node = child[0]
#       edge_key = f"{node_lineNum}^{child_node['lineNum']}"
#       if edges.get(edge_key, 0):
#         edges[edge_key] += 1
#         continue
#       else:
#         edges[edge_key] = 1
#
#       # Traverse without collecting dependencies
#       countUniqueDescendants(child_node, edges)
#
#     # Update the node's 'deps' with collected descendants
#     node['deps'] = node_deps
#   else:
#     # Leaf node: return its own lineNum
#     return [node_lineNum]
#
#   # Return a list of the node's lineNum and its unique descendants
#   return [node_lineNum] + list(node['deps'])
#
#
# def updateDepCounts(node, edges):
#   """
#   Convert the 'deps' set to a count for easier sorting.
#   """
#   # print(f"{node['inst']}")
#
#   node_lineNum = node['lineNum']
#   children = node.get('children')
#
#   if children:
#     for child in children:
#       child_node = child[0]
#       edge_key = f"{node_lineNum}^{child_node['lineNum']}"
#       if edges.get(edge_key, 0):
#         edges[edge_key] += 1
#         continue
#       else:
#         edges[edge_key] = 1
#
#       updateDepCounts(child_node, edges)
#
#   # Convert 'deps' from set to count
#   if isinstance(node.get('deps'), set):
#     node['deps'] = len(node['deps'])
#   else:
#     node['deps'] = node.get('deps', 0)
#
#
# def registerHealth(reuseHistory, reuseFlags, capData, instAddr, inst, nowarn):
#   """
#   Detect register bank conflicts and calculate reuse stats.
#   """
#   banks = [None] * 4  # Initialize banks for modulo-4 bank IDs
#   conflicts = []
#
#   for slot in ['r8', 'r20', 'r39']:
#     r = capData.get(slot)
#     if not r:
#       continue
#     if r == 'RZ':
#       continue
#
#     slotHist = reuseHistory.setdefault(slot, {})
#
#     reuseHistory['total'] = reuseHistory.get('total', 0) + 1
#
#     # If this register is in active reuse, ignore for bank conflict checking
#     if r in slotHist:
#       reuseHistory['reuse'] = reuseHistory.get('reuse', 0) + 1
#     else:
#       # Extract number from reg and take modulo-4 value (bank ID)
#       reg_num = int(r[1:])  # Assuming 'r' is in format 'R<number>'
#       bank = reg_num & 3
#
#       # Check for conflict
#       if banks[bank] and banks[bank] != r:
#         if not conflicts:
#           conflicts.append(banks[bank])
#         conflicts.append(r)
#         reuseHistory['conflicts'] = reuseHistory.get('conflicts', 0) + 1
#       banks[bank] = r
#
#     # Update the history
#     if reuseFlags & reuseSlots.get(slot, 0):
#       slotHist[r] = 1
#     else:
#       slotHist.pop(r, None)
#
#   if inst and conflicts and not nowarn:
#     print(f"CONFLICT at 0x{instAddr:04x} ({', '.join(conflicts)}): {inst}")
#
#   return len(conflicts)
