from tinygrad.dtype import dtypes, PtrDType
from tinygrad.renderer.sass import SASSRenderer
from tinygrad.ops import UOps, UOp
from tinygrad.codegen.uopgraph import linearize_uop, full_graph_rewrite

def vec(srcs): return UOp(UOps.VECTORIZE, dtypes.int.vec(4), srcs)
def gep(v): return [v.gep(i) for i in range(4)]
def dot(va, vb): return [va[i] * vb[i] for i in range(4)]

loads = [UOp.load(UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int.vec(4)), arg=1+i), UOp.const(dtypes.int, i), dtype=dtypes.int.vec(4)) for i in range(3)]
ast = gep(loads[0])
for i in range(1, len(loads)):
  ast = dot(ast, gep(loads[i]))
ast = UOp.store(UOp(UOps.DEFINE_GLOBAL, PtrDType(dtypes.int.vec(4)), arg=0), UOp.const(dtypes.int, 0), vec(ast)).sink()

sass = SASSRenderer("sm_89")
sass.render("test", linearize_uop(full_graph_rewrite(ast, opts=sass)) )