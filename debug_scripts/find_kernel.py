import json
from tqdm import tqdm
from tinygrad import Device
from tinygrad.helpers import getenv
from extra.optimization.helpers import load_worlds, ast_str_to_lin
from tinygrad.runtime.ops_cuda import CUDACompiler
from tinygrad.renderer.sass import SASSRenderer
from tinygrad.ops import UOps, UPat, _match

if __name__ == "__main__":
  ast_strs = load_worlds(filter_reduce=False, filter_novariable=True)
  if cache := getenv("CACHE", ""):
    with open(cache) as f: cache_nums = json.load(f)

  # no bfloat16 for ptx at the moment
  ast_strs = [x for x in ast_strs if "dtypes.bfloat16" not in x]
  dev = Device["CUDA"]
  sass = SASSRenderer(dev.arch)

  find = UPat(UOps.VECTORIZE)

  blacklist = []
  results = []
  max_size = getenv("MAX_SIZE", 1000)
  start, end = getenv("START", 0), getenv("END", len(ast_strs))
  for num,ast in tqdm(enumerate(ast_strs), total=min(end, len(ast_strs))):
    if num in blacklist or not (start <= num < end) or (cache and num not in cache_nums): continue

    # cuda compile
    dev.compiler = CUDACompiler(dev.arch)
    lin = ast_str_to_lin(ast, opts=sass)
    lin.hand_coded_optimizations()
    lin.linearize()

    skip = False
    if len(lin.uops) <= max_size and any(_match(u, find, {}) for u in lin.uops):
      results.append((num, len(lin.uops)))

  if write_cache := getenv("WRITE_CACHE", ""):
    with open(write_cache, "w") as f: json.dump([r[0] for r in results], f)
  results.sort(key=lambda x: x[1])
  for num,sz in results:
    print(f"{num:>4}: {sz} nodes")
  if not results:
    print("No matches")