from typing import List
from tinygrad.dtype import dtypes
from tinygrad.ops import PatternMatcher, UOp
from tinygrad.renderer import Renderer
from tinygrad.renderer.cstyle import CUDARenderer

sass_matcher = PatternMatcher([])

class SASSRenderer(Renderer):
  device = "CUDA"
  suffix = "SASS"
  global_max, local_max, shared_max = CUDARenderer.global_max, CUDARenderer.local_max, CUDARenderer.shared_max
  tensor_cores = [tc for tc in CUDARenderer.tensor_cores if tc.dtype_in == dtypes.half]
  code_for_op = {}
  extra_matcher = sass_matcher
  def __init__(self, arch:str, device="CUDA"): self.device, self.tensor_cores = device, SASSRenderer.tensor_cores if int(arch[3:]) >= 80 else []

  def render(self, name:str, uops:List[UOp]) -> str: return ""
