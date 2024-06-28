import math
from typing import Union, Tuple

from tinygrad import Tensor, nn, dtypes
from tinygrad.multi import MultiLazyBuffer
from tinygrad.helpers import prod, argfix, getenv
from examples.hlb_cifar10 import UnsyncedBatchNorm

# rejection sampling truncated randn
def rand_truncn(*shape, dtype=None, truncstds=2, **kwargs) -> Tensor:
  CNT=8
  x = Tensor.randn(*(*shape, CNT), dtype=dtype, **kwargs)
  ctr = Tensor.arange(CNT).reshape((1,) * len(x.shape[:-1]) + (CNT,)).expand(x.shape)
  take = (x.abs() <= truncstds).where(ctr, CNT).min(axis=-1, keepdim=True)  # set to 0 if no good samples
  return (ctr == take).where(x, 0).sum(axis=-1)

# https://github.com/keras-team/keras/blob/v2.15.0/keras/initializers/initializers.py#L1026-L1065
def he_normal(*shape, a: float = 0.00, **kwargs) -> Tensor:
  std = math.sqrt(2.0 / (1 + a ** 2)) / math.sqrt(prod(argfix(*shape)[1:])) / 0.87962566103423978
  return std * rand_truncn(*shape, **kwargs)

class Conv2dHeNormal(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.in_channels, self.out_channels = in_channels, out_channels  # for testing
    self.weight = he_normal(out_channels, in_channels//groups, *self.kernel_size, a=0.0, dtype=dtypes.float32)
    if bias: self.bias = self.bias.cast(dtypes.float32)
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True):
    super().__init__(in_features, out_features, bias=bias)
    self.weight = Tensor.normal((out_features, in_features), mean=0.0, std=0.01, dtype=dtypes.float32)
    if bias: self.bias = Tensor.zeros(out_features, dtype=dtypes.float32)
  def __call__(self, x:Tensor):
    return x.linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)

class Conv2dRetina(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.normal(out_channels, in_channels//groups, *self.kernel_size, std=0.01)
    self.bias = Tensor.zeros(out_channels) if bias else None
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Conv2dClsRetina(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.normal(out_channels, in_channels//groups, *self.kernel_size, std=0.01)
    self.bias = Tensor.full(out_channels, -math.log((1 - 0.01) / 0.01)) if bias else None
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class Conv2dFPN(nn.Conv2d):
  def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
    super().__init__(in_channels, out_channels, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
    self.weight = Tensor.kaiming_uniform(out_channels, in_channels//groups, *self.kernel_size, a=1)
    self.bias = Tensor.zeros(out_channels) if bias else None
  def __call__(self, x: Tensor):
    return x.conv2d(self.weight.cast(dtypes.default_float), self.bias.cast(dtypes.default_float) if self.bias is not None else None,
                    padding=self.padding, stride=self.stride, dilation=self.dilation, groups=self.groups)

class FrozenBatchNorm(nn.BatchNorm2d):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1):
    super().__init__(sz, eps=eps, affine=affine, track_running_stats=track_running_stats, momentum=momentum)
  def __call__(self, x:Tensor):
    batch_invstd = self.running_var.reshape(1, -1, 1, 1).expand(x.shape).add(self.eps).rsqrt()
    return x.batchnorm(self.weight, self.bias, self.running_mean, batch_invstd)

class FrozenUnsyncedBatchNorm(UnsyncedBatchNorm):
  def __init__(self, sz:int, eps=1e-5, affine=True, track_running_stats=True, momentum=0.1, num_devices=None):
    super().__init__(sz, eps=eps, affine=affine, track_running_stats=track_running_stats, momentum=momentum, num_devices=num_devices)
    self.scale, self.bias_term = None, None

  def __call__(self, x:Tensor):
    if isinstance(x.lazydata, MultiLazyBuffer): assert x.lazydata.axis is None or x.lazydata.axis == 0 and len(x.lazydata.lbs) == self.num_devices
    # scale = weight * invstd
    # bias_term = bias - mean * scale
    # ret = x * scale + bias_term
    xr = x.reshape(self.num_devices, -1, *x.shape[1:]).cast(dtypes.float32)
    from tqdm import tqdm
    tqdm.write(f"{x.shape=}")
    tqdm.write(f"{xr.shape=}")
    tqdm.write(f"{self.running_mean.shape=}")
    tqdm.write(f"{self.running_var.shape=}")
    if self.scale is None or self.bias_term is None:
      batch_mean, batch_invstd = self.calc_stats(xr)
      weight = self.weight.reshape(1, -1).expand((self.num_devices, -1))
      bias = self.bias.reshape(1, -1).expand((self.num_devices, -1))
      tqdm.write(f"{batch_mean.shape=}")
      tqdm.write(f"{batch_invstd.shape=}")
      tqdm.write(f"{self.weight.shape=}")
      tqdm.write(f"{self.bias.shape=}")
      self.scale = weight.reshape(xr.shape) * batch_invstd.reshape(xr.shape)
      self.bias_term = bias.reshape(xr.shape) - self.running_mean.reshape(xr.shape) * self.scale
    return xr * self.scale + self.bias_term

    # batch_mean, batch_invstd = self.calc_stats(xr)
    # ret = xr.batchnorm(
    #   self.weight.reshape(1, -1).expand((self.num_devices, -1)),
    #   self.bias.reshape(1, -1).expand((self.num_devices, -1)),
    #   batch_mean, batch_invstd, axis=(0, 2))
    # return ret.reshape(x.shape).cast(x.dtype)

class LinearBert(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, std=0.02):
    self.weight = std * rand_truncn(out_features, in_features, dtype=dtypes.float32)
    self.bias = Tensor.zeros(out_features, dtype=dtypes.float32) if bias else None
  
  def __call__(self, x:Tensor):
    return x.cast(dtypes.default_float).linear(self.weight.cast(dtypes.default_float).transpose(), self.bias.cast(dtypes.default_float) if self.bias is not None else None)
class EmbeddingBert(nn.Embedding):
  def __init__(self, vocab_size:int, embed_size:int, std=0.02):
    self.vocab_sz, self.embed_sz = vocab_size, embed_size
    self.weight = std * rand_truncn(vocab_size, embed_size, dtype=dtypes.float32)

  def __call__(self, idx:Tensor) -> Tensor:
    if idx.numel() == 0: return Tensor.empty(idx.shape+(self.embed_sz,), dtype=self.weight.dtype, device=self.weight.device)
    arange_shp, weight_shp, big_shp = (1, 1, self.vocab_sz, 1), (1, 1, self.vocab_sz, self.embed_sz), idx.shape+(self.vocab_sz, self.embed_sz,)
    if not hasattr(self, 'arange'): self.arange = Tensor.arange(self.vocab_sz, requires_grad=False, device=self.weight.device).reshape(arange_shp)
    arange, idx, vals = self.arange.expand(big_shp), idx.reshape(idx.shape+(1, 1,)).expand(big_shp), self.weight.cast(dtypes.default_float).reshape(weight_shp).expand(big_shp)
    return (arange == idx).mul(vals).sum(2, acc_dtype=vals.dtype)

class LayerNormBert:
  def __init__(self, normalized_shape:Union[int, Tuple[int, ...]], eps:float=1e-12, elementwise_affine:bool=True):
    self.normalized_shape = (normalized_shape,) if isinstance(normalized_shape, int) else tuple(normalized_shape)
    self.axis, self.eps, self.elementwise_affine = tuple(-1-i for i in range(len(self.normalized_shape))), eps, elementwise_affine
    self.weight, self.bias = (Tensor.ones(*self.normalized_shape, dtype=dtypes.float32), Tensor.zeros(*self.normalized_shape, dtype=dtypes.float32)) if elementwise_affine else (None, None)

  def __call__(self, x:Tensor):
    assert self.normalized_shape == x.shape[-len(self.normalized_shape):], f"last dimensions of {x.shape} must match {self.normalized_shape}"
    xn = x.cast(dtypes.float32).layernorm(eps=self.eps, axis=self.axis).cast(x.dtype)
    if not self.elementwise_affine: return xn
    return (xn * self.weight.cast(dtypes.default_float) + self.bias.cast(dtypes.default_float))
