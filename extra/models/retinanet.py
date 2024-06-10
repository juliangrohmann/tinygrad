import copy
import math
import numpy as np
from itertools import product
from typing import List, Dict
from tinygrad import Tensor
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import TinyJit
from tinygrad.helpers import flatten, get_child
import tinygrad.nn as nn
from extra.models.resnet import ResNet
from extra.datasets.preprocess_openimages import preprocess_image, preprocess_target
from tqdm import tqdm

# allow monkeypatching in layer implementations
Conv2d = nn.Conv2d
Conv2dCls = nn.Conv2d
Conv2dFPN = nn.Conv2d

def sigmoid_focal_loss(inputs, targets, alpha=0.25, gamma=2, reduction="none"):
  p = inputs.sigmoid()
  ce_loss = binary_crossentropy_logits(inputs, targets)
  p_t = p * targets + (1 - p) * (1 - targets)
  loss = ce_loss * ((1 - p_t) ** gamma)

  if alpha >= 0:
    alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
    loss = alpha_t * loss

  if reduction == "mean":
    loss = loss.mean()
  elif reduction == "sum":
    loss = loss.sum()
  elif reduction is not None:
    raise NotImplementedError
  return loss

def binary_crossentropy_logits(x, y):
  return x.maximum(0) - y * x + (1 + x.abs().neg().exp()).log()

def l1_loss(inputs, targets, reduction="sum"):
  loss = (inputs - targets).abs()
  if reduction == "sum":
    loss = loss.sum()
  elif reduction is not None:
    raise NotImplementedError
  return loss

def nms(boxes, scores, thresh=0.5):
  x1, y1, x2, y2 = np.rollaxis(boxes, 1)
  areas = (x2 - x1 + 1) * (y2 - y1 + 1)
  to_process, keep = scores.argsort()[::-1], []
  while to_process.size > 0:
    cur, to_process = to_process[0], to_process[1:]
    keep.append(cur)
    inter_x1 = np.maximum(x1[cur], x1[to_process])
    inter_y1 = np.maximum(y1[cur], y1[to_process])
    inter_x2 = np.minimum(x2[cur], x2[to_process])
    inter_y2 = np.minimum(y2[cur], y2[to_process])
    inter_area = np.maximum(0, inter_x2 - inter_x1 + 1) * np.maximum(0, inter_y2 - inter_y1 + 1)
    iou = inter_area / (areas[cur] + areas[to_process] - inter_area)
    to_process = to_process[np.where(iou <= thresh)[0]]
  return keep

def encode_boxes(ref_boxes: Tensor, gt_boxes: Tensor, weights: Tensor) -> Tensor:
  gt_lengths = ref_boxes[:, 2:] - ref_boxes[:, :2]
  gt_centers = ref_boxes[:, :2] + 0.5 * gt_lengths
  pred_lengths = gt_boxes[:, 2:] - gt_boxes[:, :2]
  pred_centers = gt_boxes[:, :2] + 0.5 * pred_lengths
  mask = gt_lengths != 0
  targets_centers = mask.where(weights[:2] * (gt_centers - pred_centers) / pred_lengths, 0)
  targets_lengths = mask.where(weights[2:] * (gt_lengths / pred_lengths).log(), 0)
  ret = targets_centers.cat(targets_lengths, dim=1)
  return ret

def decode_bbox(offsets, anchors):
  dx, dy, dw, dh = np.rollaxis(offsets, 1)
  widths, heights = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
  cx, cy = anchors[:, 0] + 0.5 * widths, anchors[:, 1] + 0.5 * heights
  pred_cx, pred_cy = dx * widths + cx, dy * heights + cy
  pred_w, pred_h = np.exp(dw) * widths, np.exp(dh) * heights
  pred_x1, pred_y1 = pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h
  pred_x2, pred_y2 = pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h
  return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1, dtype=np.float32)

def generate_anchors(input_size, grid_sizes, scales, aspect_ratios):
  assert len(scales) == len(aspect_ratios) == len(grid_sizes)
  anchors = []
  for s, ar, gs in zip(scales, aspect_ratios, grid_sizes):
    s, ar = np.array(s), np.array(ar)
    h_ratios = np.sqrt(ar)
    w_ratios = 1 / h_ratios
    ws = (w_ratios[:, None] * s[None, :]).reshape(-1)
    hs = (h_ratios[:, None] * s[None, :]).reshape(-1)
    base_anchors = (np.stack([-ws, -hs, ws, hs], axis=1) / 2).round()
    stride_h, stride_w = input_size[0] // gs[0], input_size[1] // gs[1]
    shifts_x, shifts_y = np.meshgrid(np.arange(gs[1]) * stride_w, np.arange(gs[0]) * stride_h)
    shifts_x = shifts_x.reshape(-1)
    shifts_y = shifts_y.reshape(-1)
    shifts = np.stack([shifts_x, shifts_y, shifts_x, shifts_y], axis=1, dtype=np.float32)
    anchors.append((shifts[:, None] + base_anchors[None, :]).reshape(-1, 4))
  return anchors

def compute_grid_sizes(input_size):
  return np.ceil(np.array(input_size)[None, :] / 2 ** np.arange(3, 8)[:, None])

def match(match_quality_matrix, high=0.5, low=0.5, allow_low_quality_matches=False):
  assert match_quality_matrix.numel() > 0, "empty targets or proposals not supported during training"
  matched_vals, matches = match_quality_matrix.max(axis=0), match_quality_matrix.argmax(axis=0)
  all_matches = copy.copy(matches) if allow_low_quality_matches else None
  below_low_threshold = matched_vals < low
  matches = below_low_threshold.where(-1, matches)

  if allow_low_quality_matches:
    between_thresholds = (matched_vals >= low) * (matched_vals < high)
    matches = between_thresholds.where(-2, matches)
    pred_inds_to_update = match_quality_matrix.argmax(axis=1).tolist()
    for i in pred_inds_to_update:
      matches[i] = all_matches[i]

  return matches

def box_area(x: Tensor) -> Tensor:
  x = _upcast(x)
  return (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])

def _torch_max(x: Tensor, y: Tensor):
  return _torch_binary_map(x, y, "max")

def _torch_min(x: Tensor, y: Tensor):
  return _torch_binary_map(x, y, "min")

def _torch_binary_map(x: Tensor, y: Tensor, op): # (3, 1, 2), (3, 2)
  x = x.expand((x.shape[0], y.shape[0], x.shape[2]))
  y = y.unsqueeze(0).expand(x.shape)

  if op == "max":
    return (x > y).where(x, y)
  elif op == "min":
    return (x < y).where(x, y)
  raise NotImplementedError

def _clamp(x: Tensor, x_min, x_max) -> Tensor:
  return Tensor(np.clip(x.numpy(), x_min, x_max), device=x.device, dtype=x.dtype)

def _pad_dims(dims, n):
  if len(dims) >= n: return dims
  return (1,) * (n - len(dims)) + dims

def _upcast(t: Tensor) -> Tensor:
  return t
#   # protects from numerical overflows in multiplications by upcasting to the equivalent higher type
#   if dtypes.is_float(t.dtype):
#     return t if t.dtype in (dtypes.float32, dtypes.float64) else t.float()
#   else:
#     return t if t.dtype in (dtypes.int32, dtypes.int64) else t.int()

def _sum(x: List[Tensor]) -> Tensor:
  res = x[0]
  for i in x[1:]:
    res = res + i
  return res

class RetinaNet:
  def __init__(self, backbone: ResNet, num_classes=264, num_anchors=9, scales=None, aspect_ratios=None):
    assert isinstance(backbone, ResNet)
    scales = tuple((i, int(i*2**(1/3)), int(i*2**(2/3))) for i in 2**np.arange(5, 10)) if scales is None else scales
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(scales) if aspect_ratios is None else aspect_ratios
    self.num_anchors, self.num_classes = num_anchors, num_classes
    assert len(scales) == len(aspect_ratios) and all(self.num_anchors == len(s) * len(ar) for s, ar in zip(scales, aspect_ratios))

    self.backbone = ResNetFPN(backbone)
    self.head = RetinaHead(self.backbone.out_channels, num_anchors=num_anchors, num_classes=num_classes)
    self.anchor_gen = lambda input_size: generate_anchors(input_size, compute_grid_sizes(input_size), scales, aspect_ratios)

  def __call__(self, x):
    return self.forward(x)
  def forward(self, x):
    return self.head(self.backbone(x))

  def compute_loss(self, targets:List[Dict[str, np.ndarray]], head_outputs:Dict[str, Tensor], prep_dat:Tensor=None, input_size=(800, 800)) -> Dict[str, Tensor]:
    if prep_dat is None:
      dat = []
      anchors = np.concatenate(self.anchor_gen(input_size))
      for t in targets:
        dat.append(preprocess_target(t, anchors)[None, ...])
      prep_dat = Tensor(np.concatenate(dat, axis=0), head_outputs['cls_logits'].device, head_outputs['cls_logits'].dtype)

    losses = self.head.compute_loss(targets, head_outputs, prep_dat)
    return losses['classification'] + losses['bbox_regression']

  def load_from_pretrained(self):
    model_urls = {
      (50, 1, 64): "https://download.pytorch.org/models/retinanet_resnet50_fpn_coco-eeacb38b.pth",
      (50, 32, 4): "https://zenodo.org/record/6605272/files/retinanet_model_10.zip",
    }
    self.url = model_urls[(self.backbone.body.num, self.backbone.body.groups, self.backbone.body.base_width)]
    from torch.hub import load_state_dict_from_url
    state_dict = load_state_dict_from_url(self.url, progress=True, map_location='cpu')
    state_dict = state_dict['model'] if 'model' in state_dict.keys() else state_dict
    for k, v in state_dict.items():
      obj = get_child(self, k)
      dat = v.detach().numpy()
      assert obj.shape == dat.shape, (k, obj.shape, dat.shape)
      obj.assign(dat)

  # # predictions: (BS, (H1W1+...+HmWm)A, 4 + K)
  # def postprocess_detections(self, head_outputs, input_size=(800, 800), image_sizes=None, orig_image_sizes=None, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5):
  #   predictions = head_outputs['bbox_regression'].cat(head_outputs['cls_logits'].sigmoid(), dim=-1).numpy()
  #   anchors = self.anchor_gen(input_size)
  #   grid_sizes = compute_grid_sizes(input_size)
  #   split_idx = np.cumsum([int(self.num_anchors * sz[0] * sz[1]) for sz in grid_sizes[:-1]])
  #   detections = []
  #   for i, predictions_per_image in enumerate(predictions):
  #     h, w = input_size if image_sizes is None else image_sizes[i]
  #
  #     predictions_per_image = np.split(predictions_per_image, split_idx)
  #     offsets_per_image = [br[:, :4] for br in predictions_per_image]
  #     scores_per_image = [cl[:, 4:] for cl in predictions_per_image]
  #
  #     image_boxes, image_scores, image_labels = [], [], []
  #     for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
  #       # remove low scoring boxes
  #       scores_per_level = scores_per_level.flatten()
  #       keep_idxs = scores_per_level > score_thresh
  #       scores_per_level = scores_per_level[keep_idxs]
  #
  #       # keep topk
  #       topk_idxs = np.where(keep_idxs)[0]
  #       num_topk = min(len(topk_idxs), topk_candidates)
  #       sort_idxs = scores_per_level.argsort()[-num_topk:][::-1]
  #       topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]
  #
  #       # bbox coords from offsets
  #       anchor_idxs = topk_idxs // self.num_classes
  #       labels_per_level = topk_idxs % self.num_classes
  #       boxes_per_level = decode_bbox(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
  #       # clip to image size
  #       clipped_x = boxes_per_level[:, 0::2].clip(0, w)
  #       clipped_y = boxes_per_level[:, 1::2].clip(0, h)
  #       boxes_per_level = np.stack([clipped_x, clipped_y], axis=2).reshape(-1, 4)
  #
  #       image_boxes.append(boxes_per_level)
  #       image_scores.append(scores_per_level)
  #       image_labels.append(labels_per_level)
  #
  #     image_boxes = np.concatenate(image_boxes)
  #     image_scores = np.concatenate(image_scores)
  #     image_labels = np.concatenate(image_labels)
  #
  #     # nms for each class
  #     keep_mask = np.zeros_like(image_scores, dtype=bool)
  #     for class_id in np.unique(image_labels):
  #       curr_indices = np.where(image_labels == class_id)[0]
  #       curr_keep_indices = nms(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
  #       keep_mask[curr_indices[curr_keep_indices]] = True
  #     keep = np.where(keep_mask)[0]
  #     keep = keep[image_scores[keep].argsort()[::-1]]
  #
  #     # resize bboxes back to original size
  #     image_boxes = image_boxes[keep]
  #     if orig_image_sizes is not None:
  #       resized_x = image_boxes[:, 0::2] * orig_image_sizes[i][1] / w
  #       resized_y = image_boxes[:, 1::2] * orig_image_sizes[i][0] / h
  #       image_boxes = np.stack([resized_x, resized_y], axis=2).reshape(-1, 4)
  #     # xywh format
  #     image_boxes = np.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], axis=1)
  #
  #     detections.append({"boxes":image_boxes, "scores":image_scores[keep], "labels":image_labels[keep]})
  #     print(f"{image_boxes.shape=}")
  #     print(f"{image_scores.shape=}")
  #     print(f"{image_labels.shape=}")
  #   return detections

class ClassificationHead:
  def __init__(self, in_channels, num_anchors, num_classes):
    self.num_classes = num_classes
    self.conv = flatten([(Conv2d(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.cls_logits = Conv2dCls(in_channels, num_anchors * num_classes, kernel_size=3, padding=1)
  def __call__(self, x):
    out = [self.cls_logits(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, self.num_classes) for feat in x]
    return out[0].cat(*out[1:], dim=1)
  def compute_loss(self, targets:List[Dict[str, np.ndarray]], head_outputs:Dict[str, Tensor], prep_dat:Tensor) -> Dict[str, Tensor]:
    cls_logits = head_outputs['cls_logits']
    gt_classes, fg_mask = prep_dat[:, :, 0].one_hot(self.num_classes), prep_dat[:, :, -1:]
    diff = sigmoid_focal_loss(cls_logits, gt_classes, reduction=None).sum(axis=(1, 2))
    losses = diff / fg_mask.squeeze(dim=2).sum(axis=1).maximum(1)
    return losses.sum() / max(1, len(targets))

class RegressionHead:
  def __init__(self, in_channels, num_anchors):
    self.conv = flatten([(Conv2d(in_channels, in_channels, kernel_size=3, padding=1), lambda x: x.relu()) for _ in range(4)])
    self.bbox_reg = Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
  def __call__(self, x):
    out = [self.bbox_reg(feat.sequential(self.conv)).permute(0, 2, 3, 1).reshape(feat.shape[0], -1, 4) for feat in x]
    return out[0].cat(*out[1:], dim=1)
  def compute_loss(self, targets:List[Dict[str, np.ndarray]], head_outputs:Dict[str, Tensor], prep_dat:Tensor):
    target_regr, fg_mask = prep_dat[:, :, -5:-1], prep_dat[:, :, -1:]
    out_regr = head_outputs['bbox_regression'] * fg_mask
    diff = l1_loss(out_regr, target_regr, reduction=None).sum(axis=(1, 2))
    losses = diff / fg_mask.squeeze(dim=2).sum(axis=1).maximum(1)
    return losses.sum() / max(1, len(targets))

class RetinaHead:
  def __init__(self, in_channels, num_anchors, num_classes):
    self.classification_head = ClassificationHead(in_channels, num_anchors, num_classes)
    self.regression_head = RegressionHead(in_channels, num_anchors)
  def __call__(self, x):
    return {
      'cls_logits': self.classification_head(x),
      'bbox_regression': self.regression_head(x)
    }
  def compute_loss(self, targets, head_outputs, prep_dat):
    return {
      'classification': self.classification_head.compute_loss(targets, head_outputs, prep_dat),
      'bbox_regression': self.regression_head.compute_loss(targets, head_outputs, prep_dat),
    }

class ResNetFPN:
  def __init__(self, resnet, out_channels=256, returned_layers=[2, 3, 4]):
    self.out_channels = out_channels
    self.body = resnet
    in_channels_list = [(self.body.in_planes // 8) * 2 ** (i - 1) for i in returned_layers]
    self.fpn = FPN(in_channels_list, out_channels)

  def __call__(self, x):
    out = self.body.bn1(self.body.conv1(x)).relu()
    out = out.pad2d([1,1,1,1]).max_pool2d((3,3), 2)
    out = out.sequential(self.body.layer1)
    p3 = out.sequential(self.body.layer2)
    p4 = p3.sequential(self.body.layer3)
    p5 = p4.sequential(self.body.layer4)
    return self.fpn([p3, p4, p5])

class ExtraFPNBlock:
  def __init__(self, in_channels, out_channels):
    self.p6 = Conv2dFPN(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.p7 = Conv2dFPN(out_channels, out_channels, kernel_size=3, stride=2, padding=1)
    self.use_P5 = in_channels == out_channels

  def __call__(self, p, c):
    p5, c5 = p[-1], c[-1]
    x = p5 if self.use_P5 else c5
    p6 = self.p6(x)
    p7 = self.p7(p6.relu())
    p.extend([p6, p7])
    return p

class FPN:
  def __init__(self, in_channels_list, out_channels, extra_blocks=None):
    self.inner_blocks, self.layer_blocks = [], []
    for in_channels in in_channels_list:
      self.inner_blocks.append(Conv2dFPN(in_channels, out_channels, kernel_size=1))
      self.layer_blocks.append(Conv2dFPN(out_channels, out_channels, kernel_size=3, padding=1))
    self.extra_blocks = ExtraFPNBlock(256, 256) if extra_blocks is None else extra_blocks

  def __call__(self, x):
    last_inner = self.inner_blocks[-1](x[-1])
    results = [self.layer_blocks[-1](last_inner)]
    for idx in range(len(x) - 2, -1, -1):
      inner_lateral = self.inner_blocks[idx](x[idx])

      # upsample to inner_lateral's shape
      (ih, iw), (oh, ow), prefix = last_inner.shape[-2:], inner_lateral.shape[-2:], last_inner.shape[:-2]
      eh, ew = math.ceil(oh / ih), math.ceil(ow / iw)
      inner_top_down = last_inner.reshape(*prefix, ih, 1, iw, 1).expand(*prefix, ih, eh, iw, ew).reshape(*prefix, ih*eh, iw*ew)[:, :, :oh, :ow]

      last_inner = inner_lateral + inner_top_down
      results.insert(0, self.layer_blocks[idx](last_inner))
    if self.extra_blocks is not None:
      results = self.extra_blocks(results, x)
    return results
