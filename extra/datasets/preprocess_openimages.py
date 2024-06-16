import numpy as np
import copy
import random
import warnings
from PIL import Image, ImageOps
from typing import Tuple, Dict, Any
from extra.datasets.openimages import MLPERF_CLASSES
from tinygrad import Tensor

def preprocess_target(target:Dict[str, Any], anchors:np.ndarray, mirror:bool, img_width=800):
  boxes = mirror_boxes(target['boxes'], img_width) if mirror else target['boxes']
  matched_idxs = compute_matched_idxs(boxes, anchors)
  is_fg = matched_idxs >= 0
  is_valid = matched_idxs != -2

  fg_idxs = matched_idxs[is_fg]
  gt_classes = np.full(matched_idxs.shape, -1, dtype=np.float32)
  gt_classes[is_fg] = target['labels'][fg_idxs]

  masked_anchors = anchors * is_fg[:, None]
  matched_gt_boxes = np.zeros(anchors.shape, dtype=np.float32)
  matched_gt_boxes[is_fg] = boxes[fg_idxs]
  weights = np.array([1.0, 1.0, 1.0], dtype=np.float32)
  bbox_regr = encode_boxes(matched_gt_boxes, masked_anchors, weights)
  return np.concatenate([gt_classes[:, None], bbox_regr, is_fg.astype(np.float32)[:, None], is_valid.astype(np.float32)[:, None]], axis=1)

def preprocess_image(fn:str, mirror:bool, img_size:Tuple[int, int]=(800, 800)):
  if fn:
    img = Image.open(fn)
    img = img.convert('RGB') if img.mode != "RGB" else img
  else:
    img = None

  if img:
    img = img.resize(img_size, Image.BILINEAR)
    if mirror:
      img = ImageOps.mirror(img)
  else:
    img = np.tile(np.array([[[123.68, 116.78, 103.94]]], dtype=np.uint8), (*img_size, 1)) # pad data with training mean
  return img

def compute_matched_idxs(boxes:np.ndarray, anchors:np.ndarray):
  match_quality_matrix = box_iou(boxes, anchors)
  return match(match_quality_matrix)

def match(match_quality_matrix:np.ndarray, high:float=0.5, low:float=0.5, allow_low_quality_matches=True):
  assert match_quality_matrix.size > 0, "empty targets or proposals not supported during training"
  matched_vals, matches = np.max(match_quality_matrix, axis=0), np.argmax(match_quality_matrix, axis=0)
  all_matches = copy.copy(matches) if allow_low_quality_matches else None
  below_low_threshold = matched_vals < low
  between_thresholds = (matched_vals >= low) & (matched_vals < high)
  matches[below_low_threshold] = -1
  matches[between_thresholds] = -2

  if allow_low_quality_matches:
    highest_quality_foreach_gt = match_quality_matrix.max(axis=1)
    pred_inds_to_update = np.where(match_quality_matrix == highest_quality_foreach_gt[:, None])[1]
    matches[pred_inds_to_update] = all_matches[pred_inds_to_update]

  return matches

def box_iou(x:np.ndarray, y:np.ndarray) -> np.ndarray:
  area1 = box_area(x)
  area2 = box_area(y)
  lt = np.maximum(x[:, None, :2], y[:, :2])  # [N,M,2]
  rb = np.minimum(x[:, None, 2:], y[:, 2:])  # [N,M,2]
  wh = np.clip(_upcast(rb - lt), 0, None)  # [N,M,2]
  inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]
  union = area1[:, None] + area2 - inter
  iou = inter / union
  return iou

def box_area(x:np.ndarray) -> np.ndarray:
  x = _upcast(x)
  return (x[:, 2] - x[:, 0]) * (x[:, 3] - x[:, 1])

def _upcast(t:np.ndarray) -> np.ndarray:
  return t
#   # TODO: protect from numerical overflows in multiplications by upcasting to the equivalent higher type
#   if dtypes.is_float(t.dtype):
#     return t if t.dtype in (dtypes.float32, dtypes.float64) else t.float()
#   else:
#     return t if t.dtype in (dtypes.int32, dtypes.int64) else t.int()

def encode_boxes(ref_boxes:np.ndarray, gt_boxes:np.ndarray, weights:np.ndarray) -> np.ndarray:
  gt_lengths = ref_boxes[:, 2:] - ref_boxes[:, :2]
  gt_centers = ref_boxes[:, :2] + 0.5 * gt_lengths
  pred_lengths = gt_boxes[:, 2:] - gt_boxes[:, :2]
  pred_centers = gt_boxes[:, :2] + 0.5 * pred_lengths
  mask = pred_lengths != 0
  with warnings.catch_warnings():
    warnings.filterwarnings('ignore')
    targets_centers = np.where(mask, weights[:2] * (gt_centers - pred_centers) / pred_lengths, 0)
    targets_lengths = np.where(mask, weights[2:] * np.log(gt_lengths / pred_lengths), 0)
  ret = np.concatenate([targets_centers, targets_lengths], axis=1)
  return ret

def mirror_boxes(boxes:np.ndarray, img_width):
  ret = np.empty(boxes.shape)
  ret[:, (0, 2)] = -boxes[:, (2, 0)] + img_width
  ret[:, (1, 3)] = boxes[:, (1, 3)]
  return ret
