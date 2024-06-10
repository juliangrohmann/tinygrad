from multiprocessing import Process, Queue, shared_memory, cpu_count
import numpy as np

from tinygrad import  Tensor
from tinygrad.dtype import dtypes
from extra.datasets.openimages import MLPERF_CLASSES
from extra.models.retinanet import compute_grid_sizes

class Postprocessor:
  def __init__(self, anchors, num_classes=None, max_size=128, buf_name="retina_post"):
    self.anchors, self.max_size, self.buf_name = anchors, max_size, buf_name
    self.num_classes = num_classes if num_classes else len(MLPERF_CLASSES)
    self.q_in, self.q_out = Queue(maxsize=max_size), Queue()
    self.pred_buffer, self.predictions, self.shm_anchors, self.shm_detections, self.procs = [], None, None, None, None
    self.counter = 0
    self._shutdown = False

  def __del__(self):
    self.shutdown()

  def add(self, prediction, orig_size):
    if self.predictions is not None and self.counter < self.max_size:
      self.enqueue(prediction, orig_size)
    else:
      self.pred_buffer.append((prediction, orig_size))

  def launch(self):
    try:
      self.predictions = Tensor.empty((self.max_size, self.anchors.shape[0], self.num_classes + 4),
                                      dtype=dtypes.float32, device=f"disk:{self.buf_name}_in")
      det_shp = (self.max_size, -1) # TODO: Shape
      self.shm_detections = shared_memory.SharedMemory(create=True, size=np.empty(det_shp).nbytes)
      self.shm_anchors = shared_memory.SharedMemory(create=True, size=self.anchors.nbytes)
      self.buf_detections = np.ndarray(det_shp, dtype=np.float32, buffer=self.shm_anchors.buf)
      buf_anchors = np.ndarray(self.anchors.shape, dtype=np.float32, buffer=self.shm_anchors.buf)
      buf_anchors[:] = self.anchors[:]

      self.procs = []
      for _ in range(cpu_count()):
        p = Process(target=pp_process,
                    args=(self.q_in, self.q_out, self.predictions, self.shm_anchors, self.shm_detections, self.max_size))
        p.daemon = True
        p.start()
        self.procs.append(p)
      for pred, orig_size in self.pred_buffer:
        self.enqueue(pred, orig_size)
        if self.counter == self.max_size:
          break
    except:
      self.shutdown()
      # TODO

  def enqueue(self, prediction, orig_size):
    if not self._shutdown:
      # faster than X[idx].assign(img.tobytes())
      self.predictions[self.idx].contiguous().realize().lazydata.realized.as_buffer(force_zero_copy=True)[:] = prediction.cast(dtypes.float32).tobytes()
      self.q_in.put((self.idx, orig_size))
      self.counter += 1

  def receive(self):
    idx = self.q_out.get()
    if idx is None:
      return None
    ret = {"boxes": np.empty(), "scores": np.empty(), "labels": np.empty()} # TODO shapes
    ret['boxes'][:] = self.buf_detections[idx, :, :4] # TODO shapes
    ret['scores'][:] = self.buf_detections[idx, :, 4:5] # TODO shapes
    ret['labels'][:] = self.buf_detections[idx, :, 5:] # TODO shapes
    return ret, self.Cookie(idx, self)

  def shutdown(self):
    pass # TODO

  class Cookie:
    def __init__(self, idx, post_proc):
      self.idx, self.post_proc = idx, post_proc
    def __del__(self):
      if not self._shutdown:
        self.post_proc.enqueue(self.idx)

def pp_process(q_in, q_out, predictions, shm_anch_id, shm_det_id, max_size):
  import signal
  signal.signal(signal.SIGINT, lambda _, __: exit(0))

  shm_anchors = shared_memory.SharedMemory(name=shm_anch_id)
  anchors = np.ndarray((predictions.shape[1], 4), dtype=np.float32, buffer=shm_anchors.buf)
  shm_det = shared_memory.SharedMemory(name=shm_det_id)
  detections = np.ndarray((max_size, 4), dtype=np.float32, buffer=shm_det.buf) # TODO shape
  while (_recv := q_in.get()) is not None:
    idx, orig_size = _recv
    detections[idx] = postprocess_detection(predictions[idx].numpy(), anchors, orig_size=orig_size)
    q_out.put(idx)
  q_out.put(None)

def postprocess_detection(prediction, anchors, input_size=(800, 800), orig_size=None, score_thresh=0.05, topk_candidates=1000, nms_thresh=0.5, num_classes=None):
  num_classes = num_classes if num_classes else len(MLPERF_CLASSES)
  grid_sizes = compute_grid_sizes(input_size)
  split_idx = np.cumsum([int(anchors.shape[0] * sz[0] * sz[1]) for sz in grid_sizes[:-1]])
  h, w = input_size
  prediction = np.split(prediction, split_idx)
  offsets_per_image = [br[:, :4] for br in prediction]
  scores_per_image = [cl[:, 4:] for cl in prediction]

  image_boxes, image_scores, image_labels = [], [], []
  for offsets_per_level, scores_per_level, anchors_per_level in zip(offsets_per_image, scores_per_image, anchors):
    # remove low scoring boxes
    scores_per_level = scores_per_level.flatten()
    keep_idxs = scores_per_level > score_thresh
    scores_per_level = scores_per_level[keep_idxs]

    # keep topk
    topk_idxs = np.where(keep_idxs)[0]
    num_topk = min(len(topk_idxs), topk_candidates)
    sort_idxs = scores_per_level.argsort()[-num_topk:][::-1]
    topk_idxs, scores_per_level = topk_idxs[sort_idxs], scores_per_level[sort_idxs]

    # bbox coords from offsets
    anchor_idxs = topk_idxs // num_classes
    labels_per_level = topk_idxs % num_classes
    boxes_per_level = decode_bbox(offsets_per_level[anchor_idxs], anchors_per_level[anchor_idxs])
    # clip to image size
    clipped_x = boxes_per_level[:, 0::2].clip(0, w)
    clipped_y = boxes_per_level[:, 1::2].clip(0, h)
    boxes_per_level = np.stack([clipped_x, clipped_y], axis=2).reshape(-1, 4)

    image_boxes.append(boxes_per_level)
    image_scores.append(scores_per_level)
    image_labels.append(labels_per_level)

  image_boxes = np.concatenate(image_boxes)
  image_scores = np.concatenate(image_scores)
  image_labels = np.concatenate(image_labels)

  # nms for each class
  keep_mask = np.zeros_like(image_scores, dtype=bool)
  for class_id in np.unique(image_labels):
    curr_indices = np.where(image_labels == class_id)[0]
    curr_keep_indices = nms(image_boxes[curr_indices], image_scores[curr_indices], nms_thresh)
    keep_mask[curr_indices[curr_keep_indices]] = True
  keep = np.where(keep_mask)[0]
  keep = keep[image_scores[keep].argsort()[::-1]]

  # resize bboxes back to original size
  image_boxes = image_boxes[keep]
  if orig_size is not None:
    resized_x = image_boxes[:, 0::2] * orig_size[1] / w
    resized_y = image_boxes[:, 1::2] * orig_size[0] / h
    image_boxes = np.stack([resized_x, resized_y], axis=2).reshape(-1, 4)
  # xywh format
  image_boxes = np.concatenate([image_boxes[:, :2], image_boxes[:, 2:] - image_boxes[:, :2]], axis=1)

  return {"boxes":image_boxes, "scores":image_scores[keep], "labels":image_labels[keep]}

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

def decode_bbox(offsets, anchors):
  dx, dy, dw, dh = np.rollaxis(offsets, 1)
  widths, heights = anchors[:, 2] - anchors[:, 0], anchors[:, 3] - anchors[:, 1]
  cx, cy = anchors[:, 0] + 0.5 * widths, anchors[:, 1] + 0.5 * heights
  pred_cx, pred_cy = dx * widths + cx, dy * heights + cy
  pred_w, pred_h = np.exp(dw) * widths, np.exp(dh) * heights
  pred_x1, pred_y1 = pred_cx - 0.5 * pred_w, pred_cy - 0.5 * pred_h
  pred_x2, pred_y2 = pred_cx + 0.5 * pred_w, pred_cy + 0.5 * pred_h
  return np.stack([pred_x1, pred_y1, pred_x2, pred_y2], axis=1, dtype=np.float32)
