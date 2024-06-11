import concurrent.futures
import glob
import json
import math
import multiprocessing
import os
import pathlib
import pickle
import random
import sys
import boto3, botocore
import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from pycocotools.coco import COCO
from tqdm import tqdm
import extra.datasets.get_image_size as get_image_size
from tinygrad.helpers import fetch, diskcache
from tinygrad import Tensor

BASEDIR = pathlib.Path(__file__).parent / "open-images-v6-mlperf"
BUCKET_NAME = "open-images-dataset"
MAP_CLASSES_URL = "https://storage.googleapis.com/openimages/v5/class-descriptions-boxable.csv"
ANNOTATIONS_URLS = {
  'train': "https://storage.googleapis.com/openimages/v6/oidv6-train-annotations-bbox.csv",
  'validation': "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv"
}
MLPERF_CLASSES = ['Airplane', 'Antelope', 'Apple', 'Backpack', 'Balloon', 'Banana',
  'Barrel', 'Baseball bat', 'Baseball glove', 'Bee', 'Beer', 'Bench', 'Bicycle',
  'Bicycle helmet', 'Bicycle wheel', 'Billboard', 'Book', 'Bookcase', 'Boot',
  'Bottle', 'Bowl', 'Bowling equipment', 'Box', 'Boy', 'Brassiere', 'Bread',
  'Broccoli', 'Bronze sculpture', 'Bull', 'Bus', 'Bust', 'Butterfly', 'Cabinetry',
  'Cake', 'Camel', 'Camera', 'Candle', 'Candy', 'Cannon', 'Canoe', 'Carrot', 'Cart',
  'Castle', 'Cat', 'Cattle', 'Cello', 'Chair', 'Cheese', 'Chest of drawers', 'Chicken',
  'Christmas tree', 'Coat', 'Cocktail', 'Coffee', 'Coffee cup', 'Coffee table', 'Coin',
  'Common sunflower', 'Computer keyboard', 'Computer monitor', 'Convenience store',
  'Cookie', 'Countertop', 'Cowboy hat', 'Crab', 'Crocodile', 'Cucumber', 'Cupboard',
  'Curtain', 'Deer', 'Desk', 'Dinosaur', 'Dog', 'Doll', 'Dolphin', 'Door', 'Dragonfly',
  'Drawer', 'Dress', 'Drum', 'Duck', 'Eagle', 'Earrings', 'Egg (Food)', 'Elephant',
  'Falcon', 'Fedora', 'Flag', 'Flowerpot', 'Football', 'Football helmet', 'Fork',
  'Fountain', 'French fries', 'French horn', 'Frog', 'Giraffe', 'Girl', 'Glasses',
  'Goat', 'Goggles', 'Goldfish', 'Gondola', 'Goose', 'Grape', 'Grapefruit', 'Guitar',
  'Hamburger', 'Handbag', 'Harbor seal', 'Headphones', 'Helicopter', 'High heels',
  'Hiking equipment', 'Horse', 'House', 'Houseplant', 'Human arm', 'Human beard',
  'Human body', 'Human ear', 'Human eye', 'Human face', 'Human foot', 'Human hair',
  'Human hand', 'Human head', 'Human leg', 'Human mouth', 'Human nose', 'Ice cream',
  'Jacket', 'Jeans', 'Jellyfish', 'Juice', 'Kitchen & dining room table', 'Kite',
  'Lamp', 'Lantern', 'Laptop', 'Lavender (Plant)', 'Lemon', 'Light bulb', 'Lighthouse',
  'Lily', 'Lion', 'Lipstick', 'Lizard', 'Man', 'Maple', 'Microphone', 'Mirror',
  'Mixing bowl', 'Mobile phone', 'Monkey', 'Motorcycle', 'Muffin', 'Mug', 'Mule',
  'Mushroom', 'Musical keyboard', 'Necklace', 'Nightstand', 'Office building',
  'Orange', 'Owl', 'Oyster', 'Paddle', 'Palm tree', 'Parachute', 'Parrot', 'Pen',
  'Penguin', 'Personal flotation device', 'Piano', 'Picture frame', 'Pig', 'Pillow',
  'Pizza', 'Plate', 'Platter', 'Porch', 'Poster', 'Pumpkin', 'Rabbit', 'Rifle',
  'Roller skates', 'Rose', 'Salad', 'Sandal', 'Saucer', 'Saxophone', 'Scarf', 'Sea lion',
  'Sea turtle', 'Sheep', 'Shelf', 'Shirt', 'Shorts', 'Shrimp', 'Sink', 'Skateboard',
  'Ski', 'Skull', 'Skyscraper', 'Snake', 'Sock', 'Sofa bed', 'Sparrow', 'Spider', 'Spoon',
  'Sports uniform', 'Squirrel', 'Stairs', 'Stool', 'Strawberry', 'Street light',
  'Studio couch', 'Suit', 'Sun hat', 'Sunglasses', 'Surfboard', 'Sushi', 'Swan',
  'Swimming pool', 'Swimwear', 'Tank', 'Tap', 'Taxi', 'Tea', 'Teddy bear', 'Television',
  'Tent', 'Tie', 'Tiger', 'Tin can', 'Tire', 'Toilet', 'Tomato', 'Tortoise', 'Tower',
  'Traffic light', 'Train', 'Tripod', 'Truck', 'Trumpet', 'Umbrella', 'Van', 'Vase',
  'Vehicle registration plate', 'Violin', 'Wall clock', 'Waste container', 'Watch',
  'Whale', 'Wheel', 'Wheelchair', 'Whiteboard', 'Window', 'Wine', 'Wine glass', 'Woman',
  'Zebra', 'Zucchini',
]

def openimages(split, dataset_dir=None):
  dataset_dir = _valargs(split, dataset_dir)
  ann_file = dataset_dir / f"{split}/labels/openimages-mlperf.json"
  if not ann_file.is_file():
    fetch_openimages(ann_file, split, dataset_dir=dataset_dir)
  return ann_file

@diskcache
def get_files(split, dataset_dir=None):
  dataset_dir = _valargs(split, dataset_dir)
  if not (files := glob.glob(p := str(dataset_dir / split / "data/*"))): raise FileNotFoundError(f"No {split} files in {p}")
  return files

def get_targets(split, dataset_dir=None, cache=False):
  dataset_dir = _valargs(split, dataset_dir)
  if cache:
    (cache_dir := dataset_dir / "cache").mkdir(parents=True, exist_ok=True)
    cache_fn = cache_dir / f"{split}_targets.pkl" if cache_dir else None
    if cache_fn.is_file():
      with open(cache_fn, 'rb') as f:
        return pickle.load(f)

  dataset = json.load(open(openimages(split, dataset_dir=dataset_dir)))
  images, annotations = dataset['images'], dataset['annotations']
  entries, targets = [], []
  prev_id = annotations[0]['image_id']

  def process_img(idx):
    file_name = str(pathlib.Path(split) / "data" / images[idx - 1]['file_name'])
    img_size = extract_dims(dataset_dir / file_name)
    target = prepare_target(entries, prev_id, img_size, new_size=(800, 800))
    target['file_name'] = file_name
    targets.append(target)

  for ann in tqdm(annotations, desc="Processing targets"):
    if (idx := ann['image_id']) != prev_id and entries:
      process_img(prev_id)
      entries = []
      prev_id = idx
    entries.append(ann)
  process_img(idx)

  if cache_dir:
    with open(cache_fn, 'wb') as f:
      pickle.dump(targets, f)
  return targets

def _valargs(split, dataset_dir):
  assert split in (valid_splits := ['train', 'validation']), f"{split=} must be one of {valid_splits}"
  return pathlib.Path(dataset_dir) if dataset_dir else BASEDIR

def extract_dims(path):
  dims = get_image_size.Image.to_str_row(get_image_size.get_image_metadata(str(path)))
  return tuple(int(i) for i in dims.split()[1::-1])

def export_to_coco(class_map, annotations, image_list, dataset_path, output_path, classes=MLPERF_CLASSES):
  output_path.parent.mkdir(parents=True, exist_ok=True)
  cats = [{"id": i, "name": c, "supercategory": None} for i, c in enumerate(classes)]

  valid_ids = set(annotations["ImageID"].tolist()).intersection(set(image_list))
  annotations = annotations[[row["ImageID"] in valid_ids for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Filtering images")]]
  categories_map = pd.DataFrame([(i, c) for i, c in enumerate(classes)], columns=["category_id", "category_name"])
  class_map = class_map.merge(categories_map, left_on="DisplayName", right_on="category_name", how="inner")
  annotations = annotations.merge(class_map, on="LabelName", how="inner")
  annotations["image_id"] = pd.factorize(annotations["ImageID"].tolist())[0]
  dims = pd.DataFrame([extract_dims(dataset_path / f"{row['ImageID']}.jpg")
    for _, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Extracting dimensions")], columns=["height", "width"])
  annotations = annotations.join(dims)

  # Images
  imgs = [{"id": int(id + 1), "file_name": f"{image_id}.jpg", "height": row["height"], "width": row["width"], "license": None, "coco_url": None}
    for (id, image_id), row in tqdm(annotations.groupby(["image_id", "ImageID"]).first().iterrows(), total=len(set(annotations["ImageID"])), desc="Loading images")
  ]

  # Annotations
  annots = []
  for i, row in tqdm(annotations.iterrows(), total=len(annotations), desc="Loading annotations"):
    xmin, ymin, xmax, ymax, img_w, img_h = [row[k] for k in ["XMin", "YMin", "XMax", "YMax", "width", "height"]]
    x, y, w, h = xmin * img_w, ymin * img_h, (xmax - xmin) * img_w, (ymax - ymin) * img_h
    coco_annot = {"id": int(i) + 1, "image_id": int(row["image_id"] + 1), "category_id": int(row["category_id"]), "bbox": [x, y, w, h], "area": w * h}
    coco_annot.update({k: row[k] for k in ["IsOccluded", "IsInside", "IsDepiction", "IsTruncated", "IsGroupOf"]})
    coco_annot["iscrowd"] = int(row["IsGroupOf"])
    annots.append(coco_annot)

  print("Writing coco annotations...")
  info = {"dataset": "openimages_mlperf", "version": "v6"}
  coco_annotations = {"info": info, "licenses": [], "categories": cats, "images": imgs, "annotations": annots}
  with open(output_path, "w") as fp:
    json.dump(coco_annotations, fp)

def get_image_list(class_map, annotations, classes=MLPERF_CLASSES):
  labels = class_map[np.isin(class_map["DisplayName"], classes)]["LabelName"]
  image_ids = annotations[np.isin(annotations["LabelName"], labels)]["ImageID"].unique()
  return image_ids

def download_image(bucket, split, image_id, data_dir):
  try:
    bucket.download_file(f"{split}/{image_id}.jpg", f"{data_dir}/{image_id}.jpg")
  except botocore.exceptions.ClientError as exception:
    sys.exit(f"ERROR when downloading image `{split}/{image_id}`: {str(exception)}")

def fetch_openimages(output_fn, split, dataset_dir=BASEDIR):
  bucket = boto3.resource("s3", config=botocore.config.Config(signature_version=botocore.UNSIGNED)).Bucket(BUCKET_NAME)
  (annotations_dir := dataset_dir / "annotations").mkdir(parents=True, exist_ok=True)
  (data_dir := dataset_dir / f"{split}/data").mkdir(parents=True, exist_ok=True)

  if not (annotations_fn := annotations_dir / ANNOTATIONS_URLS[split].split('/')[-1]).is_file():
    fetch(ANNOTATIONS_URLS[split], annotations_fn)
  else:
    print(f"Already exists: {annotations_fn}")
  print("Loading annotations...")
  annotations = pd.read_csv(annotations_fn)

  if not (classmap_fn := annotations_dir / MAP_CLASSES_URL.split('/')[-1]).is_file():
    fetch(MAP_CLASSES_URL, classmap_fn)
  else:
    print(f"Already exists: {classmap_fn}")
  print("Loading class map...")
  class_map = pd.read_csv(classmap_fn, names=["LabelName", "DisplayName"])

  print("Loading images...")
  image_list = get_image_list(class_map, annotations)
  with concurrent.futures.ThreadPoolExecutor() as executor:
    futures = [executor.submit(download_image, bucket, split, image_id, data_dir) for image_id in tqdm(image_list, desc="Submitting downloads")
               if not pathlib.Path(f"{data_dir}/{image_id}.jpg").is_file()]
    for future in (t := tqdm(concurrent.futures.as_completed(futures), total=len(futures))):
      t.set_description("Downloading images")
      future.result()

  print("Converting annotations to COCO format...")
  export_to_coco(class_map, annotations, image_list, data_dir, output_fn)

def image_load(fn):
  img = Image.open(fn).convert('RGB')
  import torchvision.transforms.functional as F
  ret = F.resize(img, size=(800, 800))
  ret = np.array(ret)
  return ret, img.size[::-1]

def iterate(coco, split, bs=8, dataset_dir=BASEDIR, progress=False):
  image_ids = sorted(coco.imgs.keys())
  for i in range(0, len(image_ids), bs):
    X, targets = [], []
    for img_id in tqdm(image_ids[i:i+bs], disable=not progress, desc="Loading samples"):
      x, original_size = image_load(dataset_dir / split / "data" / coco.loadImgs(img_id)[0]["file_name"])
      X.append(x)
      annotations = coco.loadAnns(coco.getAnnIds(img_id))
      targets.append(prepare_target(annotations, img_id, original_size))
    yield np.array(X), targets

def prepare_target(annotations, img_id, img_size, new_size=None):
  boxes = [annot["bbox"] for annot in annotations]
  boxes = np.array(boxes, dtype=np.float32).reshape(-1, 4)
  boxes[:, 2:] += boxes[:, :2]
  boxes[:, 0::2] = boxes[:, 0::2].clip(0, img_size[1])
  boxes[:, 1::2] = boxes[:, 1::2].clip(0, img_size[0])
  keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
  boxes = boxes[keep]
  if new_size:
    h, w = img_size
    boxes = resize_boxes(boxes, (h, w), new_size)

  classes = [annot["category_id"] for annot in annotations]
  classes = np.array(classes, dtype=np.int64)
  classes = classes[keep]
  return {"boxes": boxes, "labels": classes, "image_id": img_id, "image_size": img_size}

def resize_boxes(boxes, original_size, new_size) -> Tensor:
  ratios = [s / s_orig for s, s_orig in zip(new_size, original_size)]
  ratio_height, ratio_width = ratios
  xmin, ymin, xmax, ymax = boxes.transpose()
  xmin = xmin * ratio_width
  xmax = xmax * ratio_width
  ymin = ymin * ratio_height
  ymax = ymax * ratio_height
  return np.stack((xmin, ymin, xmax, ymax), axis=1)
