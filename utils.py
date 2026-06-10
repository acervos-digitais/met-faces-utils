import json
import numpy as np

from os import listdir
from PIL import Image as PImage, ImageDraw as PImageDraw


def get_masks_definitions():
  mask_definitions = {}
  with open("./data/json/mp_masks_definitions.json", "r") as ifp:
    mask_definitions = json.load(ifp)
  return mask_definitions


def export_combined_jsons(input_json_dir, output_json_dir, filename, filter_keys=None):
  json_files = sorted(f for f in listdir(input_json_dir) if f.endswith("json"))
  combined_data = []
  for jf in json_files:
    if filter_keys is None:
      with open(f"{input_json_dir}/{jf}", "r") as ifp:
        combined_data.append(json.load(ifp))
    else:
      with open(f"{input_json_dir}/{jf}", "r") as ifp:
        json_data = json.load(ifp)
        json_keys = set(json_data.keys())
        if len(json_keys.intersection(set(filter_keys))) > 0:
          combined_data.append(json_data)

  with open(f"{output_json_dir}/{filename}.json", "w") as ofp:
    json.dump({ filename : combined_data }, ofp)


def px_to_pct(box, img_w, img_h, xyxy=True):
  x0,y0,x1,y1 = box
  if xyxy:
    return [x0 / img_w, y0 / img_h, x1 / img_w, y1 / img_h]
  else:
    return [x0 / img_w, y0 / img_h, (x1 - x0) / img_w, (y1 - y0) / img_h]


def pxs_to_pcts(boxes, img_w, img_h, xyxy=True):
  return np.array([px_to_pct(box, img_w, img_h, xyxy) for box in boxes])


def pct_to_px(box, img_w, img_h):
  x0,y0,x1,y1 = box
  return [int(x0 * img_w), int(y0 * img_h), int(x1 * img_w), int(y1 * img_h)]

def pcts_to_pxs(boxes, img_w, img_h):
  return np.array([pct_to_px(box, img_w, img_h) for box in boxes])


def pct_to_sq(box, img_w, img_h, xyxy=True):
  x0,y0,x1,y1 = box

  if xyxy:
    wpx = (x1 - x0) * img_w
    hpx = (y1 - y0) * img_h
  else:
    wpx = x1 * img_w
    hpx = y1 * img_h

  dim_px = 1.8 * max(wpx, hpx)
  dim_x_2 = (dim_px / img_w) / 2
  dim_y_2 = (dim_px / img_h) / 2

  cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
  cx = min(max(cx, dim_x_2), 1 - dim_x_2)
  cy = min(max(cy, dim_y_2), 1 - dim_y_2)

  if xyxy:
    return [cx - dim_x_2, cy - dim_y_2, cx + dim_x_2, cy + dim_y_2]
  else:
    return [cx - dim_x_2, cy - dim_y_2, 2 * dim_x_2, 2 * dim_y_2]


def pcts_to_sqs(boxes, img_w, img_h, xyxy=True):
  return np.array([pct_to_sq(box, img_w, img_h, xyxy) for box in boxes])


def pct_to_px_sq(box, img_w, img_h):
  x0,y0,x1,y1 = pct_to_px(box, img_w, img_h)
  dim = 1.1 * max(x1 - x0, y1 - y0)
  dim_2 = dim / 2
  cx, cy = (x0 + x1) / 2, (y0 + y1) / 2

  cx = min(max(cx, dim_2), img_w - dim_2)
  cy = min(max(cy, dim_2), img_h - dim_2)

  return [int(cx - dim_2), int(cy - dim_2), int(cx + dim_2), int(cy + dim_2)]


def draw_boxes(img, boxes):
  dimg = img.convert("RGB").copy()
  draw = PImageDraw.Draw(dimg)
  iw,ih = img.size
  for x0,y0,x1,y1 in boxes:
    draw.rectangle((x0*iw, y0*ih, x1*iw, y1*ih),
                   outline=(0, 220, 0),
                   width=(min(img.size) // 128))
  dimg.thumbnail((300,300))
  return dimg


def curve_t(a,b,c,d,t):
  t3 = t * t * t
  t2 = t * t
  f1 = -0.5 * t3 + t2 - 0.5 * t
  f2 = 1.5 * t3 - 2.5 * t2 + 1
  f3 = -1.5 * t3 + 2 * t2 + 0.5 * t
  f4 = 0.5 * t3 -0.5 * t2
  return a * f1 + b * f2 + c * f3 + d * f4


def curve_segment(a,b,c,d,detail=4):
  return np.array([
    curve_t(a,b,c,d,t)
    for t in np.linspace(0.0, 1.0, num=detail)
  ])


def curve(points,detail=4):
  points_np = np.vstack((points[:1], points, points[-1:]))
  curve_points = np.empty(shape=(0,2))
  for idx in range(0, len(points_np) - 3):
    curve_points = np.vstack((curve_points, curve_segment(*points_np[idx:idx+4], detail=detail)))
  return curve_points


def crop_with_polygons(img, pointses, detail=4):
  polygons = [
    curve(np.vstack((points, points[:1])), detail=detail)
    for points in pointses
  ]

  nimg = img.convert("RGBA")
  mask = PImage.new("L", nimg.size, 0)
  draw = PImageDraw.Draw(mask)
  for polygon in polygons:
    draw.polygon(np.array(polygon).tolist(), fill=255)
  nimg.putalpha(mask)
  return nimg
