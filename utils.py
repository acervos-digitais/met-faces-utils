import json
import numpy as np

from os import listdir
from PIL import ImageDraw as PImageDraw

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


def pxs_to_pcts(boxes, img_w, img_h):
  xywhn = []
  xyxyn = []
  for x0,y0,x1,y1 in boxes:
    xywhn.append([x0/img_w, y0/img_h, (x1-x0)/img_w, (y1-y0)/img_h])
    xyxyn.append([x0/img_w, y0/img_h, x1/img_w, y1/img_h])
  return (np.array(xywhn), np.array(xyxyn))


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
