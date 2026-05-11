import json
import numpy as np

from os import listdir
from PIL import ImageDraw as PImageDraw


MP_FACE_LANDMARKS = {
  "EYE_0_L": [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398],
  "EYE_0_R": [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246],

  "EYE_1_L": [463, 341, 256, 252, 253, 254, 339, 255, 359, 467, 260, 259, 257, 258, 286, 414],
  "EYE_1_R": [130, 25, 110, 24, 23, 22, 26, 112, 243, 190, 56, 28, 27, 29, 30, 247],

  "EYE_2_L": [417, 464, 453, 452, 451, 450, 449, 448, 261, 446, 342, 445, 444, 443, 442, 441],
  "EYE_2_R": [226, 31, 228, 229, 230, 231, 232, 233, 244, 193, 221, 222, 223, 224, 225, 113],

  "EYE_3_L": [168, 465, 357, 350, 349, 348, 347, 346, 340, 265, 353, 276, 283, 282, 295, 285],
  "EYE_3_R": [35, 111, 117, 118, 119, 120, 121, 128, 245, 168, 55, 65, 52, 53, 46, 124],

  "EYE_4": [143, 116, 123, 50, 101, 100, 47, 114, 188, 122, 6, 351, 412, 343, 277, 329, 330,
            280, 352, 345, 372, 383, 300, 293, 334, 296, 336, 8, 107, 66, 105, 63, 70, 156],

  "EYE_3_4": [35, 111, 117, 118, 119, 120, 121, 128, 122, 6, 351, 357, 350, 349, 348, 347,
              346, 340, 265, 353, 276, 283, 282, 295, 285, 8, 55, 65, 52, 53, 46, 124],

  "EYE_4_3": [143, 111, 117, 118, 119, 120, 121, 128, 245, 6, 465, 357, 350, 349, 348, 347,
              346, 340, 372, 383, 300, 293, 334, 296, 336, 8, 107, 66, 105, 63, 70, 156],

  "LIP_OUTER": [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
  "LIP_INNER": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],
}


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

  dim_px = 1.1 * max(wpx, hpx)
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
