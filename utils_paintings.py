import json
import numpy as np
import requests

from os import makedirs
from time import sleep, time as timestamp

from huggingface_hub import hf_hub_download

from mediapipe import Image as mpImage, ImageFormat as mpImageFormat
from mediapipe.tasks.python.core.base_options import BaseOptions as mpBaseOptions
from mediapipe.tasks.python.vision import FaceDetector as mpFaceDetector, FaceLandmarker as mpFaceLandmarker
from mediapipe.tasks.python.vision import FaceDetectorOptions as mpFaceDetectorOptions
from mediapipe.tasks.python.vision import FaceLandmarkerOptions as mpFaceLandmarkerOptions
from mediapipe.tasks.python.vision import RunningMode as mpRunningMode

from ultralytics import YOLO

from utils import pxs_to_pcts, pcts_to_sqs, pct_to_px

class PaintingsUtils:
  MET_URL = "https://collectionapi.metmuseum.org/public/collection/v1"

  OBJ_FIELDS = [
    "objectID",
    "accessionNumber",
    "objectName",
    "title",
    "department",
    "primaryImage",
    "artistRole",
    "artistDisplayName",
    "objectDate",
    "objectBeginDate",
    "objectEndDate",
    "medium",
    "dimensions"
  ]

  landmarker_model_path = "./face_landmarker.task"

  landmarker_options = mpFaceLandmarkerOptions(
    base_options=mpBaseOptions(model_asset_path=landmarker_model_path),
    running_mode=mpRunningMode.IMAGE
  )


  def __init__(self, json_dir, image_dir):
    self.image_dir = image_dir
    self.json_dir = json_dir
    self.json_objs_dir = f"{json_dir}/objects"
    self.json_faces_dir = f"{json_dir}/faces"
    self.json_landmarks_dir = f"{json_dir}/landmarks"
    self.image_eyes_dir = f"{image_dir}/eyes"

    makedirs(self.json_objs_dir, exist_ok=True)
    makedirs(self.json_faces_dir, exist_ok=True)
    makedirs(self.json_landmarks_dir, exist_ok=True)
    makedirs(self.image_eyes_dir, exist_ok=True)

    yolo_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    self.face_detector = YOLO(yolo_model_path)
    self.face_landmarker = mpFaceLandmarker.create_from_options(self.landmarker_options)
    self.last_req = int(timestamp())

    with open(f"{self.json_dir}/mp_masks_definitions.json", "r") as ifp:
      mp_ldk_defs = json.load(ifp)
      self.EYES_IDXS = [mp_ldk_defs["A2b_R"], mp_ldk_defs["A2b_L"]]


  @classmethod
  def get_object_ids(cls):
    response = requests.get(f"{cls.MET_URL}/search?medium=Paintings&hasImages=true&q=*")
    return sorted(list(set(response.json()["objectIDs"])))


  @classmethod
  def get_measurement(cls, obj_data):
    if "measurements" in obj_data and obj_data["measurements"] and len(obj_data["measurements"]) > 0:
      img_meas = [m["elementMeasurements"] for m in obj_data["measurements"] if m["elementName"] == "Image"]
      ovr_meas = [m["elementMeasurements"] for m in obj_data["measurements"] if m["elementName"] == "Overall"]
      otr_meas = [m["elementMeasurements"] for m in obj_data["measurements"] if m["elementName"] == "Other"]
      smt_meas = [m["elementMeasurements"] for m in obj_data["measurements"] if "Height" in m["elementMeasurements"] and "Width" in m["elementMeasurements"]]

      if len(img_meas) > 0:
        return img_meas[0]
      elif len(ovr_meas) > 0:
        return ovr_meas[0]
      elif len(otr_meas) > 0:
        return otr_meas[0]
      elif len(smt_meas) > 0:
        return smt_meas[0]
      else:
        return None
    else:
      return None


  def get_obj_data(self, oid):
    json_obj_path = f"{self.json_objs_dir}/{oid}.json"
    try:
      with open(json_obj_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      tdiff = timestamp() - self.last_req
      if tdiff < 0.75:
        sleep(0.75 - tdiff)
      obj_response = requests.get(f"{self.MET_URL}/objects/{oid}")
      obj_data = obj_response.json()
      self.last_req = timestamp()

      if not ("primaryImage" in obj_data and obj_data["primaryImage"].startswith("http")):
        return None

      obj_filtered_data = { f: obj_data[f] for f in self.OBJ_FIELDS }

      obj_measurements = type(self).get_measurement(obj_data)
      if obj_measurements:
        obj_filtered_data["measurements"] = obj_measurements

      if "tags" in obj_data and obj_data["tags"] and len(obj_data["tags"]) > 0:
        obj_filtered_data["tags"] = [t["term"].lower() for t in obj_data["tags"]]

      with open(json_obj_path, "w") as ofp:
        json.dump(obj_filtered_data, ofp, ensure_ascii=False)
        return obj_filtered_data


  def get_face_data(self, obj_data, img):
    obj_data = json.loads(json.dumps(obj_data))
    oid = obj_data["objectID"]
    json_face_path = f"{self.json_faces_dir}/{oid}.json"
    try:
      with open(json_face_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      iw,ih = img.size
      nh = 256
      nw = int(nh * iw // ih)
      nimg = img.resize((nw, nh))

      faces = self.face_detector.predict(nimg, verbose=False, device="cuda")
      if len(faces) < 1 or len(faces[0]) < 1:
        obj_data["faces"] = {
          "yolo": {
            "count": 0,
            "xyxyn": [],
            "xyxyn_sq": [],
          }
        }
        return obj_data

      faces_xyxyn = faces[0].boxes.xyxyn.cpu().numpy().astype(np.float64)
      faces_xyxyn_sq = pcts_to_sqs(faces_xyxyn, iw, ih)

      obj_data["faces"] = {
        "yolo": {
          "count": len(faces_xyxyn),
          "xyxyn": faces_xyxyn.round(4).tolist(),
          "xyxyn_sq": faces_xyxyn_sq.round(4).tolist(),
        }
      }

      with open(json_face_path, "w") as ofp:
        json.dump(obj_data, ofp, ensure_ascii=False)
        return obj_data


  def get_landmark_data(self, obj_data, img):
    obj_data = json.loads(json.dumps(obj_data))
    oid = obj_data["objectID"]
    json_landmark_path = f"{self.json_landmarks_dir}/{oid}.json"
    try:
      with open(json_landmark_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      iw,ih = img.size
      mp_results = {
        "count": 0,
        "landmarks": [],
      }

      if obj_data["faces"]["yolo"]["count"] < 1:
        obj_data["faces"]["mp"] = mp_results
        with open(json_landmark_path, "w") as ofp:
          json.dump(obj_data, ofp, ensure_ascii=False)
        return obj_data

      for fcnt,fbox in enumerate(obj_data["faces"]["yolo"]["xyxyn_sq"]):
        x0,y0,x1,y1 = pct_to_px(fbox, iw, ih)
        fx0,fy0,fx1,fy1 = fbox
        fw, fh = (fx1 - fx0), (fy1 - fy0)

        face_landmarks = np.array([]).astype(np.float64)
        fimg = img.crop((x0,y0,x1,y1)).convert("RGB")

        for dim in [512, 256, 128, 64]:
          fimg = fimg.resize((dim, dim))
          mp_image = mpImage(image_format=mpImageFormat.SRGB, data=np.array(fimg))
          landmarks = self.face_landmarker.detect(mp_image)

          if len(landmarks.face_landmarks) > 0:
            mp_results["count"] += 1
            face_landmarks = np.array([[fx0 + lm.x * fw, fy0 + lm.y * fh] for lm in landmarks.face_landmarks[0]]).astype(np.float64)
            break

        mp_results["landmarks"].append(face_landmarks.round(4).tolist())

      obj_data["faces"]["mp"] = mp_results

      with open(json_landmark_path, "w") as ofp:
        json.dump(obj_data, ofp, ensure_ascii=False)
        return obj_data


  def get_eye_images(self, obj_data, img):
    oid = obj_data["objectID"]
    iw,ih = img.size

    if not ("faces" in obj_data and "mp" in obj_data["faces"] and obj_data["faces"]["mp"]["count"] > 0):
      return

    for fcnt,landmarks in enumerate(obj_data["faces"]["mp"]["landmarks"]):
      if len(landmarks) < 1:
        continue

      face_cnt_str = f"000{fcnt}"[-3:]
      img_path = f"{self.image_eyes_dir}/{oid}_{face_cnt_str}.avif"
      if path.isfile(img_path):
        continue

      landmarks_np = np.array(landmarks) * img.size
      pair_points = [landmarks_np[eye_idxs].tolist() for eye_idxs in self.EYES_IDXS]

      pair_points_np = np.array(pair_points).reshape(-1, 2)
      minx, miny = (pair_points_np.min(axis=0)).tolist()
      maxx, maxy = (pair_points_np.max(axis=0)).tolist()
      cx, cy = (maxx + minx) / 2, (maxy + miny) / 2
      sx, sy = maxx - minx, maxy - miny
      w_2, h_2 = 1.25 * (maxx - minx) / 2, 1.25 * (maxy - miny) / 2

      scale = 2 if h_2 > 200 else (4 if h_2 > 100 else 8)
      mimg = mask_with_polygons(img, pair_points, detail=32, scale=scale)

      pair_img = mimg.crop((cx - w_2, cy - h_2, cx + w_2, cy + h_2))
      pair_img.save(img_path)
