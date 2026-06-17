import json
import numpy as np
import requests

from os import listdir, makedirs, path
from PIL import Image as PImage
from time import sleep, time as timestamp

try:
  from huggingface_hub import hf_hub_download

  from mediapipe import Image as mpImage, ImageFormat as mpImageFormat
  from mediapipe.tasks.python.core.base_options import BaseOptions as mpBaseOptions
  from mediapipe.tasks.python.vision import FaceDetector as mpFaceDetector, FaceLandmarker as mpFaceLandmarker
  from mediapipe.tasks.python.vision import FaceDetectorOptions as mpFaceDetectorOptions
  from mediapipe.tasks.python.vision import FaceLandmarkerOptions as mpFaceLandmarkerOptions
  from mediapipe.tasks.python.vision import RunningMode as mpRunningMode

  from ultralytics import YOLO
except:
  pass

from utils import pxs_to_pcts, pcts_to_sqs, pct_to_px
from utils import get_masks_definitions, mask_with_polygons

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

  def __init__(self, json_dir, image_dir):
    self.image_dir = image_dir
    self.json_dir = json_dir
    self.json_objs_dir = f"{json_dir}/objects"
    self.json_faces_dir = f"{json_dir}/faces"
    self.json_landmarks_dir = f"{json_dir}/landmarks"
    self.image_eyes_dir = f"{image_dir}/eyes"

    makedirs(self.json_objs_dir, exist_ok=True)
    makedirs(self.image_eyes_dir, exist_ok=True)

    self.last_req = int(timestamp())

    mp_ldk_defs = get_masks_definitions()
    self.EYES_IDXS = [mp_ldk_defs["A2b_R"], mp_ldk_defs["A2b_L"]]

    try:
      with open(f"{self.json_dir}/no_imgs.json", "r") as ifp:
        self.no_imgs = json.load(ifp)
    except:
      self.no_imgs = []

    try:
      with open(f"{self.json_dir}/no_faces.json", "r") as ifp:
        self.no_faces = json.load(ifp)
    except:
      self.no_faces = []

    try:
      with open(f"{self.json_dir}/no_landmarks.json", "r") as ifp:
        self.no_landmarks = json.load(ifp)
    except:
      self.no_landmarks = []


  @classmethod
  def get_object_ids(cls):
    response = requests.get(f"{cls.MET_URL}/search?medium=Paintings&hasImages=true&q=*")
    return sorted(list(set(response.json()["objectIDs"])))


  def init_face_detector(self):
    makedirs(self.json_faces_dir, exist_ok=True)

    yolo_model_path = hf_hub_download(repo_id="AdamCodd/YOLOv11n-face-detection", filename="model.pt")
    self.face_detector = YOLO(yolo_model_path)


  def init_face_landmarker(self):
    makedirs(self.json_landmarks_dir, exist_ok=True)

    landmarker_model_path = "./face_landmarker.task"

    landmarker_options = mpFaceLandmarkerOptions(
      base_options=mpBaseOptions(model_asset_path=landmarker_model_path),
      running_mode=mpRunningMode.IMAGE
    )

    self.face_landmarker = mpFaceLandmarker.create_from_options(landmarker_options)


  def get_measurement(obj_data):
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
    if oid in self.no_imgs:
      return None

    json_obj_path = f"{self.json_objs_dir}/{oid}.json"
    try:
      with open(json_obj_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      tdiff = timestamp() - self.last_req
      if tdiff < 0.75:
        sleep(0.75 - tdiff)

      try:
        obj_response = requests.get(f"{self.MET_URL}/objects/{oid}", timeout=16)
      except requests.exceptions.Timeout:
        return None

      obj_data = obj_response.json()
      self.last_req = timestamp()

      if not ("primaryImage" in obj_data and obj_data["primaryImage"].startswith("http")):
        self.no_imgs.append(oid)
        with open(f"{self.json_dir}/no_imgs.json", "w") as ofp:
          json.dump(self.no_imgs, ofp, ensure_ascii=False)
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


  def is_done(self, obj_data):
    oid = obj_data["objectID"]
    json_face_path = f"{self.json_faces_dir}/{oid}.json"
    json_landmark_path = f"{self.json_landmarks_dir}/{oid}.json"

    # Haven't run face detection
    if not (path.isfile(json_face_path) or oid in self.no_faces):
      return False

    # Haven't run landmarks
    if not (path.isfile(json_landmark_path) or oid in self.no_landmarks):
      return False

    # Have run, but there are no faces nor landmarks
    if oid in self.no_faces or oid in self.no_landmarks:
      return True

    # Have landmarks, but not all images
    if path.isfile(json_landmark_path):
      with open(json_landmark_path, "r") as ifp:
        landmark_data = json.load(ifp)["faces"]["mp"]

      img_files = [f for f in listdir(self.image_eyes_dir) if f.startswith(f"{oid}_")]
      return len(img_files) >= landmark_data["count"]

    # Have run face detection, have run landmarking, have image files
    return True


  def get_face_data(self, obj_data, img):
    oid = obj_data["objectID"]
    json_face_path = f"{self.json_faces_dir}/{oid}.json"

    if oid in self.no_imgs or oid in self.no_faces:
      return None

    try:
      with open(json_face_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      obj_data = json.loads(json.dumps(obj_data))
      iw,ih = img.size
      nh = 256
      nw = int(nh * iw // ih)
      nimg = img.resize((nw, nh))

      faces = self.face_detector.predict(nimg, verbose=False, device="cuda")
      if len(faces) < 1 or len(faces[0]) < 1:
        self.no_faces.append(oid)
        with open(f"{self.json_dir}/no_faces.json", "w") as ofp:
          json.dump(self.no_faces, ofp, ensure_ascii=False)
        return None

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
    oid = obj_data["objectID"]
    json_landmark_path = f"{self.json_landmarks_dir}/{oid}.json"

    if oid in self.no_imgs or oid in self.no_faces or oid in self.no_landmarks:
      return None

    try:
      with open(json_landmark_path, "r") as ifp:
        return json.load(ifp)
    except FileNotFoundError:
      obj_data = json.loads(json.dumps(obj_data))
      iw,ih = img.size
      mp_results = {
        "count": 0,
        "landmarks": [],
      }

      if obj_data["faces"]["yolo"]["count"] < 1:
        self.no_landmarks.append(oid)
        with open(f"{self.json_dir}/no_landmarks.json", "w") as ofp:
          json.dump(self.no_landmarks, ofp, ensure_ascii=False)
        return None

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

      if obj_data["faces"]["mp"]["count"] < 1:
        self.no_landmarks.append(oid)
        with open(f"{self.json_dir}/no_landmarks.json", "w") as ofp:
          json.dump(self.no_landmarks, ofp, ensure_ascii=False)
        return None

      with open(json_landmark_path, "w") as ofp:
        json.dump(obj_data, ofp, ensure_ascii=False)
        return obj_data


  def get_image(self, img_url):
    tdiff = timestamp() - self.last_req
    if tdiff < 0.75:
      sleep(0.75 - tdiff)

    img_response = requests.get(img_url, stream=True)
    self.last_req = timestamp()

    if img_response.status_code > 399 or img_response.status_code < 200:
      return None
    else:
      return PImage.open(img_response.raw)


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
