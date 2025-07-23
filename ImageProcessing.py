# FastAPI service for face quality evaluation using SCRFD
# Requires: pip install fastapi uvicorn opencv-python onnxruntime numpy pillow

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import onnxruntime as ort
from typing import List, Optional, Dict
from io import BytesIO
from PIL import Image
import math

app = FastAPI(title="SCRFD Face Detection API")

# ---- Models ----
class FaceResponse(BaseModel):
    score: float
    landmarks: List[List[float]]
    aligned_base64: Optional[str] = None

class ImageRequest(BaseModel):
    image_base64: str

class Landmark(BaseModel):
    name: str
    coordinates: List[float]  # [x, y]

class FaceInfo(BaseModel):
    score: float
    landmarks: List[Landmark]
    aligned_base64: Optional[str] = None
    wearing_sunglasses: Optional[bool] = None

class MultiFaceResponse(BaseModel):
    faces: List[FaceInfo]

class SingleFaceResponse(BaseModel):
    score: float
    landmarks: List[Landmark]
    aligned_base64: Optional[str] = None
    wearing_sunglasses: Optional[bool] = None
    template: Optional[str] = None

# ---- Load SCRFD Model ----
scrfd_session = ort.InferenceSession("scrfd_10g_bnkps.onnx", providers=["CPUExecutionProvider"])
input_size = (640, 640)

# ---- Utility Functions ----
def base64_to_image(base64_str: str) -> np.ndarray:
    img_data = base64.b64decode(base64_str)
    image = Image.open(BytesIO(img_data)).convert("RGB")
    return np.array(image)

def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")

def generate_anchors_per_stride(input_size=(640, 640), strides=[8, 16, 32]):
    all_anchors = []
    all_strides = []
    for stride in strides:
        fm_h = math.ceil(input_size[1] / stride)
        fm_w = math.ceil(input_size[0] / stride)
        anchors = []
        strides_arr = []
        for i in range(fm_h):
            for j in range(fm_w):
                anchor_cx = j * stride
                anchor_cy = i * stride
                anchors.append([anchor_cx, anchor_cy])
                anchors.append([anchor_cx, anchor_cy])
                strides_arr.append(stride)
                strides_arr.append(stride)
        all_anchors.append(np.array(anchors, dtype=np.float32))
        all_strides.append(np.array(strides_arr, dtype=np.float32))
    return all_anchors, all_strides

def decode_bboxes(anchors, deltas, strides):
    anchor_cx, anchor_cy = anchors[:, 0], anchors[:, 1]
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_cx = dx * strides + anchor_cx
    pred_cy = dy * strides + anchor_cy
    pred_w = np.exp(dw) * strides
    pred_h = np.exp(dh) * strides
    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2
    return np.stack([x1, y1, x2, y2], axis=1)

def decode_landmarks(anchors, lms_deltas, strides):
    lms = []
    for i in range(5):
        dx = lms_deltas[:, 2 * i]
        dy = lms_deltas[:, 2 * i + 1]
        x = dx * strides + anchors[:, 0]
        y = dy * strides + anchors[:, 1]
        lms.append(np.stack([x, y], axis=1))
    lms = np.stack(lms, axis=1)  # (N, 5, 2)
    return lms

def align_face(img: np.ndarray, lms: np.ndarray, size: int = 112) -> np.ndarray:
    std = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(lms, std)[0]
    return cv2.warpAffine(img, M, (size, size), borderValue=0)

def numpy_nms(boxes, scores, iou_threshold=0.2):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)
        inds = np.where(ovr <= iou_threshold)[0]
        order = order[inds + 1]
    return np.array(keep)

def cluster_boxes_by_center(boxes, scores, distance_thresh=60):
    centers = np.column_stack(((boxes[:, 0] + boxes[:, 2]) / 2, (boxes[:, 1] + boxes[:, 3]) / 2))
    keep = []
    used = set()
    for i in np.argsort(-scores):
        if i in used:
            continue
        keep.append(i)
        dists = np.linalg.norm(centers - centers[i], axis=1)
        close = np.where(dists < distance_thresh)[0]
        used.update(close)
    return np.array(keep)

def is_wearing_sunglasses(aligned_face: np.ndarray, aligned_landmarks: np.ndarray, debug_prefix=None, face_idx=None) -> bool:
    left_eye = tuple(aligned_landmarks[0].astype(int))
    right_eye = tuple(aligned_landmarks[1].astype(int))
    eye_size = 12

    def eye_darkness(center, label):
        x, y = center
        x1, x2 = max(0, x - eye_size // 2), min(aligned_face.shape[1], x + eye_size // 2)
        y1, y2 = max(0, y - eye_size // 2), min(aligned_face.shape[0], y + eye_size // 2)
        region = aligned_face[y1:y2, x1:x2]
        if region.size == 0 or region.shape[0] == 0 or region.shape[1] == 0:
            return 0.0, 255.0  # Return non-dark, very bright if region is invalid
        gray = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        gray = np.array(gray, dtype=np.float32)  # Ensure it's a numpy ndarray of float32
        if debug_prefix is not None and face_idx is not None:
            cv2.imwrite(f"{debug_prefix}_eye_{label}_{face_idx}.jpg", region)
        frac_dark = float(np.mean(gray < 60))
        mean_val = float(np.mean(gray))
        return frac_dark, mean_val

    l_frac, l_mean = eye_darkness(left_eye, 'left')
    r_frac, r_mean = eye_darkness(right_eye, 'right')
    # Sunglasses if both eyes are mostly dark and mean value is low
    return bool(l_frac > 0.5 and r_frac > 0.5) and (l_mean < 80 and r_mean < 80)

# ---- Face Detection Endpoint ----
@app.post("/detectFace", response_model=SingleFaceResponse)
def detect(req: ImageRequest):
    img = base64_to_image(req.image_base64)
    orig_h, orig_w = img.shape[:2]
    # Save original image
    cv2.imwrite("debug_original.jpg", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    resized = cv2.resize(img, input_size)
    # Save resized image
    cv2.imwrite("debug_resized.jpg", cv2.cvtColor(resized, cv2.COLOR_RGB2BGR))
    input_blob = resized[:, :, ::-1].astype(np.float32)
    input_blob = (input_blob - 127.5) / 128
    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis]

    input_name = scrfd_session.get_inputs()[0].name
    outputs = scrfd_session.run(None, {input_name: input_blob})

    def only_arrays(lst):
        return [x for x in lst if isinstance(x, np.ndarray)]
    scores = [s.squeeze(-1) if len(s.shape) == 2 and s.shape[1] == 1 else s for s in only_arrays([outputs[0], outputs[1], outputs[2]])]
    bboxes = only_arrays([outputs[3], outputs[4], outputs[5]])
    lms_deltas = only_arrays([outputs[6], outputs[7], outputs[8]])
    anchors_per_stride, strides_per_stride = generate_anchors_per_stride(input_size=input_size, strides=[8, 16, 32])

    synced = [
        (s, b, l, a, st)
        for s, b, l, a, st in zip(scores, bboxes, lms_deltas, anchors_per_stride, strides_per_stride)
        if isinstance(s, np.ndarray) and isinstance(b, np.ndarray) and isinstance(l, np.ndarray)
        and s.shape[0] == b.shape[0] == l.shape[0] == a.shape[0] == st.shape[0] and s.shape[0] > 0
    ]
    if not synced:
        raise HTTPException(status_code=404, detail="No valid predictions")

    all_scores = np.concatenate([s.reshape(-1) for s, _, _, _, _ in synced])
    all_bboxes = np.concatenate([b.reshape(-1, 4) for _, b, _, _, _ in synced])
    all_lms_deltas = np.concatenate([l.reshape(-1, 10) for _, _, l, _, _ in synced])
    all_anchors = np.concatenate([a for _, _, _, a, _ in synced])
    all_strides = np.concatenate([st for _, _, _, _, st in synced])

    # Decode
    decoded_bboxes = decode_bboxes(all_anchors, all_bboxes, all_strides)
    decoded_landmarks = decode_landmarks(all_anchors, all_lms_deltas, all_strides)

    # Filter by score
    mask = all_scores > 0.1
    if not np.any(mask):
        raise HTTPException(status_code=404, detail="No face detected")
    indices = np.where(mask)[0]

    # Optionally: NMS (not applied here, but can be added)
    boxes = decoded_bboxes[indices]
    scores_filtered = all_scores[indices]

    # Use numpy NMS only
    nms_indices = numpy_nms(boxes, scores_filtered, iou_threshold=0.2)
    if len(nms_indices) == 0:
        raise HTTPException(status_code=404, detail="No face detected after NMS")
    indices = indices[nms_indices]

    # Cluster by center distance to suppress repeated detections
    cluster_indices = cluster_boxes_by_center(boxes[nms_indices], scores_filtered[nms_indices], distance_thresh=60)
    indices = indices[cluster_indices]

    # Before NMS
    img_before_nms = img.copy()
    for box in boxes.astype(int):
        cv2.rectangle(img_before_nms, (box[0], box[1]), (box[2], box[3]), (255, 0, 0), 2)
    cv2.imwrite("debug_boxes_before_nms.jpg", cv2.cvtColor(img_before_nms, cv2.COLOR_RGB2BGR))

    # After NMS and clustering, select the highest score face only
    if len(indices) > 1:
        best_idx = np.argmax(all_scores[indices])
        indices = indices[[best_idx]]

    idx = indices[0]
    lms = decoded_landmarks[idx]
    scale_x = orig_w / input_size[0]
    scale_y = orig_h / input_size[1]
    lms[:, 0] *= scale_x
    lms[:, 1] *= scale_y

    std_landmarks = np.array([[38.2946, 51.6963],
                              [73.5318, 51.5014],
                              [56.0252, 71.7366],
                              [41.5493, 92.3655],
                              [70.7299, 92.2041]], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(lms, std_landmarks)[0]
    aligned = cv2.warpAffine(img, M, (112, 112), borderValue=0)
    aligned_landmarks = cv2.transform(lms[None, :, :], M)[0]
    aligned_base64 = image_to_base64(aligned)
    wearing_sunglasses = is_wearing_sunglasses(aligned, aligned_landmarks, debug_prefix="debug", face_idx=0)
    landmark_names = ["left_eye", "right_eye", "nose", "left_mouth", "right_mouth"]
    landmarks = [Landmark(name=landmark_names[j], coordinates=lms[j].tolist()) for j in range(5)]
    return SingleFaceResponse(
        score=float(all_scores[idx]),
        landmarks=landmarks,
        aligned_base64=aligned_base64,
        wearing_sunglasses=wearing_sunglasses,
        template=None
    )
