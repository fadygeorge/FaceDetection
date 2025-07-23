# FastAPI service for face quality evaluation using SCRFD
# Requires: pip install fastapi uvicorn opencv-python onnxruntime numpy pillow

import cv2
import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import base64
import onnxruntime as ort
from typing import List, Optional
from io import BytesIO
from PIL import Image
import math

app = FastAPI(title="Face Quality Evaluation API")


class FaceQualityResponse(BaseModel):
    is_acceptable: bool
    score: int
    reasons: List[str]
    image_base64: Optional[str] = None


class ImageRequest(BaseModel):
    image_base64: str


# === Load SCRFD ONNX model ===
scrfd_session = ort.InferenceSession("scrfd_10g_bnkps.onnx", providers=["CPUExecutionProvider"])
input_size = (640, 640)


def base64_to_image(base64_str: str) -> np.ndarray:
    try:
        img_data = base64.b64decode(base64_str)
        image = Image.open(BytesIO(img_data)).convert("RGB")
        return np.array(image)
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid base64 image")


def image_to_base64(image: np.ndarray) -> str:
    _, buffer = cv2.imencode('.jpg', image)
    return base64.b64encode(buffer).decode("utf-8")


def generate_anchors_per_stride(input_size=(640, 640), strides=[8, 16, 32]):
    all_anchors = []
    for stride in strides:
        fm_h = math.ceil(input_size[1] / stride)
        fm_w = math.ceil(input_size[0] / stride)
        anchors = []
        for i in range(fm_h):
            for j in range(fm_w):
                anchor_cx = (j + 0.5) * stride
                anchor_cy = (i + 0.5) * stride
                # SCRFD uses 2 anchors per cell
                anchors.append([anchor_cx, anchor_cy])
                anchors.append([anchor_cx, anchor_cy])
        all_anchors.append(np.array(anchors, dtype=np.float32))
    return all_anchors


def decode_bboxes(anchors, deltas):
    # deltas: (N, 4), anchors: (N, 2)
    # SCRFD uses [dx, dy, dw, dh]
    anchor_cx, anchor_cy = anchors[:, 0], anchors[:, 1]
    dx, dy, dw, dh = deltas[:, 0], deltas[:, 1], deltas[:, 2], deltas[:, 3]
    pred_cx = dx * anchor_cx + anchor_cx
    pred_cy = dy * anchor_cy + anchor_cy
    pred_w = np.exp(dw) * anchor_cx
    pred_h = np.exp(dh) * anchor_cy
    x1 = pred_cx - pred_w / 2
    y1 = pred_cy - pred_h / 2
    x2 = pred_cx + pred_w / 2
    y2 = pred_cy + pred_h / 2
    return np.stack([x1, y1, x2, y2], axis=1)


def decode_landmarks(anchors, lms_deltas):
    # lms_deltas: (N, 10), anchors: (N, 2)
    lms = []
    for i in range(5):
        dx = lms_deltas[:, 2 * i]
        dy = lms_deltas[:, 2 * i + 1]
        x = dx * anchors[:, 0] + anchors[:, 0]
        y = dy * anchors[:, 1] + anchors[:, 1]
        lms.append(np.stack([x, y], axis=1))
    lms = np.stack(lms, axis=1)  # (N, 5, 2)
    return lms


def detect_faces(image: np.ndarray) -> Optional[np.ndarray]:
    orig_h, orig_w = image.shape[:2]
    resized = cv2.resize(image, input_size)
    input_blob = resized[:, :, ::-1].astype(np.float32)
    input_blob = (input_blob - 127.5) / 128
    input_blob = input_blob.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    input_name = scrfd_session.get_inputs()[0].name
    outputs = scrfd_session.run(None, {input_name: input_blob})

    def only_arrays(lst):
        return [x for x in lst if isinstance(x, np.ndarray)]

    anchors_per_stride = generate_anchors_per_stride(input_size=input_size, strides=[8, 16, 32])
    scores = [s.squeeze(-1) if len(s.shape) == 2 and s.shape[1] == 1 else s for s in only_arrays([outputs[0], outputs[1], outputs[2]])]
    bboxes = only_arrays([outputs[3], outputs[4], outputs[5]])
    lms_deltas = only_arrays([outputs[6], outputs[7], outputs[8]])
    # anchors_per_stride is already correct

    print("scores shapes:", [s.shape for s in scores])
    print("bboxes shapes:", [b.shape for b in bboxes])
    print("lms_deltas shapes:", [l.shape for l in lms_deltas])
    print("anchors shapes:", [a.shape for a in anchors_per_stride])

    # Synchronize all lists by shape
    synced = [
        (s, b, l, a)
        for s, b, l, a in zip(scores, bboxes, lms_deltas, anchors_per_stride)
        if isinstance(s, np.ndarray) and isinstance(b, np.ndarray) and isinstance(l, np.ndarray)
        and s.shape[0] == b.shape[0] == l.shape[0] == a.shape[0] and s.shape[0] > 0
    ]

    print("Number of synced predictions:", len(synced))

    if not synced:
        return None  # No valid predictions

    all_scores = np.concatenate([s.reshape(-1) for s, _, _, _ in synced])
    all_bboxes = np.concatenate([b.reshape(-1, 4) for _, b, _, _ in synced])
    all_lms_deltas = np.concatenate([l.reshape(-1, 10) for _, _, l, _ in synced])
    all_anchors = np.concatenate([a for _, _, _, a in synced])

    # Now decode using all_anchors and the concatenated outputs
    decoded_bboxes = decode_bboxes(all_anchors, all_bboxes)
    decoded_landmarks = decode_landmarks(all_anchors, all_lms_deltas)

    print("Top 10 scores:", all_scores[:10])
    print("Max score:", np.max(all_scores))

    # Filter by score threshold
    mask = all_scores > 0.5
    if not np.any(mask):
        return None
    best_idx = np.argmax(all_scores * mask)
    best_score = all_scores[best_idx]

    print("Best score:", best_score)
    print("Decoded landmarks (pixel):", decoded_landmarks[best_idx])
    print("Decoded box (pixel):", decoded_bboxes[best_idx])

    lms = decoded_landmarks[best_idx]  # (5, 2)
    # Scale landmarks to original image size if needed
    scale_x = orig_w / input_size[0]
    scale_y = orig_h / input_size[1]
    lms[:, 0] *= scale_x
    lms[:, 1] *= scale_y
    return lms
    

def align_face(img: np.ndarray, lms: np.ndarray, size: int = 112) -> np.ndarray:
    std = np.array([[38.2946, 51.6963],
                    [73.5318, 51.5014],
                    [56.0252, 71.7366],
                    [41.5493, 92.3655],
                    [70.7299, 92.2041]], dtype=np.float32)
    M = cv2.estimateAffinePartial2D(lms, std)[0]
    print("Affine matrix:", M)
    return cv2.warpAffine(img, M, (size, size), borderValue=0)


def is_blurry(img: np.ndarray, thresh=0.5) -> tuple[bool, float]:
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return lap < thresh, lap


def is_frontal(lms: np.ndarray, max_dev: float = 30.0) -> tuple[bool, float]:
    eye_diff = abs(lms[0][1] - lms[1][1])
    return eye_diff < max_dev, eye_diff


def is_wearing_glasses(img: np.ndarray, lms: np.ndarray) -> bool:
    # Use the first two landmarks as eye centers
    left_eye = lms[0].astype(int)
    right_eye = lms[1].astype(int)
    eye_box_size = 15
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    h, w = img.shape[:2]
    left_eye[0] = np.clip(left_eye[0], 0, w-1)
    left_eye[1] = np.clip(left_eye[1], 0, h-1)
    right_eye[0] = np.clip(right_eye[0], 0, w-1)
    right_eye[1] = np.clip(right_eye[1], 0, h-1)
    left_eye_region = gray[
        max(0, left_eye[1] - eye_box_size):left_eye[1] + eye_box_size,
        max(0, left_eye[0] - eye_box_size):left_eye[0] + eye_box_size
    ]
    right_eye_region = gray[
        max(0, right_eye[1] - eye_box_size):right_eye[1] + eye_box_size,
        max(0, right_eye[0] - eye_box_size):right_eye[0] + eye_box_size
    ]
    if left_eye_region.size > 0:
        print("Left eye region stats: min", left_eye_region.min(), "max", left_eye_region.max(), "mean", left_eye_region.mean())
    if right_eye_region.size > 0:
        print("Right eye region stats: min", right_eye_region.min(), "max", right_eye_region.max(), "mean", right_eye_region.mean())
    print("Left eye coords:", lms[0])
    print("Right eye coords:", lms[1])
    left_dark = np.mean(left_eye_region < 80) if left_eye_region.size > 0 else 0
    right_dark = np.mean(right_eye_region < 80) if right_eye_region.size > 0 else 0
    return bool((left_dark > 0.25) or (right_dark > 0.25))


def compute_score(blur_val, frontal, glasses):
    score = 100
    if blur_val < 80:
        score -= 30
    if not frontal:
        score -= 30
    if glasses:
        score -= 10
    return max(0, min(100, score))


def draw_landmarks(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    for (x, y) in landmarks.astype(int):
        cv2.circle(image, (x, y), radius=3, color=(0, 255, 0), thickness=-1)
    return image



@app.post("/evaluate", response_model=FaceQualityResponse)
def evaluate(req: ImageRequest):
    img = base64_to_image(req.image_base64)
    img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

    lms = detect_faces(img_bgr)
    if lms is None:
        return FaceQualityResponse(is_acceptable=False, score=0, reasons=["No face detected"])
    
    print("Detected landmarks (before alignment):", lms)
    img_with_landmarks = draw_landmarks(img_bgr.copy(), lms)
    cv2.imwrite("original_with_landmarks.jpg", img_with_landmarks)

    # Example: reorder indices to match std
    lms = lms[[1, 0, 2, 4, 3], :]

    aligned = align_face(img_bgr, lms.astype(np.float32))
    cv2.imwrite("aligned.jpg", aligned)

    blurry, blur_val = is_blurry(aligned)
    frontal, eye_dev = is_frontal(lms)
    glasses = is_wearing_glasses(aligned, lms)

    reasons = []
    if blurry:
        reasons.append(f"Blurry image (score={blur_val:.2f})")
    if not frontal:
        reasons.append(f"Face not frontal (eye diff={eye_dev:.2f})")
    if glasses:
        reasons.append("Person is wearing glasses")

    score = compute_score(blur_val, frontal, glasses)
    return FaceQualityResponse(
        is_acceptable=len(reasons) == 0,
        score=score,
        reasons=reasons,
        image_base64=image_to_base64(aligned) if len(reasons) == 0 else None
    )
