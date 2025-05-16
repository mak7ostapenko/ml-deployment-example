import base64

import cv2
import numpy as np
from fastapi import HTTPException

from src.model.model import DepthEstimationModel


def predict_depth(img_b64: bytes, model: DepthEstimationModel):
    
    img_bytes = base64.b64decode(img_b64)

    if img_bytes is None or len(img_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Image bytes are empty."
        )
    
    img_array = np.frombuffer(img_bytes, dtype=np.uint8)

    if not isinstance(img_array, np.ndarray):
        raise HTTPException(
            status_code=400,
            detail="Image is not a numpy array."
        )
    if len(img_array.shape) < 1:
        raise HTTPException(
            status_code=400,
            detail="Image is not a valid numpy array."
        )
    
    img = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if len(img.shape) < 2:
        raise HTTPException(
            status_code=400,
            detail="Image is has less then 2 dimensions."
        )
    
    points_3d, confidence, intrinsics, depth = model.predict_img(img=img)

    if points_3d is None or confidence is None or intrinsics is None or depth is None:
        raise HTTPException(
            status_code=500,
            detail="Model prediction failed."
        )
    
    return points_3d, confidence, intrinsics, depth