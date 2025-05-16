import cv2
import pytest
import numpy as np

from app.config import app_cfg
from src.model.model import DepthEstimationModel 


@pytest.fixture
def rec_model():
    model = DepthEstimationModel(checkpoint_path=app_cfg.checkpoints_path)
    yield model

def load_img():
    img_path = 'assets/room.jpg'
    return cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)


def test_predict_img(rec_model: DepthEstimationModel):
    img_rgb: np.ndarray = load_img()
    
    assert len(img_rgb.shape) > 1 
    
    points_3d, confidence, intrinsics, depth = rec_model.predict_img(img=img_rgb)

    h, w = img_rgb.shape[:2]
    assert isinstance(points_3d, np.ndarray)
    assert isinstance(confidence, np.ndarray)
    assert isinstance(intrinsics, np.ndarray)
    assert isinstance(depth, np.ndarray)
    
    assert points_3d.shape == (h, w, 3)
    assert confidence.shape == (h, w)
    assert intrinsics.shape == (3, 3)
    assert depth.shape == (h, w)


def test_predict_batch(rec_model: DepthEstimationModel):
    img_rgb: np.ndarray = load_img()
    
    assert len(img_rgb.shape) > 1 
    
    img_batch = np.array([img_rgb, img_rgb])
    points_3d, confidence, intrinsics, depth = rec_model.predict_batch(img_batch=img_batch)

    h, w = img_rgb.shape[:2]
    assert isinstance(points_3d, np.ndarray)
    assert isinstance(confidence, np.ndarray)
    assert isinstance(intrinsics, np.ndarray)
    assert isinstance(depth, np.ndarray)
    
    assert points_3d.shape == (2, h, w, 3)
    assert confidence.shape == (2, h, w)
    assert intrinsics.shape == (2, 3, 3)
    assert depth.shape == (2, h, w)
