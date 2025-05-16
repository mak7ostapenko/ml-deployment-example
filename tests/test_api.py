import cv2
import numpy as np
from fastapi.testclient import TestClient

from app.config import app_cfg
from app.main import app as server_app

from src.utils.image import encode_to_base64, decode_from_base64


def load_and_encode_img():
    img_path = 'assets/room.jpg'
    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    img_b64 = encode_to_base64(img)
    return img_b64


def test_api_root():
    with TestClient(server_app) as client:
        resp = client.get("/")

        assert resp.status_code == 200
        data = resp.json()
        assert data["msg"] == app_cfg.app_name
    

def test_depth_predict_api():
    img_b64 = load_and_encode_img()
    assert isinstance(img_b64, str)

    with TestClient(server_app) as client:
        resp = client.post(
            "/depth/predict", 
            json={'img_b64': img_b64}
        )
        
        assert resp.status_code == 200

        data = resp.json()
        assert data["server_name"] == app_cfg.app_name

        assert isinstance(data["points_3d_b64"], str)
        assert isinstance(data["confidence_b64"], str)
        assert isinstance(data["intrinsics_b64"], str)
        assert isinstance(data["depth_b64"], str)

        points_3d_b64 = decode_from_base64(data["points_3d_b64"])
        confidence_b64 = decode_from_base64(data["confidence_b64"])
        intrinsics_b64 = decode_from_base64(data["intrinsics_b64"])
        depth_b64 = decode_from_base64(data["depth_b64"])
        
        assert isinstance(points_3d_b64, np.ndarray)
        assert isinstance(confidence_b64, np.ndarray)
        assert isinstance(intrinsics_b64, np.ndarray)
        assert isinstance(depth_b64, np.ndarray)
        

        