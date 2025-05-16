import sys
import pathlib

project_root = pathlib.Path(__file__).resolve().parent.parent
sys.path.append(str(project_root))

import os
import cv2
import locust 

from app.config import app_cfg
from src.utils.image import encode_to_base64

img_path = project_root / "assets" / "room.jpg"
img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
img_b64 = encode_to_base64(img) 

class DepthEstimationCheck(locust.HttpUser):
    host = f"http://{app_cfg.hostname}:{app_cfg.http_port}"  
    wait_time = locust.between(1, 5)  

    @locust.task
    def api_depth_estimation_check(self):
        headers = {
            "Content-Type": "application/json",
        }
        self.client.post(
            "/depth/predict",  
            json={'img_b64': img_b64}, 
            headers=headers
        )
