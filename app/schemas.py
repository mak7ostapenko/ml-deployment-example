from typing import List

from pydantic import BaseModel


class ImgItem(BaseModel):
    img_b64: str    # img in base 64

class DepthEstimationResponse(BaseModel):
    server_name: str
    points_3d_b64: str        
    confidence_b64: str      
    intrinsics_b64: str      
    depth_b64: str     
