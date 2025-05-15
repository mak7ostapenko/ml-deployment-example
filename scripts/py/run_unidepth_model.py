import os
import sys

proj_root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
sys.path.append(proj_root_dir)

import argparse

import cv2

from src.model.model import DepthEstimationModel
from src.utils import visualization


def main(img_path: str, model_path: str):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model = DepthEstimationModel(checkpoint_path=model_path)
    points_3d, confidence, intrinsics, depth = model.predict_img(img=img)

    print(f"Depth prediction shape: {depth.shape}")
    print('in shape:', img.shape)
    depth_pred_col = visualization.colorize(depth, vmin=0.01, vmax=10.0, cmap="magma_r")
    
    cv2.imshow("Input Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Depth Prediction", cv2.cvtColor(depth_pred_col, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the UniDepth model.")
    parser.add_argument(
        "--model_path",
        type=str,
        default="checkpoints/unidepthv2_vits_462_630.onnx",
        help="Path to the weights file.",
    )
    parser.add_argument(
        "--image_path",
        type=str,
        default="data/coridor.jpg",
        help="Path to the input image.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    img_path = args.image_path

    main(img_path=img_path, model_path=model_path)
    
    

