import sys
sys.path.append("../../")

import os
import argparse

import cv2

import numpy as np

from src.utils.image import scaled_resize


import os
from typing import List, Tuple

import onnxruntime as ort


import math


def get_paddings(original_shape, target_shape):
    H_ori, W_ori = original_shape
    orig_aspect_ratio = W_ori / H_ori

    H_target, W_target = target_shape
    target_aspect_ratio = W_target / H_target

    if orig_aspect_ratio > target_aspect_ratio:  # Too wide
        W_new = W_ori
        H_new = int(W_ori / target_aspect_ratio)
        pad_top = (H_new - H_ori) // 2
        pad_bottom = H_new - H_ori - pad_top
        pad_left, pad_right = 0, 0
    else:  # Too tall
        H_new = H_ori
        W_new = int(H_ori * target_aspect_ratio)
        pad_left = (W_new - W_ori) // 2
        pad_right = W_new - W_ori - pad_left
        pad_top, pad_bottom = 0, 0

    return (pad_left, pad_right, pad_top, pad_bottom), (H_new, W_new)


def get_resize_factor(original_shape: Tuple[int, int], target_shape: Tuple[int, int]) -> Tuple[int, int]:
    resize_factor_h = target_shape[0] / original_shape[0]
    resize_factor_w = target_shape[1] / original_shape[1]
    return (resize_factor_h, resize_factor_w)


IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)

def _postprocess(batch, shapes, paddings, interpolation_mode="bilinear"):
    # Map interpolation mode to OpenCV constants
    if interpolation_mode == "bilinear":
        cv_interp = cv2.INTER_LINEAR
    elif interpolation_mode == "nearest":
        cv_interp = cv2.INTER_NEAREST
    elif interpolation_mode == "bicubic":
        cv_interp = cv2.INTER_CUBIC
    else:
        cv_interp = cv2.INTER_LINEAR
    
    pad_left, pad_right, pad_top, pad_bottom = paddings

    result = []
    batch_size = batch.shape[0]
    for i in range(batch_size):
        img = batch[i].transpose(1, 2, 0)  # Convert from (C, H, W) to (H, W, C)
        item_resized = cv2.resize(img, (shapes[1], shapes[0]), interpolation=cv_interp)
        # Remove paddings
        item_unpad = item_resized[pad_top:shapes[0]-pad_bottom, pad_left:shapes[1]-pad_right]   
        result.append(item_unpad)
    
    result = np.array(result)
    return result

def _postprocess_intrinsics(K, resize_factors, paddings):
    batch_size = K.shape[0]
    K_new = K.copy()

    for i in range(batch_size):
        scale_h, scale_w = resize_factors[i]
        pad_l, _, pad_t, _ = paddings[i]

        K_new[i, 0, 0] /= scale_w  # fx
        K_new[i, 1, 1] /= scale_h  # fy
        K_new[i, 0, 2] /= scale_w  # cx
        K_new[i, 1, 2] /= scale_h  # cy

        K_new[i, 0, 2] -= pad_l  # cx
        K_new[i, 1, 2] -= pad_t  # cy

    return K_new


class DepthEstimationModel:
    input_size: Tuple[int] = (462, 630)
    model_name: str = "unidepthv2_vits_462_630"

    def __init__(self, checkpoint_path: str) -> None:
        self.checkpoint_path = checkpoint_path

        # Start from ORT 1.10, ORT requires explicitly setting the providers parameter if you want to use execution providers
        # other than the default CPU provider (as opposed to the previous behavior of providers getting set/registered by default
        # based on the build flags) when instantiating InferenceSession.
        # Following code assumes NVIDIA GPU is available, you can specify other execution providers or don't include providers parameter
        # to use default CPU provider.
        self._model = ort.InferenceSession(self.checkpoint_path, providers=["CUDAExecutionProvider"])
        # self._model = ort.InferenceSession(self._model_path)

    def predict_img(self, img: np.ndarray) -> np.ndarray:
        img_batch = np.array([img])

        out = self.predict_batch(img_batch=img_batch)
        
        points_3d, confidence, intrinsics, depth = out[0][0], out[1][0], out[2][0], out[3][0]
        
        # depth = depth.squeeze(0)
        return depth
    
    def predict_batch(self, img_batch: np.ndarray) -> np.ndarray:
        # Process batch maintaining original format until the end
        B, H, W, C = img_batch.shape
        
        paddings, (padded_H, padded_W) = get_paddings((H, W), target_shape=self.input_size)
        (pad_left, pad_right, pad_top, pad_bottom) = paddings
        resize_factor_H, resize_factor_W = get_resize_factor(
            (padded_H, padded_W), target_shape=self.input_size
        )

        # Pad and resize each image in the batch
        processed_batch = []
        for i in range(B):
            # Work directly with (H, W, C) format
            padded_img = np.pad(
                img_batch[i], 
                pad_width=((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)), 
                mode='constant', 
                constant_values=0.0
            )
            resized_img = cv2.resize(
                padded_img, (self.input_size[1], self.input_size[0]), interpolation=cv2.INTER_LINEAR
            )
            processed_batch.append(resized_img)

        img_batch = np.stack(processed_batch, axis=0)  # (N, H, W, C)
        
        # Normalize
        normalize = True
        if normalize:
            img_batch = img_batch / 255.0  # Scale to [0, 1]
            mean = np.array(IMAGENET_DATASET_MEAN)
            std = np.array(IMAGENET_DATASET_STD)
            img_batch = (img_batch - mean) / std
        
        img_batch = np.transpose(img_batch, (0, 3, 1, 2))
        
        # run model
        preds = self._model.run(
            None, {self._model.get_inputs()[0].name: img_batch.astype(np.float32)}
        ) # -> pts_3d, outputs["confidence"], outputs["intrinsics"]

        points_3d, confidence, intrinsics = preds
        
        # postprocessing
        self.interpolation_mode = "bilinear"

        confidence = _postprocess(
            confidence,
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        points_3d = _postprocess(
            points_3d,
            (padded_H, padded_W),
            paddings=paddings,
            interpolation_mode=self.interpolation_mode,
        )
        intrinsics = _postprocess_intrinsics(
            intrinsics, [(resize_factor_H, resize_factor_W)] * B, [paddings] * B
        )
        depth = points_3d[:, :, :, 2]
        return points_3d, confidence, intrinsics, depth

import matplotlib.pyplot as plt


def colorize(
    value: np.ndarray, vmin: float = None, vmax: float = None, cmap: str = "magma_r"
):
    # if already RGB, do nothing
    if value.ndim > 2:
        if value.shape[-1] > 1:
            return value
        value = value[..., 0]
    invalid_mask = value < 0.0001
    # normalize
    vmin = value.min() if vmin is None else vmin
    vmax = value.max() if vmax is None else vmax
    value = (value - vmin) / (vmax - vmin)  # vmin..vmax

    # set color
    cmapper = plt.get_cmap(cmap)
    value = cmapper(value, bytes=True)  # (nxmx4)
    value[invalid_mask] = 0
    img = value[..., :3]
    return img


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
        # default="data/room.jpg",
        default="data/coridor.jpg",
        help="Path to the input image.",
    )
    args = parser.parse_args()

    model_path = args.model_path
    img_path = args.image_path

    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    model = DepthEstimationModel(checkpoint_path=model_path)
    depth_pred = model.predict_img(img=img)
    # depth_pred = model.predict_batch(np.array([img, ]))
    print(f"Depth prediction shape: {depth_pred.shape}")
    print('in shape:', img.shape)
    depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
    
    cv2.imshow("Input Image", cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imshow("Depth Prediction", cv2.cvtColor(depth_pred_col, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

