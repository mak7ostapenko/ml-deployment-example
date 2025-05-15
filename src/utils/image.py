import base64
from typing import Tuple

import cv2
import numpy as np


def encode_to_byteimg(
    bgr_img: np.ndarray, img_ext: str = "jpg"
) -> bytes:
    _, buffer = cv2.imencode("." + img_ext, bgr_img)
    img_in_base64 = base64.b64encode(buffer).decode("utf-8")
    return img_in_base64


def decode_byteimg(byte_img: np.ndarray) -> np.ndarray:
    """ NOTE: returns in BGR format
    """
    byte_img = base64.b64decode(byte_img)
    byte_img = np.frombuffer(byte_img, dtype=np.uint8)
    bgr_img = cv2.imdecode(byte_img, cv2.IMREAD_COLOR)
    return bgr_img


def scaled_resize(
    img: np.ndarray, target_size: Tuple, grayscale: bool = False,
) -> np.ndarray:
    if img.shape[0:2] == target_size: 
        return img

    if img.shape[0] > 0 and img.shape[1] > 0:
        factor_0 = target_size[0] / img.shape[0]
        factor_1 = target_size[1] / img.shape[1]
        factor = min(factor_0, factor_1)

        dsize = (
            int(img.shape[1] * factor),
            int(img.shape[0] * factor),
        )
        img = cv2.resize(img, dsize)

        diff_0 = target_size[0] - img.shape[0]
        diff_1 = target_size[1] - img.shape[1]
        if grayscale is False:
            # Put the base image in the middle of the padded image
            img = np.pad(
                img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                    (0, 0),
                ),
                "constant",
            )
        else:
            img = np.pad(
                img,
                (
                    (diff_0 // 2, diff_0 - diff_0 // 2),
                    (diff_1 // 2, diff_1 - diff_1 // 2),
                ),
                "constant",
            )

    # double check: if target image is not still the same size with target.
    if img.shape[0:2] != target_size:
        img = cv2.resize(img, target_size)

    return img


def make_square_img_with_borders(
    img: np.ndarray, return_padding: bool = False
):
    height, width = img.shape[:2]

    if height > width:
        target_size = (height, height)
    else:
        target_size = (width, width)
    
    diff_h = target_size[0] - img.shape[0]
    diff_w = target_size[1] - img.shape[1]
    
    padd_top = diff_h // 2
    padd_bottom = diff_h - padd_top
    padd_left = diff_w // 2
    padd_right = diff_w - padd_left

    img = np.pad(
        img,
        (
            (padd_top, padd_bottom),
            (padd_right, padd_left),
            (0, 0),
        ),
        "constant",
    )
    assert img.shape[0] == img.shape[1], "Image is not square."
    
    if return_padding:
        return img, (padd_top, padd_bottom, padd_left, padd_right)
    else:
        return img


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