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
