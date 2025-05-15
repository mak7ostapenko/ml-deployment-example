import cv2
import numpy as np

from src.utils.image import get_paddings, get_resize_factor


IMAGENET_DATASET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DATASET_STD = (0.229, 0.224, 0.225)


def preprocess_unidepth_input(
    target_shape: tuple, img_batch: np.ndarray, normalize: bool = True
):
    B, H, W, C = img_batch.shape
        
    paddings, (padded_H, padded_W) = get_paddings((H, W), target_shape=target_shape)
    (pad_left, pad_right, pad_top, pad_bottom) = paddings
    resize_factor_H, resize_factor_W = get_resize_factor(
        (padded_H, padded_W), target_shape=target_shape
    )

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
            padded_img, (target_shape[1], target_shape[0]), interpolation=cv2.INTER_LINEAR
        )
        processed_batch.append(resized_img)

    img_batch = np.stack(processed_batch, axis=0)  # (N, H, W, C)

    if normalize:
        img_batch = img_batch / 255.0  # Scale to [0, 1]
        mean = np.array(IMAGENET_DATASET_MEAN)
        std = np.array(IMAGENET_DATASET_STD)
        img_batch = (img_batch - mean) / std
    
    img_batch = np.transpose(img_batch, (0, 3, 1, 2))
    return img_batch, (paddings, (padded_H, padded_W), (resize_factor_H, resize_factor_W))
    

def postprocess_unidepth_out(batch, shapes, paddings, interpolation_mode="bilinear"):
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
    
    return np.array(result)

def postprocess_intrinsics(K, resize_factors, paddings):
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