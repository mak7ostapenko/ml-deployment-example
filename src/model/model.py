from typing import Tuple

import numpy as np
import onnxruntime as ort

from src.model.unidepth_utils import (
    preprocess_unidepth_input, 
    postprocess_intrinsics, 
    postprocess_unidepth_out
)


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

    def predict_img(
        self, 
        img: np.ndarray,
        normalize: bool = True,
        interpolation_mode: str = "bilinear"
    ) -> np.ndarray:
        img_batch = np.array([img])

        out = self.predict_batch(
            img_batch=img_batch,
            normalize=normalize,
            interpolation_mode=interpolation_mode
        )
        
        points_3d, confidence, intrinsics, depth = out[0][0], out[1][0], out[2][0], out[3][0]
        return points_3d, confidence, intrinsics, depth
    
    def predict_batch(
        self, 
        img_batch: np.ndarray,
        normalize: bool = True,
        interpolation_mode: str = "bilinear"
    ) -> np.ndarray:
        B, H, W, C = img_batch.shape

        # preprocess
        img_batch, (paddings, padded_dims, resize_factors) = preprocess_unidepth_input(
            self.input_size, img_batch, normalize=normalize
        )

        # run model
        preds = self._model.run(
            None, {self._model.get_inputs()[0].name: img_batch.astype(np.float32)}
        ) # -> pts_3d, outputs["confidence"], outputs["intrinsics"]

        points_3d, confidence, intrinsics = preds
        
        # postprocessing
        confidence = postprocess_unidepth_out(
            confidence,
            padded_dims,
            paddings=paddings,
            interpolation_mode=interpolation_mode,
        )
        points_3d = postprocess_unidepth_out(
            points_3d,
            padded_dims,
            paddings=paddings,
            interpolation_mode=interpolation_mode,
        )
        intrinsics = postprocess_intrinsics(
            intrinsics, [resize_factors] * B, [paddings] * B
        )
        depth = points_3d[:, :, :, 2]

        return points_3d, confidence, intrinsics, depth