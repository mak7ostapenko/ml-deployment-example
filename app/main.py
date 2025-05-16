import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Body

from app import predict  
from app.config import app_cfg 
from app.schemas import ImgItem, DepthEstimationResponse

from src.utils.logging import get_logger
from src.utils.image import encode_to_base64
from src.model.model import DepthEstimationModel


logger: logging.Logger = get_logger(
    log_dir=app_cfg.log_dir, 
    log_name=app_cfg.log_name
)
logger.info('Application is starting...')
logger.info(f'{app_cfg = }')


@asynccontextmanager
async def lifespan(app_: FastAPI):
    try:
        app_.state.rec_model = DepthEstimationModel(
            checkpoint_path=app_cfg.checkpoints_path,
        )
    except Exception as e:
        logger.critical(f"During models startrap following exception occured '{str(e)}'")

    logger.info('Model is loaded.')
    yield 


app = FastAPI(
    title=app_cfg.app_name,
    version=app_cfg.app_version,
    lifespan=lifespan
)


@app.get("/")
async def root():
    return {"msg": app_cfg.app_name}


@app.post("/depth/predict")
async def predict_depth(img_item: ImgItem = Body(...)) -> DepthEstimationResponse:

    points_3d, confidence, intrinsics, depth = predict.predict_depth(
        img_b64=img_item.img_b64, model=app.state.rec_model
    )

    response = DepthEstimationResponse(
        server_name=app_cfg.app_name,
        points_3d_b64=encode_to_base64(points_3d),
        confidence_b64=encode_to_base64(confidence),
        intrinsics_b64=encode_to_base64(intrinsics),
        depth_b64=encode_to_base64(depth)
    )
    return response.model_dump()