import pathlib

from app.structures import DepthEsimationAppCfg

IS_IN_DOCKER = True
project_root = pathlib.Path(__file__).parent.parent.resolve()

base_config = {
    "app_name": "depth_estimation_app",
    "app_version": "0.1",
    "log_dir": "logs/",
    "log_name": "depth_esimation_app.log",
    "resources_url": None,
    "checkpoints_path": "checkpoints/unidepthv2_vits_462_630.onnx",
    "proj_root": project_root,
    "data_folder": "data/",
    "device": "cuda",
}

env_config = {
    "hostname": "0.0.0.0" if IS_IN_DOCKER else "localhost",
    "http_port": 80 if IS_IN_DOCKER else 8001,
}

app_cfg = DepthEsimationAppCfg(**base_config, **env_config)
