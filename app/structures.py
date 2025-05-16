from dataclasses import dataclass


@dataclass
class BaseAppCfg:
    app_name: str
    app_version: str = '0.1'
    hostname: str = 'localhost'
    http_port: int = 8000
    proj_root: str = ''
    data_folder: str = 'data/'
    log_dir: str = ''
    log_name: str = ''


@dataclass
class DepthEsimationAppCfg(BaseAppCfg):
    resources_url: str = None
    checkpoints_path: str = 'checkpoints/model.onnx'
    model_cfg_path: str = ''
    device: str = 'cpu'
