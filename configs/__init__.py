import importlib
import os.path as osp


def get_config(config_file=None):
    config = importlib.import_module("configs.base")
    cfg = config.config
    if config_file is None:
        return cfg
    assert config_file.startswith('configs/'), 'config file setting must start with configs/'
    temp_config_name = config_file.replace('/', '.')
    temp_module_name = osp.splitext(temp_config_name)[0]
    config = importlib.import_module(temp_module_name)
    job_cfg = config.config
    cfg.update(job_cfg)
    return cfg
