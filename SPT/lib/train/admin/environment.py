import importlib


class EnvSettings:
    """Base environment settings. Values are intended to be overridden in local.py."""

    def __init__(self):
        self.workspace_dir = ''
        self.tensorboard_dir = ''
        self.pretrained_models = ''
        self.unimod1k_dir = ''
        self.unimod1k_dir_nlp = ''


def env_settings():
    """Load user-defined EnvironmentSettings from lib.train.admin.local."""
    env_module = importlib.import_module('lib.train.admin.local')
    env_cls = getattr(env_module, 'EnvironmentSettings')
    return env_cls()
