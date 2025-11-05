import importlib
import os


def env_settings():
    env_module = importlib.import_module('lib.train.admin.local')
    env_cls = getattr(env_module, 'EnvironmentSettings')
    return env_cls()


def create_default_local_file_ITP_train(workspace_dir, data_dir):
    path = os.path.join(os.path.dirname(__file__), 'local.py')
    tensorboard_dir = os.path.join(workspace_dir, 'tensorboard')
    pretrained_dir = os.path.join(workspace_dir, 'pretrained_models')
    content = f"""class EnvironmentSettings:\n    def __init__(self):\n        self.workspace_dir = '{workspace_dir}'\n        self.tensorboard_dir = '{tensorboard_dir}'\n        self.pretrained_models = '{pretrained_dir}'\n        self.unimod1k_dir = '{data_dir}'\n        self.unimod1k_dir_nlp = '{data_dir}'\n"""
    with open(path, 'w') as f:
        f.write(content)
