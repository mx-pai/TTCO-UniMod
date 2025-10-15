from lib.train.admin.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.description = 'Default train settings'
    settings.project_path = 'train/{script_name}/{config_name}'
    settings.workspace_dir = '/root/autodl-tmp/UniMod1K/SPT'    # Base directory for saving network checkpoints.
    settings.tensorboard_dir = '/root/autodl-tmp/UniMod1K/SPT/tensorboard'    # Directory for tensorboard files.
    settings.pretrained_networks = '/root/autodl-tmp/UniMod1K/SPT/pretrained_networks'

    # UniMod1K dataset paths - 根据您的实际路径
    settings.unimod1k_dir = '/root/autodl-tmp/1-训练验证集/TrainSet'
    settings.unimod1k_dir_nlp = '/root/autodl-tmp/1-训练验证集/TrainSet'

    return settings