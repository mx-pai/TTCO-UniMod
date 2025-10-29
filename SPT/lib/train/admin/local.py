from lib.train.admin.environment import EnvSettings


class EnvironmentSettings(EnvSettings):
    """Local environment paths for training/evaluation."""

    def __init__(self):
        super().__init__()
        # Base directories -------------------------------------------------
        self.workspace_dir = '/root/autodl-tmp/spt_runs'
        self.tensorboard_dir = f"{self.workspace_dir}/tensorboard"
        self.pretrained_models = '/root/autodl-tmp'

        # Dataset roots ----------------------------------------------------
        self.unimod1k_dir = '/root/autodl-tmp/data/1-训练验证集/TrainSet'
        self.unimod1k_dir_nlp = '/root/autodl-tmp/data/1-训练验证集/TrainSet'

        # Default save dir for evaluation/test scripts ---------------------
        self.prj_dir = '/root/autodl-tmp/TTCo-UniMod/SPT'
        self.save_dir = self.workspace_dir
        self.results_path = f"{self.workspace_dir}/lib/test/tracking_results"
        self.segmentation_path = f"{self.workspace_dir}/lib/test/segmentation_results"
        self.network_path = f"{self.workspace_dir}/lib/test/networks"
        self.result_plot_path = f"{self.workspace_dir}/lib/test/result_plots"
