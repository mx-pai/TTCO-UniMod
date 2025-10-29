class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/root/autodl-tmp/spt_runs'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = f"{self.workspace_dir}/tensorboard"    # Directory for tensorboard files.
        self.pretrained_models = '/root/autodl-tmp/pretrained_models'
        self.unimod1k_dir = '/root/autodl-tmp/data/1-训练验证集/TrainSet'
        self.unimod1k_dir_nlp = '/root/autodl-tmp/data/1-训练验证集/TrainSet'
        self.prj_dir = '/root/autodl-tmp/TTCO-UniMod/SPT'
        self.save_dir = self.workspace_dir
        self.results_path = f"{self.workspace_dir}/lib/test/tracking_results"
        self.segmentation_path = f"{self.workspace_dir}/lib/test/segmentation_results"
        self.network_path = f"{self.workspace_dir}/lib/test/networks"
        self.result_plot_path = f"{self.workspace_dir}/lib/test/result_plots"

