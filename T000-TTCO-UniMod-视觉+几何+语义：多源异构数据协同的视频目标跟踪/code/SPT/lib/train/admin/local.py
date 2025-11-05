from pathlib import Path


class EnvironmentSettings:
    def __init__(self):
        spt_root = Path(__file__).resolve().parents[3]
        package_root = spt_root.parents[1]

        experiments_root = package_root / 'results' / 'experiments'
        models_root = package_root / 'models'
        evaluations_root = package_root / 'results' / 'evaluations'

        self.workspace_dir = str(experiments_root)
        self.tensorboard_dir = str(experiments_root / 'tensorboard')
        self.pretrained_models = str(models_root / 'pretrained')
        self.unimod1k_dir = str(package_root / 'data' / 'UniMod1K' / 'TrainSet')
        self.unimod1k_dir_nlp = str(package_root / 'data' / 'UniMod1K' / 'TrainSet')
        self.prj_dir = str(spt_root)
        self.save_dir = self.workspace_dir
        self.results_path = str(evaluations_root / 'tracking_results')
        self.segmentation_path = str(evaluations_root / 'segmentation_results')
        self.network_path = str(models_root / 'checkpoints')
        self.result_plot_path = str(package_root / 'results' / 'reports' / 'figures')
