from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = '/home/mx2004/AiC/paper_project/UniMod1K/SPT'
    settings.save_dir = '/home/mx2004/AiC/paper_project/UniMod1K/SPT'
    settings.unimod1k_path = '/home/mx2004/AiC/paper_project/1-训练验证集/ValidationSet'
    settings.results_path = '/home/mx2004/AiC/paper_project/UniMod1K/SPT/test/tracking_results'
    settings.segmentation_path = '/home/mx2004/AiC/paper_project/UniMod1K/SPT/test/segmentation_results'
    settings.network_path = '/home/mx2004/AiC/paper_project/UniMod1K/SPT/test/networks'
    settings.result_plot_path = '/home/mx2004/AiC/paper_project/UniMod1K/SPT/test/result_plots'

    return settings