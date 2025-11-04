from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    settings.prj_dir = '/root/autodl-tmp/TTCO-UniMod/SPT'
    settings.save_dir = '/root/autodl-tmp/TTCO-UniMod/SPT'
    settings.unimod1k_path = '/root/autodl-tmp/data/fusai'
    settings.results_path = '/root/autodl-tmp/TTCO-UniMod/SPT/tracking_results'
    settings.network_path   = '/root/autodl-tmp/TTCO-UniMod/SPT/networks'

    return settings