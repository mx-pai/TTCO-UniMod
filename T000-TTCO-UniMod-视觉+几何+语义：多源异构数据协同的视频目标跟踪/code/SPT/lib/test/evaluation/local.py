from pathlib import Path

from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.
    spt_root = Path(__file__).resolve().parents[3]
    package_root = spt_root.parents[1]

    settings.prj_dir = str(spt_root)
    settings.save_dir = str(package_root / 'results')
    settings.unimod1k_path = str(package_root / 'data' / 'UniMod1K' / 'TestSet')
    settings.results_path = str(package_root / 'results' / 'evaluations' / 'tracking_results')
    settings.network_path = str(package_root / 'models' / 'checkpoints')

    return settings
