import json
import os
import shutil
import subprocess
from datetime import datetime

import torch
from torch.utils.data.distributed import DistributedSampler

# datasets related
from lib.train.dataset import UniMod1K
from lib.train.data import sampler, opencv_loader, processing, LTRLoader
import lib.train.data.transforms as tfm
from lib.utils.misc import is_main_process
import yaml


def configure_paths(settings, cfg):
    paths_cfg = getattr(cfg, 'PATHS', None)

    def _abs_path(path):
        if path is None or path == '':
            return None
        return os.path.abspath(os.path.expanduser(path))

    env = settings.env

    run_name_cfg = getattr(paths_cfg, 'RUN_NAME', None) if paths_cfg is not None else None
    if not getattr(settings, 'run_name', None):
        settings.run_name = run_name_cfg or datetime.now().strftime('%Y%m%d-%H%M%S')

    base_output = _abs_path(getattr(paths_cfg, 'OUTPUT_DIR', None)) if paths_cfg is not None else None
    if base_output is None:
        fallback_root = getattr(settings, 'save_dir', None)
        if fallback_root:
            fallback_root = os.path.abspath(os.path.expanduser(fallback_root))
        else:
            fallback_root = getattr(env, 'workspace_dir', None)
            if fallback_root:
                fallback_root = os.path.abspath(os.path.expanduser(fallback_root))
        if fallback_root is None:
            fallback_root = os.path.join(os.getcwd(), 'outputs')
        else:
            fallback_root = os.path.join(fallback_root, 'outputs')
        base_output = fallback_root

    run_dir_override = _abs_path(getattr(paths_cfg, 'RUN_DIR', None)) if paths_cfg is not None else None
    if run_dir_override is not None:
        run_root = run_dir_override
    else:
        run_root = os.path.join(base_output, settings.config_name, settings.run_name)
    os.makedirs(run_root, exist_ok=True)

    tensorboard_dir = _abs_path(getattr(paths_cfg, 'TENSORBOARD_DIR', None)) if paths_cfg is not None else None
    if tensorboard_dir is None:
        tensorboard_dir = os.path.join(run_root, 'tensorboard')

    checkpoint_dir = _abs_path(getattr(paths_cfg, 'CHECKPOINT_DIR', None)) if paths_cfg is not None else None
    if checkpoint_dir is None:
        checkpoint_dir = os.path.join(run_root, 'checkpoints')

    log_dir = _abs_path(getattr(paths_cfg, 'LOG_DIR', None)) if paths_cfg is not None else None
    if log_dir is None:
        log_dir = os.path.join(run_root, 'logs')

    pretrained_dir = _abs_path(getattr(paths_cfg, 'PRETRAINED_DIR', None)) if paths_cfg is not None else None
    data_root = _abs_path(getattr(paths_cfg, 'DATA_ROOT', None)) if paths_cfg is not None else None
    nlp_root = _abs_path(getattr(paths_cfg, 'NLP_ROOT', None)) if paths_cfg is not None else None

    settings.save_dir = run_root
    settings.checkpoint_dir = checkpoint_dir
    settings.log_dir = log_dir
    settings.paths = {
        'run_root': run_root,
        'log_dir': log_dir,
        'tensorboard_dir': tensorboard_dir,
        'checkpoint_dir': checkpoint_dir,
        'base_output': base_output
    }
    settings.tensorboard_dir = tensorboard_dir

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)

    if hasattr(env, 'workspace_dir'):
        env.workspace_dir = run_root
    if tensorboard_dir is not None and hasattr(env, 'tensorboard_dir'):
        env.tensorboard_dir = tensorboard_dir
    if pretrained_dir is not None and hasattr(env, 'pretrained_models'):
        env.pretrained_models = pretrained_dir
    if data_root is not None and hasattr(env, 'unimod1k_dir'):
        env.unimod1k_dir = data_root
    if nlp_root is not None and hasattr(env, 'unimod1k_dir_nlp'):
        env.unimod1k_dir_nlp = nlp_root


def _edict_to_dict(ed):
    if isinstance(ed, dict):
        return {k: _edict_to_dict(v) for k, v in ed.items()}
    if isinstance(ed, (list, tuple)):
        return [_edict_to_dict(v) for v in ed]
    return ed


def _collect_git_info():
    info = {}
    try:
        commit = subprocess.check_output(['git', 'rev-parse', 'HEAD'], stderr=subprocess.DEVNULL).decode().strip()
        info['git_commit'] = commit
    except Exception:
        info['git_commit'] = None
    try:
        branch = subprocess.check_output(['git', 'rev-parse', '--abbrev-ref', 'HEAD'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        info['git_branch'] = branch
    except Exception:
        info['git_branch'] = None
    return info


def _collect_git_status():
    try:
        status = subprocess.check_output(['git', 'status', '--short'],
                                         stderr=subprocess.DEVNULL).decode().strip()
        return status
    except Exception:
        return None


def snapshot_run(settings, cfg, extra_meta=None):
    run_root = settings.paths.get('run_root', settings.save_dir)
    metadata_dir = os.path.join(run_root, 'metadata')
    os.makedirs(metadata_dir, exist_ok=True)

    # raw config
    cfg_src = getattr(settings, 'cfg_file', None)
    if cfg_src and os.path.exists(cfg_src):
        shutil.copy2(cfg_src, os.path.join(metadata_dir, 'config_raw.yaml'))

    # resolved config
    cfg_dict = _edict_to_dict(cfg)
    resolved_path = os.path.join(metadata_dir, 'config_resolved.yaml')
    with open(resolved_path, 'w') as f:
        yaml.safe_dump(cfg_dict, f, sort_keys=False, allow_unicode=True)

    meta = {
        'run_name': settings.run_name,
        'script_name': settings.script_name,
        'config_name': settings.config_name,
        'timestamp': datetime.now().isoformat(),
        'run_root': run_root,
        'checkpoint_dir': settings.checkpoint_dir,
        'tensorboard_dir': settings.tensorboard_dir,
        'log_file': getattr(settings, 'log_file', None),
        'launch_cmd': getattr(settings, 'launch_cmd', None),
    }
    if extra_meta:
        meta.update(extra_meta)
    meta.update(_collect_git_info())

    meta_path = os.path.join(metadata_dir, 'run_meta.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    status = _collect_git_status()
    if status:
        with open(os.path.join(metadata_dir, 'git_status.txt'), 'w') as f:
            f.write(status + '\n')


def update_settings(settings, cfg):
    settings.print_interval = cfg.TRAIN.PRINT_INTERVAL
    settings.search_area_factor = {'template': cfg.DATA.TEMPLATE.FACTOR,
                                   'search': cfg.DATA.SEARCH.FACTOR}
    settings.output_sz = {'template': cfg.DATA.TEMPLATE.SIZE,
                          'search': cfg.DATA.SEARCH.SIZE}
    settings.center_jitter_factor = {'template': cfg.DATA.TEMPLATE.CENTER_JITTER,
                                     'search': cfg.DATA.SEARCH.CENTER_JITTER}
    settings.scale_jitter_factor = {'template': cfg.DATA.TEMPLATE.SCALE_JITTER,
                                    'search': cfg.DATA.SEARCH.SCALE_JITTER}
    settings.grad_clip_norm = cfg.TRAIN.GRAD_CLIP_NORM
    settings.print_stats = None
    settings.batchsize = cfg.TRAIN.BATCH_SIZE
    settings.scheduler_type = cfg.TRAIN.SCHEDULER.TYPE
    settings.lang_loss_weight = getattr(cfg.TRAIN, 'LANG_WEIGHT', 0.0)


def names2datasets(name_list: list, settings, image_loader):
    assert isinstance(name_list, list)
    datasets = []
    for name in name_list:
        assert name in ["UniMod1K"]
        if name == 'UniMod1K':
            root_dir = getattr(settings.env, 'unimod1k_dir', None)
            if not root_dir:
                raise ValueError("UniMod1K root directory is not configured. Set PATHS.DATA_ROOT or update local.py.")
            datasets.append(UniMod1K(root=root_dir,
                                     nlp_root=getattr(settings.env, 'unimod1k_dir_nlp', None),
                                     dtype='rgbcolormap',
                                     image_loader=image_loader))


    return datasets


def build_dataloaders(cfg, settings):
    # Data transform
    transform_joint = tfm.Transform(tfm.ToGrayscale(probability=0.05),
                                    tfm.RandomHorizontalFlip(probability=0.5))

    aug_transforms = [tfm.ToTensorAndJitter(0.2)]

    aug_cfg = getattr(cfg.TRAIN, 'AUG', None)
    if aug_cfg is not None:
        color_cfg = getattr(aug_cfg, 'COLOR_JITTER', None)
        if color_cfg is not None and getattr(color_cfg, 'ENABLED', False):
            aug_transforms.append(
                tfm.TensorColorJitter(
                    brightness=getattr(color_cfg, 'BRIGHTNESS', 0.2),
                    contrast=getattr(color_cfg, 'CONTRAST', 0.2),
                    saturation=getattr(color_cfg, 'SATURATION', 0.2),
                    hue=getattr(color_cfg, 'HUE', 0.02),
                    probability=getattr(color_cfg, 'PROBABILITY', 0.8)
                )
            )
        blur_cfg = getattr(aug_cfg, 'GAUSSIAN_BLUR', None)
        if blur_cfg is not None and getattr(blur_cfg, 'ENABLED', False):
            sigma = getattr(blur_cfg, 'SIGMA', (0.1, 2.0))
            if isinstance(sigma, list):
                sigma = tuple(sigma)
            aug_transforms.append(
                tfm.RandomGaussianBlur(
                    kernel_size=getattr(blur_cfg, 'KERNEL_SIZE', 3),
                    sigma=sigma,
                    probability=getattr(blur_cfg, 'PROBABILITY', 0.1)
                )
            )

    aug_transforms.append(tfm.RandomHorizontalFlip_Norm(probability=0.5))

    if aug_cfg is not None:
        er_cfg = getattr(aug_cfg, 'RANDOM_ERASE', None)
        if er_cfg is not None and getattr(er_cfg, 'ENABLED', False):
            scale = getattr(er_cfg, 'SCALE', (0.02, 0.33))
            ratio = getattr(er_cfg, 'RATIO', (0.3, 3.3))
            if isinstance(scale, list):
                scale = tuple(scale)
            if isinstance(ratio, list):
                ratio = tuple(ratio)
            aug_transforms.append(
                tfm.TensorRandomErasing(
                    probability=getattr(er_cfg, 'PROBABILITY', 0.2),
                    scale=scale,
                    ratio=ratio
                )
            )

    aug_transforms.append(tfm.Normalize(mean=cfg.DATA.MEAN, std=cfg.DATA.STD))

    transform_train = tfm.Transform(aug_transforms)


    # The tracking pairs processing module
    output_sz = settings.output_sz
    search_area_factor = settings.search_area_factor

    data_processing_train = processing.SPTProcessing(search_area_factor=search_area_factor,
                                                       output_sz=output_sz,
                                                       center_jitter_factor=settings.center_jitter_factor,
                                                       scale_jitter_factor=settings.scale_jitter_factor,
                                                       mode='sequence',
                                                       transform=transform_train,
                                                       joint_transform=transform_joint,
                                                       settings=settings)


    # Train sampler and loader
    settings.num_template = getattr(cfg.DATA.TEMPLATE, "NUMBER", 1)
    settings.num_search = getattr(cfg.DATA.SEARCH, "NUMBER", 1)
    sampler_mode = getattr(cfg.DATA, "SAMPLER_MODE", "causal")
    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    print("sampler_mode", sampler_mode)
    total_samples = cfg.DATA.TRAIN.SAMPLE_PER_EPOCH
    long_seq_ratio = getattr(cfg.DATA.TRAIN, "LONG_SEQ_RATIO", 0.0)
    long_seq_ratio = max(0.0, min(1.0, long_seq_ratio))
    long_seq_len = getattr(cfg.DATA.TRAIN, "LONG_SEQ_LENGTH", 4)

    short_samples = int(round(total_samples * (1.0 - long_seq_ratio)))
    long_samples = total_samples - short_samples

    base_datasets = names2datasets(cfg.DATA.TRAIN.DATASETS_NAME, settings, opencv_loader)
    datasets_list = []

    if short_samples > 0:
        dataset_short = sampler.VLTrackingSampler(
            datasets=base_datasets,
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=short_samples,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=settings.num_search,
            num_template_frames=settings.num_template,
            processing=data_processing_train,
            frame_sample_mode=sampler_mode,
            train_cls=train_cls,
            max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
            bert_model=cfg.MODEL.LANGUAGE.TYPE,
            bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH
        )
        datasets_list.append(dataset_short)

    if long_samples > 0:
        from lib.train.data.sampler_longseq import LongSeqTrackingSampler

        dataset_long = LongSeqTrackingSampler(
            datasets=base_datasets,
            p_datasets=cfg.DATA.TRAIN.DATASETS_RATIO,
            samples_per_epoch=long_samples,
            max_gap=cfg.DATA.MAX_SAMPLE_INTERVAL,
            num_search_frames=1,
            num_template_frames=settings.num_template,
            processing=data_processing_train,
            seq_length=long_seq_len,
            max_seq_len=cfg.DATA.MAX_SEQ_LENGTH,
            bert_model=cfg.MODEL.LANGUAGE.TYPE,
            bert_path=cfg.MODEL.LANGUAGE.VOCAB_PATH
        )
        datasets_list.append(dataset_long)

    if not datasets_list:
        raise ValueError("No training samples configured. Please check SAMPLE_PER_EPOCH and LONG_SEQ_RATIO.")

    if len(datasets_list) == 1:
        dataset_train = datasets_list[0]
    else:
        from torch.utils.data import ConcatDataset
        dataset_train = ConcatDataset(datasets_list)

    train_sampler = DistributedSampler(dataset_train) if settings.local_rank != -1 else None
    shuffle = False if settings.local_rank != -1 else True

    loader_train = LTRLoader('train', dataset_train, training=True, batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=shuffle,
                             num_workers=cfg.TRAIN.NUM_WORKER, drop_last=True, stack_dim=1, sampler=train_sampler)

    return loader_train


def get_optimizer_scheduler(net, cfg):

    VISUAL_LR = getattr(cfg.TRAIN, "LR", 10e-5)
    LANGUAGE_LR = getattr(cfg.MODEL.LANGUAGE.BERT, "LR", 10e-5)

    train_cls = getattr(cfg.TRAIN, "TRAIN_CLS", False)
    if train_cls:
        print("Only training classification head. Learnable parameters are shown below.")
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "cls" in n and p.requires_grad]}
        ]

        for n, p in net.named_parameters():
            if "cls" not in n:
                p.requires_grad = False
            else:
                print(n)
    else:
        param_dicts = [
            {"params": [p for n, p in net.named_parameters() if "backbone" not in n and 'nl_pos_embed' not in n
                        and 'text_proj' not in n and p.requires_grad]},
            {
                "params": [p for n, p in net.named_parameters() if "backbone" in n and "language_backbone" not in n
                           and p.requires_grad],
                "lr": VISUAL_LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            },
            {
                "params": [p for n, p in net.named_parameters() if ("language_backbone" in n or 'nl_pos_embed' in n
                                                                    or 'text_proj' in n) and p.requires_grad],
                "lr": LANGUAGE_LR * cfg.TRAIN.BACKBONE_MULTIPLIER,
            }
        ]
        if is_main_process():
            print("Learnable parameters are shown below.")
            for n, p in net.named_parameters():
                if p.requires_grad:
                    print(n)

    if cfg.TRAIN.OPTIMIZER == "ADAMW":
        optimizer = torch.optim.AdamW(param_dicts, lr=cfg.TRAIN.LR,
                                      weight_decay=cfg.TRAIN.WEIGHT_DECAY)
    else:
        raise ValueError("Unsupported Optimizer")
    if cfg.TRAIN.SCHEDULER.TYPE == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.TRAIN.LR_DROP_EPOCH)
    elif cfg.TRAIN.SCHEDULER.TYPE == "Mstep":
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                            milestones=cfg.TRAIN.SCHEDULER.MILESTONES,
                                                            gamma=cfg.TRAIN.SCHEDULER.GAMMA)
    else:
        raise ValueError("Unsupported scheduler")
    return optimizer, lr_scheduler
