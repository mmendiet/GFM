# --------------------------------------------------------
# Based from SimMIM codebase
# https://github.com/microsoft/SimMIM
# --------------------------------------------------------

import os
import torch.distributed as dist
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from data.mixup import Mixup
from timm.data import create_transform
from timm.data.transforms import str_to_pil_interp
import data.ben_data as bend
import data.ucmerced as merd


def build_loader_finetune(config, logger):
    config.defrost()
    dataset_train, config.MODEL.NUM_CLASSES = build_dataset(is_train=True, config=config, logger=logger)
    config.freeze()
    dataset_val, _ = build_dataset(is_train=False, config=config, logger=logger)
    logger.info(f"Build dataset: train images = {len(dataset_train)}, val images = {len(dataset_val)}")

    num_tasks = dist.get_world_size()
    global_rank = dist.get_rank()
    sampler_train = DistributedSampler(
        dataset_train, num_replicas=num_tasks, rank=global_rank, shuffle=True
    )
    sampler_val = DistributedSampler(
        dataset_val, num_replicas=num_tasks, rank=global_rank, shuffle=False
    )

    data_loader_train = DataLoader(
        dataset_train, sampler=sampler_train,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=True,
    )

    data_loader_val = DataLoader(
        dataset_val, sampler=sampler_val,
        batch_size=config.DATA.BATCH_SIZE,
        num_workers=config.DATA.NUM_WORKERS,
        pin_memory=config.DATA.PIN_MEMORY,
        drop_last=False,
    )

    # setup mixup / cutmix
    mixup_fn = None
    mixup_active = config.AUG.MIXUP > 0 or config.AUG.CUTMIX > 0. or config.AUG.CUTMIX_MINMAX is not None
    if mixup_active:
        mixup_fn = Mixup(
            mixup_alpha=config.AUG.MIXUP, cutmix_alpha=config.AUG.CUTMIX, cutmix_minmax=config.AUG.CUTMIX_MINMAX,
            prob=config.AUG.MIXUP_PROB, switch_prob=config.AUG.MIXUP_SWITCH_PROB, mode=config.AUG.MIXUP_MODE,
            label_smoothing=config.MODEL.LABEL_SMOOTHING, num_classes=config.MODEL.NUM_CLASSES)

    return dataset_train, dataset_val, data_loader_train, data_loader_val, mixup_fn


def build_dataset(is_train, config, logger):
    if 'bigearthnet' in config.DATA.DATA_PATH:
        prefix = 'train' if is_train else 'val'
        transform = bend.build_transform(config, split=prefix)
        nb_classes = 19
        dataset = bend.Bigearthnet(config.DATA.DATA_PATH, split=prefix, img_size=config.DATA.IMG_SIZE,
                        bands=config.MODEL.SWIN.IN_CHANS, transform=transform)
        if config.TRAIN_FRAC < 1.0 and prefix=='train':
            dataset = bend.random_subset(dataset, config.TRAIN_FRAC, 42)
    elif 'merced' in config.DATA.DATA_PATH.lower():
        prefix = 'train' if is_train else 'val'
        transform = merd.build_transform(config, split=prefix)
        nb_classes = 21
        dataset = merd.UCMerced(config.DATA.DATA_PATH, split=prefix, transform=transform, download=True)
        if config.TRAIN_FRAC < 1.0 and prefix=='train':
            dataset = bend.random_subset(dataset, config.TRAIN_FRAC, 42)
    elif config.DATA.DATASET == 'imagenet':
        transform = build_transform(is_train, config)
        logger.info(f'Fine-tune data transform, is_train={is_train}:\n{transform}')
        prefix = 'train' if is_train else 'val'
        root = os.path.join(config.DATA.DATA_PATH, prefix)
        dataset = datasets.ImageFolder(root, transform=transform)
        nb_classes = 1000
    else:
        raise NotImplementedError("We only support ImageNet Now.")
    logger.info('Downstream data transform:\n{}'.format(transform))
    return dataset, nb_classes


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=str_to_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=str_to_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD))
    return transforms.Compose(t)