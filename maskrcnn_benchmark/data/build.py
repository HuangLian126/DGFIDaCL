# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import bisect
import copy
import logging
import numpy as np
import torch.utils.data
from maskrcnn_benchmark.utils.comm import get_world_size
from maskrcnn_benchmark.utils.imports import import_file
from . import datasets as D
from . import samplers
from .collate_batch import BatchCollator, BBoxAugCollator
from .transforms import build_transforms, build_closeup_transforms

def build_dataset(dataset_list, transforms, dataset_catalog, is_train=True, closeup_transforms=None):
    if not isinstance(dataset_list, (list, tuple)):
        raise RuntimeError("dataset_list should be a list of strings, got {}".format(dataset_list))
    datasets = []
    #  dataset_name:  voc_2007_trainval_split1_base_closeup
    for dataset_name in dataset_list:
        data = dataset_catalog.get(dataset_name)

        # data:  {'factory': 'PascalVOCDataset', 'args': {'data_dir': 'datasets/voc/VOC2007', 'split': 'test_split1_base'}}

        factory = getattr(D, data["factory"])
        args = data["args"]

        if data["factory"] == "COCODataset":
            args["remove_images_without_annotations"] = is_train
        if data["factory"] == "PascalVOCDataset":
            args["use_difficult"] = not is_train
        args["transforms"] = transforms
        if data["factory"] == "CloseupDataset":
            args["transforms"] = closeup_transforms
        dataset = factory(**args)
        datasets.append(dataset)
    # print('datasets: ', len(datasets))

    '''
    if not is_train:  # 指测试
        if len(datasets) > 1:
            dataset = D.ConcatDataset(datasets)
            return [dataset]
        else:
            return datasets
    '''

    if not is_train:  # 指测试
        if len(datasets) > 1:
            dataset = D.ConcatDataset(datasets)
            return [dataset]
        else:
            return datasets

    dataset = datasets[0]
    if len(datasets) > 1:
        dataset = D.ConcatDataset(datasets)
    
    return [dataset]

def make_data_sampler(dataset, shuffle, distributed):
    if distributed:
        return samplers.DistributedSampler(dataset, shuffle=shuffle)
    if shuffle:
        sampler = torch.utils.data.sampler.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.sampler.SequentialSampler(dataset)
    return sampler

def _quantize(x, bins):
    bins = copy.copy(bins)
    bins = sorted(bins)
    quantized = list(map(lambda y: bisect.bisect_right(bins, y), x))
    return quantized

def _compute_aspect_ratios(dataset):
    aspect_ratios = []
    for i in range(len(dataset)):
        img_info = dataset.get_img_info(i)
        aspect_ratio = float(img_info["height"]) / float(img_info["width"])
        aspect_ratios.append(aspect_ratio)
    return aspect_ratios

def _init_fn(worker_id):
    np.random.seed(1+worker_id)

def make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_batch, num_iters=None, start_iter=0):
    if aspect_grouping:
        if not isinstance(aspect_grouping, (list, tuple)):
            aspect_grouping = [aspect_grouping]
        aspect_ratios = _compute_aspect_ratios(dataset)
        group_ids = _quantize(aspect_ratios, aspect_grouping)
        batch_sampler = samplers.GroupedBatchSampler(sampler, group_ids, images_per_batch, drop_uneven=False)
    else:
        batch_sampler = torch.utils.data.sampler.BatchSampler(sampler, images_per_batch, drop_last=False)
    if num_iters is not None:
        batch_sampler = samplers.IterationBasedBatchSampler(batch_sampler, num_iters, start_iter)
    return batch_sampler

def make_data_loader(cfg, is_train=True, is_distributed=False, start_iter=0, is_closeup=False):
    num_gpus = get_world_size()
    if is_train:
        if is_closeup:
            images_per_batch = 1
        else:
            images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (images_per_batch % num_gpus == 0), "SOLVER.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus  # 1
        shuffle = True
        num_iters = cfg.SOLVER.MAX_ITER
    else:
        if is_closeup:
            images_per_batch = 1
        else:
            images_per_batch = cfg.SOLVER.IMS_PER_BATCH
        assert (images_per_batch % num_gpus == 0), "TEST.IMS_PER_BATCH ({}) must be divisible by the number of GPUs ({}) used.".format(images_per_batch, num_gpus)
        images_per_gpu = images_per_batch // num_gpus
        shuffle = False if not is_distributed else True
        num_iters = None
        start_iter = 0

    if images_per_gpu > 1:
        logger = logging.getLogger(__name__)
        logger.warning(
            "When using more than one image per GPU you may encounter an out-of-memory (OOM) error if your GPU does not have sufficient memory. If this happens, you can reduce SOLVER.IMS_PER_BATCH (for training) or TEST.IMS_PER_BATCH (for inference)."
            "For training, you must also adjust the learning rate and schedule length according to the linear scaling rule."
            "See for example: "
            "https://github.com/facebookresearch/Detectron/blob/master/configs/getting_started/tutorial_1gpu_e2e_faster_rcnn_R-50-FPN.yaml#L14"
        )

    aspect_grouping = [1] if cfg.DATALOADER.ASPECT_RATIO_GROUPING else []
    paths_catalog = import_file("maskrcnn_benchmark.config.paths_catalog", cfg.PATHS_CATALOG, True)
    DatasetCatalog = paths_catalog.DatasetCatalog
    dataset_list = cfg.DATASETS.TRAIN if is_train else cfg.DATASETS.TEST
    dataset_list = cfg.DATASETS.CLOSEUP if is_closeup else dataset_list

    transforms = None if not is_train and cfg.TEST.BBOX_AUG.ENABLED else build_transforms(cfg, is_train, is_sup=False)
    closeup_transforms = build_closeup_transforms(cfg, is_sup=True) if not is_train else build_closeup_transforms(cfg, is_sup=True)
    datasets = build_dataset(dataset_list, transforms, DatasetCatalog, is_train, closeup_transforms)

    if 'closeup' in dataset_list[0] or 'standard' in dataset_list[0]:
        aspect_grouping = False

    data_loaders = []
    for dataset in datasets:
        sampler = make_data_sampler(dataset, shuffle, is_distributed)
        batch_sampler = make_batch_data_sampler(dataset, sampler, aspect_grouping, images_per_gpu, num_iters, start_iter)
        collator = BBoxAugCollator() if not is_train and cfg.TEST.BBOX_AUG.ENABLED else BatchCollator(cfg.DATALOADER.SIZE_DIVISIBILITY)
        num_workers = cfg.DATALOADER.NUM_WORKERS
        data_loader = torch.utils.data.DataLoader(dataset, num_workers=num_workers, batch_sampler=batch_sampler, collate_fn=collator, worker_init_fn=np.random.seed(1))
        data_loaders.append(data_loader)

    # print('len(data_loaders): ', len(data_loaders))

    if not is_train and is_closeup:
        # print('len(data_loaders[0]): ', len(data_loaders[0]))
        return data_loaders[0]

    if is_train:
        assert len(data_loaders) == 1
        return data_loaders[0]

    return data_loaders