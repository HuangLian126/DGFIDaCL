# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import logging
import time
import os, sys
import cv2
import torch
from tqdm import tqdm
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.data.datasets.evaluation import evaluate
from ..utils.comm import is_main_process, get_world_size
from ..utils.comm import all_gather
from ..utils.comm import synchronize
from ..utils.timer import Timer, get_time_str
from .bbox_aug import im_detect_bbox_aug

import collections
import pickle
from grad_cam import GradCAM
from torch import nn

def get_last_conv_name(net):
    """
    获取网络的最后一个卷积层的名字
    :param net:
    :return:
    """
    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name

def compute_on_dataset(model, data_loader, data_loader_sup, device, timer=None):
    # model.eval()
    results_dict = {}
    cpu_device = torch.device("cpu")
    if not os.path.exists('demopics'):
        os.mkdir('demopics')
    data_loader_sup = iter(data_loader_sup)
    oneStage = False

    logit = []
    logit_target = []

    # layer_name = get_last_conv_name(model)
    # layer_name = 'DAFIV4_opt.combine'
    layer_name = 'conv_before'
    print(layer_name)
    grad_cam = GradCAM(model, layer_name)


    for _, batch in enumerate(tqdm(data_loader)):
        images, targets, image_ids = batch
        # with torch.no_grad():
        if timer:
            timer.tic()
        if cfg.TEST.BBOX_AUG.ENABLED:
            output = im_detect_bbox_aug(model, images, device)
        else:
            sups, supTarget = next(data_loader_sup)

            mask = grad_cam(images.to(device), [target.to(device) for target in targets], [sup.to(device) for sup in sups], supTarget.to(device), oneStage)  # cam mask

        if timer:
            if not cfg.MODEL.DEVICE == 'cpu':
                torch.cuda.synchronize()
            timer.toc()



def inference(model, data_loader, data_loader_sup, dataset_name, iou_types=("bbox",), box_only=False, device="cuda", expected_results=(),
              expected_results_sigma_tol=4, output_folder=None):
    device = torch.device(device)
    num_devices = get_world_size()
    logger = logging.getLogger("maskrcnn_benchmark.inference")
    dataset = data_loader.dataset

    logger.info("Start evaluation on {} dataset({} images).".format(dataset_name, len(dataset)))
    total_timer = Timer()
    inference_timer = Timer()
    total_timer.tic()
    predictions = compute_on_dataset(model, data_loader, data_loader_sup, device, inference_timer)

