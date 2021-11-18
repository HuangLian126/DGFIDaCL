#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
configfile=configs/fewshot/base/e2e_coco_base.yaml
python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}