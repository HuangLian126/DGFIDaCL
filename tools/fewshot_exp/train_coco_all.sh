#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SHOT=(10)
for shot in ${SHOT[*]}
do
  configfile=configs/fewshot/base/e2e_coco_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  rm last_checkpoint
  python caoGao4_coco.py
  configfile=configs/fewshot/standard/e2e_coco_${shot}shot_finetune.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  rm last_checkpoint
done