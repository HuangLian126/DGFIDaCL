#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SHOT=(30)
for shot in ${SHOT[*]} 
do
  configfile=configs/fewshot/standard/e2e_coco_${shot}shot_finetune.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
done