#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
SHOT=(3)
for shot in ${SHOT[*]}
do
  for split in ${SPLIT[*]}
  do
    configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune.yaml
    python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/demo_cam.py --config-file ${configfile}
  done
done