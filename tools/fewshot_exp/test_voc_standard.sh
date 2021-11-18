#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
SHOT=(10)
mkdir fs_exp/voc_standard_results
for shot in ${SHOT[*]}
do
  for split in ${SPLIT[*]}
  do
    configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune.yaml
    python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/demo.py --config-file ${configfile}
  done
done