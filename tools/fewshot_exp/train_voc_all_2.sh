#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(2)
SHOT=(1 2 3 5 10)
for split in ${SPLIT[*]}
do
  configfile=configs/fewshot/base/e2e_voc_split${split}_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  rm last_checkpoint
  python caoGao_split${split}.py
  for shot in ${SHOT[*]}
  do
    configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune.yaml
    python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
    rm last_checkpoint
  done
done