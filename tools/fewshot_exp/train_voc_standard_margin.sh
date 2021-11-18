#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
SHOT=(1)
MARGIN=(0.1 0.2 0.3 0.4 0.5)
mkdir fs_exp/voc_standard_results
for shot in ${SHOT[*]} 
do
  for split in ${SPLIT[*]} 
  do
    for margin in ${MARGIN[*]}
    do
      configfile=configs/fewshot/standard/e2e_voc_split${split}_${shot}shot_finetune_${margin}.yaml
      python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
      rm last_checkpoint
    done
  done
done