#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(1)
for split in ${SPLIT[*]} 
do
  configfile=configs/fewshot/base/e2e_voc_split${split}_base_dcn.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
done
