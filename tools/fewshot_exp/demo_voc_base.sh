#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
SPLIT=(2)
for split in ${SPLIT[*]} 
do
  configfile=configs/fewshot/base/e2e_voc_split${split}_base.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/demo.py --config-file ${configfile}
done