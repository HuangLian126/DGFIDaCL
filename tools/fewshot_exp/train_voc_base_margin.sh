#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
export NGPUS=1
MARGIN=(0.1 0.2 0.3 0.4 0.5)
for margin in ${MARGIN[*]}
do
  configfile=configs/fewshot/base/e2e_voc_split1_base_${margin}.yaml
  python -m torch.distributed.launch --nproc_per_node=$NGPUS ./tools/train_net.py --config-file ${configfile}
  rm last_checkpoint
done