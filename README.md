
Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare datasets

### Prepare Pascal VOC
First, you need to download the VOC datasets.
Then, put "datasets" into this 


### Prepare base and few-shot datasets
For a fair comparison, we use the few-shot data splits from [Few-shot Object Detection via Feature Reweighting](https://github.com/bingykang/Fewshot_Detection) as a standard evaluation.
To download their data splits and transfer it into VOC/COCO style, you need to run this script:
```bash
bash tools/fewshot_exp/datasets/init_fs_dataset_standard.sh
```
This will also generate the datasets on base classes for base training.

## Training and Evaluation
4 scripts are used for full splits experiments and you can modify them later. 
They will crop objects and store them (e.g. `datasets/voc/VOC2007/Crops_standard`) before training.
You may need to change GPU device which is `export CUDA_VISIBLE_DEVICES=0,1` by default.
```bash
tools/fewshot_exp/
├── train_voc_base.sh
├── train_voc_standard.sh
├── train_coco_base.sh
└── train_coco_standard.sh
```

Configurations of base & few-shot experiments are:
```base
configs/fewshot/
├── base
│   ├── e2e_coco_base.yaml
│   └── e2e_voc_split*_base.yaml
└── standard
    ├── e2e_coco_*shot_finetune.yaml
    └── e2e_voc_split*_*shot_finetune.yaml
```
Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.

### Perform few-shot training on VOC dataset
1. Run the following for base training on 3 VOC splits
```bash
bash tools/fewshot_exp/train_voc_base.sh
```
This will generate base models (e.g. `model_voc_split1_base.pth`) and corresponding pre-trained models (e.g. `voc0712_split1base_pretrained.pth`).

2. Run the following for few-shot fine-tuning
```bash
bash tools/fewshot_exp/train_voc_standard.sh
```
This will perform evaluation on 1/2/3/5/10 shot of 3 splits. 
Result folder is `fs_exp/voc_standard_results` by default, and you can get a quick summary by:
```bash
python tools/fewshot_exp/cal_novel_voc.py fs_exp/voc_standard_results
```

3. For more general experiments, refer to `tools/fewshot_exp/train_voc_series.sh`. In this script, only few-shot classes are limited to N-shot. This may lead to a drop in performance but more natural conditions.
