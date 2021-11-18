Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.
 

## Installation
Check INSTALL.md for installation instructions.

## Prepare datasets

### Prepare Pascal VOC datasets
First, you need to download the VOC datasets.
Then, put VOC datasets into file folder "datasets".

```bash
datasets/voc/VOC2007
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── Crops
            ├── Crops_standard-1shot
            ├── Crops_standard-2shot
            ├── Crops_standard-3shot
            ├── Crops_standard-5shot
            ├── Crops_standard-10shot
datasets/voc/VOC2007
            ├── Annotations
            ├── ImageSets
            ├── JPEGImages
            ├── Crops
            ├── Crops_standard-1shot
            ├── Crops_standard-2shot
            ├── Crops_standard-3shot
            ├── Crops_standard-5shot
            ├── Crops_standard-10shot
```

### Prepare base and few-shot datasets
We upload the few-shot data splits on ImageSets:

## Training and Evaluation
4 scripts are used for full splits experiments and you can modify them later. 
They will crop objects and store them (e.g. `datasets/voc/VOC2007/Crops_standard`) before training.
You may need to change GPU device which is `export CUDA_VISIBLE_DEVICES=0,1` by default.
```bash
datasets/voc/
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


