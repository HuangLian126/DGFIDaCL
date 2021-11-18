
Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare Pascal VOC datasets
First, you need to download the VOC datasets.
Then, put "datasets" into this repositories. The "datasets" contains the few-shot data splits. The "datasets" is shown below:
```bash
datasets/voc/
            ├──VOC2007
                  ├── Annotations
                  ├── ImageSets
                  ├── JPEGImages
                  ├── Crops
                  ├── Crops_standard-1shot
                  ├── Crops_standard-2shot
                  ├── Crops_standard-3shot
                  ├── Crops_standard-5shot
                  ├── Crops_standard-10shot
            ├──VOC2012
                  ├── Annotations
                  ├── ImageSets
                  ├── JPEGImages
                  ├── Crops
                  ├── Crops_standard-1shot
                  ├── Crops_standard-2shot
                  ├── Crops_standard-3shot
                  ├── Crops_standard-5shot
                  ├── Crops_standard-10shot
'''

## Training and Evaluation
1. Run the following for base training and novel training on Pascal VOC splits-1.

```bash
bash tools/fewshot_exp/train_voc_all.sh 
```

Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.



3. For more general experiments, refer to `tools/fewshot_exp/train_voc_series.sh`. In this script, only few-shot classes are limited to N-shot. This may lead to a drop in performance but more natural conditions.
