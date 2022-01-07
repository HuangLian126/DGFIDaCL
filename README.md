
[![LICENSE](https://img.shields.io/badge/license-MIT-green)](https://github.com/CSer-Tang-hao/FS-KTN/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.6-blue.svg)](https://www.python.org/)
![PyTorch](https://img.shields.io/badge/pytorch-1.2.0-%237732a8)

Our code is based on  [https://github.com/facebookresearch/maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) and developed with Python 3.7 & PyTorch 1.1.0.

## Installation
Check INSTALL.md for installation instructions. Since maskrcnn-benchmark has been deprecated, please follow these instructions carefully (e.g. version of Python packages).

## Prepare Pascal VOC datasets
First, you need to download the VOC datasets [here](https://drive.google.com/file/d/14muqZUdbpnYQ_30ZpAP9KqrVVHSkJOhU/view?usp=sharing).
Then, put "datasets" into this repository. The "datasets" contains the original VOC2007/2012 datasets and correspondiing class split. The "datasets" is shown below:

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
```

## Training and Evaluation
1. Run the following for base training and novel training on Pascal VOC splits-1.

```bash
bash tools/fewshot_exp/train_voc_all.sh 
```

2. Modify them if needed. If you have any question about these parameters (e.g. batchsize), please refer to [maskrcnn-benchmark](https://github.com/facebookresearch/maskrcnn-benchmark) for quick solutions.
