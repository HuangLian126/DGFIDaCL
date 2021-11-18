## Installation

### Requirements:
- PyTorch 1.1.0
- torchvison 0.3.0
- torchvision
- cocoapi
- yacs
- matplotlib
- GCC >= 4.9
- OpenCV
- CUDA = 10.0


### Step-by-step installation

```bash

# install pycocotools
# modify "cocoeval.py" to store raw results in "~/coco_result.txt"
cd $INSTALL_DIR
git clone https://github.com/cocodataset/cocoapi.git
cp MPSR/tools/fewshot_exp/cocoeval.py cocoapi/PythonAPI/pycocotools/
cd cocoapi/PythonAPI
python setup.py build_ext install

# install apex-96b017a8b40f137abb971c4555d61b2fcbb87648
cd $INSTALL_DIR
git clone https://github.com/NVIDIA/apex.git
cd apex
git checkout 96b017a
python setup.py install --cuda_ext --cpp_ext

# install PyTorch Detection
cd $INSTALL_DIR
cd DGFI_DaCL

python setup.py build develop


```
