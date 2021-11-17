import os
import random
import torch
import torch.utils.data
from PIL import Image
import sys

if sys.version_info[0] == 2:
    import xml.etree.cElementTree as ET
else:
    import xml.etree.ElementTree as ET

from maskrcnn_benchmark.structures.bounding_box import BoxList


class CloseupDataset(torch.utils.data.Dataset):
    CLASSES = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "boat",
        "bottle",
        "bus",
        "car",
        "cat",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "sofa",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_1 = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "boat",
        "bottle",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_2 = (
        "__background__ ",
        "bicycle",
        "bird",
        "boat",
        "bus",
        "car",
        "cat",
        "chair",
        "diningtable",
        "dog",
        "motorbike",
        "person",
        "pottedplant",
        "sheep",
        "train",
        "tvmonitor",
    )

    CLASSES_SPLIT_3 = (
        "__background__ ",
        "aeroplane",
        "bicycle",
        "bird",
        "bottle",
        "bus",
        "car",
        "chair",
        "cow",
        "diningtable",
        "dog",
        "horse",
        "person",
        "pottedplant",
        "train",
        "tvmonitor",
    )

    CLASSES_COCO_NOVEL = (
        'airplane', 'bicycle', 'bird', 'boat',
        'bottle', 'bus', 'car', 'cat', 'chair',
        'cow', 'dining table', 'dog', 'horse',
        'motorcycle', 'person', 'potted plant',
        'sheep', 'couch', 'train', 'tv'
    )

    CLASSES_COCO_BASE = (
        '__background__', 'truck', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'elephant', 'bear', 'zebra',
        'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
        'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
        'tennis racket', 'wine glass', 'cup', 'fork', 'knife',
        'spoon', 'bowl', 'banana', 'apple', 'sandwich',
        'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'toilet', 'laptop', 'microwave',
        'mouse', 'remote', 'keyboard', 'cell phone', 'book',
        'oven', 'toaster', 'sink', 'refrigerator', 'hair drier',
        'clock', 'vase', 'scissors', 'teddy bear', 'toothbrush', 'bed',
    )

    CLASSES_COCO = (
        '__background__', 'person', 'bicycle', 'car', 'motorcycle',
        'airplane', 'bus', 'train', 'truck', 'boat',
        'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog',
        'horse', 'sheep', 'cow', 'elephant', 'bear',
        'zebra', 'giraffe', 'backpack', 'umbrella',
        'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard',
        'surfboard', 'tennis racket', 'bottle', 'wine glass',
        'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
        'potted plant', 'bed', 'dining table', 'toilet', 'tv',
        'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator',
        'book', 'clock', 'vase', 'scissors', 'teddy bear',
        'hair drier', 'toothbrush'
    )

    def __init__(self, data_dir, split, transforms=None, ):
        self.root = data_dir
        self.image_set = split
        self.transforms = transforms

        # print('split: ', split)  # split:  coco_base

        path = "Crops_standard" if 'standard' in split else "Crops"
        if 'extreme' in split:
            path = "Crops_extreme"

        if '10shot' in split:
            shot_ = 9
            path = "Crops_standard-10shot"
            self._imgpath = os.path.join(self.root, path, "%s")

        elif '5shot' in split:
            shot_ = 4
            path = "Crops_standard-5shot"
            self._imgpath = os.path.join(self.root, path, "%s")

        elif '3shot' in split:
            shot_ = 2
            path = "Crops_standard-3shot"
            self._imgpath = os.path.join(self.root, path, "%s")
        elif '2shot' in split:
            shot_ = 1
            path = "Crops_standard-2shot"
            self._imgpath = os.path.join(self.root, path, "%s")
        elif '1shot' in split:
            shot_ = 0
            path = "Crops_standard-1shot"
            self._imgpath = os.path.join(self.root, path, "%s")
        else:
            self._imgpath = os.path.join(self.root, path, "%s")

        self.ids = []
        self.labels = []

        if 'split1_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_1

        elif 'split2_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_2
        elif 'split3_base' in split:
            cls = CloseupDataset.CLASSES_SPLIT_3
        elif 'coco_base' in split:
            cls = CloseupDataset.CLASSES_COCO_BASE
        elif 'coco_standard' in split:
            cls = CloseupDataset.CLASSES_COCO
        else:
            cls = CloseupDataset.CLASSES

        self.class_to_ind = dict(zip(cls, range(len(cls))))
        self.categories = dict(zip(range(len(cls)), cls))
        self.oneOrTwo = len(cls)

        self.count = {c: 0 for c in cls[1:]}
        for i, c in enumerate(cls[1:]):
            class_dir = os.path.join(self.root, path, c)
            for img_id in os.listdir(class_dir):
                self.ids.append(c + "/" + img_id)
                self.labels.append(i + 1)
                self.count[c] += 1

        self.count2 = {c: [] for c in cls[1:]}
        for i, c in enumerate(cls[1:]):
            class_dir2 = os.path.join(self.root, path, c)
            for img_id2 in os.listdir(class_dir2):
                self.count2[c].append(c + "/" + img_id2)

        '''
        if self.oneOrTwo == 16:  # spilt3
            self.idsxxx = []
            self.ranges = 200
            for xxx in range(3773):
                vvv = []
                vvv.append(self.count2['aeroplane'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bird'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bottle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bus'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cow'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['horse'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)
        else:
            self.idsxxx = []
            self.ranges = int(shot_)
            for xxx in range(4952):
                vvv = []
                vvv.append(self.count2['aeroplane'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bird'][random.randint(0, self.ranges)])
                vvv.append(self.count2['boat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bottle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bus'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cow'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['horse'][random.randint(0, self.ranges)])
                vvv.append(self.count2['motorbike'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sheep'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sofa'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)
        '''

        '''
        if self.oneOrTwo == 16:  # spilt2
            self.idsxxx = []
            self.ranges = 200
            for xxx in range(3773):
                vvv = []
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bird'][random.randint(0, self.ranges)])
                vvv.append(self.count2['boat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bus'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['motorbike'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sheep'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)
        else:
            self.idsxxx = []
            self.ranges = int(shot_)
            for xxx in range(4952):
                vvv = []
                vvv.append(self.count2['aeroplane'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bird'][random.randint(0, self.ranges)])
                vvv.append(self.count2['boat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bottle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bus'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cow'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['horse'][random.randint(0, self.ranges)])
                vvv.append(self.count2['motorbike'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sheep'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sofa'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)
        '''

        if self.oneOrTwo == 16:  # spilt1
            self.idsxxx = []
            self.ranges = 199
            for xxx in range(3773):
                vvv = []
                vvv.append(self.count2['aeroplane'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['boat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bottle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, 160)])
                vvv.append(self.count2['cat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['horse'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sheep'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)
        else:
            self.idsxxx = []
            self.ranges = int(shot_)
            for xxx in range(4952):
                vvv = []
                vvv.append(self.count2['aeroplane'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bicycle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bird'][random.randint(0, self.ranges)])
                vvv.append(self.count2['boat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bottle'][random.randint(0, self.ranges)])
                vvv.append(self.count2['bus'][random.randint(0, self.ranges)])
                vvv.append(self.count2['car'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cat'][random.randint(0, self.ranges)])
                vvv.append(self.count2['chair'][random.randint(0, self.ranges)])
                vvv.append(self.count2['cow'][random.randint(0, self.ranges)])
                vvv.append(self.count2['diningtable'][random.randint(0, self.ranges)])
                vvv.append(self.count2['dog'][random.randint(0, self.ranges)])
                vvv.append(self.count2['horse'][random.randint(0, self.ranges)])
                vvv.append(self.count2['motorbike'][random.randint(0, self.ranges)])
                vvv.append(self.count2['person'][random.randint(0, self.ranges)])
                vvv.append(self.count2['pottedplant'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sheep'][random.randint(0, self.ranges)])
                vvv.append(self.count2['sofa'][random.randint(0, self.ranges)])
                vvv.append(self.count2['train'][random.randint(0, self.ranges)])
                vvv.append(self.count2['tvmonitor'][random.randint(0, self.ranges)])
                self.idsxxx.append(vvv)

        # too few ids lead to an unfixed bug
        # if len(self.ids) < 50:
            # self.ids = self.ids * (int(100 / len(self.ids)) + 1)
            # self.labels = self.labels * (int(100 / len(self.labels)) + 1)

        self.id_to_img_map = {k: v for k, v in enumerate(self.ids)}

    def __getitem__(self, index):

        if self.oneOrTwo == 16:
            ff0 = self.idsxxx[index][0]
            ff1 = self.idsxxx[index][1]
            ff2 = self.idsxxx[index][2]
            ff3 = self.idsxxx[index][3]
            ff4 = self.idsxxx[index][4]
            ff5 = self.idsxxx[index][5]
            ff6 = self.idsxxx[index][6]
            ff7 = self.idsxxx[index][7]
            ff8 = self.idsxxx[index][8]
            ff9 = self.idsxxx[index][9]
            ff10 = self.idsxxx[index][10]
            ff11 = self.idsxxx[index][11]
            ff12 = self.idsxxx[index][12]
            ff13 = self.idsxxx[index][13]
            ff14 = self.idsxxx[index][14]

            img0 = Image.open(self._imgpath % ff0).convert("RGB")
            img1 = Image.open(self._imgpath % ff1).convert("RGB")
            img2 = Image.open(self._imgpath % ff2).convert("RGB")
            img3 = Image.open(self._imgpath % ff3).convert("RGB")
            img4 = Image.open(self._imgpath % ff4).convert("RGB")
            img5 = Image.open(self._imgpath % ff5).convert("RGB")
            img6 = Image.open(self._imgpath % ff6).convert("RGB")
            img7 = Image.open(self._imgpath % ff7).convert("RGB")
            img8 = Image.open(self._imgpath % ff8).convert("RGB")
            img9 = Image.open(self._imgpath % ff9).convert("RGB")
            img10 = Image.open(self._imgpath % ff10).convert("RGB")
            img11 = Image.open(self._imgpath % ff11).convert("RGB")
            img12 = Image.open(self._imgpath % ff12).convert("RGB")
            img13 = Image.open(self._imgpath % ff13).convert("RGB")
            img14 = Image.open(self._imgpath % ff14).convert("RGB")

            imgs = []
            if self.transforms is not None:
                imgs.append(self.transforms(img0, None))
                imgs.append(self.transforms(img1, None))
                imgs.append(self.transforms(img2, None))
                imgs.append(self.transforms(img3, None))
                imgs.append(self.transforms(img4, None))
                imgs.append(self.transforms(img5, None))
                imgs.append(self.transforms(img6, None))
                imgs.append(self.transforms(img7, None))
                imgs.append(self.transforms(img8, None))
                imgs.append(self.transforms(img9, None))
                imgs.append(self.transforms(img10, None))
                imgs.append(self.transforms(img11, None))
                imgs.append(self.transforms(img12, None))
                imgs.append(self.transforms(img13, None))
                imgs.append(self.transforms(img14, None))

            target = torch.LongTensor([int(1), int(2), int(3), int(4), int(5), int(6), int(7), int(8), int(9), int(10), int(11),
                 int(12), int(13), int(14), int(15)], device=imgs[0].device)
            return imgs, target
        else:
            ff0 = self.idsxxx[index][0]
            ff1 = self.idsxxx[index][1]
            ff2 = self.idsxxx[index][2]
            ff3 = self.idsxxx[index][3]
            ff4 = self.idsxxx[index][4]
            ff5 = self.idsxxx[index][5]
            ff6 = self.idsxxx[index][6]
            ff7 = self.idsxxx[index][7]
            ff8 = self.idsxxx[index][8]
            ff9 = self.idsxxx[index][9]
            ff10 = self.idsxxx[index][10]
            ff11 = self.idsxxx[index][11]
            ff12 = self.idsxxx[index][12]
            ff13 = self.idsxxx[index][13]
            ff14 = self.idsxxx[index][14]
            ff15 = self.idsxxx[index][15]
            ff16 = self.idsxxx[index][16]
            ff17 = self.idsxxx[index][17]
            ff18 = self.idsxxx[index][18]
            ff19 = self.idsxxx[index][19]

            img0 = Image.open(self._imgpath % ff0).convert("RGB")
            img1 = Image.open(self._imgpath % ff1).convert("RGB")
            img2 = Image.open(self._imgpath % ff2).convert("RGB")
            img3 = Image.open(self._imgpath % ff3).convert("RGB")
            img4 = Image.open(self._imgpath % ff4).convert("RGB")
            img5 = Image.open(self._imgpath % ff5).convert("RGB")
            img6 = Image.open(self._imgpath % ff6).convert("RGB")
            img7 = Image.open(self._imgpath % ff7).convert("RGB")
            img8 = Image.open(self._imgpath % ff8).convert("RGB")
            img9 = Image.open(self._imgpath % ff9).convert("RGB")
            img10 = Image.open(self._imgpath % ff10).convert("RGB")
            img11 = Image.open(self._imgpath % ff11).convert("RGB")
            img12 = Image.open(self._imgpath % ff12).convert("RGB")
            img13 = Image.open(self._imgpath % ff13).convert("RGB")
            img14 = Image.open(self._imgpath % ff14).convert("RGB")
            img15 = Image.open(self._imgpath % ff15).convert("RGB")
            img16 = Image.open(self._imgpath % ff16).convert("RGB")
            img17 = Image.open(self._imgpath % ff17).convert("RGB")
            img18 = Image.open(self._imgpath % ff18).convert("RGB")
            img19 = Image.open(self._imgpath % ff19).convert("RGB")

            imgs = []
            if self.transforms is not None:
                imgs.append(self.transforms(img0, None))
                imgs.append(self.transforms(img1, None))
                imgs.append(self.transforms(img2, None))
                imgs.append(self.transforms(img3, None))
                imgs.append(self.transforms(img4, None))
                imgs.append(self.transforms(img5, None))
                imgs.append(self.transforms(img6, None))
                imgs.append(self.transforms(img7, None))
                imgs.append(self.transforms(img8, None))
                imgs.append(self.transforms(img9, None))
                imgs.append(self.transforms(img10, None))
                imgs.append(self.transforms(img11, None))
                imgs.append(self.transforms(img12, None))
                imgs.append(self.transforms(img13, None))
                imgs.append(self.transforms(img14, None))
                imgs.append(self.transforms(img15, None))
                imgs.append(self.transforms(img16, None))
                imgs.append(self.transforms(img17, None))
                imgs.append(self.transforms(img18, None))
                imgs.append(self.transforms(img19, None))

            target = torch.LongTensor([int(1), int(2), int(3), int(4),
                                       int(5), int(6), int(7), int(8), int(9),
                                       int(10), int(11), int(12), int(13), int(14),
                                       int(15), int(16), int(17), int(18), int(19), int(20)], device=imgs[0].device)
            return imgs, target

    def __len__(self):
        return len(self.idsxxx)