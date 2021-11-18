from PIL import Image
from maskrcnn_benchmark.data.datasets.coco import COCODataset
from maskrcnn_benchmark.data.datasets.closeup import CloseupDataset
import os, shutil, sys
#crop the object from original image and save them in original shape
#save object under categorized folders


#here we do not crop the original size, but crop a (8 / 7) larger closeup
def get_closeup(image, target):
    closeup = []
    closeup_target = target.get_field('labels').tolist()
    for t in range(len(target)):
        x1, y1, x2, y2 = target.bbox[t].tolist()
        if min(x2 - x1, y2 - y1) < 8:
            continue
        cutsize = max(x2 - x1, y2 - y1) / 2
        midx = (x1 + x2) / 2
        midy = (y1 + y2) / 2
        crop_img = image.crop((int(midx - cutsize), int(midy - cutsize), int(midx + cutsize), int(midy + cutsize)))
        closeup.append(crop_img)
    return closeup, closeup_target


imgdirs = ['/home/hl/hl/ourMetaWithoutFPN/datasets/coco/train2014', '/home/hl/hl/ourMetaWithoutFPN/datasets/coco/val2014']
annofiles = ["/home/hl/hl/ourMetaWithoutFPN/datasets/coco/annotations/instances_train2014_30shot_novel_standard.json",
             "/home/hl/hl/ourMetaWithoutFPN/datasets/coco/annotations/instances_val2014_30shot_novel_standard.json"]


cls_count = {cls: 0 for cls in CloseupDataset.CLASSES_COCO}
for s in range(2):
    dataset = COCODataset(annofiles[s], imgdirs[s], True)
    for index in range(len(dataset)):
        img, annos, _ = dataset.__getitem__(index)
# /home/hl/hl/ourMetaWithoutFPN/datasets/coco/cocosplit/seed0/full_box_30shot_zebra_trainval.json

        crops, crop_labels = get_closeup(img, annos)
        for crop, label in list(zip(crops, crop_labels)):
            label = CloseupDataset.CLASSES_COCO[label]
            cls_count[label] += 1
            crop.save('/home/hl/hl/ourMetaWithoutFPN/datasets/coco/Crops_standard/%s/%d.jpg' % (label, cls_count[label]))
print(cls_count)
print('crop amount:%d' % sum(list(cls_count.values())))