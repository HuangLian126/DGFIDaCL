import torch
import numpy as np
import random
import torch.backends.cudnn as cudnn

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.deterministic = True
    cudnn.benchmark = False

setup_seed(1)

checkpoint = torch.load('/home/hl/hl/ourMetaWithoutFPN/model_final_base_split1.pth', map_location=torch.device("cpu"))
model = checkpoint['model']

change = [('roi_heads.box.predictor.cls_score.weight', (21, 1024)), ('roi_heads.box.predictor.cls_score.bias', 21)]
t = torch.empty(change[0][1])
torch.nn.init.normal_(t, std=0.001)
model[change[0][0]] = t

t = torch.empty(change[1][1])
torch.nn.init.constant_(t, 0)
model[change[1][0]] = t

'''
change2 = [('sup_classfication.weight', (20, 1024)), ('sup_classfication.bias', 20)]
t = torch.empty(change2[0][1])
torch.nn.init.normal_(t, std=0.01)
model[change2[0][0]] = t

t = torch.empty(change2[1][1])
torch.nn.init.constant_(t, 0)
model[change2[1][0]] = t

change3 = [('weight', (20, 1024))]
t = torch.empty(change3[0][1])
torch.nn.init.normal_(t, std=0.01)
model[change3[0][0]] = t

'''

checkpoint = dict(model=model)
torch.save(checkpoint, 'voc0712_split_1_base_pretrained.pth')