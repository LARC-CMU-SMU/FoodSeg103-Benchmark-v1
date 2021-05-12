import torch

pthfile = r'/SEG/mmsegmentation/pretrained_model/recipe1MP_R50.pth'


net=torch.load(pthfile)

for k in net.keys():
    print(k)