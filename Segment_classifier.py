import torchvision.models as models
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
class Segment_Classifier:
    def __init__(self):
        self.device = torch.device("cuda:0")
        self.label2word = {
        0 : 'HANDWRITTEN',
        1 : 'PRINTED'
        }
        self.model_e10 = models.resnet18(pretrained=True)
        num_ftrs = self.model_e10.fc.in_features
        self.model_e10.fc = nn.Sequential(nn.Linear(num_ftrs,500),
        nn.ReLU(),
        nn.Dropout(), nn.Linear(500,2))
        self.model_e10.load_state_dict(torch.load('Checkpoint_e14_227x227.pth'))
        if torch.cuda.device_count() > 1:
            print("Using", torch.cuda.device_count(), "GPUs!")
            self.model_e10 = nn.DataParallel(self.model_e10)
        self.model_e10 = self.model_e10.to(self.device)
        self.model_e10.eval()
    def preprocess(self,folder):
        #normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
         #                                std=[0.229, 0.224, 0.225])
        normalize=transforms.Normalize(mean=0.449,std= 0.226)
        trans = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((227, 227)),
            transforms.ToTensor(),
            normalize
        ])
        Images=datasets.ImageFolder(folder,transform=trans)
        self.data_loader = torch.utils.data.DataLoader(Images,batch_size=1040,num_workers=2,shuffle=True)
    def test(self):
        c = 0
        n = 0
        c_not=0
        n_not=0
        for x, y in self.data_loader:
            x=x.repeat_interleave(3, dim=1)
            x=x.to(self.device)
            out = self.model_e10(x)
            _, predicted = torch.max(out.data, 1)
            pr = predicted.detach().cpu().numpy()
            y=y.numpy()
            for i in range(len(pr)):
                if(self.label2word[pr[i]]=='PRINTED' and self.label2word[y[i]]=='PRINTED'):
                    c=c+1
                elif(self.label2word[pr[i]]=='HANDWRITTEN' and self.label2word[y[i]]=='HANDWRITTEN'):
                    n=n+1
                elif(self.label2word[pr[i]]=='PRINTED' and self.label2word[y[i]]=='HANDWRITTEN'):
                    n_not=n_not+1
                elif(self.label2word[pr[i]]=='HANDWRITTEN' and self.label2word[y[i]]=='PRINTED'):
                    c_not=c_not+1
        print(c," ",c_not)
        print(n_not, " ", n)
        return pr


import time
start=time.time()
sc=Segment_Classifier()
sc.preprocess('/mnt/user-workspace/debjit.chowdhury/multi-processing_gpu_pytorch/Multi GPU model/test')
sc.test()
print(time.time()-start)