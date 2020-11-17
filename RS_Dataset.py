from path import Path
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision.datasets import ImageFolder
import cv2
import numpy as np
from PIL import Image
from random import randint,sample

class RS_Dataset(ImageFolder):
    # split image into 5 parts each part's partion is 0.6
    def __init__(self, root, transform=None, partion = 0.6, size = 224 ,):
        super(RS_Dataset, self).__init__(root, transform)
        self.indices = range(len(self)) 
        self.transform = transform
        self.partion = partion
        self.size = size
        self.width,self.length = int(self.size*self.partion),int(self.size*self.partion)



        
    def __getitem__(self, index):

        img = np.array(Image.open(self.imgs[index][0]))
        upper_left   =   cv2.resize(img[:self.width,:self.length],(self.size,self.size))
        upper_right  =   cv2.resize(img[:self.width,-self.length:],(self.size,self.size))
        bottom_right =   cv2.resize(img[-self.width:,-self.length:],(self.size,self.size))
        bottom_left  =   cv2.resize(img[-self.width:,:self.length],(self.size,self.size))
        mid = img[int((1-self.partion)/2*(img.shape[0])):-int((1-self.partion)/2*(img.shape[0])),int((1-self.partion)/2*(img.shape[1])):-int((1-self.partion)/2*(img.shape[1]))]
        cluster_data = upper_left,upper_right,bottom_right,bottom_left,mid
        img = self.transform(img)
        cluster_data = [self.transform(i) for i in cluster_data]
        label = self.imgs[index][1]

        return img, cluster_data, label