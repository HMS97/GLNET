
import torch
from torch import nn
import torchvision.models as models
from torchvision.models import vgg16,alexnet,resnet50
import numpy as np 



torch.nn.Module.dump_patches = True
class SiameseNetwork(nn.Module):
    def __init__(self,base_model ='vgg16',num_classes = 5 , fixed = False,out_features_dim = 128):
        super(SiameseNetwork, self).__init__()
        self.base_model = base_model
        self.num_classes = num_classes
        self.fixed = fixed
        self.out_features_dim = out_features_dim

        self.lower_model = self.make_back_bone(self.base_model)
        self.upper_backbone = self.make_back_bone(self.base_model)
        # for param in self.parameters():
        #     param.requires_grad = False
        self.fc1 = nn.Linear(in_features=6*8*4, out_features=self.num_classes, bias=True)
        self.prelu = nn.PReLU()
        self.avgpool = nn.AvgPool2d(2)

    def make_back_bone(self,base_model):
        if base_model == 'vgg16':
            # model = torch.load('new_saved_models/2020-07-07_vgg16_93.79_baseline.pth')
            # model = torch.load('saved_models/2020-07-07_vgg16_cloud_baseline.pth')
            model = vgg16()
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
            model.classifier[-1] =nn.Linear(in_features=4096,out_features=self.out_features_dim)
            return model

        if base_model == 'alexnet':
            # model = torch.load('new_saved_models/2020-07-07_alexnet_91.57_baseline.pth')
            model = torch.load('new_saved_models/2020-08-20_alexnet_88.85_cloud_baseline.pth')
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
            model.classifier[-1] = nn.Linear(in_features=4096,out_features=self.out_features_dim)
            return model
        
        if base_model == 'resnet50':
           # model = torch.load('new_saved_models/2020-07-07_resnet50_94.21_baseline.pth')
            model = torch.load('new_saved_models/2020-08-21_resnet50_91.57_cloud_baseline.pth')
            for param in model.parameters():
                if self.fixed:
                    param.requires_grad = False
            model.fc = nn.Linear(in_features=2048,out_features=self.out_features_dim)

            return model 

    def forward(self, img, cluster_data):
        output_list = []
        for index,image in enumerate(cluster_data):
            output_list.append(self.upper_backbone(image))
         
            output_list[index] = torch.unsqueeze(output_list[index],1)

        x_upper = torch.cat((output_list),1)
        x_upper = x_upper.view(x_upper.shape[0],5,32,4)

        x_lower = self.lower_model(img)
        x_lower = torch.unsqueeze(x_lower,1)
        x_lower = x_lower.view(x_lower.shape[0],1,32,4)

        x = torch.cat((x_upper,x_lower), dim = 1)
        x = self.avgpool(x)

        x = x.view(x.size(0), -1) 
        # print(x.shape)
        x = self.fc1(x)
    


        return x


if __name__ == '__main__':
    model = SiameseNetwork(7)
