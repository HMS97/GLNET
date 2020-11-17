from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset, DataLoader, Dataset,SubsetRandomSampler
from torchvision import models
import time
from RS_Dataset import RS_Dataset
from tqdm import tqdm
import os 
import shutil
from datetime import date
import argparse
from torchvision.models import resnet50,alexnet,vgg16
from model import SiameseNetwork

#offline



def train(PARAMS, model, criterion, device, train_loader, optimizer, epoch):
    t0 = time.time()
    model.train()
    correct = 0

    for batch_idx, (img, cluster,  target) in enumerate(tqdm(train_loader)):
        img,  target = img.to(device),  target.to(device)
        cluster =  [item.to(device) for item in cluster ]
        optimizer.zero_grad()
        output = model(img,cluster)
        # output = model(img)

        loss = criterion(output, target )
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    
        # if batch_idx % config.log_interval == 0:

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} , {:.2f} seconds'.format(
        epoch, batch_idx * len(img), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(),time.time() - t0))

    print('train_loss', epoch, loss.data.cpu().numpy())
    print('Train Accuracy', epoch ,100. * correct / len(train_loader.dataset))
    return 100. * correct / len(train_loader.dataset)




def test(PARAMS, model,criterion, device, test_loader,optimizer,epoch,best_acc):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for batch_idx, (img, cluster,  target) in enumerate(tqdm(test_loader)):
            img, target = img.to(device), target.to(device)
            cluster =  [item.to(device) for item in cluster ]
            output = model(img,cluster)
            # output = model(img)

            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Save the first input tensor in each test batch as an example image

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    print('Test Accuracy ',  100. * correct / len(test_loader.dataset))
    print('Test Loss ',  test_loss)

    current_acc = 100. * correct / len(test_loader.dataset)

    checkpoint = {
        'best_acc': best_acc,    
        'epoch': epoch + 1,
        'model': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }

    return current_acc

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def main():
    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=str, default = 'vgg16')
    parser.add_argument('--partion', type=float, default=0.5)
    parser.add_argument('--bs', type=int, default=8)
    parser.add_argument('--fixed',type=boolean_string, default=False)
    parser.add_argument('--Augmentation',type=boolean_string, default=False)
    parser.add_argument('--debug',type=boolean_string, default=False)
    args = parser.parse_args()

    


    
    PARAMS = {'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'bs': args.bs,
                'epochs':50,
                'lr': 0.0006,
                'momentum': 0.5,
                'log_interval':10,
                'criterion':F.cross_entropy,
                'partion':args.partion,
                'model_name': str(args.model) ,
                'fixed':args.fixed,
                'Augmentation': args.Augmentation,
                }
    tags =   PARAMS['model_name']   +'_'+ "fixed_" +str(PARAMS['fixed']) +'_'+ 'aug_' + str(PARAMS['Augmentation'])


 # Training settings

    if PARAMS['Augmentation']:
        train_transform = transforms.Compose(
                        [ 
                            transforms.ToPILImage(),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(0.4, 0.4, 0.4),
                            transforms.Resize((224,224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])
    else:
        train_transform = transforms.Compose(
                [ 
                    transforms.ToPILImage(),
                    transforms.Resize((224,224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])
    test_transform = transforms.Compose(
                    [ 
                        transforms.ToPILImage(),
                        transforms.Resize((224,224)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])


    train_dataset = RS_Dataset(
        root='thin_cloud/train_img',transform = train_transform)
    test_dataset = RS_Dataset(
        root='thin_cloud/test_img',transform = test_transform)

    print(PARAMS)
    train_loader = DataLoader(train_dataset,  batch_size=PARAMS['bs'], shuffle=True, num_workers=4, pin_memory = True )
    test_loader =  DataLoader(test_dataset, batch_size=PARAMS['bs'], shuffle=True,  num_workers=4, pin_memory = True  )




    num_classes = len(train_dataset.classes)
    # model = SiameseNetwork(base_model = PARAMS['model_name'], num_classes = num_classes).to(PARAMS['DEVICE'])
    model = SiameseNetwork(base_model = PARAMS['model_name'], num_classes = num_classes, fixed = PARAMS['fixed']).to(PARAMS['DEVICE'] )

    model = model.to(PARAMS['DEVICE'])   

    optimizer = optim.SGD(model.parameters(), lr=PARAMS['lr'], momentum=PARAMS['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.9)
    criterion =  F.cross_entropy
    current_acc = 0

    for epoch in range(1, PARAMS['epochs'] + 1):
        train(PARAMS, model,criterion, PARAMS['DEVICE'], train_loader, optimizer, epoch)
        current_acc = test(PARAMS, model,criterion, PARAMS['DEVICE'], test_loader,optimizer,epoch,current_acc)
        scheduler.step()
    torch.save(model, 'new_saved_models/{}_{}_{}_proposed_nodiff.pth'.format(date.today(),PARAMS['model_name'],round(current_acc,2)))



if __name__ == '__main__':
    main()