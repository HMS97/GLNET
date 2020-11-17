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
from torchvision.models import resnet50,alexnet,vgg16


def train(PARAMS, model, criterion, device, train_loader, optimizer, epoch):
    t0 = time.time()
    model.train()
    correct = 0

    for batch_idx, (img, target) in enumerate(tqdm(train_loader)):
        img,  target = img.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(img)

        loss = criterion(output, target )
        loss.backward()
        optimizer.step()
        pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
        correct += pred.eq(target.view_as(pred)).sum().item()
    

    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f} , {:.2f} seconds'.format(
        epoch, batch_idx * len(img), len(train_loader.dataset),
        100. * batch_idx / len(train_loader), loss.item(),time.time() - t0))


def test(PARAMS, model,criterion, device, test_loader,optimizer,epoch,best_acc):
    model.eval()
    test_loss = 0
    correct = 0

    example_images = []
    with torch.no_grad():
        for batch_idx, (img, target) in enumerate(tqdm(test_loader)):
            img, target = img.to(device), target.to(device)
            output = model(img)

            test_loss += criterion(output, target).item() # sum up batch loss
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            # Save the first input tensor in each test batch as an example image

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
 

    current_acc = 100. * correct / len(test_loader.dataset)
    return current_acc

def main():


    parser = argparse.ArgumentParser(description='manual to this script')
    parser.add_argument('--model', type=str, default = 'vgg16')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--evaluate_model', type=str)
    parser.add_argument('--dataset', type=str, default='rsscn7')

    args = parser.parse_args()

    PARAMS = {'DEVICE': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
                'bs': 8,
                'epochs':50,
                'lr': 0.0006,
                'momentum': 0.5,
                'log_interval':10,
                'criterion':'cross_entropy',
                'model_name': args.model,
                'dataset': args.dataset,
                }


    # Training settings
    train_transform = transforms.Compose(
                    [ 
                        transforms.RandomHorizontalFlip(),
                         transforms.ColorJitter(0.4, 0.4, 0.4),
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])
    test_transform = transforms.Compose(
                    [ 
                        transforms.Resize((256,256)),
                        transforms.ToTensor(),
                        transforms.Normalize([0.4850, 0.4560, 0.4060], [0.2290, 0.2240, 0.2250])])


    if args.dataset == 'rsscn7':
  
        # train_dataset = datasets.ImageFolder(root='data/thick_removal',transform = train_transform)
        # test_dataset = datasets.ImageFolder(root='data/thick_removal',transform = test_transform)

        train_dataset = datasets.ImageFolder(root='data/rsscn7/train_dataset/',transform = train_transform)
        test_dataset = datasets.ImageFolder(root='data/rsscn7/test_dataset/',transform = test_transform)
    elif args.dataset == 'ucm':
    
        train_dataset = datasets.ImageFolder(root='data/ucm/train_dataset/',transform = train_transform)
        test_dataset = datasets.ImageFolder(root='data/ucm/test_dataset/',transform = test_transform)
    print(PARAMS)
    train_loader = DataLoader(train_dataset,  batch_size=PARAMS['bs'], shuffle=True, num_workers=4, pin_memory = True )
    test_loader =  DataLoader(test_dataset, batch_size=PARAMS['bs'], shuffle=True,  num_workers=4, pin_memory = True  )



    num_classes = len(train_dataset.classes)
    if PARAMS['model_name'] == 'vgg16':
        model = models.vgg16(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'resnet50':
        model = models.resnet50(pretrained=True)
        model.fc =  nn.Linear(in_features=2048, out_features=num_classes, bias=True)
    elif PARAMS['model_name'] == 'alexnet':
        model = models.alexnet(pretrained=True)
        model.classifier[-1] =  nn.Linear(in_features=4096, out_features=num_classes, bias=True)    

    
   
    model = model.to(PARAMS['DEVICE'])   
    optimizer = optim.SGD(model.parameters(), lr=PARAMS['lr'], momentum=PARAMS['momentum'])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = 7, gamma = 0.9)
    criterion =  F.cross_entropy
    acc = 0

    if not args.evaluate_model:
        for epoch in range(1, PARAMS['epochs'] + 1):
            train(PARAMS, model,criterion, PARAMS['DEVICE'], train_loader, optimizer, epoch)
            acc = test(PARAMS, model,criterion, PARAMS['DEVICE'], test_loader,optimizer,epoch,acc)
            scheduler.step()
        torch.save(model.state_dict(), 'saved_models/{}_{}_{}_{}_baseline.pth'.format(args.dataset, date.today(), PARAMS['model_name'], round(acc,2)))
        # torch.save(model, 'saved_models/{}_{}_{}_{}_baseline.pth'.format(args.dataset, date.today(), PARAMS['model_name'], round(acc,2)))
    else:
        model = torch.load(args.evaluate_model)
        acc = test(PARAMS, model,criterion, PARAMS['DEVICE'], test_loader, optimizer, 0, acc)
        print(f'the evalutaion acc is {acc}')


if __name__ == '__main__':
    main()