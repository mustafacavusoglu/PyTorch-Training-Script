import gc
import os
import torch
import numpy as np
import torch.nn as nn
from tqdm import tqdm
from pprint import pprint
from torch.cuda import amp
import torch.optim as optim
import matplotlib.pyplot as plt
from collections import defaultdict
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from  segmentation_models_pytorch import utils 
from segmentation_models_pytorch.datasets import SimpleOxfordPetDataset


root = "."
#SimpleOxfordPetDataset.download(root)


# init train, val, test sets
train_dataset = SimpleOxfordPetDataset(root, "train")
valid_dataset = SimpleOxfordPetDataset(root, "valid")
test_dataset = SimpleOxfordPetDataset(root, "test")

# It is a good practice to check datasets don`t intersects with each other
assert set(test_dataset.filenames).isdisjoint(set(train_dataset.filenames))
assert set(test_dataset.filenames).isdisjoint(set(valid_dataset.filenames))
assert set(train_dataset.filenames).isdisjoint(set(valid_dataset.filenames))

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")
print(f"Test size: {len(test_dataset)}")

n_cpu = os.cpu_count()
train_dataloader = DataLoader(train_dataset, batch_size=2, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=2, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=2, shuffle=False)


# sample = train_dataset[0]
# plt.subplot(1,2,1)
# plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
# plt.subplot(1,2,2)
# plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

# sample = valid_dataset[0]
# plt.subplot(1,2,1)
# plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
# plt.subplot(1,2,2)
# plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

# sample = test_dataset[0]
# plt.subplot(1,2,1)
# plt.imshow(sample["image"].transpose(1, 2, 0)) # for visualization we have to transpose back to HWC
# plt.subplot(1,2,2)
# plt.imshow(sample["mask"].squeeze())  # for visualization we have to remove 3rd dimension of mask
# plt.show()

loss_fn = smp.losses.SoftBCEWithLogitsLoss()
jaccard = utils.metrics.IoU(threshold = .5)

device = torch.device('cuda')
model = smp.Unet(in_channels=3, classes=1, activation='sigmoid', encoder_name='resnet18')
model.to(device)
model.train()
optimizer = optim.Adam(model.parameters(), lr = 0.0007)


JaccardLoss = smp.losses.JaccardLoss(mode='multilabel')
DiceLoss    = smp.losses.DiceLoss(mode='multilabel')
BCELoss     = smp.losses.SoftBCEWithLogitsLoss()
LovaszLoss  = smp.losses.LovaszLoss(mode='multilabel', per_image=False)
TverskyLoss = smp.losses.TverskyLoss(mode='multilabel', log_loss=False)

def dice_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    den = y_true.sum(dim=dim) + y_pred.sum(dim=dim)
    dice = ((2*inter+epsilon)/(den+epsilon)).mean(dim=(1,0))
    return dice

def iou_coef(y_true, y_pred, thr=0.5, dim=(2,3), epsilon=0.001):
    y_true = y_true.to(torch.float32)
    y_pred = (y_pred>thr).to(torch.float32)
    inter = (y_true*y_pred).sum(dim=dim)
    union = (y_true + y_pred - y_true*y_pred).sum(dim=dim)
    iou = ((inter+epsilon)/(union+epsilon)).mean(dim=(1,0))
    return iou

def loss_fn(y_pred, y_true):
    return 0.5*BCELoss(y_pred, y_true) + 0.5*TverskyLoss(y_pred, y_true)


def toNumpy(tensor):
    return tensor.cpu().detach().numpy()

def trainOneEpoch(model, optimizer, dataloader, device, epoch):
    model.train()
    scaler = amp.GradScaler()
    datasetSize = 0
    runningLoss = .0
    runningJaccard = .0

    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Train ')
    for step, sample in pbar:     
        images = sample['image']    
        masks = sample['mask']    
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = loss_fn(y_pred, masks)
            jaccard = toNumpy(iou_coef(y_pred, masks))

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        runningLoss += (loss.item() * batch_size)
        runningJaccard += jaccard
        datasetSize += batch_size
        
        epochLoss = runningLoss / datasetSize
        epochJaccard = runningJaccard / datasetSize
        pbar.set_postfix(Step = f'{step}', Epoch =  f'{epoch}', Train_Loss= f'{epochLoss:.4f}', Train_IoU =  f'{epochJaccard:.4f}')

        torch.cuda.empty_cache()
        gc.collect()
    
    return epochLoss, epochJaccard, epoch


@torch.no_grad()
def validOneEpoch(model, dataloader, device, epoch):
    model.eval()

    datasetSize = 0
    runningLoss = .0
    runningJaccard = .0


    pbar = tqdm(enumerate(dataloader), total=len(dataloader), desc='Validaton')
    for step, sample in pbar:     
        images = sample['image']    
        masks = sample['mask']    
        images = images.to(device, dtype=torch.float)
        masks  = masks.to(device, dtype=torch.float)
        
        optimizer.zero_grad()
        batch_size = images.size(0)
        
        with amp.autocast(enabled=True):
            y_pred = model(images)
            loss   = loss_fn(y_pred, masks)
            jaccard = toNumpy(iou_coef(y_pred, masks))


        runningLoss += (loss.item() * batch_size)
        runningJaccard += jaccard
        datasetSize += batch_size
        
        epochLoss = runningLoss / datasetSize
        epochJaccard = runningJaccard / datasetSize

        pbar.set_postfix(Step = f'{step}', Epoch =  f'{epoch}', Val_Loss= f'{epochLoss:.4f}', Val_IoU =  f'{epochJaccard:.4f}')

        torch.cuda.empty_cache()
        gc.collect()
    
    return epochLoss, epochJaccard, epoch




history = defaultdict(list) 
for i in range(0, 5):
    loss, iou, epoch = trainOneEpoch(model, optimizer, train_dataloader, device, i)
    valLoss, valIou, epoch = validOneEpoch(model, valid_dataloader, device, i)

    history['trainIoU'].append(iou)
    history['trainLoss'].append(loss)
    history['valIoU'].append(valIou)
    history['valLoss'].append(valLoss)

plt.figure(figsize=(30, 5))
plt.subplot(121)
plt.plot(history['trainIoU'])
plt.plot(history['valIoU'])
plt.title('Model Accuracy')
plt.ylabel('IOU Score')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')

# Plot training & validation loss values
plt.subplot(122)
plt.plot(history['trainLoss'])
plt.plot(history['valLoss'])
plt.title(f'Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Val'], loc='upper left')
plt.savefig(f'deneme.png')
   