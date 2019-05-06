from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from tensorboardX import SummaryWriter
import logging

logging.basicConfig(filename="app.log", filemode="w", level = logging.INFO, format="%(name)s - %(levelname)s - %(message)s")

plt.ion() 

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"


#TODO Test the tensorboardX code
writer = SummaryWriter('tensorboardX/exp-1')


# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

data_dir = "/data6/SuryaKrishnan/personal-trainer/data"
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val']}

dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
class_names = image_datasets['train'].classes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

logging.info("Device: {}".format(device))

def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(10)  # pause a bit so that plots are updated


# Get a batch of training data
inputs, classes = next(iter(dataloaders['train']))

# Make a grid from batch
out = torchvision.utils.make_grid(inputs)

# imshow(out, title=[class_names[x] for x in classes])


def data_stats(data_path):
    stats_dict = dict()
    dirs = os.listdir( data_path )
    for sub_dir in dirs:
        stats_dict[sub_dir] = dict()
        sub_dir_path = os.path.join(data_path, sub_dir)
        for each in os.listdir(sub_dir_path):
            class_path = os.path.join(sub_dir_path, each)
            stats_dict[sub_dir][each] = len(os.listdir(class_path))
    return stats_dict

print(data_stats(data_dir))
logging.info("Data Stats: {}".format(data_stats(data_dir)))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.savefig('confusion-matrix.png')
    return ax









def train_model(model, criterion, optimizer, scheduler, num_epochs=25):

    #TODO implement early stopping

    logging.info("Model Architecture - {}".format(model))
    logging.info("Criterion - {}".format(criterion))
    logging.info("Optimizer - {}".format(optimizer))
    logging.info("Scheduler - {}".format(scheduler))
    logging.info("Num epochs - {}".format(num_epochs))

    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = 0.0
    best_labels = np.array([])
    best_preds = np.array([])
    total_train_acc = np.array([])
    total_val_acc = np.array([])
    total_train_loss = np.array([])
    tota_val_loss = np.array([])

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                scheduler.step()

                # Set model to training mode
                model.train()  
            else:
                # Set model to evaluate mode
                model.eval()   

            running_loss = 0.0
            running_corrects = 0

            running_labels = np.array([])
            running_preds = np.array([])

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

                running_preds = np.append(running_preds, preds)
                running_labels = np.append(running_labels, labels)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == "train":
                total_train_acc = np.append(total_train_acc, epoch_acc)
                total_train_loss = np.append(total_train_loss, epoch_loss)

                writer.add_scalar('train-acc', epoch_acc, epoch)
                writer.add_scalar('train-loss', epoch_loss, epoch)

            else:
                total_val_acc = np.append(total_val_acc, epoch_acc)
                tota_val_loss = np.append(tota_val_loss, epoch_loss)

                writer.add_scalar('val-acc', epoch_acc, epoch)
                writer.add_scalar('val-loss', epoch_loss, epoch)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())
                torch.save(model.state_dict(), "./best_model.pth")
                best_preds = running_preds
                best_labels = running_labels
                print("type: {}".format(type(best_labels)))
                print("shape: {}".format(best_labels.shape))

        print()

    conf_matrix = plot_confusion_matrix(best_labels, best_preds, classes=class_names,
                      title='Confusion matrix, without normalization')
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    print('Best val Loss: {:4f}'.format(best_loss))
    print('Confusion Matrix: {}'.format(conf_matrix))

    logging.info('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    logging.info('Best val Acc: {:4f}'.format(best_acc))
    logging.info('Best val Loss: {:4f}'.format(best_loss))
    logging.info('Confusion Matrix: {}'.format(conf_matrix))

    # load best model weights
    model.load_state_dict(best_model_wts)
    plt.show()
    plt.pause(10)
    return model

model_conv = torchvision.models.resnet18(pretrained=True)

logging.info("Model Name: {}".format("resnet18"))

# use this code if you want to freeze the conv layer weights
for param in model_conv.parameters():
    param.requires_grad = False

logging.info("Froze Conv Layers: {}".format("True"))

# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model_conv.fc.in_features
model_conv.fc = nn.Linear(num_ftrs, 2)

model_conv = model_conv.to(device)

criterion = nn.CrossEntropyLoss()

# optimizer_conv = optim.SGD(model_conv.fc.parameters(), lr=0.001, momentum=0.9)
optimizer_conv = optim.Adam(model_conv.fc.parameters(), lr=0.001)

# Decay LR by a factor of 0.1 every 7 epochs
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_conv, step_size=7, gamma=0.1)

model_conv = train_model(model_conv, criterion, optimizer_conv,
                         exp_lr_scheduler, num_epochs=50)









