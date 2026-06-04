from data_preprocess import annotate
from EyeData_preparation import EyeData, visualize
from model import init_model_tuned
from sklearn.utils import shuffle
import numpy as np
import pandas as pd
import torch
import torchvision
from torchvision import transforms, datasets
import albumentations as A
from torch.utils.data import Dataset
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import copy
import cv2
from tqdm import tqdm_notebook as tqdm
import random
import time
import sys
import os
import math
import shutil
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from sklearn.metrics import roc_auc_score, f1_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold
from model import optimizer_parameters
torch.cuda.empty_cache()
from model import initialize_model
from model0 import Model
import torch.nn.functional as F
# import EarlyStopping
from typing_extensions import Concatenate
from pytorchtools import EarlyStopping
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available. Training on CPU...')
    device = torch.device('cpu')
else:
    print('CUDA is available. Training on GPU...')
    device = torch.device('cuda:0')

# set seed
def seed_everything(seed = 23):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

seed = 27
seed_everything(seed)

batch_size = 16
image_size = 224

train_df = annotate('./DATASET/train')
train_df = shuffle(train_df)
test_df = annotate('./DATASET/validation')
test_df = shuffle(test_df)
criterion = nn.CrossEntropyLoss()

# train transformations
train_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomRotation((-360, 360)),
                                  transforms.RandomHorizontalFlip(),
                                  transforms.RandomVerticalFlip(),
                                  transforms.ToTensor()
                                 ])

# validation transformations
valid_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.ToTensor()
                                 ])

# create datasets
train_dataset = EyeData(data      = train_df,
                             directory = './DATASET/train',
                             transform = train_trans)
valid_dataset = EyeData(data       = test_df,
                            directory  = './DATASET/validation',
                            transform  = valid_trans)

# create data loaders
train_loader = torch.utils.data.DataLoader(train_dataset,
                                           batch_size  = batch_size,
                                           shuffle     = True)
valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                           batch_size  = batch_size,
                                           shuffle     = False)





def select_optimizer(parameters, optimize, eta):
    if(optimize == "adam"):
        optimizer =  torch.optim.Adam(parameters, lr=eta)
    if(optimize == "SGD"):
        optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=0.0001)
    if(optimize == "RMSprop"):
        optimizer = torch.optim.RMSprop(parameters, lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0.001, momentum=0, centered=False)
    return optimizer



def mean_over_epochs(accuracy, folds, epochs):
    epochs_loss = []

    for i in range(epochs):
        total = 0
        for x in range(folds):
            total += accuracy[epochs*x + i]
        epochs_loss.append(total/folds)
    return epochs_loss

def save_file(avg_train_losses, folds, max_epochs, lr, mode):
    for fold in range(folds):

        with open('./metadata/{}/{}_{}_{}_fold_{}'.format(model_name, mode, model_name, lr, fold+1), 'w') as f:
            for item in avg_train_losses[max_epochs*fold: max_epochs*(fold+1)]:
                f.write("%s\n" % item)
def reset_weights(m):
  '''
    Try resetting model weights to avoid
    weight leakage.
  '''
  for layer in m.children():
   if hasattr(layer, 'reset_parameters'):
    print(f'Reset trainable parameters of layer = {layer}')
    layer.reset_parameters()

def init_weights(m):
    if isinstance(m, nn.Embedding):
        nn.init.normal_(m.weight, mean=0.0, std=0.1) ## or simply use your layer.reset_parameters()
    if isinstance(m, nn.Linear):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(1 / m.in_features))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    if isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight, mean=0.0, std=np.sqrt(4 / m.in_channels))
        if m.bias is not None:
            nn.init.zeros_(m.bias)
def training_with_folds(model_name, optimize, lr, max_epochs=5, folds=5):
    num_folds = folds

    # creating splits
    skf = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=seed)
    splits = list(skf.split(train_df['Images'], train_df['Labels']))

    # placeholders
    oof_preds = np.zeros((len(train_df), 2))

    # timer
    cv_start = time.time()

    ##### PARAMETERS

    # learning rates
    eta = lr

    # scheduler
    step = 5
    gamma = 0.5
    torch.manual_seed(42)


    val_accuracy = []
    train_accuracy = []

    train_losses = []
    # to track the validation loss as the model trains
    valid_losses = []
    # to track the average training loss per epoch as the model trains
    avg_train_losses = []
    # to track the average validation loss per epoch as the model trains
    avg_valid_losses = []

    train_f1_scores = []
    val_f1_scores = []

    train_con_mat = []
    val_con_mat = []
    best_accu = -1
    # optimizer


    # initialize the early_stopping object
    early_stopping = EarlyStopping(patience=5, verbose=True)

    ##### CROSS-VALIDATION LOOP
    for fold in range(num_folds):

        best_accu = -1
        model = init_model_tuned(model_name=model_name)
        model = model.to(device)

        #model.apply(reset_weights)


        ####### DATA PREPARATION

        # display information
        print('-' * 30)
        print('FOLD {}/{}'.format(fold + 1, num_folds))
        print('-' * 30)


        # load splits
        data_train = train_df.iloc[splits[fold][0]].reset_index(drop=True)
        data_valid = train_df.iloc[splits[fold][1]].reset_index(drop=True)

        print(len(data_train),  "  ", len(data_valid))

        # create datasets
        train_dataset = EyeData(data=data_train,
                                directory='./DATASET/train',
                                transform=train_trans)
        valid_dataset = EyeData(data=data_valid,
                                directory='./DATASET/train',
                                transform=valid_trans)

        # create data loaders
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=True,
                                                   num_workers=1)
        valid_loader = torch.utils.data.DataLoader(valid_dataset,
                                                   batch_size=batch_size,
                                                   shuffle=False,
                                                   num_workers=1)

        parameters = optimizer_parameters(model_ft=model)
        optimizer = select_optimizer(parameters, optimize, eta)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step, gamma=gamma)


    ####### TRAINING AND VALIDATION LOOP
        for epoch in range(max_epochs):

            ##### PREPARATION

            # timer
            epoch_start = time.time()

            # reset losses
            trn_loss = 0.0
            val_loss = 0.0

            fold_preds = np.zeros((len(data_valid), 2))

            fold_train = np.zeros((len(data_train), 2))

            ##### TRAINING
            train_pred = []
            train_actual = []
            val_pred = []
            val_actual = []
            # switch regime
            model.train()
            train_correct = 0
            infer_correct = 0
            # loop through batches
            for batch_i, data in enumerate(train_loader):
                # extract inputs and labels
                inputs = data['image']
                labels = data['label'].view(-1)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)
                optimizer.zero_grad()

                # forward and backward pass
                with torch.set_grad_enabled(True):
                    preds = model(inputs).squeeze()
                    loss = criterion(preds, labels)

                    preds = torch.softmax(preds, 1).detach()
                    _, class_preds = preds.topk(1)
                    train_correct += torch.sum((class_preds.squeeze(1) == labels))
                    #print(labels, class_preds.squeeze(1))
                    loss.backward()
                    optimizer.step()




                # compute loss
                trn_loss += loss.item() * inputs.size(0)

                train_losses.append(trn_loss)
                for i in labels:
                    train_actual.append(i.item())

                for i in class_preds:
                    train_pred.append(i.item())


            ##### INFERENCE

            # initialize
            model.eval()

            # loop through batches
            for batch_i, data in enumerate(valid_loader):
                # extract inputs and labels
                inputs = data['image']
                labels = data['label'].view(-1)
                inputs = inputs.to(device, dtype=torch.float)
                labels = labels.to(device, dtype=torch.long)

                # predictions
                with torch.set_grad_enabled(False):
                    preds = torch.softmax(model(inputs).squeeze(), 1).detach()
                    _, class_preds = preds.topk(1)
                    infer_correct += torch.sum((class_preds.squeeze(1) == labels))


                # loss
                loss = criterion(preds, labels)
                val_loss += loss.item() * inputs.size(0)
                valid_losses.append(val_loss)

                for i in labels:
                    val_actual.append(i.item())

                for i in class_preds:
                    val_pred.append(i.item())

            # scheduler step
            scheduler.step()

            ##### EVALUATION
            print(infer_correct, len(valid_dataset))
            val_accu = infer_correct/len(data_valid)
            train_accu = train_correct/len(data_train)

            train_f1 = f1_score(train_actual, train_pred)
            val_f1 = f1_score(val_actual, val_pred)


            train_f1_scores.append(train_f1)
            val_f1_scores.append(val_f1)



            tn, fp, fn, tp = confusion_matrix(train_actual, train_pred).ravel()
            train_con_mat.append(str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\t"+str(tp))
            tn, fp, fn, tp = confusion_matrix(val_actual, val_pred).ravel()
            val_con_mat.append(str(tn)+"\t"+str(fp)+"\t"+str(fn)+"\t"+str(tp))

            """
            fold_preds_round = np.argmax(fold_preds, axis=1)
            val_train = metrics.accuracy_score(data_train['Labels'], np.argmax(fold_train, axis=1))
            val_frac = metrics.accuracy_score(data_valid['Labels'], fold_preds_round)
            """



            if (val_accu > best_accu):
                best_accu = val_accu
                torch.save(copy.deepcopy(model.state_dict()), './models/model_{}.pth'.format(model_name))
                print("Saving................", best_accu)


            val_accuracy.append(val_accu.item())
            train_accuracy.append(train_accu.item())
            #valid_losses.append(val_loss / len(test_df))
            #train_losses.append(trn_loss / len(train_df))

            #valid_loss = np.sum(valid_losses) / len(data_valid)
            #train_loss = np.sum(train_losses) / len(data_train)
            valid_loss = val_loss/len(data_valid)
            train_loss = trn_loss/len(data_train)

            avg_train_losses.append(train_loss)
            avg_valid_losses.append(valid_loss)



            # display info
            print(
                '- epoch {}/{} | lr = {} | trn_loss = {:.4f} | val_loss = {:.4f} | train_acc = {:.4f}  | valid_acc = {:.4f} | time = {:.4f}'.format(
                    epoch + 1, max_epochs, scheduler.get_lr()[len(scheduler.get_lr()) - 1],
                    train_loss, valid_loss, train_accu, val_accu,
                    (time.time() - epoch_start) / 60))

            valid_losses = []
            train_losses = []

            early_stopping(valid_loss, model)
            """
            if early_stopping.early_stop:
                print("Early stopping")
                break
            """
        print('Total elapsed time in {:.2f} minutes'.format((time.time() - cv_start) / 60))


    print('')
    print('Finished in {:.2f} minutes'.format((time.time() - cv_start) / 60))

    print("Best accuracy--------------", best_accu)
    modes = ["train_accu", "val_accu", "train_loss", "val_loss", "train_f1", "val_f1",
             "train_confusion_matrix", "val_confusion_matrix"]

    save_file(train_accuracy, folds, max_epochs, lr, modes[0])
    save_file(val_accuracy, folds, max_epochs, lr, modes[1])
    save_file(avg_train_losses, folds, max_epochs, lr, modes[2])
    save_file(avg_valid_losses, folds, max_epochs, lr, modes[3])
    save_file(train_f1_scores, folds, max_epochs, lr, modes[4])
    save_file(val_f1_scores, folds, max_epochs, lr, modes[5])
    save_file(train_con_mat, folds, max_epochs, lr, modes[6])
    save_file(val_con_mat, folds, max_epochs, lr, modes[7])

    avg_train_losses = mean_over_epochs(avg_train_losses, folds, max_epochs)
    avg_valid_losses = mean_over_epochs(avg_valid_losses, folds, max_epochs)
    train_accuracy = mean_over_epochs(train_accuracy, folds, max_epochs)
    val_accuracy = mean_over_epochs(val_accuracy, folds, max_epochs)
    #Saving files



    return avg_train_losses, avg_valid_losses, train_accuracy, val_accuracy





def inference(model_name, lr):
    model = init_model_tuned(train=False, model_name=model_name).to(device)
    model.load_state_dict(torch.load('./models/model_{}.pth'.format(model_name)))
    model = model.to(device)
    model.eval()

    count = 0

    test_predicted = []
    test_actual = []
    for i, data in enumerate(valid_loader):
      inputs = data['image']
      labels = data['label'].view(-1)

      inputs = inputs.to(device, dtype=torch.float)
      labels = labels.to(device, dtype=torch.long)

      with torch.set_grad_enabled(False):
          preds = model(inputs).squeeze().detach()
          _, class_preds = preds.topk(1)
          print(preds)
          count += torch.sum((class_preds.squeeze(1) == labels))

      for i in class_preds:
          test_predicted.append(i.item())

      for i in labels:
          test_actual.append(i.item())
    test_con_mat = []
    tn, fp, fn, tp = confusion_matrix(test_actual, test_predicted).ravel()
    test_con_mat.append(str(tn) + "\t" + str(fp) + "\t" + str(fn) + "\t" + str(tp))
    save_file(test_con_mat, 1, 1, lr, "test_confusion_matrix")
    print(count)

def plot(train_losses, valid_losses, train_accu, val_accu, number):
    # plot size
    fig = plt.figure(figsize=(15, 5))

    # plot loss dynamics
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses) + 1), train_losses, 'red', label='Training')
    plt.plot(range(1, len(train_losses) + 1), valid_losses, 'green', label='Validation')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    #plt.ylim(0, 2)  # consistent scale
    plt.xlim(0, len(train_losses) + 1)  # consistent scale
    plt.legend()

    # plot kappa dynamics
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accu) + 1), train_accu, 'red', label='train accuracy')
    plt.plot(range(1, len(train_accu) + 1), val_accu, 'blue', label='validation accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.xlim(0, len(train_accu) + 1)
    plt.legend()
    plt.savefig("./reports/{}/serial{}.pdf".format(model_name, number), bbox_inches='tight')
    plt.savefig("./reports/{}/serial{}.svg".format(model_name, number), bbox_inches='tight')


if __name__ == "__main__":
    #custom_model is bad
    models = ["squeezenet", "net1", "binaryclfr", "resnet", "resnet_drop", "cnnnet", "densenet", "idrid", "inception-v3", "enet-b4", "custom_model", "vgg", "custom_resnet", "CNN_Net", "resnet_v0", "Binary_Classifier", "alexnet"]
    model_name = models[5]
    print(len(train_df), len(test_df))
    lr = 1e-2
    train_losses, valid_losses, train_accu, val_accu = training_with_folds(model_name=model_name, optimize="adam", lr=lr, max_epochs=50, folds=10)
    #train_losses, valid_losses, train_accu, val_accu = training(model_name=model_name, optimize="adam", lr=1e-3, max_epochs=50)

    inference(model_name=model_name, lr=lr)
    print(train_accu)
    print("--------------------------")
    print(val_accu)
    print("--------------------------")
    print("--------------------------")
    print(max(train_accu))
    print(max(val_accu))

    plot(train_losses, valid_losses, train_accu, val_accu, 1002)
