#import Libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch.nn as nn
from efficientnet_pytorch import EfficientNet
from torchvision.models import inception_v3
from torchvision.models import vgg16
import torchvision.models as models
import torch
from EyeData_preparation import EyeData
import pretrainedmodels
import numpy as np
import torch.nn.functional as F
from model0 import Net, DenseNet, Net1, BinaryClassifier
#Optimizer learning parameters
def optimizer_parameters(feature_extract=True, model_ft=None):
    params_to_update = model_ft.parameters()
    print("Params to learn:")
    if feature_extract:
        params_to_update = []
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                print("\t", name)
    else:
        for name, param in model_ft.named_parameters():
            if param.requires_grad == True:
                print("\t", name)

    return params_to_update




#make requires_grad false for pretrained layers
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False



class Binary_Classifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(720, 1024)
        self.fc2 = nn.Linear(1024, 2)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(x.shape[0],-1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return x

class IDRiDClassifier(nn.Module):

    def __init__(self, n_classes=2):
        super(IDRiDClassifier, self).__init__()

        self.classes = n_classes
        self.Avg = nn.AvgPool2d(4)
        self.ReLU = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm1d(1000)
        self.batchnorm2 = nn.BatchNorm1d(512)
        self.linear1 = nn.Linear(1000, 1000)
        self.linear2 = nn.Linear(1000, 512)
        self.linear3 = nn.Linear(512, self.classes)

        ## ResNet101 features extractor
        self.res50 = models.resnet50(pretrained=True)
        # Freezing all layers
        for child in list(self.res50.children()):
            for param in child.parameters():
                param.requires_grad = False
        # Removing the softmax layer
        self.res101 = nn.Sequential(*list(self.res50.children())[:-1])

    def forward(self, x):
        x = self.res50(x)
        x = x.view(-1, 1000)
        x = nn.Dropout(0.5)(self.ReLU(self.batchnorm1(self.linear1(x))))
        x = nn.Dropout(0.7)(self.ReLU(self.batchnorm2(self.linear2(x))))
        return torch.nn.functional.softmax(self.linear3(x))




# Inception-V3 Model

def Inception_v3_model(num_classes=3):
    model_ft = inception_v3(pretrained=True, aux_logits=False)
    set_parameter_requires_grad(model_ft, feature_extracting=True)
    # Handle the auxilary net
    #num_ftrs = model_ft.AuxLogits.fc.in_features
    #model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
    # Handle the primary net
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, num_classes)
    input_size = 299
    return model_ft


# EfficientNet
def EfficientNet_model(num_classes=2, trn_layers=2, model_name="enet-b4"):
    # load pre-trained model
    model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=2)
    #model.load_state_dict(torch.load('./models/model_{}.bin'.format(model_name)))

    # freeze first layers
    for child in list(model.children())[:-trn_layers]:
        for param in child.parameters():
            param.requires_grad = False



    return model



def init_model_tuned(train=True, trn_layers=2, epochs=1, model_name=None):
    ### training mode
    if train == True:

        # load pre-trained model
        if(model_name == "binaryclfr"):
            model = BinaryClassifier()

        if (model_name == "net1"):
            model = Net1()
        if(model_name == "cnnnet"):
            model = Net()
        if(model_name == "densenet"):
            model = DenseNet()
        if (model_name == "idrid"):
            model = IDRiDClassifier()
        #model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=3)
        if(model_name == "custom_model"):
            model = CustomNet()
        if(model_name == "inception-v3"):
            model = Inception_v3_model()
        if(model_name == "enet-b4"):
            model = EfficientNet_model()
        if(model_name == "Binary_Classifier"):
            model = BinaryClassifier()

        models = ["squeezenet", "densenet", "resnet_drop", "resnet", "resnet_v0", "vgg", "alexnet"]

        if (model_name in models):
            model, input_size = initialize_model(model_name=model_name, num_classes=2, feature_extract=True,
                                                 use_pretrained=True)




        # freeze first layers
        """
        if (epochs >= 1):
            model.load_state_dict(torch.load('./models/model_{}.bin'.format(model_name)))
        for child in list(model.children())[:-trn_layers]:
            for param in child.parameters():
                param.requires_grad = False"""

    ### inference mode
    if train == False:
        """
        # load pre-trained model
        model = EfficientNet.from_pretrained('efficientnet-b4', num_classes=3)
        model.load_state_dict(torch.load('./models/model_{}.bin'.format(model_name)))

        # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        """
        if (model_name == "binaryclfr"):
            model = BinaryClassifier()

        if (model_name == "net1"):
            model = Net1()
        if(model_name == "cnnnet"):
            model = Net()
        if (model_name == "densenet"):
            model = DenseNet()

        if (model_name == "idrid"):
            model = IDRiDClassifier()

        if (model_name == "custom_model"):
            model = CustomNet()

        if(model_name == "inception-v3"):
            model = Inception_v3_model()


        if(model_name == "enet-b4"):
            model = EfficientNet_model()

        if (model_name == "Binary_Classifier"):
            model = BinaryClassifier()

        models = ["squeezenet", "densenet", "resnet", "resnet_drop", "resnet_v0", "vgg", "alexnet"]

        if (model_name in models):
            model, input_size = initialize_model(model_name=model_name, num_classes=2, feature_extract=True, use_pretrained=True)


            # freeze all layers
        for param in model.parameters():
            param.requires_grad = False
        #model.load_state_dict(torch.load('./models/model_{}.bin'.format(model_name)))

    return model

class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.inception = inception_v3(pretrained=True)
        self.inception.aux_logits = False
        self.inception.AuxLogits.fc = nn.ReLU()
        features = list(self.inception.children())[:15]
        self.inception = nn.Sequential(*features)
        set_parameter_requires_grad(self.inception, True)
        self.cn1 = nn.Conv2d(768, 2048, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0), bias=True)
        self.bn1 = nn.BatchNorm2d(2048)
        self.dp1 = nn.Dropout(0.5)
        self.cn2 = nn.Conv2d(2048, 64, kernel_size=(1, 1), padding=(0, 0))
        self.relu = nn.ReLU()
        self.cn3 = nn.Conv2d(64, 16, kernel_size=(1, 1), padding=(0, 0))
        self.cn4 = nn.Conv2d(16, 8, kernel_size=(1, 1), padding=(0, 0))
        self.cn5 = nn.Conv2d(8, 1, kernel_size=(1, 1), padding=(0, 0))
        self.sig = nn.Sigmoid()

        up_c2_w = torch.ones((2048, 1, 1, 1))
        self.up_c2 = nn.Conv2d(1, 2048, kernel_size=(1, 1), padding=(0, 0))
        self.up_c2.weight = nn.Parameter(up_c2_w, requires_grad=False)
        self.l1 = nn.Linear(2048, 128)
        self.f = nn.Linear(128,3)



    def forward(self, x):
        x = self.inception(x)
        x = self.cn1(x)
        bn1 = self.bn1(x)
        x = self.relu(self.cn2(self.dp1(bn1)))
        x = self.relu(self.cn3(x))
        x = self.relu(self.cn4(x))
        x = self.sig(self.cn5(x))
        att_n = self.up_c2(x)
        mask_features = torch.mul(att_n, bn1)


        gap_features = torch.mean(mask_features.view(mask_features.size(0), mask_features.size(1), -1), dim=2)
        gap_mask = torch.mean(att_n.view(att_n.size(0), att_n.size(1), -1), dim=2)
        x = gap_features/gap_mask
        x = nn.Dropout(0.25)(x)
        x = nn.Dropout(0.25)(self.relu(self.l1(x)))
        x = nn.Softmax(dim=1)(self.f(x))

        return x

def initialize_model(model_name, num_classes, feature_extract, use_pretrained=True):
    # Initialize these variables which will be set in this if statement. Each of these
    #   variables is model specific.
    model_ft = None
    input_size = 0

    if model_name == "resnet":
        model_ft = models.resnet101(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(num_ftrs, num_classes))

        input_size = 224

    elif model_name == "resnet_drop":
        model_ft = models.resnet18(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(num_ftrs, num_classes))

    elif model_name == "resnet_v0":
        """ Resnet18
        """
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 256), nn.Linear(256, 128), nn.Dropout(0.3), nn.Linear(128, 64), nn.Dropout(0.3), nn.Linear(64, num_classes))

        input_size = 224

    elif model_name == "alexnet":
        """ Alexnet
        """
        model_ft = models.alexnet(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "vgg":
        """ VGG11_bn
        """
        model_ft = models.vgg16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[6].in_features
        model_ft.classifier[6] = nn.Sequential(nn.Dropout(0.3), nn.Linear(num_ftrs, num_classes))
        input_size = 224

    elif model_name == "squeezenet":
        """ Squeezenet
        """
        model_ft = models.squeezenet1_0(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1,1), stride=(1,1))
        model_ft.num_classes = num_classes
        input_size = 224

    elif model_name == "densenet":
        """ Densenet
        """
        model_ft = models.densenet121(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier.in_features
        model_ft.classifier = nn.Linear(num_ftrs, num_classes)
        input_size = 224

    elif model_name == "inception":
        """ Inception v3
        Be careful, expects (299,299) sized images and has auxiliary output
        """
        model_ft = models.inception_v3(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        # Handle the auxilary net
        num_ftrs = model_ft.AuxLogits.fc.in_features
        model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
        # Handle the primary net
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs,num_classes)
        input_size = 299

    elif model_name == "custom_resnet": #-----------Not working fine....bad model
        model_ft = models.resnet152(pretrained=False)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Sequential(nn.Linear(num_ftrs, 3), nn.ReLU(), nn.Linear(3, num_classes), nn.LogSoftmax(dim=1))

        for name, child in model_ft.named_children():
            if name in ['fc']:
                # print(name + 'is unfrozen')
                for param in child.parameters():
                    param.requires_grad = True
            else:
                # print(name + 'is frozen')
                for param in child.parameters():
                    param.requires_grad = False


    else:
        print("Invalid model name, exiting...")
        exit()

    return model_ft, input_size

if __name__ == "__main__":
    dummy_input = torch.rand((16, 3, 224, 224)).to('cuda')
    models_list = ["squeezenet", "net1", "binaryclfr", "resnet", "resnet_drop", "cnnnet", "densenet", "idrid",
              "inception-v3", "enet-b4", "custom_model", "vgg", "custom_resnet", "CNN_Net", "resnet_v0",
              "Binary_Classifier", "alexnet"]
    model_name = models_list[-6]
    #model, input_size = initialize_model(model_name="custom_resnet", num_classes=3, feature_extract=True, use_pretrained=True)
    model = init_model_tuned(model_name=model_name).to('cuda')
    input_names = ['Image']
    output_names = ['Eye State']
    #torch.onnx.export(model, dummy_input, "./visualize/"+model_name+'.onnx', verbose=True, input_names=input_names, output_names=output_names)
    print(model)
    print(model(dummy_input).size())

    resnet = models.resnet34(pretrained=True)
    # freezing parameters
    model1 = nn.Sequential(*list(resnet.children()))
    print(model1)






