#import libraries
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import metrics
import torch.nn as nn
from torch import ones
from efficientnet_pytorch import EfficientNet
from torchvision.models import inception_v3
from torchvision.models import vgg16
import torchvision.models as models
import torch
from EyeData_preparation import EyeData
import pretrainedmodels
import numpy as np
import torch.nn.functional as F
from torch.nn.init import kaiming_normal
channels = 3
nb_filters = 32
kernel_size = (8, 8)
nb_classes = 3


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        self.mod = nn.Sequential(
            nn.Conv2d(3, nb_filters, (kernel_size[0], kernel_size[1]), stride=(4, 4), padding="valid"),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, (kernel_size[0], kernel_size[1])),
            nn.ReLU(),
            nn.Conv2d(nb_filters, nb_filters, (kernel_size[0], kernel_size[1])),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2)),
            nn.Conv2d(nb_filters, nb_filters, (16, 16)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(2592, 128),
            nn.Sigmoid(),
            nn.Dropout(0.25),
            nn.Linear(128, nb_classes),
            nn.Softmax()

        )



    def forward(self, x):
        x = self.mod(x)
        return x


import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
# from .utils import load_state_dict_from_url
from torch import Tensor
from typing import Any, List, Tuple
# from torchsummary import summary
from torchsummary import summary


class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, memory_efficient=False):
        super(_DenseLayer, self).__init__()
        self.add_module('norm1', nn.BatchNorm2d(num_input_features)),
        self.add_module('relu1', nn.ReLU(inplace=True)),
        self.add_module('conv1', nn.Conv2d(num_input_features, bn_size *
                                           growth_rate, kernel_size=1, stride=1,
                                           bias=False)),
        self.add_module('norm2', nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module('relu2', nn.ReLU(inplace=True)),
        self.add_module('conv2', nn.Conv2d(bn_size * growth_rate, growth_rate,
                                           kernel_size=3, stride=1, padding=1,
                                           bias=False)),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        "Bottleneck function"
        # type: (List[Tensor]) -> Tensor
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))  # noqa: T484
        return bottleneck_output

    def forward(self, input):  # noqa: F811
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        bottleneck_output = self.bn_function(prev_features)
        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate,
                                     training=self.training)
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, memory_efficient=False):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module('denselayer%d' % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for name, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module('norm', nn.BatchNorm2d(num_input_features))
        self.add_module('relu', nn.ReLU(inplace=True))
        self.add_module('conv', nn.Conv2d(num_input_features, num_output_features,
                                          kernel_size=1, stride=1, bias=False))
        self.add_module('pool', nn.AvgPool2d(kernel_size=2, stride=2))


class S4nd(nn.Module):
    def __init__(self, growth_rate=[16, 16, 32, 32], block_config=(6, 6, 6, 6),
                 num_init_features=64, bn_size=4, drop_rate=[0.2, 0.2, 0.5, 0, 5], num_classes=5,
                 memory_efficient=False):

        super(S4nd, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=(3, 3), stride=1,
                                padding=1, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            # ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        ]))

        # Add multiple denseblocks based on config
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(growth_rate):
            block = _DenseBlock(
                num_layers=block_config[i],
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate[i],
                drop_rate=drop_rate[i],
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + block_config[i] * growth_rate[i]
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2

        # Final batch norm
        self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, 1024)
        self.classifier1 = nn.Linear(1024, 1024)
        self.classifier2 = nn.Linear(1024, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        features = self.features(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = self.classifier1(out)
        out = self.classifier2(out)
        return out


class DenseNet(nn.Module):
    def __init__(self, growth_rate=[32, 32, 32, 32], block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=[0, 0, 0, 0, 0], num_classes=3, memory_efficient=False):

        super(DenseNet, self).__init__()

        # Convolution and pooling part from table-1
        self.features = nn.Sequential(OrderedDict([
            ('conv0', nn.Conv2d(3, num_init_features, kernel_size=7, stride=2,
                                padding=3, bias=False)),
            ('norm0', nn.BatchNorm2d(num_init_features)),
            ('relu0', nn.ReLU(inplace=True)),
            ('pool0', nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),

        ]))

        # Add multiple denseblocks based on config
        # for densenet-121 config: [6,12,24,16]
        num_features = num_init_features
        for i, num_layers in enumerate(growth_rate):
            block = _DenseBlock(
                num_layers=block_config[i],
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate[i],
                drop_rate=drop_rate[i],
                memory_efficient=memory_efficient
            )
            self.features.add_module('denseblock%d' % (i + 1), block)
            num_features = num_features + block_config[i] * growth_rate[i]
            if i != len(block_config) - 1:
                # add transition layer between denseblocks to
                # downsample
                trans = _Transition(num_input_features=num_features,
                                    num_output_features=num_features // 2)
                self.features.add_module('transition%d' % (i + 1), trans)
                num_features = num_features // 2
                # print(num_features)

        # Final batch norm
        # self.features.add_module('norm5', nn.BatchNorm2d(num_features))

        # Linear layer
        self.classifier = nn.Linear(num_features, num_classes)

        # Official init from torch repo.
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        sigmoid = torch.nn.Sigmoid()
        # softmax = torch.nn.softmax()
        # print(x.shape)
        features = self.features(x)
        # out = F.relu(features, inplace=True)
        # print(features.shape)
        out = F.adaptive_avg_pool2d(features, (1, 1))
        out = F.dropout(out, p=0.5,
                        training=self.training)
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        out = sigmoid(out)
        return out


class Net(nn.Module):
    def __init__(self, input_size=(3, 224, 224), nb_classes=2):  # not sure what nb_classes refers to

        super(Net, self).__init__()  # super class initialized

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d((3, 3))
        )

        ## Computing linear layer size
        self.flat_feats = self._get_flat_feats(input_size, self.features)

        self.classifer = nn.Sequential(
            nn.Linear(self.flat_feats, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(64, nb_classes)
        )



    ## Initializing weights
    def _weights_init(m):
        if isinstance(m, nn.Conv2d or nn.Linear):
            kaiming_normal(m.weight)
        elif isinstance(m, nn.BatchNorm2d or nn.BatchNorm1d):
            m.weight.data.fill_1(1)
            m.bias.data.zero_()





    def _get_flat_feats(self, in_size, feats):
        f = feats(torch.autograd.Variable(ones(1, *in_size)))
        return int(np.prod(f.size()[1:]))



    def forward(self, x):
        feats = self.features(x)
        flat_feats = feats.view(-1, self.flat_feats)
        out = self.classifer(flat_feats)
        return out


# This model will be right after the 8th layer of resnet.
class Net1(nn.Module):
    def __init__(self):
        super(Net1, self).__init__()
        resnet = models.resnet34(pretrained=True)
        # freezing parameters
        for param in resnet.parameters():  # resnet.parameters() is a generator object.
            param.requires_grad = False
        # convolutional layers of resnet34
        layers = list(resnet.children())[:9]
        #print(layers)
        # burada top modeldan kasti image'a yakin olan taraf.
        self.top_model = nn.Sequential(*layers).cuda()
        self.fc = nn.Linear(512, 2)

    def forward(self, x):

        x = F.relu(self.top_model(x))
        x = nn.AdaptiveAvgPool2d((1, 1))(x)
        x = x.view(x.size(0), -1)  # flattening
        x = self.fc(x)
        return x


import torch.nn as nn
import torch.utils.data as data
from PIL import Image
from torch.utils.data.sampler import RandomSampler, SequentialSampler
##import img.transformer as transformer
import csv
import torchvision
import torchvision.transforms as transforms
from torch.autograd import Variable
import torch.nn.functional as F
import torch




class BinaryClassifier(nn.Module):
    def __init__(self):
        super(BinaryClassifier, self).__init__()
        self.block1 = self.conv_block(c_in=3, c_out=16, dropout=0.1, kernel_size=5, stride=1, padding=2)
        self.block2 = self.conv_block(c_in=16, c_out=8, dropout=0.9, kernel_size=3, stride=1, padding=1)
        self.block3 = self.conv_block(c_in=8, c_out=4, dropout=0.1, kernel_size=3, stride=1, padding=1)
        self.lastcnn = nn.Conv2d(in_channels=4, out_channels=2, kernel_size=56, stride=1, padding=0)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
    def forward(self, x):
        x = self.block1(x)
        x = self.maxpool(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.maxpool(x)
        x = self.lastcnn(x)
        return x
    def conv_block(self, c_in, c_out, dropout,  **kwargs):
        seq_block = nn.Sequential(
            nn.Conv2d(in_channels=c_in, out_channels=c_out, **kwargs),
            nn.BatchNorm2d(num_features=c_out),
            nn.ReLU(),
            nn.Dropout2d(p=dropout)
        )
        return seq_block


if __name__ == "__main__":
    x = torch.rand((12, 3, 224, 224)).to("cuda")
    model = BinaryClassifier().cuda()
    print(model)
    print(torch.unsqueeze(model(x).squeeze(), 0).size())