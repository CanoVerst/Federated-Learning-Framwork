import pickle
import torch
import numpy as np
import torchvision
from typing import OrderedDict
import collections
import torch.nn as nn
import torch.nn.functional as F

class Lenet5Model(nn.Module):
    """The LeNet-5 model.
    Arguments:
        num_classes (int): The number of classes. Default: 10.
    """
    def __init__(self, num_classes=10):
        super().__init__()

        # We pad the image to get an input size of 32x32 as for the
        # original network in the LeCun paper
        self.conv1 = nn.Conv2d(in_channels=1,
                               out_channels=6,
                               kernel_size=5,
                               stride=1,
                               padding=2,
                               bias=True)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(in_channels=6,
                               out_channels=16,
                               kernel_size=5,
                               stride=1,
                               padding=0,
                               bias=True)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(in_channels=16,
                               out_channels=120,
                               kernel_size=5,
                               bias=True)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(120, 84)
        self.relu4 = nn.ReLU()
        self.fc5 = nn.Linear(84, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['relu1'] = self.relu1
        self.layerdict['pool1'] = self.pool1
        self.layerdict['conv2'] = self.conv2
        self.layerdict['relu2'] = self.relu2
        self.layerdict['pool2'] = self.pool2
        self.layerdict['conv3'] = self.conv3
        self.layerdict['relu3'] = self.relu3
        self.layerdict['flatten'] = self.flatten
        self.layerdict['fc4'] = self.fc4
        self.layerdict['relu4'] = self.relu4
        self.layerdict['fc5'] = self.fc5
        self.layers.append('conv1')
        self.layers.append('relu1')
        self.layers.append('pool1')
        self.layers.append('conv2')
        self.layers.append('relu2')
        self.layers.append('pool2')
        self.layers.append('conv3')
        self.layers.append('relu3')
        self.layers.append('flatten')
        self.layers.append('fc4')
        self.layers.append('relu4')
        self.layers.append('fc5')
    
    def flatten(self, x):
        #Flatten the tensor.
        return x.view(x.size(0), -1)
    """
    def forward(self, x):
        #Forward pass.
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.flatten(x)
        x = self.fc4(x)
        x = self.relu4(x)
        x = self.fc5(x)

        return F.log_softmax(x, dim=1)

    def forward_to(self, x, cut_layer):
        #Forward pass, but only to the layer specified by cut_layer.
        layer_index = self.layers.index(cut_layer)
        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    def forward_from(self, x, cut_layer):
        #Forward pass, starting from the layer specified by cut_layer.
        layer_index = self.layers.index(cut_layer)
        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)
        return F.log_softmax(x, dim=1)
    """
    @staticmethod
    def get_model(num_classes = 10):
        """Obtaining an instance of this model."""
        return Lenet5Model(num_classes)
        
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes,
                               planes,
                               kernel_size=3,
                               stride=stride,
                               padding=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes,
                               self.expansion * planes,
                               kernel_size=1,
                               bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes,
                          self.expansion * planes,
                          kernel_size=1,
                          stride=stride,
                          bias=False), nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResnetModel(nn.Module):

    def __init__(self, block, num_blocks, num_classes=10):
        super().__init__()

        self.in_planes = 64

        self.conv1 = nn.Conv2d(3,
                               64,
                               kernel_size=3,
                               stride=1,
                               padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512 * block.expansion, num_classes)

        # Preparing named layers so that the model can be split and straddle
        # across the client and the server
        self.layers = []
        self.layerdict = collections.OrderedDict()
        self.layerdict['conv1'] = self.conv1
        self.layerdict['bn1'] = self.bn1
        self.layerdict['relu'] = F.relu
        self.layerdict['layer1'] = self.layer1
        self.layerdict['layer2'] = self.layer2
        self.layerdict['layer3'] = self.layer3
        self.layerdict['layer4'] = self.layer4
        self.layers.append('conv1')
        self.layers.append('bn1')
        self.layers.append('relu')
        self.layers.append('layer1')
        self.layers.append('layer2')
        self.layers.append('layer3')
        self.layers.append('layer4')

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    def forward_to(self, x, cut_layer):
        """Forward pass, but only to the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)

        for i in range(0, layer_index + 1):
            x = self.layerdict[self.layers[i]](x)
        return x

    def forward_from(self, x, cut_layer):
        """Forward pass, starting from the layer specified by cut_layer."""
        layer_index = self.layers.index(cut_layer)
        for i in range(layer_index + 1, len(self.layers)):
            x = self.layerdict[self.layers[i]](x)

        out = F.avg_pool2d(x, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

    @staticmethod
    def is_valid_model_type(model_type):
        return (model_type.startswith('resnet_')
                and len(model_type.split('_')) == 2
                and int(model_type.split('_')[1]) in [18, 34, 50, 101, 152])

    @staticmethod
    def get_model(model_type, num_classes = 10):
        if not ResnetModel.is_valid_model_type(model_type):
            raise ValueError(
                'Invalid Resnet model type: {}'.format(model_type))

        resnet_type = int(model_type.split('_')[1])

        if resnet_type == 18:
            return ResnetModel(BasicBlock, [2, 2, 2, 2], num_classes)
        elif resnet_type == 34:
            return ResnetModel(BasicBlock, [3, 4, 6, 3], num_classes)
        elif resnet_type == 50:
            return ResnetModel(Bottleneck, [3, 4, 6, 3], num_classes)
        elif resnet_type == 101:
            return ResnetModel(Bottleneck, [3, 4, 23, 3], num_classes)
        elif resnet_type == 152:
            return ResnetModel(Bottleneck, [3, 8, 36, 3], num_classes)

class Inceptionv3Model():
    """The Inception v3 model."""
    @staticmethod
    def get_model(*args):
        """Obtaining an instance of this model."""

        return torchvision.models.inception_v3(pretrained=False,
                                               aux_logits=False,
                                               transform_input=False)


def get_model_ins(model_name, num_classes):
    model_ins = None
    if model_name == 'lenet5':
        model_ins = Lenet5Model.get_model(num_classes)
        return model_ins
    elif model_name == 'resnet_18':
        model_ins = ResnetModel.get_model(model_name, num_classes)
        return model_ins

    return model_ins

def len_from_shape(shapes_dict):
    #Provided with a dict of shapes, the function returns a list of length
    len_list = []
    for shape in shapes_dict.values():
        len_list.append(int(np.prod(shape)))
    return len_list

def get_shapes_dict(model_name, num_classes):
    model_ins = get_model_ins(model_name, num_classes)
    state_dict = model_ins.cpu().state_dict()
    shapes_dict = OrderedDict()
    for weight_name, weight in state_dict.items():
        shapes_dict[weight_name] = weight.size()
    return shapes_dict

def export_target_model(model_name, num_classes):
    checkpoint_path = "./checkpoints"
    filename = f"{checkpoint_path}/{model_name}_est_{28154}_{2}.pth"
    with open(filename, 'rb') as est_file:
        est_model = pickle.load(est_file)
        est_file.close()
    
    shapes_dict = get_shapes_dict(model_name, num_classes)
    len_list = len_from_shape(shapes_dict)
    est_model = torch.split(est_model, len_list)
    weight_index = 0
    rebuilt_model = OrderedDict()

    for name, shape in shapes_dict.items():
        rebuilt_model[name] = est_model[weight_index].reshape(shape)
        weight_index = weight_index + 1

    return rebuilt_model


exported_model = export_target_model('resnet_18', 100)

"""
print(rebuilt_model.keys())
for value in rebuilt_model.values():
    print(value.size())
"""