"""
resnet for 1-d signal data, pytorch version

SeonWoo Lee, Oct 2020
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader



def calc_out_len(i, k, s, p, d=1):
    o = (i + 2*p - k - (k-1)*(d-1))/s + 1
    return o

def pad_len_for_the_same_length(i, k, s, d=1):
    p = ((i-1)*s -i + k + (k-1)*(d-1)) / 2
    return p


class MyConv1dPadSame(nn.Module):
    """
    extend nn.Conv1d to support SAME padding
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride, groups=1):
        super(MyConv1dPadSame, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.groups = groups
        self.conv = torch.nn.Conv1d(
            in_channels=self.in_channels, 
            out_channels=self.out_channels, 
            kernel_size=self.kernel_size, 
            stride=self.stride, 
            groups=self.groups)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.conv(net)

        return net

class MyMaxPool1dPadSame(nn.Module):
    """
    extend nn.MaxPool1d to support SAME padding
    """
    def __init__(self, kernel_size):
        super(MyMaxPool1dPadSame, self).__init__()
        self.kernel_size = kernel_size
        self.stride = 1
        self.max_pool = torch.nn.MaxPool1d(kernel_size=self.kernel_size)

    def forward(self, x):
        
        net = x
        
        # compute pad shape
        in_dim = net.shape[-1]
        out_dim = (in_dim + self.stride - 1) // self.stride
        p = max(0, (out_dim - 1) * self.stride + self.kernel_size - in_dim)
        pad_left = p // 2
        pad_right = p - pad_left
        net = F.pad(net, (pad_left, pad_right), "constant", 0)
        
        net = self.max_pool(net)
        
        return net
    
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv1d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(planes)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(planes)
        self.conv3 = nn.Conv1d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm1d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



    
class BasicBlock(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm1d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv1d(planes, planes, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(planes)
        self.downsample = downsample
        self.stride = stride
        """
        # the first conv
        self.bn1 = nn.BatchNorm1d(inplanes)
        self.relu1 = nn.ReLU()
        self.do1 = nn.Dropout(p=0.5)
        self.conv1 = MyConv1dPadSame(
            in_channels=inplanes, 
            out_channels=planes, 
            kernel_size=3, 
            stride=self.stride,
            groups=self.groups)

        # the second conv
        self.bn2 = nn.BatchNorm1d(planes)
        self.relu2 = nn.ReLU()
        self.do2 = nn.Dropout(p=0.5)
        self.conv2 = MyConv1dPadSame(
            in_channels=planes, 
            out_channels=planes, 
            kernel_size=3, 
            stride=1,
            groups=self.groups)
                
        self.max_pool = MyMaxPool1dPadSame(kernel_size=self.stride)
        """
    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet1d(nn.Module):
    def __init__(self, block, layers, n_air=8, 
                    n_sym = 7,
                    input_channels=9):
        self.inplanes = 32
        super(ResNet1d, self).__init__()
        self.n_air = n_air
        self.n_sym = n_sym
        self.conv1 = nn.Conv1d(
            input_channels, 32, kernel_size=7, stride=1, padding=2, bias=False)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.avgpool1 = nn.AdaptiveAvgPool1d(output_size=(n_sym))
        self.avgpool2 = nn.AdaptiveAvgPool1d(output_size=(self.n_air))
        self.air = nn.Linear(512 * block.expansion, self.n_air)
        self.sym = nn.Linear(512 * block.expansion, n_sym)
        self.out = nn.Linear(720, self.n_air+n_sym)
        self.softmax  = nn.Softmax(dim=-1) #
        self.rnn = nn.Sequential(
                nn.LSTM(input_size = 512*block.expansion, 
                        hidden_size = 720, 
                        num_layers =2))
        self.rnn2 = nn.Sequential(
                nn.LSTM(input_size = 512*block.expansion, 
                        hidden_size = 720, 
                        num_layers =2))
        if 1 == n_air:
            # compatible with nn.BCELoss
            self.softmax = nn.Sigmoid()
        else:
            # compatible with nn.CrossEntropyLoss
            self.softmax = nn.LogSoftmax(dim=1)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if not (stride == 1 and self.inplanes == planes * block.expansion):
            downsample = nn.Sequential(
                nn.Conv1d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    
    def forward(self, x):
        if len(x.size())>3:
            x = x.squeeze(1)
            x = x.permute(0, 2, 1)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x) #(bs, 2048(fs), 90)
        
        x = self.avgpool(x)
        """
        #RNN Type
        x1 = self.avgpool1(x) #(bs, fs, 7)n_sym
        x2 = self.avgpool2(x) #(bs, fs, 7)
        x1 = x1.permute(0, 2, 1) #bs, 7, fsn_sym
        x2 = x2.permute(0, 2, 1) #bs, 8, fs
        x1, _ = self.rnn(x1)
        x2, _ = self.rnn2(x2)
        x1 = x1.permute(0, 2, 1) #bs, 7, fsn_sym
        x2 = x2.permute(0, 2, 1) #bs, 8, fs
        """
        # x = torch.cat((x1, x2), 1)
        # x = self.out(x)
        # print(x.shape)
        x = x.view(x.size(0), -1)
        x1 = self.air(x)
        x2 = self.sym(x)
        
        return {'sym':x2, 'air':x1}


def resnet1d_10(pretrained=False, **kwargs):
    model = ResNet1d(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet1d_18(pretrained=False, **kwargs):
    model = ResNet1d(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model

def resnet50_1d( **kwargs):
    model = ResNet1d(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model

def resnet1d(model_name, num_classes, **kwargs):
    return{
        'resnet1d_18': resnet1d_18(num_classes=num_classes, input_channels=1),
        'resnet1d_10': resnet1d_10(num_classes=num_classes, input_channels=1),
    }[model_name]

def parameter_count(model):
    para_cnt = 0
    for item in model.state_dict().keys():
        para_size = model.state_dict()[item].size()
        cnt = 1
        for s in para_size:
            cnt = cnt * s

        para_cnt += cnt

    return para_cnt

if __name__ == "__main__":
    """
    pd =pad_len_for_the_same_length(1440,k=25, s=1)
    print(pd)
    a = calc_out_len(1440, k=25, s=1, p=pd)
    print(a)
    """
    model = resnet50_1d(n_sym=6, n_air=9, input_channels=9)
    x = torch.FloatTensor(2,1,9,1440)
    
    y = model(x)
    
    
    print(y)