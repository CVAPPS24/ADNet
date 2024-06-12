import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial
import numpy as np


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def conv1x1x1(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=1,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class MHSA_3D(nn.Module):
    def __init__(self, n_dims, width=5, height=5,depth=5, heads=4):
        super(MHSA_3D, self).__init__()
        self.heads = heads

        self.query = nn.Conv3d(n_dims, n_dims, kernel_size=1)
        self.key = nn.Conv3d(n_dims, n_dims, kernel_size=1)
        self.value = nn.Conv3d(n_dims, n_dims, kernel_size=1)

        self.rel_d = nn.Parameter(torch.randn([1, heads, n_dims // heads, depth, 1 , 1]), requires_grad=True)
        self.rel_h = nn.Parameter(torch.randn([1, heads, n_dims // heads, 1 , 1, height]), requires_grad=True)
        self.rel_w = nn.Parameter(torch.randn([1, heads, n_dims // heads ,1 , width, 1]), requires_grad=True)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        n_batch, C, depth, width, height = x.size()
        q = self.query(x).view(n_batch, self.heads, C // self.heads, -1)
        k = self.key(x).view(n_batch, self.heads, C // self.heads, -1)
        v = self.value(x).view(n_batch, self.heads, C // self.heads, -1)
        content_content = torch.matmul(q.permute(0, 1, 3, 2), k)
        content_position = (self.rel_h + self.rel_w + self.rel_d).view(1, self.heads, C // self.heads, -1).permute(0, 1, 3, 2)
        content_position = torch.matmul(content_position, q)

        energy = content_content + content_position
        attention = self.softmax(energy)
        out = torch.matmul(v, attention.permute(0, 1, 3, 2))
        out = out.view(n_batch, C, depth, width, height)

        return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None, heads=4, mhsa=False, resolution=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        if not mhsa:
            self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        else:
            self.conv2 = nn.ModuleList()
            if stride == 2:
                self.conv2.append(MHSA_3D(planes, depth=int(resolution[0]//2), width=int(resolution[1]//2), height=int(resolution[2]//2), heads=heads))
            else:
                self.conv2.append(MHSA_3D(planes, depth=int(resolution[0]), width=int(resolution[1]),height=int(resolution[2]), heads=heads))

            self.conv2 = nn.Sequential(*self.conv2)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

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


class Temp(nn.Module):
    def __init__(self, block, layers,  shortcut_type='B'):
        self.inplanes=128
        super(Temp, self).__init__()
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, dilation=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return x

class Temp2(nn.Module):
    def __init__(self, block, layers, heads,resolution,mhsa, shortcut_type='B'):
        self.inplanes=256
        self.r = resolution
        super(Temp2, self).__init__()
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=2, dilation=2,heads=heads,mhsa=mhsa)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1,heads=4,mhsa=False):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample, heads=heads, mhsa=mhsa, resolution=self.r))
        if stride == 2:
            self.r[0] /= 2
            self.r[1] /= 2
            self.r[2] /= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation, heads=heads, mhsa=mhsa, resolution=self.r))

        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.layer4(x)
        x = self.avgpool(x)
        return x



class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda="cuda:1",
                 resolution=np.array([160., 160., 160.]),
                 heads=4,
                 mhsa=False):
        self.inplanes = 64
        self.no_cuda = no_cuda
        self.resolution = resolution
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        if self.conv1.stride[0] == 2:
            self.resolution[0] /= 2
        if self.conv1.stride[1] == 2:
            self.resolution[1] /= 2
        if self.conv1.stride[2] == 2:
            self.resolution[2] /= 2

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)

        if self.maxpool.stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2
            self.resolution[2] /= 2

        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, dilation=2)
        self.resolution2 = self.resolution.copy()
        self.branch_1 = Temp2(block,layers,heads,self.resolution,mhsa=mhsa).to(self.no_cuda)
        self.branch_2 = Temp2(block,layers,heads,self.resolution2,mhsa=mhsa).to(self.no_cuda)

        self.fc1 = nn.Linear(512 * block.expansion, 1)
        self.fc2 = nn.Linear(512 * block.expansion, num_seg_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = []
        layers.append(block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample))
        if stride == 2:
            self.resolution[0] /= 2
            self.resolution[1] /= 2
            self.resolution[2] /= 2
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x1=self.branch_1(x)
        x2=self.branch_2(x)

        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)

        tensor_1 = x1
        tensor_2 = x2

        x1 = self.fc1(x1)
        x1=x1.squeeze(-1)
        x2 = self.fc2(x2)

        return x1,x2,tensor_1,tensor_2


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def resnet18(num_classes=3,no_cuda="cuda:1", resolution=np.array([160,160,160]), heads=4,**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2],no_cuda=no_cuda, num_seg_classes=num_classes, resolution=resolution, heads=heads,mhsa=True, **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model

def get_n_params(model):
    pp=0
    for p in list(model.parameters()):
        nn=1
        for s in list(p.size()):
            nn = nn*s
        pp += nn
    return pp


