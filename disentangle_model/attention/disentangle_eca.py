import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


# class EfficientChannelAttention(nn.Module):           # Efficient Channel Attention module
#     def __init__(self, c, b=1, gamma=2):
#         super(EfficientChannelAttention, self).__init__()
#         t = int(abs((math.log(c, 2) + b) / gamma))
#         k = t if t % 2 else t + 1
#
#         self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
#         self.conv1 = nn.Conv1d(1, 1, kernel_size=k, padding=int(k/2), bias=False)
#         self.sigmoid = nn.Sigmoid()
#
#     def forward(self, x):
#         x = self.avg_pool(x)
#         x = self.conv1(x.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
#         out = self.sigmoid(x)
#         return out

class EfficientChannelAttention(nn.Module):
    def __init__(self,in_channel, gamma = 2 ,b = 1):
        super(EfficientChannelAttention, self).__init__()
      #use in_channel to adaptly calculate the kernel size ,more big in_channel more big kernel
        kernel_size = int(abs((math.log(in_channel,2) + b) / gamma))
        kernel_size = kernel_size if kernel_size % 2 else kernel_size + 1
        self.avg_pool = nn.AdaptiveAvgPool3d((1,1,1))
        self.conv = nn.Conv1d( 1, 1, kernel_size,padding=(kernel_size - 1) // 2, bias = False)
        self.sigmod = nn.Sigmoid()

    def forward(self, x):
        if len(x.shape) == 5 :
            b, c, d, w, h = x.size()
            avg = self.avg_pool(x).view([b,1,c]) #adjust to Sequence
        else :
            print("false")
            return None
        out = self.conv(avg)
        out = self.sigmod(out).view([b,c,1,1,1])
        return out * x



def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)  # 因为有BN层就不需要偏置了


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.channel = EfficientChannelAttention(planes).to("cuda:1")
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
        out = self.channel(out)
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
    def __init__(self, block, layers,  shortcut_type='B'):
        self.inplanes=256
        super(Temp2, self).__init__()
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
        x = self.layer4(x)
        x = self.avgpool(x)
        return x



class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 # sample_input_D,
                 # sample_input_H,
                 # sample_input_W,
                 num_seg_classes=3,
                 shortcut_type='B',
                 no_cuda="cuda:0"):
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=2, dilation=2)

        self.branch_1 = Temp2(block,layers).to(self.no_cuda) #cuda:1
        self.branch_2 = Temp2(block,layers).to(self.no_cuda) #cuda:1

        #self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
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

        return x1,x2,tensor_1,tensor_2 #x1回归 x2分类


def resnet10(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [1, 1, 1, 1], **kwargs)
    return model


def eca_resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


def resnet34(**kwargs):
    """Constructs a ResNet-34 model.
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    return model


if __name__ == '__main__':
    temp=torch.randn([1,1,160,160,160]).to("cuda:1")
    model = eca_resnet18(num_seg_classes=3).to("cuda:1")
    # model(temp)
    # print(y1.shape)
    # print(y2.shape)
    for name, module in model.named_children():
            print(name,module)

