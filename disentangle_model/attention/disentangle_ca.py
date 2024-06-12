import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import math
from functools import partial


class CA_Block(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CA_Block, self).__init__()

        self.reduction = reduction

        self.conv_1x1 = nn.Conv2d(in_channels=channel, out_channels=channel // reduction, kernel_size=1, stride=1,
                                  bias=False)

        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm2d(channel // reduction)

        self.F_d = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_h = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)
        self.F_w = nn.Conv3d(in_channels=channel // reduction, out_channels=channel, kernel_size=1, stride=1,
                             bias=False)

        self.sigmoid_d = nn.Sigmoid()
        self.sigmoid_h = nn.Sigmoid()
        self.sigmoid_w = nn.Sigmoid()

    def forward(self, x):
        b, c, d, h, w = x.size()
        c_out = c//self.reduction

        x_d = torch.mean(x, dim=2, keepdim=True).view(b,c,1,h*w)#b,c,1,h*w
        if not x_d.is_contiguous():
            x_d.contiguous()
        x_h = torch.mean(x, dim=4, keepdim=True).permute(0, 1, 4, 2,3).view(b,c,1,d*h) #b,c,d,h,1 --> b,c,1,d*h
        if not x_h.is_contiguous():
            x_h.contiguous()
        x_w = torch.mean(x, dim=3, keepdim=True).permute(0, 1, 3, 2,4).view(b,c,1,d*w)#b,c,d,1,w -->b,c,1,d*w
        if not x_w.is_contiguous():
            x_w.contiguous()

        x_cat_conv_relu = self.relu(self.bn(self.conv_1x1(torch.cat((x_d,x_h, x_w), 3))))

        x_cat_conv_split_d, x_cat_conv_split_h, x_cat_conv_split_w = x_cat_conv_relu.split([h*w,d*h, d*w], 3)
        #b,c,1,h*w,b,c,1,d*h
        s_d = self.sigmoid_d(self.F_d(x_cat_conv_split_d.view(b,c_out,1,h,w).contiguous()))
        s_h = self.sigmoid_h(self.F_h(x_cat_conv_split_h.view(b,c_out,1,d,h).permute(0, 1, 3, 4, 2).contiguous()))
        s_w = self.sigmoid_w(self.F_w(x_cat_conv_split_w.view(b,c_out,1,d,w).permute(0, 1, 3, 2, 4).contiguous()))

        out = x * s_d.expand_as(x) * s_h.expand_as(x) * s_w.expand_as(x)
        return out


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
        self.channel = CA_Block(planes).to("cuda:0")
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


def ca_resnet18(**kwargs):
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
    temp=torch.randn([1,1,160,160,160]).to("cuda:0")
    model = eca_resnet18(num_seg_classes=3).to("cuda:0")
    model(temp)
    # print(y1.shape)
    # print(y2.shape)
    # for name, module in model.named_children():
    #         print(name,module)

