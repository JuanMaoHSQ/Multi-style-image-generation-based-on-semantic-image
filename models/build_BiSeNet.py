import torch
from torch import nn
from .build_contextpath import build_contextpath
from .sync_batchnorm import SynchronizedBatchNorm2d
import warnings
import torch.nn.utils.spectral_norm as spectral_norm
warnings.filterwarnings(action='ignore')

class Spatial_path(nn.Module):
    def __init__(self,input_nc):
        super(Spatial_path, self).__init__()
        self.convblock1 = nn.Conv2d(input_nc, 64, kernel_size=3, stride=2, padding=1)
        self.convblock2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self.convblock3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.bn1 = SynchronizedBatchNorm2d(64)
        self.bn2 = SynchronizedBatchNorm2d(128)
        self.bn3 = SynchronizedBatchNorm2d(256)
        self.relu3 = nn.ReLU()

    def forward(self, input):
        x = self.convblock1(input)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.convblock2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.convblock3(x)
        x = self.bn3(x)
        x = self.relu3(x)

        return x


class AttentionRefinementModule(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.bn = SynchronizedBatchNorm2d(out_channels)
        self.sigmoid = nn.Sigmoid()
        self.in_channels = in_channels
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input):
        # global average pooling
        x = self.avgpool(input)
        assert self.in_channels == x.size(1), 'in_channels and out_channels should all be {}'.format(x.size(1))
        x = self.conv(x)
        # x = self.sigmoid(self.bn(x))
        x = self.sigmoid(x)
        # channels of input and x should be same
        x = torch.mul(input, x)
        return x


class FeatureFusionModule(torch.nn.Module):
    def __init__(self, num_classes, in_channels):
        super().__init__()
        # self.in_channels = input_1.channels + input_2.channels
        # resnet101 3328 = 256(from context path) + 1024(from spatial path) + 2048(from spatial path)
        # resnet18  1024 = 256(from context path) + 256(from spatial path) + 512(from spatial path)
        self.in_channels = in_channels

        self.convblock = nn.Conv2d(in_channels=self.in_channels, out_channels=num_classes, stride=1, kernel_size=3,
                                   padding=1)
        self.bn1 = SynchronizedBatchNorm2d(num_classes)
        self.relu1 = nn.ReLU()
        self.conv1 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(num_classes, num_classes, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=(1, 1))

    def forward(self, input_1, input_2):
        x = torch.cat((input_1, input_2), dim=1)
        assert self.in_channels == x.size(1), 'in_channels of ConvBlock should be {}'.format(x.size(1))
        feature = self.convblock(x)
        feature = self.bn1(feature)
        feature = self.relu1(feature)
        x = self.avgpool(feature)

        x = self.relu(self.conv1(x))
        x = self.sigmoid(self.conv2(x))
        x = torch.mul(feature, x)
        x = torch.add(x, feature)
        return x


class BiSeNet(torch.nn.Module):
    def __init__(self, num_classes, context_path,input_nc):
        super().__init__()
        # build spatial path
        self.saptial_path = Spatial_path(input_nc)
        self.up1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn1 = SynchronizedBatchNorm2d(256)
        self.relu1 = nn.ReLU()

        self.up2_1 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2_2 = nn.ConvTranspose2d(512, 512, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn2_1 = SynchronizedBatchNorm2d(512)
        self.relu2_1 = nn.ReLU()
        self.bn2_2 = SynchronizedBatchNorm2d(512)
        self.relu2_2 = nn.ReLU()

        # build context path
        self.context_path = build_contextpath(name=context_path,input_nc=input_nc)

        # build attention refinement module  for resnet 101
        if context_path == 'resnet101':
            self.attention_refinement_module1 = AttentionRefinementModule(1024, 1024)
            self.attention_refinement_module2 = AttentionRefinementModule(2048, 2048)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=1024, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=2048, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 3328)

        elif context_path == 'resnet18':
            # build attention refinement module  for resnet 18
            self.attention_refinement_module1 = AttentionRefinementModule(256, 256)
            self.attention_refinement_module2 = AttentionRefinementModule(512, 512)
            # supervision block
            self.supervision1 = nn.Conv2d(in_channels=256, out_channels=num_classes, kernel_size=1)
            self.supervision2 = nn.Conv2d(in_channels=512, out_channels=num_classes, kernel_size=1)
            # build feature fusion module
            self.feature_fusion_module = FeatureFusionModule(num_classes, 1024)
        else:
            print('Error: unspport context_path network \n')

        # build final convolution
        self.conv = nn.Conv2d(in_channels=num_classes, out_channels=num_classes, kernel_size=1)

        self.init_weight()

        self.mul_lr = []
        self.mul_lr.append(self.saptial_path)
        self.mul_lr.append(self.attention_refinement_module1)
        self.mul_lr.append(self.attention_refinement_module2)
        self.mul_lr.append(self.supervision1)
        self.mul_lr.append(self.supervision2)
        self.mul_lr.append(self.feature_fusion_module)
        self.mul_lr.append(self.conv)

    def init_weight(self):
        for name, m in self.named_modules():
            if 'context_path' not in name:
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
                elif isinstance(m, SynchronizedBatchNorm2d):
                    m.eps = 1e-5
                    m.momentum = 0.1
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)

    def forward(self, input):
        # output of spatial path
        sx = self.saptial_path(input)

        # output of context path
        cx1, cx2, tail = self.context_path(input)

        cx1 = self.attention_refinement_module1(cx1)

        cx2 = self.attention_refinement_module2(cx2)

        cx2 = torch.mul(cx2, tail)

        # upsampling

        cx1 = self.up1(cx1)
        cx1 = self.bn1(cx1)
        cx1 = self.relu1(cx1)

        # cx2 = torch.nn.functional.interpolate(cx2, size=sx.size()[-2:], mode='bilinear')
        cx2 = self.up2_1(cx2)
        cx2 = self.bn2_1(cx2)
        cx2 = self.relu2_1(cx2)
        cx2 = self.up2_2(cx2)
        cx2 = self.bn2_2(cx2)
        cx2 = self.relu2_2(cx2)

        cx = torch.cat((cx1, cx2), dim=1)
        # output of feature fusion module
        result = self.feature_fusion_module(sx, cx)
        # upsampling
        result = self.conv(result)
        # if self.training == True:
        #    return result, cx1_sup, cx2_sup
        return result   