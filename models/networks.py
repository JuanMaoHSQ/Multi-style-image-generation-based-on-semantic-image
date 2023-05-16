### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import torch
import torch.nn as nn
import torchsnooper
import functools
from torch.autograd import Variable
import numpy as np
from .base_model import BaseModel
from . import spectral as sp
from options.train_options import TrainOptions
from .build_BiSeNet import BiSeNet as bsnet
from .style_encoder import BiSeNet as en_style
from  .architecture import  SPADEResnetBlock
opt = TrainOptions().parse()
from .sync_batchnorm import SynchronizedBatchNorm2d
import torch.nn.utils.spectral_norm as spectral_norm
from utils.core import transform as feature_wct
from .ContextualLoss import Contextual_Loss

# SpectralNorm=sp.SpectralNorm
###############################################################################
# Functions
###############################################################################
def weights_init(m):  # 权值初始化，使用net.apply()进行参数初始化
    classname = m.__class__.__name__
    # print(m)
    if classname.find('Conv') != -1:
        # print('weight : %s' % m.weight)
        # print(m)
        m.weight.data.normal_(0.0, 0.02)  # m.weight.data是卷积核参数, m.bias.data是偏置项参数
    elif classname.find('BatchNorm2d') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

def get_norm_layer(norm_type='instance'):  # 归一化
    if norm_type == 'batch':
        norm_layer = functools.partial(spectral_norm, affine=True)  # BN是同一个batch中所有样本的同一层特征图抽出来一起求mean和variance
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)  # IN只是对一个样本中的每一层特征图求mean和variance
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer

def define_G(input_nc, output_nc, ngf, netG, n_downsample_global=3, n_blocks_global=9, n_local_enhancers=1,
             n_blocks_local=3, norm='instance', gpu_ids=[]):  # 全局下采样3层，残差块9块  ；局部增强1个，残差块3个  ；归一化：实例
    norm_layer = get_norm_layer(norm_type=norm)
    if netG == 'global':
        netG = GlobalGenerator(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global, norm_layer)
    #elif netG == 'local':
        #netG = LocalEnhancer(input_nc, output_nc, ngf, n_downsample_global, n_blocks_global,
        #                     n_local_enhancers, n_blocks_local, norm_layer)
    elif netG == 'encoder':
        netG = Encoder(input_nc, output_nc, ngf, n_downsample_global, norm_layer)
    else:
        raise ('generator not implemented!')
    print(netG)
    # print(netG.modules())
    if len(gpu_ids) > 0:  # GPU可用，加速
        assert (torch.cuda.is_available())
        netG = netG.cuda(opt.liby[1])
    netG.apply(weights_init)  # 卷积，，，参数初始化
    return netG

def define_D(input_nc, ndf, n_layers_D, norm='instance', use_sigmoid=False, num_D=1, getIntermFeat=False,
             gpu_ids=[]):  # S型函数（默认false）
    norm_layer = get_norm_layer(norm_type=norm)
    netD = MultiscaleDiscriminator(input_nc, ndf, n_layers_D, norm_layer, use_sigmoid, num_D, getIntermFeat)
    print(netD)
    if len(gpu_ids) > 0:
        assert (torch.cuda.is_available())
        netD = netD.cuda(opt.liby[0])
    netD.apply(weights_init)
    return netD

def print_network(net):
    if isinstance(net, list):  # 是否为列表
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    # print(net)
    print('Total number of parameters: %d' % num_params)  # 总参数个数


##############################################################################
# Losses
##############################################################################


class GANLoss(nn.Module):

    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0,
                 tensor=torch.FloatTensor):
        super(GANLoss, self).__init__()
        self.real_label = target_real_label
        self.fake_label = target_fake_label
        self.real_label_var = None
        self.fake_label_var = None
        self.Tensor = tensor  # 张量类型
        if use_lsgan:  # 使用最小二乘GAN
            self.loss = nn.MSELoss()  # 平方损失函数。其计算公式是预测值和真实值之间的平方和的平均数。
        else:
            self.loss = nn.BCELoss()  # 二分类用的交叉熵

    def get_target_tensor(self, input, target_is_real):  # 获取目标的张量（variable格式）
        # print("get")
        target_tensor = None
        if target_is_real:
            create_label = ((self.real_label_var is None) or
                            (self.real_label_var.numel() != input.numel()))
            if create_label:
                real_tensor = self.Tensor(input.size()).fill_(self.real_label)  # 创建一个维度为dims，值为value的tensor对象
                self.real_label_var = Variable(real_tensor,
                                               requires_grad=False)  # Variable可以看作是对Tensor对象周围的一个薄包装，也包含了和张量相关的梯度，以及对创建它的函数的引用。
            target_tensor = self.real_label_var
        else:  # 为假
            create_label = ((self.fake_label_var is None) or
                            (self.fake_label_var.numel() != input.numel()))
            if create_label:
                fake_tensor = self.Tensor(input.size()).fill_(self.fake_label)
                self.fake_label_var = Variable(fake_tensor, requires_grad=False)
            target_tensor = self.fake_label_var
        return target_tensor

    def __call__(self, input, target_is_real):  # input 鉴别器结果，以及训练目标
        if isinstance(input[0], list):
            loss = 0
            for input_i in input:
                pred = input_i[-1]  # 列表最后一个
                target_tensor = self.get_target_tensor(pred, target_is_real)  # 训练目标张量
                # print("target",target_tensor)
                # print(pred.device,target_tensor.device)
                pred = pred.cuda(opt.liby[1])
                loss += self.loss(pred, target_tensor)  # 实际与目标 差（损失）
            return loss
        else:  # 只有一个列表
            target_tensor = self.get_target_tensor(input[-1], target_is_real)
            return self.loss(input[-1], target_tensor)

class VGGLoss(nn.Module):
    def __init__(self, gpu_ids):
        super(VGGLoss, self).__init__()
        vgg = Vgg19().cuda(opt.liby[0])
        self.vgg = vgg
        self.criterion = nn.SmoothL1Loss()  # L1Loss 计算方法很简单，取预测值和真实值的绝对误差的平均数即可。
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]

    def forward(self, x, y):  # ？？？
        # print("vgg ",x.device,y.device)
        x_vgg, y_vgg = self.vgg(x), self.vgg(y)  # 特征图？
        loss = 0
        for i in range(len(x_vgg)):
            loss += self.weights[i] * self.criterion(x_vgg[i], y_vgg[i].detach())  # detach截断反向传播的梯度流。
        loss = loss.cpu()
        loss = loss.cuda(opt.liby[1])
        # print("loss", loss.device)
        return loss

class Styleloss2(nn.Module):
    def __init__(self, gpu_ids):
        super(Styleloss2, self).__init__()
        vgg = Vgg19().cuda(opt.liby[0])
        self.vgg = vgg
        self.criterion = nn.MSELoss()  # L1Loss 计算方法很简单，取预测值和真实值的绝对误差的平均数即可。
        self.weights = [1.0, 0.75, 0.2 ,0.2,0.1]
        self.style_weight=1e6

    def gram_matrix(self,y):
            (b, ch, h, w) = y.size()
            features = y.view(b, ch, w * h)
            features_t = features.transpose(1, 2)
            gram = features.bmm(features_t) / (ch * h * w)
            return gram

    def forward(self, fake_image,style_image):  # ？？？
        #x:fake_image y:style_image z:context_image
        # print("vgg ",x.device,y.device)
        #gram_fake=self.gram_matrix(fake_image)
        #gram_style=self.gram_matrix(style_image)
        vgg_fake=self.vgg(fake_image)
        Vgg_style=self.vgg(style_image)
        style_loss = 0
        for i in range(len(vgg_fake)):
            style_loss += self.weights[i] * self.criterion(self.gram_matrix(vgg_fake[i]), self.gram_matrix(Vgg_style[i].detach()))  # detach截断反向传播的梯度流。
        loss=style_loss*self.style_weight
        loss = loss.cuda(opt.liby[1])
        # print("loss", loss.device)
        return loss

class Styleloss(nn.Module):
    def __init__(self, gpu_ids):
        super(Styleloss, self).__init__()
        layers = {
            "conv_1_1": 0.8,
            "conv_1_2": 0.8,
            "conv_2_1": 0.8,
            "conv_2_2": 0.7,
            "conv_3_1": 0.7,
            "conv_3_2": 0.7,
            "conv_4_1": 0.9,
            "conv_4_2": 0.9,
            "conv_5_1": 1.0,
            "conv_5_2": 1.0,
        }
        self.contex_loss = Contextual_Loss(layers, max_1d_size=64).cuda()

    def forward(self, fake_image,style_image):  # ？？？
        loss=self.contex_loss(fake_image,style_image) 
        return loss

##############################################################################
# Generator
##############################################################################
class LocalEnhancer(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsample_global=3, n_blocks_global=9,
                 n_local_enhancers=1, n_blocks_local=3, norm_layer=spectral_norm, padding_type='reflect'):
        super(LocalEnhancer, self).__init__()
        self.n_local_enhancers = n_local_enhancers

        ###### global generator model #####
        ngf_global = ngf * (2 ** n_local_enhancers)
        model_global = GlobalGenerator(input_nc, output_nc, ngf_global, n_downsample_global, n_blocks_global,
                                       norm_layer).model
        model_global = [model_global[i] for i in
                        range(len(model_global) - 3)]  # get rid of final convolution layers
        self.model = nn.Sequential(*model_global)

        ###### local enhancer layers #####
        for n in range(1, n_local_enhancers + 1):
            ### downsample
            ngf_global = ngf * (2 ** (n_local_enhancers - n))
            model_downsample = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf_global, kernel_size=7, padding=0),
                                norm_layer(ngf_global), nn.ReLU(True),
                                nn.Conv2d(ngf_global, ngf_global * 2, kernel_size=3, stride=2, padding=1),
                                norm_layer(ngf_global * 2), nn.ReLU(True)]
            ### residual blocks
            model_upsample = []
            for i in range(n_blocks_local):
                model_upsample += [ResnetBlock(ngf_global * 2, padding_type=padding_type, norm_layer=norm_layer)]

            ### upsample
            model_upsample += [
                nn.ConvTranspose2d(ngf_global * 2, ngf_global, kernel_size=3, stride=2, padding=1, output_padding=1),
                norm_layer(ngf_global), nn.ReLU(True)]

            ### final convolution
            if n == n_local_enhancers:
                model_upsample += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0),
                                   nn.Tanh()]

            setattr(self, 'model' + str(n) + '_1', nn.Sequential(*model_downsample))  # 给对象的属性赋值，若属性不存在，先创建再赋值。(对象，属性，值)
            setattr(self, 'model' + str(n) + '_2', nn.Sequential(*model_upsample))

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1],
                                       count_include_pad=False)  # 对信号的输入通道，提供2维的平均池化（池化窗口为3*3，步长为2，补0，计算平均池化时不包括填充的0）

    def forward(self, input):
        ### create input pyramid
        input_downsampled = [input]
        for i in range(self.n_local_enhancers):
            input_downsampled.append(self.downsample(input_downsampled[-1]))  # input_downsampled的最后一项数据经过平均池化的结果加入其中

        ### output at coarest level
        output_prev = self.model(input_downsampled[-1])  # 更新后的最后一项经过无最后一层的全局模型的结果
        ### build up one layer at a time
        for n_local_enhancers in range(1, self.n_local_enhancers + 1):
            model_downsample = getattr(self, 'model' + str(
                n_local_enhancers) + '_1')  # 获取对象object的属性或者方法，如果存在打印出来 若是对象的方法，返回值为内存地址，使用时后加括号
            model_upsample = getattr(self, 'model' + str(n_local_enhancers) + '_2')  # 相应层次局部增强器的上、下采样模型
            input_i = input_downsampled[self.n_local_enhancers - n_local_enhancers]
            output_prev = model_upsample(
                model_downsample(input_i) + output_prev)  # 将相应局部增强网络的下采样结果与无最后层的全局结果相加作为后续残差块的输入
        return output_prev

class GlobalGenerator(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=64, n_downsampling=3, n_blocks=9, norm_layer=spectral_norm,
                 padding_type='reflect'):
        assert (n_blocks >= 0)
        super(GlobalGenerator, self).__init__()  # 继承父类的初始化
        activation = nn.ReLU(True)  # ReLU函数有个inplace参数，如果设为True,它会把输出直接覆盖到输入中，这样可以节省内反向传播的梯度。

        # model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, 64, kernel_size=7, padding=0), norm_layer(ngf), activation]  #镜像填充  对由多个输入平面组成的输入信号进行二维卷积  归一化  激活函数

        """方案0 ：defeat """
        # attention1 = SelfAttention(ngf,'relu')
        model = [bsnet(512, 'resnet18',input_nc)]
        self.enstyle=en_style(512,'resnet18',3)
        self.spade1 = SPADEResnetBlock(512,256, 'spectral')
        #self.up2 = SPADEResnetBlock(256,120, 'spectral')
        self.spade2 = SPADEResnetBlock(256,128, 'spectral')
        self.spade3 = SPADEResnetBlock(128, 128, 'spectral')
        self.up1=nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.up2=nn.ConvTranspose2d(128, 128, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.vgg = Vgg_style().cuda(opt.liby[0])
        ### resnet blocks
        mult = 2 ** n_downsampling
        models = []
        ### upsample
        #for i in range(3,4):
        #    mult = 2 ** (n_downsampling - i)
        #    models += [spectral_norm(nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=2, padding=1,output_padding=1)),
        #                #sp.SpectralNorm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1)),
        #               #norm_layer(int(ngf * mult / 2))
        #               activation]  # 二维转置卷积操作 也是3层

        #models += [nn.ReflectionPad2d(3), spectral_norm(nn.Conv2d(int(ngf/2), output_nc, kernel_size=7, padding=0)), nn.Tanh()]  # 双曲正切激活函数
        #origin
        for i in range(2, 3):
            mult = 2 ** (n_downsampling - i)
            models += [spectral_norm(nn.ConvTranspose2d(int(ngf * mult / 2), int(ngf * mult / 4), kernel_size=3, stride=2, padding=1,
                                          output_padding=1)),
                       # sp.SpectralNorm(nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1, output_padding=1)),
                       #norm_layer(int(ngf * mult / 2)),
                       activation]  # 二维转置卷积操作 也是3层

        models += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]  # 双曲正切激活函数
        #self.model = nn.Sequential(*model)  # *代表一个参数list
        self.models = nn.Sequential(*models)
        self.up = nn.Upsample(scale_factor=2)
    def forward(self, input,style,real):
        #print("intput",input.shape)
        feats=self.vgg(style)
        x=self.vgg(real)
        #self.model = self.model.cuda(opt.liby[1])
        self.enstyle=self.enstyle.cuda(opt.liby[1])
        self.models = self.models.cuda(opt.liby[1])
        input = input.cuda(opt.liby[1])
        #x = self.model(input)
        #x=self.up(x)

        if opt.style_loss==1:
            z=self.enstyle(x,feats,style)
            #print(z.shape)
            #z = feature_wct(x.detach(),feats[2],1)
        else:
            z=x[3]
        z=self.spade1(z,real)
        #z = feature_wct(x.detach(),feats[1],1)
        z=self.spade2(z,real)
        #z = feature_wct(x.detach(),feats[0],1)
        z=self.spade3(z,real)
        z=self.up(z)
        #x=self.up(x)
        z=self.models(z)        
        return z  # 将输入放到模型中应

# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type, norm_layer, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':  # 镜像填充
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':  # 复制填充
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':  # 0，补0操作在卷积建立中执行
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim),
                       activation]  # 卷积核3*3
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        '''fangan 5  no better
        attention_resnet = SelfAttention(dim, 'relu')
        conv_block += [attention_resnet]
        '''

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):  # 数据经过卷积块（残差块）
        out = x + self.conv_block(x)
        return out

class Encoder(nn.Module):
    def __init__(self, input_nc, output_nc, ngf=32, n_downsampling=4, norm_layer=spectral_norm):
        super(Encoder, self).__init__()
        self.output_nc = output_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
                 norm_layer(ngf), nn.ReLU(True)]
        ### downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1),
                      norm_layer(ngf * mult * 2), nn.ReLU(True)]

        ### upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1),
                      norm_layer(int(ngf * mult / 2)), nn.ReLU(True)]

        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0), nn.Tanh()]
        self.model = nn.Sequential(*model)

    def forward(self, input, inst):  # 数据经过encoder模型，并
        outputs = self.model(input)

        # instance-wise average pooling
        outputs_mean = outputs.clone()  #
        inst_list = np.unique(inst.cpu().numpy().astype(int))  # 去除数组中的重复数字，并进行排序之后输出   转为整型（小数部分截掉）
        for i in inst_list:
            for b in range(input.size()[0]):  # ？？  宽
                indices = (inst[b:b + 1] == int(i)).nonzero()  # n x 4        nonzero() 非0值位置列表
                for j in range(self.output_nc):
                    output_ins = outputs[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]]  # ？？？
                    mean_feat = torch.mean(output_ins).expand_as(output_ins)
                    outputs_mean[indices[:, 0] + b, indices[:, 1] + j, indices[:, 2], indices[:, 3]] = mean_feat
        return outputs_mean

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=spectral_norm,
                 use_sigmoid=False, num_D=3, getIntermFeat=False):
        super(MultiscaleDiscriminator, self).__init__()
        self.num_D = num_D
        self.n_layers = n_layers
        self.getIntermFeat = getIntermFeat

        for i in range(num_D):  # 生成3个鉴别器
            netD = NLayerDiscriminator(input_nc, ndf, n_layers, norm_layer, use_sigmoid, getIntermFeat)
            if getIntermFeat:
                for j in range(n_layers + 2):
                    setattr(self, 'scale' + str(i) + '_layer' + str(j), getattr(netD, 'model' + str(j)))
            else:
                setattr(self, 'layer' + str(i), netD.model)

        self.downsample = nn.AvgPool2d(3, stride=2, padding=[1, 1], count_include_pad=False)  # 平均池化 池化窗口3*3

    def singleD_forward(self, model, input):  # 单个鉴别器执行，若为中间结果获取，返回为列表；否则，为最终结果（列表，保证数据类型统一）
        if self.getIntermFeat:
            result = [input]
            for i in range(len(model)):
                result[-1] = result[-1].cuda(opt.liby[0])
                model[i] = model[i].cuda(opt.liby[0])
                result.append(model[i](result[-1]))

            # del result[4]  #删除注意力层的中间结果   需根据其模块位置调整
            return result[1:]
        else:
            return [model(input)]

    def forward(self, input):
        # print(input.shape)
        num_D = self.num_D
        result = []
        input_downsampled = input
        input_downsampled = input_downsampled.cuda(opt.liby[0])
        for i in range(num_D):  # 逐个鉴别器运行，并都将结果放入一个列表中
            if self.getIntermFeat:
                model = [getattr(self, 'scale' + str(num_D - 1 - i) + '_layer' + str(j)) for j in
                         range(self.n_layers + 2)]
            else:
                model = getattr(self, 'layer' + str(num_D - 1 - i))
            result.append(self.singleD_forward(model, input_downsampled))
            if i != (num_D - 1):  # 若不是最后一个鉴别器，则对数据进行一次池化
                input_downsampled = self.downsample(input_downsampled)
        return result

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=spectral_norm, use_sigmoid=False, getIntermFeat=False):
        super(NLayerDiscriminator, self).__init__()
        self.getIntermFeat = getIntermFeat
        self.n_layers = n_layers

        kw = 4
        padw = int(np.ceil((kw - 1.0) / 2))  # ceil：向上取整  2
        sequence = [[nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
                     nn.LeakyReLU(0.2, True)]]  # 4*4的卷积，步长2，    激活函数：RELU的变体

        nf = ndf
        for n in range(1, n_layers):  # 1,2
            nf_prev = nf
            nf = min(nf * 2, 512)
            sequence += [[
                # sp.SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=2, padding=padw)),
                #norm_layer(nf),
                nn.LeakyReLU(0.2, True)
            ]]  # 4*4的卷积   归一化  激活函数

        nf_prev = nf  # 更新   第3层 （步长为1）
        nf = min(nf * 2, 512)
        sequence += [[
            # sp.SpectralNorm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            spectral_norm(nn.Conv2d(nf_prev, nf, kernel_size=kw, stride=1, padding=padw)),
            #norm_layer(nf),
            nn.LeakyReLU(0.2, True)
        ]]

        sequence += [[nn.Conv2d(nf, 1, kernel_size=kw, stride=1, padding=padw)]]

        if use_sigmoid:
            sequence += [[nn.Sigmoid()]]

        if getIntermFeat:  # 获取中间特征
            for n in range(len(sequence)):
                setattr(self, 'model' + str(n), nn.Sequential(*sequence[n]))  # 分别建立各层模型
        else:
            sequence_stream = []
            for n in range(len(sequence)):
                sequence_stream += sequence[n]  # 合并为一个列表
            self.model = nn.Sequential(*sequence_stream)  # 建立总的组合模型

    def forward(self, input):  # 若要中间结果，则返回中间结果的列表；否则，直接返回该模型的最后结果
        if self.getIntermFeat:
            res = [input]
            for n in range(self.n_layers + 2):  # 加首尾两层
                model = getattr(self, 'model' + str(n))
                res.append(model(res[-1]))  # 列表最后一个模型处理后添加到列表中（中间结果）

            # del res[4]   # 删除注意力层的中间结果   需根据其模块位置调整
            return res[1:]  # 去掉原始未处理数据
        else:
            return self.model(input)

        ########################################################

# self-attention (zeng)
########################################################

class SelfAttention(nn.Module):
    # self-attention layer
    def __init__(self, in_dim, activation='relu'):
        super(SelfAttention, self).__init__()
        self.channel_in = in_dim
        self.activation = activation

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # g
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)  # f
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)  # h
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.gamma =torch.zeros(1)

        self.softmax = nn.Softmax(dim=-1)

    """
    input: 
        x:input feature maps(B X C X W X H)
    return: 
        out: self attention value + input feature
        attention : B X N X N(N = width*hight)
    """

    def forward(self, x):
        m_batchsize, c, width, height = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)  # new tensor(B X C X N)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)  # transpose check
        attention = self.softmax(energy)

        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, c, width, height)
        out = self.gamma * out + x
        return out  # , attention

from torchvision import models

class Vgg19(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features  # vgg19.features包含以正确的深度顺序对齐的序列
        self.slice1 = torch.nn.Sequential()  # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(1):  # 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
            #print("aaaaa ", vgg_pretrained_features[x])
        for x in range(1, 6):  # 2,3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
           # print("bbbbb ", vgg_pretrained_features[x])
        for x in range(6, 11):  # 4,5
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):  # 6,7,8,9
            #print("ccccc ", vgg_pretrained_features[x])
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):  # 10，11,12,13
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
            #print("dddd ", vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        X = X.cuda(opt.liby[1])
        #print(X.shape)
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        h_relu5 = self.slice5(h_relu4)
        out = [h_relu1, h_relu2, h_relu3, h_relu4, h_relu5]  # 不同深度的特征图
        return out
class Vgg_style(torch.nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg_style, self).__init__()
        vgg_pretrained_features = models.vgg19(pretrained=True).features  # vgg19.features包含以正确的深度顺序对齐的序列
        self.slice1 = torch.nn.Sequential()  # torch.nn.Sequential是一个Sequential容器，模块将按照构造函数中传递的顺序添加到模块中
        self.slice2 = torch.nn.Sequential()
        self.slice3 = torch.nn.Sequential()
        self.slice4 = torch.nn.Sequential()
        self.slice5 = torch.nn.Sequential()
        for x in range(1):  # 1
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(1, 6):  # 2,3
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(6, 11):  # 4,5
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(11, 20):  # 6,7,8,9
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        for x in range(20, 29):  # 10，11,12,13
            self.slice5.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False
    def forward(self, X):
        X = X.cuda(opt.liby[1])
        h_relu1 = self.slice1(X)
        h_relu2 = self.slice2(h_relu1)
        h_relu3 = self.slice3(h_relu2)
        h_relu4 = self.slice4(h_relu3)
        #print(h_relu3,h_relu3.requires_grad)
        out = [h_relu1,h_relu2, h_relu3, h_relu4]  # 不同深度的特征图
        return out
