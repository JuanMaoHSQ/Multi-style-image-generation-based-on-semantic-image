### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import numpy as np
import torch
import torch.nn as nn
import os
from torch.autograd import Variable
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import torch.nn.functional as F
import torchsummary
#liby001

class TVLoss(nn.Module):
    def __init__(self, tv_loss_weight=0.5):
        super(TVLoss, self).__init__()
        self.tv_loss_weight = tv_loss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self.tensor_size(x[:, :, 1:, :])
        count_w = self.tensor_size(x[:, :, :, 1:])
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.tv_loss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size
    @staticmethod
    def tensor_size(t):
        return t.size()[1] * t.size()[2] * t.size()[3]


    


class Pix2PixHDModel(BaseModel):
    def name(self):
        return 'Pix2PixHDModel'

    def init_loss_filter(self, use_gan_feat_loss, use_vgg_loss):  # 初始损失过滤器（GAN损失，鉴别器特征损失，VGG特征损失，鉴别器真，鉴别器假）
        flags = (True, use_gan_feat_loss, use_vgg_loss, True, True,True)  # 与下面五项对应

        def loss_filter(g_gan, g_gan_feat, g_vgg, d_real, d_fake,g_style):
            return [l for (l, f) in zip((g_gan, g_gan_feat, g_vgg, d_real, d_fake,g_style), flags) if f]

        return loss_filter

    def initialize(self, opt):  # 初始化模型
        BaseModel.initialize(self, opt)  # 参数初始化
        if opt.resize_or_crop != 'none' or not opt.isTrain:  # when training at full res this causes OOM  图片不处理或非训练
            torch.backends.cudnn.benchmark = True
        self.isTrain = opt.isTrain
        self.use_features = opt.instance_feat or opt.label_feat  # 实例特征或标签特征
        self.gen_features = self.use_features and not self.opt.load_features  # 使用特征，且非加载的预计算特征
        input_nc = opt.label_nc if opt.label_nc != 0 else opt.input_nc  # 输入图像通道为输入标签通道数（若不为0） ，否则为输入的输入图像通道数

        ##### define networks
        # Generator network
        netG_input_nc = input_nc
        if not opt.no_instance:  # 若使用实例图  生成器的输入图像的通道数+1
            netG_input_nc += 1
        if self.use_features:  # 若使用特征  。。。+编码特征的矢量长度
            netG_input_nc += opt.feat_num
        self.netG = networks.define_G(netG_input_nc, opt.output_nc, opt.ngf, opt.netG,
                                      opt.n_downsample_global, opt.n_blocks_global, opt.n_local_enhancers,
                                      opt.n_blocks_local, opt.norm, gpu_ids=self.gpu_ids)  # 根据opt.netG决定生成器的模型

        # Discriminator network
        if self.isTrain:  # 训练
            use_sigmoid = opt.no_lsgan
            netD_input_nc = input_nc + opt.output_nc  # 鉴别器的输入 为 输入与输出（生成器）的总和
            if not opt.no_instance:  # 若使用实例图，再加1
                netD_input_nc += 1
            self.netD = networks.define_D(netD_input_nc, opt.ndf, opt.n_layers_D, opt.norm, use_sigmoid,
                                          opt.num_D, not opt.no_ganFeat_loss, gpu_ids=self.gpu_ids)  # 定义多尺度鉴别器

        ### Encoder network
        if self.gen_features:
            self.netE = networks.define_G(opt.output_nc, opt.feat_num, opt.nef, 'encoder',
                                          opt.n_downsample_E, norm=opt.norm, gpu_ids=self.gpu_ids)
        if self.opt.verbose:  # 显示详细信息
            print('---------- Networks initialized -------------')

        # load networks
        if not self.isTrain or opt.continue_train or opt.load_pretrain:
            pretrained_path = '' if not self.isTrain else opt.load_pretrain  # 若为非训练，为空；否则为加载预训练模型的指定位置  ？？？
            self.load_network(self.netG, 'G', opt.which_epoch, pretrained_path)  # 加载生成器
            if self.isTrain:
                self.load_network(self.netD, 'D', opt.which_epoch, pretrained_path)  # 若是训练，加载鉴别器
            if self.gen_features:
                self.load_network(self.netE, 'E', opt.which_epoch, pretrained_path)  # 使用非预计算特征，加载编码器

        # set loss functions and optimizers
        if self.isTrain:
            if opt.pool_size > 0 and (len(self.gpu_ids)) > 1:
                raise NotImplementedError("Fake Pool Not Implemented for MultiGPU")
            self.fake_pool = ImagePool(opt.pool_size)  # 创建一个fake的缓冲池
            self.old_lr = opt.lr  # adam优化算法的初始学习率

            # define loss functions
            self.loss_filter = self.init_loss_filter(not opt.no_ganFeat_loss,
                                                     not opt.no_vgg_loss)  # 根据是否使用鉴别器特征损失，VGG损失 初始化

            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionFeat = torch.nn.L1Loss()  # 取预测值和真实值的绝对误差的平均数
            self.sig = nn.Sigmoid()
            self.l1_loss = nn.SmoothL1Loss()
            #self.tv_loss = TVLoss()
            #style_loss
            self.styleVGG=networks.Styleloss(self.gpu_ids)


            if not opt.no_vgg_loss:  # 使用VGG损失
                self.criterionVGG = networks.VGGLoss(self.gpu_ids)

            # Names so we can breakout loss
            self.loss_names = self.loss_filter('G_GAN', 'G_GAN_Feat', 'G_VGG', 'D_real', 'D_fake','G_style')

            # initialize optimizers  初始化优化器
            # optimizer G
            if opt.niter_fix_global > 0:  # 只训练最优局部增强器的时期数量（最后一个？？）
                import sys
                if sys.version_info >= (3, 0):  # 3.0以上Python版本
                    finetune_list = set()
                else:
                    from sets import Set
                    finetune_list = Set()

                params_dict = dict(self.netG.named_parameters())  # 生成器网络的参数字典（参数名，参数值）
                params = []
                for key, value in params_dict.items():  # 其单独训练的时期和参数名
                    if key.startswith('model' + str(opt.n_local_enhancers)):
                        params += [value]
                        finetune_list.add(key.split('.')[0])
                print(
                    '------------- Only training the local enhancer network (for %d epochs) ------------' % opt.niter_fix_global)
                print('The layers that are finetuned are ', sorted(finetune_list))
            else:
                params = list(self.netG.parameters())  # 参数值的列表
            if self.gen_features:
                params += list(self.netE.parameters())  # 使用非预计算特征，则增加编码器网络的参数
            self.optimizer_G = torch.optim.Adam(params, lr=opt.lr, betas=(opt.beta1, 0.999))  # adam优化算法（参数列表，初始学习率，动量项）

            # optimizer D
            params = list(self.netD.parameters())  # 鉴别器同样优化
            self.optimizer_D = torch.optim.Adam(params, lr=opt.lr * 0.3, betas=(opt.beta1, 0.999))

    def encode_input(self, label_map, inst_map=None, real_image=None, feat_map=None, infer=False):
        if self.opt.label_nc == 0:  # 输入标签通道数为0
            input_label = label_map.data.cuda(self.opt.liby[1])  # 标签图数据加载到GPU
        else:
            # create one-hot vector for label map
            size = label_map.size()
            oneHot_size = (size[0], self.opt.label_nc, size[2], size[3])  # batchsize ，输入channels ，图highth ，图width
            input_label = torch.FloatTensor(torch.Size(oneHot_size)).zero_()  # 创建该尺寸的tensor ，并填0
            input_label = input_label.cuda(self.opt.liby[1])
            input_label = input_label.scatter_(1, label_map.data.long().cuda(self.opt.liby[1]), 1.0)  # ？？
            if self.opt.data_type == 16:
                input_label = input_label.half()  # 若数据类型为16，转为浮点16位

        # get edges from instance map
        if not self.opt.no_instance:
            inst_map = inst_map.data.cuda(self.opt.liby[1])  # 实例图数据加载到GPU
            edge_map = self.get_edges(inst_map)
            input_label = torch.cat((input_label, edge_map), dim=1)  # 实例图张量 与 边张量拼接获得输入标签张量
        # input_label = Variable(input_label, volatile=infer)
        input_label = Variable(input_label)  # 张量转为variable类型

        # real images for training
        if real_image is not None:
            real_image = Variable(real_image.data.cuda(self.opt.liby[1]))

        # instance map for feature encoding
        if self.use_features:  # 使用特征
            # get precomputed feature maps 加载预计算特征图
            if self.opt.load_features:
                feat_map = Variable(feat_map.data.cuda(self.opt.liby[1]))
            if self.opt.label_feat:  # 输入中增加编码标签特征
                inst_map = label_map.cuda(self.opt.liby[1])

        return input_label, inst_map, real_image, feat_map

    def discriminate(self, input_label, test_image, use_pool=False):  # 鉴别，若使用缓冲池，则将拼接后数据经缓冲池后输入到鉴别器；否则，直接输入
        input_concat = torch.cat((input_label, test_image.detach()), dim=1)  # 输入标签张量，测试图片张量拼接
        if use_pool:
            fake_query = self.fake_pool.query(input_concat)
            fake_query = fake_query.cuda(self.opt.liby[0])
            return self.netD.forward(fake_query)
        else:
            input_concat = input_concat.cuda(self.opt.liby[0])
            return self.netD.forward(input_concat)

    def forward(self, label, inst, image, feat, infer=False):  # 返回损失值，或生成图像（可无）
        # Encode Inputs 编码输入数据
        input_label, inst_map, real_image, feat_map = self.encode_input(label, inst, image, feat)
        #print("label:",input_label.shape,"inst:",inst_map.shape,"feat:",feat.shape)
        #print('# $$$$$$$$$$$$$$$$$###########################$$$$$4 discriminator parameters:', sum(param.numel() for param in self.netG.parameters()))
        # Fake Generation
        if self.use_features:
            if not self.opt.load_features:  # 不加载预计算特征图，则输入与特征图经过编码器网络处理生成特征图
                feat_map = self.netE.forward(real_image, inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)  # 输入标签张量 与 特征图张量 拼接
        else:
            input_concat = input_label
        input_label = input_label.cuda(self.opt.liby[0])
        style_image=feat.cuda(self.opt.liby[0])
        real_image = real_image.cuda(self.opt.liby[0])
        fake_image = self.netG.forward(input_concat,feat,real_image)  # 输入标签经生成网络，生成假的图像
        fake_image = fake_image.cuda(self.opt.liby[0])

        # Fake Detection and Loss 假检测（生成图像）
        pred_fake_pool = self.discriminate(input_label, fake_image, use_pool=True)  # 鉴别生成图像
        loss_D_fake = self.criterionGAN(pred_fake_pool, False)  # 生成图像的假D损失函数
        
        # Real Detection and Loss    真检测 （真实图像）
        pred_real = self.discriminate(input_label, real_image)  # 鉴别真实图像
        loss_D_real = self.criterionGAN(pred_real, True)  # 真实图像的D损失
        # loss_D_real = self.criterionGAN(pred_dis,True)
        # GAN loss (Fake Passability Loss)   真检测（生成图像）
        pred_fake = self.netD.forward(torch.cat((input_label, fake_image), dim=1))        
        loss_G_GAN = self.criterionGAN(pred_fake, True)  # G的损失       
        # GAN feature matching loss
        loss_G_GAN_Feat = 0
        if not self.opt.no_ganFeat_loss:  # 使用鉴别器特征匹配损失
            feat_weights = 4.0 / (self.opt.n_layers_D + 1)  # 特征权重
            D_weights = 1.0 / self.opt.num_D  # 鉴别器权重
            for i in range(self.opt.num_D):  # 每个鉴别器遍历
                for j in range(len(pred_fake[i]) - 1):  # 每个鉴别器的结果逐层
                    loss_G_GAN_Feat += D_weights * feat_weights * self.criterionFeat(pred_fake[i][j],pred_real[i][j].detach()) * self.opt.lambda_feat  # 特征匹配损失权重
        # VGG feature matching loss 使用VGG损失
        loss_G_VGG = 0
        loss_G_style=0
        if self.opt.style_loss==1:
            loss_G_style=self.styleVGG(fake_image,style_image.detach())* self.opt.lambda_feat*0.0
           #print("####################################1.2")
        loss_G_VGG = self.criterionVGG(fake_image, real_image) * self.opt.lambda_feat
        # Only return the fake_B image if necessary to save BW  损失，若有需要，还可返回生成图像
        return [self.loss_filter(loss_G_GAN, loss_G_GAN_Feat, loss_G_VGG, loss_D_real, loss_D_fake,loss_G_style),
                None if not infer else fake_image]

    def inference(self, label, inst,feat, image=None):  # 返回生成图像
        # Encode Inputs        将输入参数转为variable后，编码输入
        image = Variable(image) if image is not None else None
        input_label, inst_map, real_image, _ = self.encode_input(Variable(label), Variable(inst), image, infer=True)

        # Fake Generation
        if self.use_features:  # 若使用特征，则将输入标签与实例图拼接；否则，直接输入标签作为生成器网络输入
            if self.opt.use_encoded_image:  # 使用编码图像
                # encode the real image to get feature map
                feat_map = self.netE.forward(real_image, inst_map)
            else:  # 不用编码图像
                # sample clusters from precomputed features
                feat_map = self.sample_features(inst_map)
            input_concat = torch.cat((input_label, feat_map), dim=1)
        else:
            input_concat = input_label

        if torch.__version__.startswith('0.4'):  # 0.4pytorch版本
            with torch.no_grad():  # 为了防止跟踪历史（和使用内存），包装代码块  模型可能有可训练参数，但我们并不需要整个模型的所有梯度。
                fake_image = self.netG.forward(input_concat,feat)
        else:
            fake_image = self.netG.forward(input_concat,feat)
        return fake_image

    def sample_features(self, inst):  # 读取与计算特征簇，并从其中随机采样，返回采样特征图张量
        # read precomputed feature clusters
        cluster_path = os.path.join(self.opt.checkpoints_dir, self.opt.name, self.opt.cluster_path)
        features_clustered = np.load(cluster_path, encoding='latin1').item()

        # randomly sample from the feature clusters
        inst_np = inst.cpu().numpy().astype(int)
        feat_map = self.Tensor(inst.size()[0], self.opt.feat_num, inst.size()[2], inst.size()[3])  # 根据实例图参数创建特征图张量
        for i in np.unique(inst_np):  # 去掉重复，并从大到小排序
            label = i if i < 1000 else i // 1000  # 若大于等于1000，除1000向下取整
            if label in features_clustered:  # 在簇内
                feat = features_clustered[label]
                cluster_idx = np.random.randint(0, feat.shape[0])  # 特征矩阵第一维的长度内随机取值

                idx = (inst == int(i)).nonzero()  # 不同的位置
                for k in range(self.opt.feat_num):  # 矢量长度内
                    feat_map[idx[:, 0], idx[:, 1] + k, idx[:, 2], idx[:, 3]] = feat[cluster_idx, k]
        if self.opt.data_type == 16:
            feat_map = feat_map.half()
        return feat_map

    def encode_features(self, image, inst):  # 返回编码特征列表   处理过程？？
        # image = Variable(image.cuda(), volatile=True)
        image = Variable(image.cuda(self.opt.liby[1]))
        feat_num = self.opt.feat_num
        h, w = inst.size()[2], inst.size()[3]
        block_num = 32
        feat_map = self.netE.forward(image, inst.cuda(self.opt.liby[1]))  # 由输入图像和实例图 经 编码器 生成特征图
        inst_np = inst.cpu().numpy().astype(int)
        feature = {}
        for i in range(self.opt.label_nc):
            feature[i] = np.zeros((0, feat_num + 1))  # 生成x行y列的0矩阵   （0，矢量长度+1）
        for i in np.unique(inst_np):
            label = i if i < 1000 else i // 1000
            idx = (inst == int(i)).nonzero()  # 不同的位置
            num = idx.size()[0]  # 第1维度数量（从1开始）
            idx = idx[num // 2, :]  # 取该维度中位的信息
            val = np.zeros((1, feat_num + 1))
            for k in range(feat_num):
                val[0, k] = feat_map[idx[0], idx[1] + k, idx[2], idx[3]].data[0]
            val[0, feat_num] = float(num) / (h * w // block_num)
            feature[label] = np.append(feature[label], val, axis=0)  # 列表拼接  为0，列相同，放在下方  ；为1，行相同，放在右边
        return feature

    def get_edges(self, t):  # ？？？
        edge = torch.ByteTensor(t.size()).zero_()  # 创建尺寸相同全为0的字节张量
        edge = edge.cuda(self.opt.liby[1])
        edge[:, :, :, 1:] = (edge[:, :, :, 1:] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte())

        edge[:, :, :, :-1] = edge[:, :, :, :-1] | (t[:, :, :, 1:] != t[:, :, :, :-1]).byte()
        edge[:, :, 1:, :] = edge[:, :, 1:, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        edge[:, :, :-1, :] = edge[:, :, :-1, :] | (t[:, :, 1:, :] != t[:, :, :-1, :]).byte()
        if self.opt.data_type == 16:
            return edge.half()
        else:
            return edge.float()

    def save(self, which_epoch):  # 保存某时期的网络模型
        self.save_network(self.netG, 'G', which_epoch, self.gpu_ids)
        self.save_network(self.netD, 'D', which_epoch, self.gpu_ids)
        if self.gen_features:
            self.save_network(self.netE, 'E', which_epoch, self.gpu_ids)

    def update_fixed_params(self):
        # after fixing the global generator for a number of iterations, also start finetuning it 生成器多次迭代后微调
        params = list(self.netG.parameters())
        if self.gen_features:
            params += list(self.netE.parameters())
        self.optimizer_G = torch.optim.Adam(params, lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
        if self.opt.verbose:
            print('------------ Now also finetuning global generator -----------')

    def update_learning_rate(self):  # 更新Adam学习率
        lrd = self.opt.lr / self.opt.niter_decay
        lr = self.old_lr - lrd
        for param_group in self.optimizer_D.param_groups:
            param_group['lr'] = lr * 0.3
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lr
        if self.opt.verbose:
            print('update learning rate: %f -> %f' % (self.old_lr, lr))
        self.old_lr = lr


class InferenceModel(Pix2PixHDModel):
    def forward(self, inp):
        label, inst = inp
        return self.inference(label, inst,feat)


