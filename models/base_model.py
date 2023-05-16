### Copyright (C) 2017 NVIDIA Corporation. All rights reserved. 
### Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).
import os
import torch
import sys

class BaseModel(torch.nn.Module):
    def name(self):          #类名
        return 'BaseModel'

    def initialize(self, opt):   #初始化
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor  #数据迁移到GPU
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):   #添加属性到self
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def get_image_paths(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_visuals(self):
        return self.input

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass

    # helper saving function that can be used by subclasses
    def save_network(self, network, network_label, epoch_label, gpu_ids):
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)    #某时期的网络的名称
        save_path = os.path.join(self.save_dir, save_filename)
        torch.save(network.cpu().state_dict(), save_path)    #保存模型（状态字典）
        if len(gpu_ids) and torch.cuda.is_available():   #GPU的CUDA可用，加速
            network.cuda()

    # helper loading function that can be used by subclasses
    def load_network(self, network, network_label, epoch_label, save_dir=''):        
        save_filename = '%s_net_%s.pth' % (epoch_label, network_label)
        if not save_dir:     #若为空，使用默认地址
            save_dir = self.save_dir
        save_path = os.path.join(save_dir, save_filename)   
        print(save_path)     
        if not os.path.isfile(save_path):     #预加载网络模型不存在
            print('%s not exists yet!' % save_path)
            if network_label == 'G':
                raise('Generator must exist!')
        else:                                #加载预训练模型
            #network.load_state_dict(torch.load(save_path))
            try:
                network.load_state_dict(torch.load(save_path))
            except:                    #预训练模型参数与网络不匹配
                pretrained_dict = torch.load(save_path)    #预训练模型状态字
                model_dict = network.state_dict()         #网络状态字
                try:   #多
                    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}  #仅使用网络中有的部分状态字
                    network.load_state_dict(pretrained_dict)   #更新
                    if self.opt.verbose:
                        print('Pretrained network %s has excessive layers; Only loading layers that are used' % network_label)
                except:   #少
                    print('Pretrained network %s has fewer layers; The following are not initialized:' % network_label)
                    for k, v in pretrained_dict.items():                      
                        if v.size() == model_dict[k].size():   #已有参数，size相同则更新
                            model_dict[k] = v

                    if sys.version_info >= (3,0):  #Python版本 3.0以上
                        not_initialized = set()   #未更新 集合
                    else:                       #2.0
                        from sets import Set
                        not_initialized = Set()                    

                    for k, v in model_dict.items():
                        if k not in pretrained_dict or v.size() != pretrained_dict[k].size():
                            not_initialized.add(k.split('.')[0])
                    
                    print(sorted(not_initialized))
                    network.load_state_dict(model_dict)       #加载更新后状态字

    def update_learning_rate(self):
        pass
